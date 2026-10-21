"""
Validación de la lectura de tickets (Donut v3) campo a campo.

El ticket correcto (ground truth) se obtiene AUTOMÁTICAMENTE del texto que llevan dentro los PDF de
Mercadona (mismo parser que preparar_dataset.py, que además comprueba que la suma de líneas = TOTAL).
Donut, en cambio, lee la IMAGEN del ticket, como haría con una foto.

Cada ticket se clasifica según su relación con el entrenamiento, y las cifras se dan por separado:
  - nuevo:       nunca visto por el modelo (los PDF de validacion/tickets_nuevos/)  -> cifra principal
  - validacion:  los 14 tickets de validación del entrenamiento (no se entrenó con ellos, pero se usaron
                 para elegir la época)
  - train:       tickets de entrenamiento (solo comprueba que el modelo los aprendió; no es una medida)

Fotos (--fotos CARPETA): cada foto debe llamarse igual que su PDF (p. ej. "20260413 Mercadona 16,46 €.jpg")
y el PDF estar en esa carpeta o en las de PDF. Así la foto se compara con el texto real del ticket.

Métricas por ticket: fecha, total, nº de líneas, ticket perfecto, ticket "casi perfecto" (≤ 2 campos mal),
suma de líneas = total (coherencia interna, que la app usa para mandar a revisión).
Por línea: F1 de líneas completas, nombre exacto, CER del nombre (errores de carácter), cantidad, importe.
Nombres con el corrector: sin ayuda del PDF (vocabulario = catálogo + nombres de train), como en una foto.

Uso (desde backend/, con el venv; la primera vez tarda: ~20-60 s por ticket en CPU):
    python -m validacion.evaluar_donut                     # PDF de dataset-mercadona + tickets_nuevos
    python -m validacion.evaluar_donut --fotos validacion/fotos
    python -m validacion.generar_fotos_sinteticas            # fotos simuladas (leve/media/fuerte) de los PDF
    python -m validacion.evaluar_donut --solo nuevo validacion --fotos validacion/fotos_sinteticas/leve validacion/fotos_sinteticas/media validacion/fotos_sinteticas/fuerte
    python -m validacion.evaluar_donut --solo nuevo validacion   # para ir más rápido
Las lecturas se guardan en resultados/donut_lecturas.jsonl y no se repiten (usa --rehacer para repetirlas).
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
BACKEND = RAIZ.parent
sys.path.insert(0, str(BACKEND))

import pdfplumber  # noqa: E402

RESULTADOS = RAIZ / "resultados"
CACHE = RESULTADOS / "donut_lecturas.jsonl"
CARPETAS_PDF = [BACKEND / "dataset-mercadona", RAIZ / "tickets_nuevos"]
SPLIT = RAIZ / "split_entrenamiento.json"
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}

# ── Ground truth desde el texto del PDF (copia del parser de preparar_dataset.py) ──
PRECIO = r"-?\d+,\d{2}"
RE_ITEM = re.compile(r"^(-?\d+)\s+(.+)$")
RE_PESO = re.compile(r"^(\d+,\d{3})\s*kg\s+(\d+,\d{2})\s*€/kg\s+(" + PRECIO + r")$")
RE_FECHA = re.compile(r"(\d{2}/\d{2}/\d{4})")
RE_TOTAL = re.compile(r"^TOTAL \(€\)\s+(" + PRECIO + r")$")


def a_centimos(valor) -> int:
    try:
        return int(round(float(str(valor).replace("€", "").replace(".", "").replace(",", ".").strip()) * 100))
    except ValueError:
        return -10 ** 9


def parsear_ticket(texto: str) -> dict:
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    fecha, total, items = None, None, []
    en_productos, i = False, 0
    while i < len(lineas):
        linea = lineas[i]
        if fecha is None:
            m = RE_FECHA.search(linea)
            if m:
                fecha = m.group(1)
        if "Descripción" in linea and "Importe" in linea:     # "Descripción…" o, desde 2026, "Cnt. Descripción…"
            en_productos, i = True, i + 1
            continue
        m_total = RE_TOTAL.match(linea)
        if m_total:
            total = m_total.group(1)
            break
        if en_productos:
            m = RE_ITEM.match(linea)
            if not m:
                i += 1
                continue
            tokens = m.group(2).split(" ")
            precios = []
            while tokens and len(precios) < 2 and re.fullmatch(PRECIO, tokens[-1]):
                precios.insert(0, tokens.pop())
            item = {"cnt": m.group(1), "nm": " ".join(tokens).strip()}
            if not precios:
                m_peso = RE_PESO.match(lineas[i + 1]) if i + 1 < len(lineas) else None
                if not m_peso:
                    raise ValueError(f"línea de producto sin precio: {linea!r}")
                item.update(weight=f"{m_peso.group(1)} kg", unitprice=f"{m_peso.group(2)} €/kg", price=m_peso.group(3))
                i += 1
            elif len(precios) == 2:
                item["unitprice"], item["price"] = precios
            else:
                item["price"] = precios[0]
            items.append(item)
        i += 1
    if fecha is None or total is None or not items:
        raise ValueError("no se encontró fecha, total o productos")
    if sum(a_centimos(it["price"]) for it in items) != a_centimos(total):
        raise ValueError("la suma de importes no cuadra con el TOTAL")
    return {"date": fecha, "menu": items, "total": {"total_price": total}}


def gt_de_pdf(ruta: Path) -> dict:
    with pdfplumber.open(ruta) as doc:
        return parsear_ticket("\n".join(p.extract_text() or "" for p in doc.pages))


# ── Relación con el entrenamiento ──
def huella_gt(gt: dict) -> str:
    return json.dumps([gt["date"], gt["total"]["total_price"], [(m["nm"], m["price"]) for m in gt["menu"]]], ensure_ascii=False)


def splits_entrenamiento() -> dict:
    """huella del ticket (o nombre del PDF) -> "train" | "validacion", según split_entrenamiento.json
    (exportado del dataset_v3 con el que se entrenó en Colab: 77 train / 14 validación, semilla 42)."""
    res = {}
    for grupo, tickets in json.loads(SPLIT.read_text(encoding="utf-8")).items():
        for t in tickets:
            res[huella_gt({"date": t["date"], "total": {"total_price": t["total"]},
                           "menu": [{"nm": n, "price": p} for n, p in t["menu"]]})] = grupo
            if t.get("origen"):
                res["pdf:" + t["origen"]] = grupo
    return res


# ── Comparación ──
def norm(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().upper()


def menu(pred: dict) -> list:
    m = pred.get("menu", []) if isinstance(pred, dict) else []
    m = [m] if isinstance(m, dict) else m
    return [x for x in m if isinstance(x, dict)]


def total_pred(pred: dict):
    t = pred.get("total") if isinstance(pred, dict) else None
    return t.get("total_price") if isinstance(t, dict) else t


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def emparejar(gt_items: list, pred_items: list) -> list:
    """Empareja líneas leídas con líneas reales por orden y parecido del nombre (las sobrantes quedan solas)."""
    a, b = [norm(x["nm"]) for x in gt_items], [norm(x.get("nm")) for x in pred_items]
    pares = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("equal", "replace"):
            for k in range(max(i2 - i1, j2 - j1)):
                pares.append((i1 + k if i1 + k < i2 else None, j1 + k if j1 + k < j2 else None))
        elif op == "delete":
            pares += [(i, None) for i in range(i1, i2)]
        else:
            pares += [(None, j) for j in range(j1, j2)]
    return pares


def linea_clave(x: dict) -> tuple:
    return (norm(x.get("nm")), str(x.get("cnt", "")).strip(), a_centimos(x.get("price")),
            a_centimos(x["unitprice"].split()[0]) if x.get("unitprice") else None, norm(x.get("weight")) or None)


def comparar(gt: dict, pred: dict, nombres_corregidos: list = None) -> dict:
    g, p = gt["menu"], menu(pred)
    fecha_ok = norm(pred.get("date") if isinstance(pred, dict) else "")[:10] == gt["date"]
    total_ok = a_centimos(total_pred(pred)) == a_centimos(gt["total"]["total_price"])
    cg, cp = Counter(map(linea_clave, g)), Counter(map(linea_clave, p))
    comunes = sum((cg & cp).values())
    prec, rec = (comunes / len(p) if p else 0), comunes / len(g)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    pares = emparejar(g, p)
    lineas, campos_mal = [], 0
    for i, j in pares:
        x = g[i] if i is not None else None
        y = p[j] if j is not None else None
        nm_g, nm_p = norm(x["nm"]) if x else "", norm(y.get("nm")) if y else ""
        nm_c = norm(nombres_corregidos[j]) if (y is not None and nombres_corregidos) else nm_p
        fila = {
            "nombre_real": nm_g, "nombre_leido": nm_p, "nombre_corregido": nm_c,
            "solo_real": y is None, "solo_leida": x is None,
            "nombre_ok": bool(x and y and nm_g == nm_p), "nombre_corregido_ok": bool(x and y and nm_g == nm_c),
            "cer": levenshtein(nm_g, nm_p) / max(1, len(nm_g)) if x else 1.0,
            "cer_corregido": levenshtein(nm_g, nm_c) / max(1, len(nm_g)) if x else 1.0,
            "cantidad_ok": bool(x and y and str(x["cnt"]).strip() == str(y.get("cnt", "")).strip()),
            "importe_ok": bool(x and y and a_centimos(x["price"]) == a_centimos(y.get("price"))),
            "linea_ok": bool(x and y and linea_clave(x) == linea_clave(y)),
        }
        if x is None or y is None:
            campos_mal += 1
        else:
            campos_mal += sum(not fila[k] for k in ("nombre_ok", "cantidad_ok", "importe_ok")) + \
                          (linea_clave(x)[3:] != linea_clave(y)[3:])
        lineas.append(fila)
    campos_mal += (not fecha_ok) + (not total_ok)
    suma = sum(a_centimos(y.get("price")) for y in p)
    return {
        "fecha_ok": fecha_ok, "total_ok": total_ok, "n_lineas_ok": len(g) == len(p),
        "lineas_real": len(g), "lineas_leidas": len(p), "f1_lineas": f1,
        "perfecto": campos_mal == 0 and cg == cp, "casi_perfecto": campos_mal <= 2,
        "campos_mal": campos_mal, "suma_cuadra": bool(p) and suma == a_centimos(total_pred(pred)),
        "lineas": lineas,
    }


# ── Intervalos ──
def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(k, n, ic=False):
    if not n:
        return "—"
    s = f"{100 * k / n:.1f} % ({k}/{n})"
    if ic:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.1f}–{100 * hi:.1f}]"
    return s


# ── Programa ──
def recoger(args, splits) -> list:
    casos, vistos = [], set()
    for carpeta in CARPETAS_PDF:
        if not carpeta.exists():
            continue
        for pdf in sorted(carpeta.glob("*.pdf"), key=lambda p: ("(" in p.stem, p.name)):
            datos = pdf.read_bytes()
            h = hashlib.md5(datos).hexdigest()
            try:
                gt = gt_de_pdf(pdf)
            except Exception as e:
                print(f"⚠️ {pdf.name}: no se puede sacar el ticket correcto del PDF ({e}); se omite")
                continue
            if h in vistos or huella_gt(gt) in vistos:
                continue
            vistos |= {h, huella_gt(gt)}
            grupo = splits.get(huella_gt(gt)) or splits.get("pdf:" + pdf.name) or \
                ("nuevo" if carpeta.name == "tickets_nuevos" else "desconocido")
            casos.append({"archivo": pdf, "tipo": "pdf", "grupo": grupo, "gt": gt, "md5": h})
    pdfs_por_nombre = {c["archivo"].stem: c for c in casos}
    for carpeta in args.fotos or []:
        carpeta = Path(carpeta)
        tipo = f"sintética {carpeta.name}" if carpeta.parent.name == "fotos_sinteticas" else "foto"
        for foto in sorted(carpeta.iterdir()):
            if foto.suffix.lower() not in EXT_IMAGEN:
                continue
            ref = pdfs_por_nombre.get(foto.stem)
            if ref is None and (foto.with_suffix(".pdf")).exists():
                gt = gt_de_pdf(foto.with_suffix(".pdf"))
                ref = {"gt": gt, "grupo": splits.get(huella_gt(gt), "nuevo")}
            if ref is None:
                print(f"⚠️ {foto.name}: no hay un PDF con el mismo nombre; se omite")
                continue
            casos.append({"archivo": foto, "tipo": tipo, "grupo": ref["grupo"], "gt": ref["gt"],
                          "md5": hashlib.md5(foto.read_bytes()).hexdigest()})
    if args.solo:
        casos = [c for c in casos if c["grupo"] in args.solo]
    return casos


def preparar_corrector(splits):
    from categorizador import categorizador
    from corrector_nombres import CorrectorNombres
    corrector = CorrectorNombres(categorizador)
    # solo nombres de TRAIN: los de validación y los tickets nuevos no pueden estar en el vocabulario
    for t in json.loads(SPLIT.read_text(encoding="utf-8"))["train"]:
        corrector.añadir_nombres(t["nombres"])
    return corrector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fotos", nargs="*", help="carpetas con fotos de tickets (mismo nombre que su PDF); "
                    "las de fotos_sinteticas/<nivel> se marcan como «sintética <nivel>»")
    ap.add_argument("--solo", nargs="*", choices=["nuevo", "validacion", "train", "desconocido"])
    ap.add_argument("--rehacer", action="store_true", help="vuelve a leer con Donut aunque haya lectura guardada")
    args = ap.parse_args()

    RESULTADOS.mkdir(exist_ok=True)
    splits = splits_entrenamiento()
    casos = recoger(args, splits)
    print(f"🧾 {len(casos)} tickets: " + ", ".join(f"{k} {v}" for k, v in Counter(f"{c['tipo']}/{c['grupo']}" for c in casos).items()))

    cache = {}
    if CACHE.exists() and not args.rehacer:
        for linea in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            cache[r["md5"]] = r
    pendientes = [c for c in casos if c["md5"] not in cache]
    if pendientes:
        print(f"🍩 Leyendo {len(pendientes)} tickets con Donut (los ya leídos se toman de {CACHE.name})...")
        import server  # carga Donut exactamente como la aplicación
        with open(CACHE, "a", encoding="utf-8") as fh:
            for n, c in enumerate(pendientes, 1):
                t0 = time.perf_counter()
                try:
                    pred = server.process_receipt_donut(c["archivo"].read_bytes())
                except Exception as e:
                    pred = {"error": f"{type(e).__name__}: {e}"}
                r = {"md5": c["md5"], "archivo": c["archivo"].name, "pred": pred, "segundos": round(time.perf_counter() - t0, 2)}
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                cache[c["md5"]] = r
                print(f"  [{n}/{len(pendientes)}] {r['segundos']:5.1f} s  {c['archivo'].name}")

    from fechas_ticket import revisar_fecha
    from lectura_donut import normalizar_prediccion
    corrector = preparar_corrector(splits)
    filas, filas_lineas = [], []
    for c in casos:
        r = cache[c["md5"]]
        pred = normalizar_prediccion(r["pred"])      # misma limpieza que hace la aplicación
        corregidos = []
        for it in menu(pred):
            res = corrector.corregir(str(it.get("nm", "")))
            corregidos.append(res["nombre"] if res and res["automatica"] else str(it.get("nm", "")))
        cmp = comparar(c["gt"], pred, corregidos)
        fecha_leida = str(pred.get("date", "")) if isinstance(pred, dict) else ""
        rev = revisar_fecha(fecha_leida, c["archivo"].read_bytes() if c["tipo"] != "pdf" else b"")
        fecha_final = (rev.get("fecha_sugerida") or fecha_leida)[:10]
        filas.append({"archivo": c["archivo"].name, "tipo": c["tipo"], "grupo": c["grupo"], "segundos": r["segundos"],
                      **{k: v for k, v in cmp.items() if k != "lineas"},
                      "lectura_incompleta": bool(pred.get("truncada") or pred.get("campos_rotos") or not total_pred(pred)),
                      "truncada": bool(pred.get("truncada")),
                      "fecha_real": c["gt"]["date"], "fecha_leida": fecha_leida, "aviso_fecha": bool(rev.get("aviso")),
                      "fecha_tras_revision_ok": fecha_final == c["gt"]["date"],
                      "total_real": c["gt"]["total"]["total_price"], "total_leido": total_pred(pred)})
        for l in cmp["lineas"]:
            filas_lineas.append({"archivo": c["archivo"].name, "tipo": c["tipo"], "grupo": c["grupo"], **l})

    marca = datetime.now().strftime("%Y%m%d_%H%M")
    for nombre, datos in (("tickets", filas), ("lineas", filas_lineas)):
        if datos:
            with open(RESULTADOS / f"donut_{nombre}_{marca}.csv", "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(datos[0]), delimiter=";")
                w.writeheader()
                w.writerows(datos)
    md = informe(filas, filas_lineas)
    (RESULTADOS / f"resumen_donut_{marca}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"📄 Resultados en {RESULTADOS}")


def informe(filas, filas_lineas) -> str:
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["tipo"], f["grupo"])].append(f)
    lin = defaultdict(list)
    for l in filas_lineas:
        lin[(l["tipo"], l["grupo"])].append(l)
    tipos = ["pdf", "foto", "sintética leve", "sintética media", "sintética fuerte"]
    orden = sorted(grupos, key=lambda g: (tipos.index(g[0]) if g[0] in tipos else 9, g[0], ["nuevo", "validacion", "desconocido", "train"].index(g[1])))
    cab = [f"{t} · {g}" for t, g in orden]
    L = [f"# Validación de la lectura de tickets (Donut v3) — {datetime.now():%d/%m/%Y %H:%M}", "",
         "Ticket correcto sacado del texto del PDF. «train» solo comprueba que el modelo aprendió (no es una medida); "
         "la cifra válida es «nuevo» (y, con reservas, «validacion»). Entre corchetes, IC 95 % de Wilson.", "",
         "## Por ticket", "", "| Métrica | " + " | ".join(cab) + " |", "|---|" + "---|" * len(cab)]

    def fila(nombre, fn):
        L.append(f"| {nombre} | " + " | ".join(fn(grupos[g]) for g in orden) + " |")

    def cuenta(clave, ic=False):
        return lambda fs: pct(sum(bool(f[clave]) for f in fs), len(fs), ic)

    fila("Tickets", lambda fs: str(len(fs)))
    fila("**Ticket perfecto**", cuenta("perfecto", True))
    fila("Casi perfecto (≤ 2 campos mal)", cuenta("casi_perfecto"))
    fila("Fecha correcta", cuenta("fecha_ok"))
    fila("Fecha correcta tras la revisión de fechas", cuenta("fecha_tras_revision_ok"))
    fila("Total correcto", cuenta("total_ok"))
    fila("Nº de líneas correcto", cuenta("n_lineas_ok"))
    fila("Suma de líneas = total (coherente)", cuenta("suma_cuadra"))
    fila("Lectura cortada (ticket demasiado largo)", cuenta("truncada"))
    fila("F1 de líneas completas (media)", lambda fs: f"{100 * sum(f['f1_lineas'] for f in fs) / len(fs):.1f} %")
    fila("Segundos por ticket (media)", lambda fs: f"{sum(f['segundos'] for f in fs) / len(fs):.1f}")

    # ¿La comprobación "suma = total" detecta los tickets mal leídos? (la app los manda a revisar)
    L += ["", "### ¿Detecta la app los tickets mal leídos?", "",
          "| Grupo | Tickets con algún error | …de ellos que van a revisión (suma ≠ total, fecha dudosa o lectura incompleta) | Correctos enviados a revisión |",
          "|---|---|---|---|"]
    for g in orden:
        fs = grupos[g]
        malos = [f for f in fs if not f["perfecto"]]
        buenos = [f for f in fs if f["perfecto"]]
        det = sum((not f["suma_cuadra"]) or f["aviso_fecha"] or f["lectura_incompleta"] for f in malos)
        fal = sum((not f["suma_cuadra"]) or f["aviso_fecha"] or f["lectura_incompleta"] for f in buenos)
        L.append(f"| {g[0]} · {g[1]} | {len(malos)} | {pct(det, len(malos))} | {pct(fal, len(buenos))} |")

    L += ["", "## Por línea de producto", "", "| Métrica | " + " | ".join(cab) + " |", "|---|" + "---|" * len(cab)]

    def fila_l(nombre, fn):
        L.append(f"| {nombre} | " + " | ".join(fn([l for l in lin[g] if not l['solo_leida']]) for g in orden) + " |")

    fila_l("Líneas reales", lambda ls: str(len(ls)))
    fila_l("Línea completa correcta", lambda ls: pct(sum(l["linea_ok"] for l in ls), len(ls)))
    fila_l("Nombre exacto", lambda ls: pct(sum(l["nombre_ok"] for l in ls), len(ls)))
    fila_l("Nombre exacto tras el corrector", lambda ls: pct(sum(l["nombre_corregido_ok"] for l in ls), len(ls)))
    fila_l("CER del nombre (media)", lambda ls: f"{100 * sum(l['cer'] for l in ls) / max(1, len(ls)):.2f} %")
    fila_l("CER tras el corrector", lambda ls: f"{100 * sum(l['cer_corregido'] for l in ls) / max(1, len(ls)):.2f} %")
    fila_l("Cantidad correcta", lambda ls: pct(sum(l["cantidad_ok"] for l in ls), len(ls)))
    fila_l("Importe correcto", lambda ls: pct(sum(l["importe_ok"] for l in ls), len(ls)))
    fila_l("Líneas no leídas", lambda ls: pct(sum(l["solo_real"] for l in ls), len(ls)))
    L.append(f"| Líneas inventadas | " + " | ".join(str(sum(l["solo_leida"] for l in lin[g])) for g in orden) + " |")

    errores = [l for l in filas_lineas if l["grupo"] != "train" and not l["linea_ok"]][:40]
    if errores:
        L += ["", "## Errores de línea (sin train, máx. 40)", "", "| Archivo | Tipo | Real | Leído | Tras corrector |", "|---|---|---|---|---|"]
        for l in errores:
            L.append(f"| {l['archivo']} | {l['tipo']} | {l['nombre_real'] or '—'} | {l['nombre_leido'] or '—'} | {l['nombre_corregido'] or '—'} |")
    malas_fechas = [f for f in filas if f["grupo"] != "train" and not f["fecha_ok"]]
    if malas_fechas:
        L += ["", "## Fechas mal leídas (sin train)", "", "| Archivo | Real | Leída | ¿Aviso? | ¿Bien tras revisión? |", "|---|---|---|---|---|"]
        for f in malas_fechas:
            L.append(f"| {f['archivo']} | {f['fecha_real']} | {f['fecha_leida']} | {'sí' if f['aviso_fecha'] else 'no'} | {'sí' if f['fecha_tras_revision_ok'] else 'no'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
