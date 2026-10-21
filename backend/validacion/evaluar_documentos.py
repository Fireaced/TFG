"""
Validación de (1) el rechazo de PDF que no son tickets y (2) la detección de tickets repetidos.

1) La app solo admite tickets: todos los PDF de tickets del proyecto deben aceptarse, y los extractos
   bancarios simulados (generar_extractos_sinteticos.py) rechazarse sin pasar por Donut.

2) Repetidos. La app comprueba, en este orden:
     a) mismo archivo (hash SHA-256 del contenido)                    -> exacto
     b) mismo nº de factura simplificada (solo PDF, se lee del texto) -> exacto
     c) misma huella del contenido leído por Donut: fecha + total + productos
     d) aviso de «posible repetido» si coinciden fecha y total pero no los productos (el usuario decide)
   Se mide con las lecturas reales de Donut (evaluar_donut.py), suponiendo que el ticket ya guardado es
   el correcto (revisado por el usuario) y que se vuelve a subir: como el mismo PDF, como otro PDF del
   mismo ticket o como foto (fotos simuladas). Falsos positivos: tickets DISTINTOS que la regla confunde.

Uso (desde backend/; no necesita Donut ni Mongo):
    python -m validacion.generar_extractos_sinteticos
    python -m validacion.evaluar_documentos
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

import pdfplumber  # noqa: E402

from deteccion_documentos import es_extracto_pdf, huella_ticket, numero_factura  # noqa: E402
from lectura_donut import normalizar_prediccion  # noqa: E402
from validacion.evaluar_donut import CARPETAS_PDF, a_centimos, gt_de_pdf, huella_gt  # noqa: E402

RESULTADOS = RAIZ / "resultados"
EXTRACTOS = RAIZ / "extractos_sinteticos"


def a_items(menu: list) -> list:
    """Líneas de Donut / ground truth -> items como los guarda la app (misma regla que parse_donut_output)."""
    items = []
    for it in menu:
        importe = a_centimos(it.get("price", "0")) / 100
        try:
            cantidad = int(str(it.get("cnt", "1")).strip())
        except ValueError:
            cantidad = 1
        if "weight" in it or cantidad == 0:
            cantidad, unitario = 1, importe
        elif it.get("unitprice"):
            unitario = a_centimos(it["unitprice"].split()[0]) / 100
        else:
            unitario = round(importe / max(cantidad, 1), 2)
        items.append({"descripcion": it.get("nm", ""), "cantidad": cantidad, "precio_unitario": unitario})
    return items


def huella_de(lectura: dict) -> str:
    total = (lectura.get("total") or {}).get("total_price")
    return huella_ticket(lectura.get("date") or "", a_centimos(total) / 100 if total else 0, a_items(lectura.get("menu", [])))


def clave_suave(lectura: dict):
    total = (lectura.get("total") or {}).get("total_price")
    return ((lectura.get("date") or "")[:10], a_centimos(total) if total else None)


def pct(a, n):
    return f"{100 * a / n:.1f} % ({a}/{n})" if n else "—"


def main():
    L = [f"# Validación: solo tickets y tickets repetidos — {datetime.now():%d/%m/%Y %H:%M}", ""]

    # ── 1) Ticket o extracto ──
    pdfs, vistos = [], set()
    for carpeta in CARPETAS_PDF:
        for p in sorted(carpeta.glob("*.pdf")) if carpeta.exists() else []:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h not in vistos:
                vistos.add(h)
                pdfs.append(p)
    mal_ticket = [p.name for p in pdfs if es_extracto_pdf(p.read_bytes())]
    L += ["## 1) Solo tickets: ¿se rechazan los PDF que no lo son?", "",
          f"- Tickets PDF de Mercadona (formatos de 2024, desde 08/2025 con tabla y desde 10/2026 con «Cnt.»): "
          f"aceptados {pct(len(pdfs) - len(mal_ticket), len(pdfs))}" + (f" — rechazados por error: {', '.join(mal_ticket)}" if mal_ticket else "")]
    if EXTRACTOS.exists():
        extractos = sorted(EXTRACTOS.glob("*.pdf"))
        rechazados = [p for p in extractos if es_extracto_pdf(p.read_bytes())]
        L.append(f"- Extractos bancarios simulados ({len(extractos)} formatos): rechazados {pct(len(rechazados), len(extractos))}"
                 + (f" — aceptados por error: {', '.join(p.stem for p in extractos if p not in rechazados)}"
                    if len(rechazados) < len(extractos) else ""))
    L.append("")

    # ── 2) Repetidos ──
    gts = {}
    for p in pdfs:
        try:
            gts[p.stem] = (gt_de_pdf(p), p)
        except Exception:
            pass
    # Falsos positivos: tickets distintos con la misma clave
    huellas = Counter(huella_de(gt) for gt, _ in gts.values())
    suaves = Counter(clave_suave(gt) for gt, _ in gts.values())
    facturas = Counter(f for _, p in gts.values() if (f := numero_factura(p.read_bytes())))
    con_factura = sum(1 for _, p in gts.values() if numero_factura(p.read_bytes()))
    L += ["## 2) Tickets repetidos", "", f"{len(gts)} tickets distintos (PDF con su ticket correcto).", "",
          "### Falsos positivos (tickets distintos que la regla tomaría por el mismo)", "",
          f"- Misma huella (fecha + total + productos): {sum(v - 1 for v in huellas.values() if v > 1)} casos",
          f"- Mismo nº de factura: {sum(v - 1 for v in facturas.values() if v > 1)} casos "
          f"({con_factura} de {len(gts)} PDF tienen nº de factura; los de devolución no)",
          f"- Misma fecha y total (solo aviso): {sum(v - 1 for v in suaves.values() if v > 1)} casos", ""]

    # Detección al volver a subir el mismo ticket (lecturas reales de Donut)
    cache = RESULTADOS / "donut_lecturas.jsonl"
    if cache.exists():
        lecturas = defaultdict(list)
        for linea in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            tipo = "PDF" if r["archivo"].lower().endswith(".pdf") else "foto simulada"
            lecturas[(Path(r["archivo"]).stem, tipo)].append(normalizar_prediccion(r["pred"]))
        res = defaultdict(Counter)
        for (stem, tipo), lista in lecturas.items():
            if stem not in gts:
                continue
            gt, pdf = gts[stem]
            for lectura in lista:
                exacta = huella_de(lectura) == huella_de(gt)
                suave = clave_suave(lectura) == clave_suave(gt)
                res[tipo]["n"] += 1
                res[tipo]["huella"] += exacta
                res[tipo]["huella o aviso"] += exacta or suave
                if tipo == "PDF":
                    res[tipo]["factura"] += numero_factura(pdf.read_bytes()) is not None
                    res[tipo]["factura o huella"] += exacta or numero_factura(pdf.read_bytes()) is not None
                    res[tipo]["todo"] += exacta or suave or numero_factura(pdf.read_bytes()) is not None
                else:
                    res[tipo]["todo"] += exacta or suave
        L += ["### Detección al volver a subir un ticket ya guardado (otro archivo, mismo ticket)", "",
              "El mismo archivo exacto se detecta siempre por su hash (100 %). Aquí el archivo es distinto.", "",
              "| Se vuelve a subir como | Por huella (antes) | + nº de factura | + aviso fecha y total | Total detectado |",
              "|---|---|---|---|---|"]
        for tipo in ("PDF", "foto simulada"):
            r = res[tipo]
            L.append(f"| {tipo} | {pct(r['huella'], r['n'])} | {pct(r['factura o huella'], r['n']) if tipo == 'PDF' else '—'} | "
                     f"{pct(r['huella o aviso'], r['n'])} | {pct(r['todo'], r['n'])} |")
        L += ["", "Nota: al guardar, la app vuelve a comprobar la huella con los datos ya revisados por el usuario; "
              "si los deja iguales que los del ticket guardado, el repetido se detecta entonces aunque Donut se haya equivocado.", ""]
    md = "\n".join(L) + "\n"
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / f"resumen_documentos_{datetime.now():%Y%m%d_%H%M}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
