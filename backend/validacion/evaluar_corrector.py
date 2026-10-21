"""
Validación del corrector de nombres de producto (corrector_nombres.py).

Dos pruebas complementarias:

A) Errores REALES de Donut: las líneas emparejadas (nombre real / nombre leído) de la última ejecución de
   evaluar_donut.py (PDF y fotos simuladas, tickets nuevos y de validación; nunca los de entrenamiento).

B) Errores SIMULADOS de tipo OCR sobre los nombres de los 91 tickets del dataset, dejando fuera cada ticket
   (su vocabulario es el catálogo + los nombres de los OTROS tickets): letras parecidas (U/L, O/0, R/F...),
   letra de más o de menos, palabra partida o dos palabras juntas. También se pasan los nombres SIN error,
   para medir cuántas veces el corrector "estropea" un nombre que estaba bien.

Escenarios de vocabulario:
  - historial: catálogo + nombres de tickets ya guardados (usuario que lleva tiempo usando la app)
  - catálogo:  solo el catálogo de Mercadona (usuario nuevo)
  - PDF:       además, los nombres exactos del texto del PDF (lo que hace la app al subir un PDF)

Resultado de cada nombre:
  bien leído  -> sin tocar (correcto) / cambiado solo (falsa corrección) / solo sugerencia
  mal leído   -> corregido solo bien / corregido solo mal / sugerencia correcta / sin corregir

Uso (desde backend/; no necesita Donut ni Mongo):
    python -m validacion.evaluar_corrector
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

from categorizador import categorizador  # noqa: E402
from corrector_nombres import _PARECIDOS, CorrectorNombres, normalizar_nombre  # noqa: E402
from validacion.evaluar_donut import levenshtein  # noqa: E402

RESULTADOS = RAIZ / "resultados"
SPLIT = RAIZ / "split_entrenamiento.json"
SEMILLA = 7


# ── Vocabularios ──
def tickets_dataset() -> list:
    datos = json.loads(SPLIT.read_text(encoding="utf-8"))
    return [t for g in ("train", "validacion") for t in datos[g]]


def corrector_con(nombres) -> CorrectorNombres:
    c = CorrectorNombres(categorizador)
    if nombres:
        c.añadir_nombres(nombres)
    return c


def clasificar(c: CorrectorNombres, real: str, leido: str, extra=()) -> str:
    real, leido_n = normalizar_nombre(real), normalizar_nombre(leido)
    r = c.corregir(leido, extra)
    if leido_n == real:
        if r is None:
            return "bien: sin tocar"
        return "bien: cambiado (falsa corrección)" if r["automatica"] else "bien: solo sugerencia"
    if r is None:
        return "mal: sin corregir"
    if r["automatica"]:
        if r["nombre"] == real:
            return "mal: corregido bien"
        # corrección parcial: el nombre queda más cerca del real aunque no idéntico ("PECHUGA DE POLLO 22%")
        return "mal: corregido a medias (mejora)" if levenshtein(real, r["nombre"]) < levenshtein(real, leido_n) \
            else "mal: corregido mal (no mejora)"
    return "mal: sugerencia correcta" if r["nombre"] == real else "mal: sugerencia incorrecta"


CATEGORIAS = ["bien: sin tocar", "bien: solo sugerencia", "bien: cambiado (falsa corrección)",
              "mal: corregido bien", "mal: corregido a medias (mejora)", "mal: sugerencia correcta",
              "mal: sugerencia incorrecta", "mal: corregido mal (no mejora)", "mal: sin corregir"]
MAX_CER_PAREJA = 0.5   # en las fotos, el emparejamiento de líneas puede juntar productos distintos: se descartan


# ── A) Errores reales de Donut ──
def prueba_real(lineas_csv: Path):
    filas = [f for f in csv.DictReader(open(lineas_csv, encoding="utf-8-sig"), delimiter=";")
             if f["grupo"] != "train" and f["nombre_real"] and f["nombre_leido"]
             and float(f["cer"]) <= MAX_CER_PAREJA]
    historial = [n for t in json.loads(SPLIT.read_text(encoding="utf-8"))["train"] for n in t["nombres"]]
    correctores = {"historial": corrector_con(historial), "catálogo": corrector_con([])}
    res = defaultdict(Counter)
    ejemplos = defaultdict(list)
    nombres_ticket = defaultdict(list)          # lo que la app saca del texto del PDF al subir un PDF
    for f in filas:
        nombres_ticket[f["archivo"]].append(f["nombre_real"])
    for f in filas:
        tipo = "PDF" if f["tipo"] == "pdf" else "foto simulada"
        if tipo == "PDF":
            res[("PDF", "texto del PDF")][clasificar(correctores["historial"], f["nombre_real"], f["nombre_leido"],
                                                     nombres_ticket[f["archivo"]])] += 1
        else:
            gravedad = "error leve (CER ≤ 0,2)" if float(f["cer"]) <= 0.2 else "error grave (CER > 0,2)"
            if float(f["cer"]) > 0:
                res[("foto · historial", gravedad)][clasificar(correctores["historial"], f["nombre_real"], f["nombre_leido"])] += 1
        for esc, c in correctores.items():
            k = clasificar(c, f["nombre_real"], f["nombre_leido"])
            res[(tipo, esc)][k] += 1
            if esc == "historial" and k in ("mal: corregido mal (no mejora)", "bien: cambiado (falsa corrección)"):
                r = c.corregir(f["nombre_leido"])
                ejemplos[k].append((f["nombre_real"], f["nombre_leido"], r["nombre"] if r else ""))
    return res, ejemplos, len(filas)


# ── B) Errores simulados ──
PARES = [(a, b) for a, b in _PARECIDOS] + [(b, a) for a, b in _PARECIDOS]


def estropear(nombre: str, rnd: random.Random) -> str:
    s = list(nombre)
    letras = [i for i, ch in enumerate(s) if ch.isalpha()]
    if not letras:
        return nombre
    tipo = rnd.choices(["parecida", "parecida2", "falta", "sobra", "partir", "juntar"], [40, 15, 15, 10, 12, 8])[0]
    i = rnd.choice(letras)
    if tipo in ("parecida", "parecida2"):
        for _ in range(1 if tipo == "parecida" else 2):
            opciones = [(j, b) for j in letras for a, b in PARES if s[j] == a]
            if opciones:
                j, b = rnd.choice(opciones)
                s[j] = b
    elif tipo == "falta":
        del s[i]
    elif tipo == "sobra":
        s.insert(i, s[i])
    elif tipo == "partir":
        palabras = [j for j in letras if 0 < j < len(s) - 1 and s[j - 1].isalpha() and s[j + 1].isalpha()]
        if palabras:
            s.insert(rnd.choice(palabras), " ")
    elif tipo == "juntar":
        espacios = [j for j, ch in enumerate(s) if ch == " "]
        if espacios:
            del s[rnd.choice(espacios)]
    return "".join(s)


def prueba_simulada():
    tickets = tickets_dataset()
    rnd = random.Random(SEMILLA)
    res = defaultdict(Counter)
    por_tipo = defaultdict(Counter)
    ejemplos = defaultdict(list)
    solo_catalogo = corrector_con([])
    for k, t in enumerate(tickets):
        otros = [n for j, u in enumerate(tickets) if j != k for n in u["nombres"]]
        c = corrector_con(otros)
        for nombre in t["nombres"]:
            mal = estropear(nombre, rnd)
            if normalizar_nombre(mal) == normalizar_nombre(nombre):
                continue
            conocido = "conocido" if normalizar_nombre(nombre) in c.nombres else "nuevo"
            for esc, corr, extra in (("historial", c, ()), ("catálogo", solo_catalogo, ()), ("PDF", c, t["nombres"])):
                res[(esc, "con error")][clasificar(corr, nombre, mal, extra)] += 1
                res[(esc, "sin error")][clasificar(corr, nombre, nombre, extra)] += 1
            res[(f"historial · producto {conocido}", "con error")][clasificar(c, nombre, mal)] += 1
            k_hist = clasificar(c, nombre, mal)
            por_tipo[conocido][k_hist] += 1
            if k_hist == "mal: corregido mal (no mejora)":
                ejemplos["simulado: corregido mal"].append((nombre, mal, c.corregir(mal)["nombre"]))
            k_bien = clasificar(c, nombre, nombre)
            if k_bien == "bien: cambiado (falsa corrección)":
                ejemplos["simulado: falsa corrección"].append((nombre, nombre, c.corregir(nombre)["nombre"]))
    return res, ejemplos


def pct(a, n):
    return f"{100 * a / n:.1f} % ({a})" if n else "—"


def tabla(res, claves, titulo):
    L = [f"### {titulo}", "", "| Resultado | " + " | ".join(" · ".join(k) for k in claves) + " |",
         "|---|" + "---|" * len(claves)]
    for cat in CATEGORIAS:
        if any(res[k][cat] for k in claves):
            L.append(f"| {cat} | " + " | ".join(
                pct(res[k][cat], sum(v for c2, v in res[k].items() if c2.split(':')[0] == cat.split(':')[0]))
                for k in claves) + " |")
    L.append("| Nombres | " + " | ".join(str(sum(res[k].values())) for k in claves) + " |")
    return L + [""]


def main():
    RESULTADOS.mkdir(exist_ok=True)
    L = [f"# Validación del corrector de nombres — {datetime.now():%d/%m/%Y %H:%M}", "",
         "Porcentajes sobre los nombres bien leídos («bien: …») o sobre los mal leídos («mal: …») de cada columna.", ""]
    csvs = sorted(RESULTADOS.glob("donut_lineas_*.csv"))
    if csvs:
        res, ejemplos, n = prueba_real(csvs[-1])
        L += [f"## A) Errores reales de Donut ({csvs[-1].name}, {n} líneas emparejadas con CER ≤ 0,5, sin tickets de entrenamiento)", ""]
        L += tabla(res, [("PDF", "texto del PDF"), ("PDF", "historial"), ("PDF", "catálogo"), ("foto simulada", "historial"),
                         ("foto simulada", "catálogo")], "Resultado por tipo de imagen y vocabulario")
        L += ["En la app, al subir un PDF se usa la columna «texto del PDF» (nombres exactos del propio ticket); "
              "las otras dos columnas de PDF muestran qué pasaría sin ese texto.", ""]
        L += tabla(res, [("foto · historial", "error leve (CER ≤ 0,2)"), ("foto · historial", "error grave (CER > 0,2)")],
                   "Fotos simuladas, según lo grave que sea el error de lectura")
        for k, ej in ejemplos.items():
            vistos = list(dict.fromkeys(ej))[:15]
            L += [f"**{k}** ({len(ej)}): " + "; ".join(f"{r} ← «{l}» → {c}" for r, l, c in vistos), ""]
    else:
        L += ["(No hay resultados de evaluar_donut.py: se omite la prueba con errores reales.)", ""]

    res, ejemplos = prueba_simulada()
    L += ["## B) Errores simulados de tipo OCR (91 tickets, dejando fuera cada ticket)", ""]
    L += tabla(res, [("historial", "con error"), ("catálogo", "con error"), ("PDF", "con error")],
               "Nombres con un error simulado")
    L += tabla(res, [("historial", "sin error"), ("catálogo", "sin error"), ("PDF", "sin error")],
               "Nombres correctos (¿los estropea?)")
    L += tabla(res, [("historial · producto conocido", "con error"), ("historial · producto nuevo", "con error")],
               "Con historial: producto ya comprado antes vs producto nuevo")
    for k, ej in ejemplos.items():
        vistos = list(dict.fromkeys(ej))[:15]
        L += [f"**{k}** ({len(ej)}): " + "; ".join(f"{r} ← «{l}» → {c}" for r, l, c in vistos), ""]
    md = "\n".join(L) + "\n"
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    (RESULTADOS / f"resumen_corrector_{marca}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
