"""
Validación de la revisión de fechas (fechas_ticket.revisar_fecha).

La app compara la fecha leída por Donut con una fecha de referencia (la de la foto en el EXIF o, si no
hay, la de última modificación del archivo) y, si no encaja, avisa y propone una corrección.

A) Fechas REALES leídas por Donut (última ejecución de evaluar_donut.py, sin tickets de entrenamiento).
   Las fotos simuladas no tienen EXIF, así que se simula la referencia en varios escenarios:
     - foto hecha el mismo día de la compra
     - foto hecha 1-3 días después
     - sin referencia (foto sin EXIF y archivo sin fecha)
   y para los PDF: descargado entre 0 y 120 días después de la compra.

B) Errores SIMULADOS sobre las fechas de todos los tickets (dataset + nuevos), con los errores típicos
   (dígitos parecidos, día/mes intercambiados, dígitos del mes o del día intercambiados, año mal, mes
   leído como «09», que es el error que más se ha visto en las fotos), en los mismos escenarios.

Métricas:
  - fecha bien leída: falsa alarma (avisa sin motivo)
  - fecha mal leída: detectada (avisa), corregida (la fecha propuesta es la real), se cuela (sin aviso)

Uso (desde backend/; no necesita Donut):
    python -m validacion.evaluar_fechas
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

from fechas_ticket import PARECIDOS, revisar_fecha  # noqa: E402

RESULTADOS = RAIZ / "resultados"
SPLIT = RAIZ / "split_entrenamiento.json"
HOY = date(2026, 10, 13)
SEMILLA = 11
FOTO = b"\xff\xd8 imagen sin exif"     # contenido de imagen sin EXIF: la referencia es la fecha del archivo
PDF = b"%PDF-1.4"


def a_fecha(txt: str):
    try:
        return datetime.strptime(txt.strip()[:10], "%d/%m/%Y").date()
    except ValueError:
        return None


def ms(d: date) -> float:
    return datetime(d.year, d.month, d.day, 20).timestamp() * 1000


ESCENARIOS = {
    "foto, mismo día": (FOTO, lambda real, rnd: real),
    "foto, 1-3 días después": (FOTO, lambda real, rnd: real + timedelta(days=rnd.randint(1, 3))),
    "foto sin referencia": (FOTO, lambda real, rnd: None),
    "PDF descargado 0-120 días después": (PDF, lambda real, rnd: real + timedelta(days=rnd.randint(0, 120))),
}


def evaluar(real: date, leida: str, contenido: bytes, referencia) -> str:
    ref = referencia if referencia is None or referencia <= HOY else HOY
    r = revisar_fecha(leida, contenido, ms(ref) if ref else None, hoy=HOY)
    bien = a_fecha(leida) == real
    if bien:
        return "bien: falsa alarma" if r["aviso"] else "bien: sin aviso"
    if not r["aviso"]:
        return "mal: se cuela sin aviso"
    return "mal: avisa y propone la real" if a_fecha(r["fecha_sugerida"] or "") == real else \
        ("mal: avisa, propuesta incorrecta" if r["fecha_sugerida"] else "mal: avisa, sin propuesta")


CATEGORIAS = ["bien: sin aviso", "bien: falsa alarma", "mal: avisa y propone la real",
              "mal: avisa, propuesta incorrecta", "mal: avisa, sin propuesta", "mal: se cuela sin aviso"]


def estropear(f: date, rnd: random.Random):
    d, m, a = f"{f.day:02d}", f"{f.month:02d}", str(f.year)
    tipo = rnd.choice(["parecido", "día/mes", "mes al revés", "día al revés", "año", "mes 09"])
    if tipo == "parecido":
        texto = list(d + m)
        i = rnd.choice([i for i, c in enumerate(texto) if PARECIDOS.get(c)])
        texto[i] = rnd.choice(PARECIDOS[texto[i]])
        d, m = "".join(texto[:2]), "".join(texto[2:])
    elif tipo == "día/mes":
        d, m = m, d
    elif tipo == "mes al revés":
        m = m[::-1]
    elif tipo == "día al revés":
        d = d[::-1]
    elif tipo == "año":
        a = str(int(a) + rnd.choice([-1, 1]))
    else:
        m = "09"
    leida = f"{d}/{m}/{a}"
    return (None, tipo) if a_fecha(leida) == f else (leida, tipo)


def pct(a, n):
    return f"{100 * a / n:.1f} % ({a})" if n else "—"


def tabla(res, claves, titulo):
    L = [f"### {titulo}", "", "| Resultado | " + " | ".join(claves) + " |", "|---|" + "---|" * len(claves)]
    for cat in CATEGORIAS:
        if any(res[k][cat] for k in claves):
            L.append(f"| {cat} | " + " | ".join(
                pct(res[k][cat], sum(v for c, v in res[k].items() if c.split(':')[0] == cat.split(':')[0]))
                for k in claves) + " |")
    L.append("| Fechas | " + " | ".join(str(sum(res[k].values())) for k in claves) + " |")
    return L + [""]


def main():
    rnd = random.Random(SEMILLA)
    L = [f"# Validación de la revisión de fechas — {datetime.now():%d/%m/%Y %H:%M}", "",
         f"Hoy = {HOY:%d/%m/%Y}. Porcentajes sobre las fechas bien leídas («bien: …») o mal leídas («mal: …»).", ""]

    csvs = sorted(RESULTADOS.glob("donut_tickets_*.csv"))
    if csvs:
        filas = [f for f in csv.DictReader(open(csvs[-1], encoding="utf-8-sig"), delimiter=";") if f["grupo"] != "train"]
        res, ejemplos = defaultdict(Counter), []
        for f in filas:
            real = a_fecha(f["fecha_real"])
            leida = f["fecha_leida"] if f["fecha_leida"] not in ("None", "") else ""
            escenarios = ["PDF descargado 0-120 días después"] if f["tipo"] == "pdf" else \
                ["foto, mismo día", "foto, 1-3 días después", "foto sin referencia"]
            for esc in escenarios:
                contenido, ref = ESCENARIOS[esc]
                k = evaluar(real, leida, contenido, ref(real, rnd))
                res[esc][k] += 1
                if esc == "foto, mismo día" and k.startswith("mal"):
                    ejemplos.append((f["fecha_real"], leida or "—", k.split(": ")[1]))
        L += [f"## A) Fechas leídas por Donut ({csvs[-1].name}; {len(filas)} tickets: PDF y fotos simuladas)", ""]
        L += tabla(res, list(ESCENARIOS)[3:] + list(ESCENARIOS)[:3], "Según la referencia disponible")
        if ejemplos:
            L += ["Fechas mal leídas (foto del mismo día): " + "; ".join(f"{r} ← «{l}» ({k})" for r, l, k in ejemplos), ""]

    # B) errores simulados
    fechas = [a_fecha(t["date"]) for g in json.loads(SPLIT.read_text(encoding="utf-8")).values() for t in g]
    nuevos = RAIZ / "tickets_nuevos"
    if nuevos.exists():
        fechas += [datetime.strptime(p.name[:8], "%Y%m%d").date() for p in nuevos.glob("*.pdf") if p.name[:8].isdigit()]
    res, por_tipo = defaultdict(Counter), defaultdict(Counter)
    for f in fechas:
        for _ in range(5):
            leida, tipo = estropear(f, rnd)
            if leida is None:
                continue
            for esc, (contenido, ref) in ESCENARIOS.items():
                k = evaluar(f, leida, contenido, ref(f, rnd))
                res[esc][k] += 1
                if esc == "foto, mismo día":
                    por_tipo[tipo][k] += 1
        for esc, (contenido, ref) in ESCENARIOS.items():
            res[esc][evaluar(f, f.strftime("%d/%m/%Y"), contenido, ref(f, rnd))] += 1
    L += [f"## B) Errores simulados ({len(fechas)} fechas de tickets × 5 errores, más la fecha correcta)", ""]
    L += tabla(res, list(ESCENARIOS), "Según la referencia disponible")
    L += tabla(por_tipo, sorted(por_tipo), "Foto del mismo día, por tipo de error")
    md = "\n".join(L) + "\n"
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / f"resumen_fechas_{datetime.now():%Y%m%d_%H%M}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
