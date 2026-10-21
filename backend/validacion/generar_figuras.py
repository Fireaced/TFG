"""
Genera las gráficas de la memoria (PDF vectorial) a partir de los ficheros de validacion/resultados.

Si se repite una validación, basta con actualizar los nombres de fichero de FUENTES (o dejar que se use
el más reciente) y volver a ejecutar:
    python -m validacion.generar_figuras                       # escribe en ../Memoria/figuras
    python -m validacion.generar_figuras --salida otra/carpeta

Estilo común: tipografía con remates (como el texto de la memoria), colores fijos por serie validados para
daltonismo (azul, naranja, verde agua), marcas finas, rejilla tenue y el valor escrito junto a cada barra o
punto final para que la gráfica se lea también impresa en escala de grises.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
SALIDA_DEFECTO = RAIZ.parent.parent / "Memoria" / "figuras"

# Ficheros usados (None = el más reciente que encaje con el patrón)
FUENTES = {
    "chat_desarrollo": {"Reglas": "chat_desarrollo_reglas_20261008_1552.csv",
                        "LLM": "chat_desarrollo_llm_qwen2.5-7b_20261013_1224.csv",
                        "LLM + RAG": "chat_desarrollo_llm_rag_qwen2.5-7b_20261013_1224.csv"},
    "chat_prueba": {"Reglas": "chat_prueba_reglas_20261008_1544.csv",
                    "LLM": "chat_prueba_llm_qwen2.5-7b_20261013_1156.csv",
                    "LLM + RAG": "chat_prueba_llm_rag_qwen2.5-7b_20261013_1156.csv"},
    "donut_tickets": None,      # donut_tickets_*.csv
    "donut_lineas": None,       # donut_lineas_*.csv
    "clustering": "clustering_20261012_1431.json",
    "hibrido": "resumen_hibrido.md",
}

AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [AZUL, NARANJA, AGUA]
MARCADORES = ["o", "s", "^"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 3, "figure.dpi": 150,
})
ANCHO = 6.0   # pulgadas (~15 cm, ancho de texto)


def ultimo(patron: str) -> Path:
    ficheros = sorted(RES.glob(patron))
    if not ficheros:
        raise FileNotFoundError(f"No hay ficheros {patron} en {RES}")
    return ficheros[-1]


def leer_csv(ruta: Path) -> list:
    with open(ruta, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(v):
    return f"{100 * v:.1f}".replace(".", ",")


def guardar(fig, salida: Path, nombre: str):
    fig.savefig(salida / nombre, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {nombre}")


# ── 1. Asistente: acierto por banco y motor, con IC 95 % de Wilson ──
def figura_chat(salida: Path):
    bancos = [("Desarrollo", FUENTES["chat_desarrollo"]), ("Prueba (reservado)", FUENTES["chat_prueba"])]
    motores = ["Reglas", "LLM", "LLM + RAG"]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    ancho = 0.24
    for j, motor in enumerate(motores):
        for i, (banco, ficheros) in enumerate(bancos):
            filas = [f for f in leer_csv(RES / ficheros[motor]) if f["categoria"] != "limites"]
            k, n = sum(int(f["acierto"]) for f in filas), len(filas)
            lo, hi = wilson(k, n)
            x = i + (j - 1) * (ancho + 0.03)
            ax.bar(x, k / n, ancho, color=SERIES[j], label=motor if i == 0 else None, zorder=2)
            ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor=TINTA_2, elinewidth=0.8,
                        capsize=2.5, zorder=3)
            ax.text(x, hi + 0.02, f"{pct(k / n)} %", ha="center", va="bottom", fontsize=7.5, color=TINTA)
    ax.set_xticks(range(len(bancos)), [f"{b}" for b, _ in bancos])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_ylabel("Preguntas acertadas")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    guardar(fig, salida, "chat_acierto.pdf")


# ── 2. Lectura de tickets: del PDF a la foto simulada (tickets nuevos) ──
def figura_donut(salida: Path):
    tickets = leer_csv(RES / FUENTES["donut_tickets"]) if FUENTES["donut_tickets"] else leer_csv(ultimo("donut_tickets_*.csv"))
    lineas = leer_csv(RES / FUENTES["donut_lineas"]) if FUENTES["donut_lineas"] else leer_csv(ultimo("donut_lineas_*.csv"))
    niveles = [("pdf", "PDF"), ("sintética leve", "Foto leve"), ("sintética media", "Foto media"), ("sintética fuerte", "Foto fuerte")]
    si = lambda v: v in ("True", "1", "true")  # noqa: E731
    medidas = {
        "Ticket perfecto": [sum(si(t["perfecto"]) for t in tickets if t["tipo"] == k and t["grupo"] == "nuevo") /
                            max(1, sum(1 for t in tickets if t["tipo"] == k and t["grupo"] == "nuevo")) for k, _ in niveles],
        "Fecha correcta": [sum(si(t["fecha_ok"]) for t in tickets if t["tipo"] == k and t["grupo"] == "nuevo") /
                           max(1, sum(1 for t in tickets if t["tipo"] == k and t["grupo"] == "nuevo")) for k, _ in niveles],
        "Línea de producto correcta": [
            sum(si(l["linea_ok"]) for l in lineas if l["tipo"] == k and l["grupo"] == "nuevo" and not si(l["solo_leida"])) /
            max(1, sum(1 for l in lineas if l["tipo"] == k and l["grupo"] == "nuevo" and not si(l["solo_leida"]))) for k, _ in niveles],
    }
    fig, ax = plt.subplots(figsize=(ANCHO, 2.7))
    xs = range(len(niveles))
    for (nombre, valores), color, marca in zip(medidas.items(), SERIES, MARCADORES):
        ax.plot(xs, valores, color=color, lw=2, marker=marca, ms=5.5, mec="white", mew=1.2, label=nombre,
                solid_capstyle="round", zorder=3)
        ax.annotate(f"{pct(valores[0])} %", (0, valores[0]), xytext=(-7, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=TINTA)
        ax.annotate(f"{pct(valores[-1])} %", (len(niveles) - 1, valores[-1]), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.5, color=TINTA)
    ax.set_xticks(list(xs), [n for _, n in niveles])
    ax.set_xlim(-0.75, len(niveles) - 0.55)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    guardar(fig, salida, "donut_degradacion.pdf")


# ── 3. Categorización: mejor variante de cada método en el escenario real (C) ──
def figura_clustering(salida: Path):
    datos = json.loads((RES / FUENTES["clustering"]).read_text(encoding="utf-8"))
    filas = [f for f in datos["filas"] if f["escenario"].startswith("C.")]
    mejor = {}
    for f in filas:
        if f["metodo"] not in mejor or f["acierto_categoria"] > mejor[f["metodo"]]["acierto_categoria"]:
            mejor[f["metodo"]] = f
    hibrido = re.search(r"confianza ≥ 0\.75\) \+ 1-NN \| ([\d.]+) %", (RES / FUENTES["hibrido"]).read_text(encoding="utf-8"))
    buscador = re.search(r"Buscador del catálogo solo \| ([\d.]+) %", (RES / FUENTES["hibrido"]).read_text(encoding="utf-8"))
    nombres = {"kmeans (k=26)": "K-Means (k = 26)", "kmeans (k=148)": "K-Means (k = 148)", "kmeans (k=444)": "K-Means (k = 444)",
               "jerarquico (k=148)": "Ward (k = 148)", "jerarquico (k=444)": "Ward (k = 444)", "hdbscan (min=3)": "HDBSCAN",
               "5-NN": "5 vecinos", "1-NN": "Vecino más cercano"}
    barras = [(nombres.get(m, m), f["acierto_categoria"], "clustering" if m[0] in "kjh" else "vecinos") for m, f in mejor.items()]
    barras.sort(key=lambda b: b[1])
    barras += [("Buscador del catálogo", float(buscador.group(1)) / 100, "buscador"),
               ("Buscador + vecino (sistema final)", float(hibrido.group(1)) / 100, "final")]
    colores = {"clustering": AGUA, "vecinos": AZUL, "buscador": "#7a7974", "final": NARANJA}
    fig, ax = plt.subplots(figsize=(ANCHO, 3.2))
    ys = range(len(barras))
    ax.barh(list(ys), [b[1] for b in barras], height=0.62, color=[colores[b[2]] for b in barras], zorder=2)
    for y, (nombre, v, _) in zip(ys, barras):
        ax.text(v + 0.01, y, f"{pct(v)} %", va="center", fontsize=7.5, color=TINTA)
    ax.set_yticks(list(ys), [b[0] for b in barras])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Acierto de categoría en los productos de los tickets (escenario C)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colores["clustering"], label="Clustering"), Patch(color=AZUL, label="Vecinos más cercanos"),
                       Patch(color=colores["buscador"], label="Buscador"), Patch(color=NARANJA, label="Sistema final")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.1))
    guardar(fig, salida, "clustering_metodos.pdf")


# ── 4. Umbral de confianza del buscador en el sistema híbrido ──
def figura_umbral(salida: Path):
    texto = (RES / FUENTES["hibrido"]).read_text(encoding="utf-8")
    datos = {"1-NN": [], "clustering": []}
    for umbral, alt, acierto, n in re.findall(r"confianza ≥ ([\d.]+)\) \+ (1-NN|clustering) \| ([\d.]+) % \| (\d+)", texto):
        datos[alt].append((float(umbral), float(acierto) / 100, int(n)))
    solo = float(re.search(r"Buscador del catálogo solo \| ([\d.]+) %", texto).group(1)) / 100
    fig, ax = plt.subplots(figsize=(ANCHO, 2.5))
    for (alt, nombre), color, marca in zip([("1-NN", "Buscador + vecino más cercano"), ("clustering", "Buscador + clustering (K-Means)")],
                                           SERIES, MARCADORES):
        xs, ys = [d[0] for d in datos[alt]], [d[1] for d in datos[alt]]
        ax.plot(xs, ys, color=color, lw=2, marker=marca, ms=5.5, mec="white", mew=1.2, label=nombre, zorder=3)
        if alt == "1-NN":
            ax.annotate(f"{pct(max(ys))} %", (xs[ys.index(max(ys))], max(ys)), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7.5, color=TINTA)
    ax.axhline(solo, color=TINTA_2, lw=1, zorder=1)
    ax.annotate(f"Buscador solo: {pct(solo)} %", (0.95, solo), xytext=(0, -10), textcoords="offset points",
                ha="right", fontsize=7.5, color=TINTA_2)
    ax.set_xlabel("Confianza mínima del buscador para aceptar su respuesta")
    ax.set_ylabel("Acierto de categoría")
    ax.set_xticks([d[0] for d in datos["1-NN"]], [f"{d[0]:.2f}".replace(".", ",") for d in datos["1-NN"]])
    ax.set_ylim(0.84, 0.98)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    guardar(fig, salida, "hibrido_umbral.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default=str(SALIDA_DEFECTO))
    args = ap.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    print(f"📊 Generando figuras en {salida}")
    for f in (figura_chat, figura_donut, figura_clustering, figura_umbral):
        try:
            f(salida)
        except FileNotFoundError as e:
            print(f"  ✗ {f.__name__}: {e}")


if __name__ == "__main__":
    main()
