"""
Genera etiquetas_tickets.csv para revisar a mano la categoría de cada producto de los tickets.

Las categorías guardadas en los tickets las puso casi siempre el buscador del catálogo, así que no sirven
para medir ese mismo buscador (saldría un 100 % por construcción). Revisándolas a mano se obtienen
etiquetas independientes ("verdad de referencia") para el experimento de clustering.

Cómo revisar (Excel o LibreOffice, separador ';'):
  - Si la categoría y subcategoría son correctas: pon "si" en la columna revisado.
  - Si no: corrígelas (copiando los nombres de categorias_validas.csv) y pon "si" en revisado.
  - Si el producto pertenece a dos categorías, usa también categoria_2 / subcategoria_2.
No hace falta revisarlas todas: el experimento usa solo las filas con revisado = si.

Uso (desde backend/):  python -m validacion.exportar_etiquetas
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))
from categorizador import categorizador  # noqa: E402


def main():
    destino = RAIZ / "etiquetas_tickets.csv"
    if destino.exists():
        sys.exit(f"Ya existe {destino.name}; bórralo o renómbralo si quieres generarlo de nuevo (perderías lo revisado).")
    cats = defaultdict(Counter)
    for t in json.loads((RAIZ / "tickets_prueba.json").read_text(encoding="utf-8")):
        for it in t["items"]:
            for c in it.get("categorias") or [{"categoria": "", "subcategoria": ""}]:
                cats[it["descripcion"]][(c["categoria"], c["subcategoria"])] += 1
    with open(destino, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["nombre", "categoria", "subcategoria", "categoria_2", "subcategoria_2", "revisado"])
        for nombre in sorted(cats):
            pares = [p for p, _ in cats[nombre].most_common() if p[0]]
            (c1, s1), (c2, s2) = (pares + [("", ""), ("", "")])[:2]
            w.writerow([nombre, c1, s1, c2, s2, ""])
    with open(RAIZ / "categorias_validas.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["categoria", "subcategoria"])
        for g in categorizador.listar_categorias():
            for s in g["subcategorias"]:
                w.writerow([g["categoria"], s])
    print(f"✔ {len(cats)} productos -> {destino}\n✔ Lista de categorías -> categorias_validas.csv")


if __name__ == "__main__":
    main()
