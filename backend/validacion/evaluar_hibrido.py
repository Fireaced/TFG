"""
Experimento complementario: categorización híbrida = buscador del catálogo + respaldo aprendido.

El buscador por palabras del catálogo es muy preciso cuando encuentra el producto, pero no responde
para algunos nombres (abreviaturas, marcas) y a veces responde con poca confianza. Aquí se mide qué
pasa si, cuando el buscador no responde o su confianza es baja, se usa en su lugar:
  - el producto más parecido (1-NN) en el espacio de embeddings bge-m3 + TF-IDF de caracteres, o
  - el grupo más cercano de un K-Means (k = 444) en ese mismo espacio (clasificación por clustering).
Validación cruzada de 5 particiones sobre los productos de los tickets (entrenando con el catálogo +
el resto de tickets), con las etiquetas revisadas a mano.

Uso (desde backend/, con la caché de embeddings de bge-m3 ya calculada):
    python -m validacion.evaluar_hibrido
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

from validacion.evaluar_clustering import (SEMILLA, KFold, categorizador, clasificar_knn,  # noqa: E402
                                           clasificar_por_clusters, datos_catalogo, datos_tickets, representar)
from embeddings import ClienteEmbeddings  # noqa: E402

UMBRALES = [0.55, 0.65, 0.75, 0.85, 0.95]


def main():
    cat, tk = datos_catalogo(), datos_tickets()
    todos = cat + tk
    n = len(cat)
    X = representar("emb:bge-m3+char", [p["nombre"] for p in todos], {"bge-m3": ClienteEmbeddings("bge-m3")})
    pred_knn, pred_cl = [None] * len(tk), [None] * len(tk)
    for tr, te in KFold(5, shuffle=True, random_state=SEMILLA).split(np.arange(len(tk))):
        TR, TE = np.concatenate([np.arange(n), n + tr]), n + te
        ytr = [todos[i] for i in TR]
        pk, _ = clasificar_knn(X[TR], ytr, X[TE], 1)
        pc, _, _ = clasificar_por_clusters(X[TR], ytr, X[TE], "kmeans", 444)
        for j, i in enumerate(te):
            pred_knn[i], pred_cl[i] = pk[j], pc[j]

    busq = [categorizador.buscar(p["nombre"]) for p in tk]

    def ok(c, p):
        return c is not None and c in p["cats"]

    lineas = ["# Categorización híbrida (buscador + respaldo) — productos de los tickets", "",
              f"{len(tk)} productos, validación cruzada de 5 particiones, embeddings bge-m3 + TF-IDF de caracteres.", "",
              "| Configuración | Acierto categoría | Productos que van al respaldo |", "|---|---|---|"]
    acierto_b = np.mean([ok(b["categorias"][0]["categoria"] if b["categorias"] else None, p) for b, p in zip(busq, tk)])
    lineas.append(f"| Buscador del catálogo solo | {100 * acierto_b:.1f} % | — (no responde en {sum(not b['categorias'] for b in busq)}) |")
    lineas.append(f"| 1-NN embeddings solo | {100 * np.mean([ok(c, p) for c, p in zip(pred_knn, tk)]):.1f} % | — |")
    lineas.append(f"| Clustering K-Means (k=444) solo | {100 * np.mean([ok(c, p) for c, p in zip(pred_cl, tk)]):.1f} % | — |")
    for t in UMBRALES:
        for nombre, alt in (("1-NN", pred_knn), ("clustering", pred_cl)):
            res, usa = [], 0
            for b, a, p in zip(busq, alt, tk):
                if b["categorias"] and b["confianza"] >= t:
                    c = b["categorias"][0]["categoria"]
                else:
                    c, usa = a, usa + 1
                res.append(ok(c, p))
            lineas.append(f"| Buscador (confianza ≥ {t}) + {nombre} | {100 * np.mean(res):.1f} % | {usa} |")
    sin = [i for i, b in enumerate(busq) if not b["categorias"]]
    lineas += ["", f"Productos sin respuesta del buscador ({len(sin)}): " + ", ".join(tk[i]["nombre"] for i in sin),
               f"De ellos, el 1-NN acierta {sum(ok(pred_knn[i], tk[i]) for i in sin)} y el clustering {sum(ok(pred_cl[i], tk[i]) for i in sin)}.",
               "", "Nota: el umbral se elige con estos mismos datos (5 valores probados), así que la mejor cifra es algo optimista."]
    md = "\n".join(lineas) + "\n"
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "resultados" / "resumen_hibrido.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
