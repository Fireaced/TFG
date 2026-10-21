"""
Experimento: clasificación de productos en categorías mediante clustering.

Se comparan varias REPRESENTACIONES del nombre del producto y varios ALGORITMOS CLÁSICOS de clustering,
y se mide cómo de bien sirve el clustering para asignar la categoría a un producto nuevo.

Representaciones
  - tfidf_char      TF-IDF de n-gramas de caracteres (3-5), reducido con SVD (LSA) a 256 dimensiones
  - tfidf_palabras  TF-IDF de palabras (+ SVD)
  - tfidf_mixto     caracteres + palabras (la representación que usa hoy la aplicación, sin la parte de categoría)
  - emb:<modelo>    embeddings de un modelo local (Ollama), p. ej. bge-m3 o nomic-embed-text
  - emb:<modelo>+char  embeddings concatenados con TF-IDF de caracteres

Algoritmos
  - K-Means (k = nº de categorías, nº de subcategorías y 3 × nº de subcategorías)
  - Clustering jerárquico aglomerativo (Ward), mismos k
  - HDBSCAN (densidad; el número de grupos sale solo)

Clasificación mediante clustering
  1. Se agrupan los productos de entrenamiento (sin mirar sus categorías).
  2. Cada grupo recibe la categoría (y subcategoría) mayoritaria de sus productos.
  3. Un producto nuevo se asigna al grupo con el centroide más cercano y hereda su categoría.
  Referencias: k vecinos más cercanos (1-NN, 5-NN) y el buscador por palabras del catálogo actual.

Escenarios
  A. catalogo_cv        validación cruzada (5 particiones) sobre los 3.931 productos del catálogo
                        (etiquetas limpias: las del propio catálogo de Mercadona)
  B. catalogo->tickets  se entrena con el catálogo (nombres completos) y se prueba con los nombres
                        abreviados de los tickets ("Q. UNTAR SUAVE", "PECHU.POLLO BRASEADA")
  C. tickets_cv         validación cruzada sobre los productos de los tickets, entrenando con el catálogo
                        + el resto de tickets (es la situación real de la aplicación)
  Además: calidad del agrupamiento sin etiquetas (silueta) y su parecido con las categorías (ARI, NMI).

Uso (desde backend/):
    python -m validacion.evaluar_clustering                         # solo TF-IDF
    python -m validacion.evaluar_clustering --modelos bge-m3 nomic-embed-text
Requiere los modelos en Ollama:  ollama pull bge-m3   /   ollama pull nomic-embed-text
"""
import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ.parent / ".env")

from categorizador import categorizador, normalizar  # noqa: E402

warnings.filterwarnings("ignore")
SEMILLA = 0


# ── Datos etiquetados ──
def datos_catalogo():
    prods = []
    for nombre, cats in categorizador.categorias_por_producto.items():
        prods.append({"nombre": nombre, "cats": {c for c, _ in cats}, "subs": {f"{c}/{s}" for c, s in cats}})
    return prods


# etiquetas_revisadas.csv: versión final (anotación manual + comprobación del usuario); si no existe, la de trabajo
RUTA_ETIQUETAS = RAIZ / "etiquetas_revisadas.csv" if (RAIZ / "etiquetas_revisadas.csv").exists() else RAIZ / "etiquetas_tickets.csv"


def datos_tickets():
    """Productos de los tickets con su categoría. Si el usuario ha revisado etiquetas_tickets.csv
    (columna revisado = si), se usan SOLO esas filas con sus categorías corregidas: son etiquetas
    independientes del buscador del catálogo, que es el que puso la mayoría de las categorías."""
    if RUTA_ETIQUETAS.exists():
        import csv
        with open(RUTA_ETIQUETAS, encoding="utf-8-sig", newline="") as fh:
            filas = [f for f in csv.DictReader(fh, delimiter=";")
                     if (f.get("revisado") or "").strip().lower() in ("si", "sí", "s", "x", "1")]
        if filas:
            print(f"✔ Usando {len(filas)} etiquetas de tickets revisadas a mano ({RUTA_ETIQUETAS.name})")
            prods = defaultdict(lambda: {"cats": set(), "subs": set()})
            for f in filas:
                for cat, sub in ((f["categoria"], f["subcategoria"]), (f.get("categoria_2", ""), f.get("subcategoria_2", ""))):
                    if cat.strip():
                        prods[f["nombre"]]["cats"].add(cat.strip())
                        prods[f["nombre"]]["subs"].add(f"{cat.strip()}/{sub.strip()}")
            return [{"nombre": n, **v} for n, v in sorted(prods.items()) if v["cats"]]
    etiquetas = defaultdict(Counter)
    for t in json.loads((RAIZ / "tickets_prueba.json").read_text(encoding="utf-8")):
        for it in t["items"]:
            for c in it.get("categorias") or []:
                etiquetas[it["descripcion"]][(c["categoria"], c["subcategoria"])] += 1
    return [{"nombre": n, "cats": {c for c, _ in cnt}, "subs": {f"{c}/{s}" for c, s in cnt}}
            for n, cnt in sorted(etiquetas.items())]


def principal(p, campo="cats"):
    return sorted(p[campo])[0]


# ── Representaciones ──
def _lsa(textos, analizador, rango, dims=256):
    vec = TfidfVectorizer(analyzer=analizador, ngram_range=rango, sublinear_tf=True,
                          token_pattern=r"[a-zñ]{2,}" if analizador == "word" else None)
    X = vec.fit_transform(textos)
    d = min(dims, X.shape[1] - 1, X.shape[0] - 1)
    return normalize(TruncatedSVD(d, random_state=SEMILLA).fit_transform(X))


def representar(nombre_rep, textos, clientes):
    norm = [normalizar(t) for t in textos]
    if nombre_rep == "tfidf_char":
        return _lsa(norm, "char_wb", (3, 5))
    if nombre_rep == "tfidf_palabras":
        return _lsa(norm, "word", (1, 1))
    if nombre_rep == "tfidf_mixto":
        return normalize(np.hstack([_lsa(norm, "char_wb", (3, 5)), 0.6 * _lsa(norm, "word", (1, 1))]))
    if nombre_rep.startswith("emb:"):
        modelo, _, extra = nombre_rep[4:].partition("+")
        E = clientes[modelo].vectores([t.lower() for t in textos])
        if extra == "char":
            return normalize(np.hstack([E, 0.7 * _lsa(norm, "char_wb", (3, 5))]))
        return E
    raise ValueError(nombre_rep)


# ── Clustering y clasificación ──
def agrupar(X, algoritmo, k=None):
    if algoritmo == "kmeans":
        return KMeans(n_clusters=k, n_init=4, random_state=SEMILLA).fit_predict(X)
    if algoritmo == "jerarquico":
        # Ward sobre vectores normalizados (distancia euclídea ≈ coseno); el enlace medio con coseno
        # daba grupos muy desequilibrados (un grupo enorme y muchos de un solo producto)
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algoritmo == "hdbscan":
        return HDBSCAN(min_cluster_size=k or 3, min_samples=2).fit_predict(X)
    raise ValueError(algoritmo)


def clasificar_por_clusters(Xtr, ytr, Xte, algoritmo, k):
    """Agrupa el entrenamiento, etiqueta cada grupo por mayoría y asigna cada producto de prueba al
    grupo con el centroide más cercano. Devuelve (pred_cat, pred_sub, nº de grupos)."""
    etiquetas = agrupar(Xtr, algoritmo, k)
    grupos = [g for g in np.unique(etiquetas) if g >= 0]          # HDBSCAN marca el ruido con -1
    centroides, cat_g, sub_g = [], [], []
    for g in grupos:
        idx = np.where(etiquetas == g)[0]
        centroides.append(Xtr[idx].mean(axis=0))
        cat_g.append(Counter(c for i in idx for c in ytr[i]["cats"]).most_common(1)[0][0])
        sub_g.append(Counter(s for i in idx for s in ytr[i]["subs"]).most_common(1)[0][0])
    C = normalize(np.array(centroides))
    cerca = (Xte @ C.T).argmax(axis=1)
    return [cat_g[j] for j in cerca], [sub_g[j] for j in cerca], len(grupos)


def clasificar_knn(Xtr, ytr, Xte, k):
    S = Xte @ Xtr.T
    pred_c, pred_s = [], []
    for fila in S:
        vecinos = np.argsort(-fila)[:k]
        cc, cs = Counter(), Counter()
        for i in vecinos:
            for c in ytr[i]["cats"]:
                cc[c] += fila[i]
            for s in ytr[i]["subs"]:
                cs[s] += fila[i]
        pred_c.append(cc.most_common(1)[0][0])
        pred_s.append(cs.most_common(1)[0][0])
    return pred_c, pred_s


def acierto(pred, reales, campo):
    return float(np.mean([p in r[campo] for p, r in zip(pred, reales)]))


def metodos(n_cat, n_sub):
    # k = nº de categorías, nº de subcategorías y 3 veces más (grupos más finos que las subcategorías)
    return ([("kmeans", n_cat), ("kmeans", n_sub), ("kmeans", 3 * n_sub),
             ("jerarquico", n_sub), ("jerarquico", 3 * n_sub), ("hdbscan", 3)])


# ── Escenarios ──
def escenario(nombre, X_all, prods, particiones):
    """particiones: lista de (idx_train, idx_test). Devuelve filas de resultados."""
    filas = []
    n_cat = len({principal(p) for p in prods})
    n_sub = len({principal(p, "subs") for p in prods})
    for algoritmo, k in metodos(n_cat, n_sub) + [("1-NN", 1), ("5-NN", 5)]:
        t0 = time.time()
        ac_c, ac_s, ngr = [], [], []
        for tr, te in particiones:
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = [prods[i] for i in tr], [prods[i] for i in te]
            if algoritmo.endswith("NN"):
                pc, ps = clasificar_knn(Xtr, ytr, Xte, k)
                ngr.append(0)
            else:
                pc, ps, g = clasificar_por_clusters(Xtr, ytr, Xte, algoritmo, k)
                ngr.append(g)
            ac_c.append(acierto(pc, yte, "cats"))
            ac_s.append(acierto(ps, yte, "subs"))
        nombre_m = algoritmo if algoritmo.endswith("NN") else \
            f"{algoritmo} (k={k})" if algoritmo != "hdbscan" else f"hdbscan (min={k})"
        filas.append({"escenario": nombre, "metodo": nombre_m, "grupos": int(np.mean(ngr)),
                      "acierto_categoria": np.mean(ac_c), "acierto_subcategoria": np.mean(ac_s),
                      "segundos": time.time() - t0})
    return filas


def calidad_sin_etiquetas(X, prods):
    """Agrupa todo con k = nº de categorías y compara con las categorías reales."""
    y = [principal(p) for p in prods]
    k = len(set(y))
    res = {}
    for algoritmo in ("kmeans", "jerarquico"):
        et = agrupar(X, algoritmo, k)
        res[algoritmo] = {"silueta": float(silhouette_score(X, et, metric="cosine")),
                          "ARI": adjusted_rand_score(y, et), "NMI": normalized_mutual_info_score(y, et)}
    return res


def baseline_buscador(prods_test):
    pc, ps = [], []
    for p in prods_test:
        r = categorizador.buscar(p["nombre"])
        cats = r["categorias"]
        pc.append(cats[0]["categoria"] if cats else None)
        ps.append(f"{cats[0]['categoria']}/{cats[0]['subcategoria']}" if cats else None)
    return acierto(pc, prods_test, "cats"), acierto(ps, prods_test, "subs"), \
        float(np.mean([x is not None for x in pc]))


def figura_tsne(X, prods, titulo, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  (sin matplotlib: no se genera la figura)")
        return None
    Y = TSNE(n_components=2, metric="cosine", perplexity=20, random_state=SEMILLA, init="random").fit_transform(X)
    cats = [principal(p) for p in prods]
    top = [c for c, _ in Counter(cats).most_common(10)]
    plt.figure(figsize=(9, 7))
    for c in top:
        idx = [i for i, x in enumerate(cats) if x == c]
        plt.scatter(Y[idx, 0], Y[idx, 1], s=14, label=c)
    otros = [i for i, x in enumerate(cats) if x not in top]
    plt.scatter(Y[otros, 0], Y[otros, 1], s=8, c="lightgray", label="otras")
    plt.legend(fontsize=7, loc="best")
    plt.title(titulo)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


# ── Programa ──
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="*", default=[], help="modelos de embeddings de Ollama (p. ej. bge-m3)")
    ap.add_argument("--particiones", type=int, default=5)
    ap.add_argument("--solo-embeddings", action="store_true",
                    help="solo calcula y guarda en caché los embeddings (rápido); el experimento se puede ejecutar después")
    args = ap.parse_args()

    cat, tk = datos_catalogo(), datos_tickets()
    print(f"Catálogo: {len(cat)} productos · Tickets: {len(tk)} productos con categoría")
    clientes = {}
    if args.modelos:
        from embeddings import ClienteEmbeddings
        for m in args.modelos:
            cli = ClienteEmbeddings(m)
            if cli.disponible():
                clientes[m] = cli
                print(f"🤖 Embeddings: {m} ({cli.url})")
            else:
                print(f"⚠️ No se puede usar el modelo de embeddings {m} en {cli.url} (¿ollama pull {m}?)")
    if args.solo_embeddings:
        textos_emb = [p["nombre"].lower() for p in cat + tk]
        for m, cli in clientes.items():
            t0 = time.time()
            cli.vectores(textos_emb)
            print(f"✔ {m}: {len(textos_emb)} embeddings en caché ({time.time() - t0:.0f} s) -> {cli._ruta_cache}")
        return
    reps = ["tfidf_char", "tfidf_palabras", "tfidf_mixto"] + \
           [f"emb:{m}" for m in clientes] + [f"emb:{m}+char" for m in clientes]

    todos = cat + tk
    textos = [p["nombre"] for p in todos]
    n_cat = len(cat)
    kf = KFold(args.particiones, shuffle=True, random_state=SEMILLA)
    part_cat = [(tr, te) for tr, te in kf.split(np.arange(n_cat))]
    part_cat_tk = [(np.arange(n_cat), np.arange(n_cat, len(todos)))]
    part_tk = [(np.concatenate([np.arange(n_cat), n_cat + tr]), n_cat + te) for tr, te in kf.split(np.arange(len(tk)))]

    filas, calidad, figuras = [], {}, []
    salida = RAIZ / "resultados"
    salida.mkdir(exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    for rep in reps:
        print(f"\n▶ {rep}")
        t0 = time.time()
        X = representar(rep, textos, clientes)
        print(f"  vectores {X.shape} en {time.time() - t0:.1f} s")
        for nombre, part in (("A. catálogo (CV)", part_cat), ("B. catálogo → tickets", part_cat_tk),
                             ("C. tickets (CV, + catálogo)", part_tk)):
            for f in escenario(nombre, X, todos, part):
                f["representacion"] = rep
                filas.append(f)
                print(f"  {nombre:30s} {f['metodo']:22s} cat {100 * f['acierto_categoria']:5.1f} %  "
                      f"sub {100 * f['acierto_subcategoria']:5.1f} %  ({f['segundos']:.1f} s)")
        calidad[rep] = {"catalogo": calidad_sin_etiquetas(X[:n_cat], cat),
                        "tickets": calidad_sin_etiquetas(X[n_cat:], tk)}
        fig = figura_tsne(X[n_cat:], tk, f"Productos de los tickets · {rep}",
                          salida / f"clustering_tsne_{rep.replace(':', '_').replace('+', '_')}_{marca}.png")
        if fig:
            figuras.append(fig)

    base_c, base_s, cobertura = baseline_buscador(tk)
    md = informe(filas, calidad, (base_c, base_s, cobertura), len(cat), len(tk), reps)
    (salida / f"resumen_clustering_{marca}.md").write_text(md, encoding="utf-8")
    with open(salida / f"clustering_{marca}.json", "w", encoding="utf-8") as fh:
        json.dump({"filas": filas, "calidad": calidad}, fh, ensure_ascii=False, indent=1, default=float)
    print("\n" + md)
    print(f"📄 Resultados en {salida}")


def informe(filas, calidad, base, n_cat, n_tk, reps):
    pct = lambda x: f"{100 * x:.1f} %"  # noqa: E731
    l = [f"# Clasificación de productos mediante clustering — {datetime.now():%d/%m/%Y %H:%M}", "",
         f"Catálogo: {n_cat} productos (26 categorías, 150 subcategorías) · Tickets: {n_tk} productos con categoría", ""]
    escenarios = list(dict.fromkeys(f["escenario"] for f in filas))
    for esc in escenarios:
        l += [f"## {esc} — acierto de categoría (subcategoría)", "",
              "| Método | " + " | ".join(reps) + " |", "|---|" + "---|" * len(reps)]
        for m in dict.fromkeys(f["metodo"] for f in filas if f["escenario"] == esc):
            celdas = []
            for r in reps:
                f = next(x for x in filas if x["escenario"] == esc and x["metodo"] == m and x["representacion"] == r)
                celdas.append(f"{pct(f['acierto_categoria'])} ({pct(f['acierto_subcategoria'])})")
            l.append(f"| {m} | " + " | ".join(celdas) + " |")
        if esc.startswith("B") or esc.startswith("C"):
            l.append(f"| *Buscador por palabras del catálogo (actual)* | {pct(base[0])} ({pct(base[1])}), "
                     f"responde en el {pct(base[2])} | " + " | ".join("" for _ in reps[1:]) + " |")
        l.append("")
    l += ["## Calidad del agrupamiento sin etiquetas (k = nº de categorías)", "",
          "| Representación | Datos | Algoritmo | Silueta | ARI | NMI |", "|---|---|---|---|---|---|"]
    for r in reps:
        for datos in ("catalogo", "tickets"):
            for alg, v in calidad[r][datos].items():
                l.append(f"| {r} | {datos} | {alg} | {v['silueta']:.3f} | {v['ARI']:.3f} | {v['NMI']:.3f} |")
    l += ["", "## Mejor configuración por escenario", ""]
    for esc in escenarios:
        cl = [f for f in filas if f["escenario"] == esc and not f["metodo"].endswith("NN")]
        b = max(cl, key=lambda f: f["acierto_categoria"])
        nn = max([f for f in filas if f["escenario"] == esc and f["metodo"].endswith("NN")], key=lambda f: f["acierto_categoria"])
        l.append(f"- **{esc}**: clustering → {b['representacion']} + {b['metodo']}: {pct(b['acierto_categoria'])} "
                 f"(subcat. {pct(b['acierto_subcategoria'])}); referencia k-NN → {nn['representacion']} + {nn['metodo']}: "
                 f"{pct(nn['acierto_categoria'])}")
    l += ["", "Notas: «acierto» = la categoría asignada está entre las del producto (un producto puede tener varias). "
          "Las etiquetas de los tickets son las de etiquetas_tickets.csv (revisadas a mano), independientes del buscador."]
    return "\n".join(l) + "\n"


if __name__ == "__main__":
    main()
