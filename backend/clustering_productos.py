"""
Clustering de los productos comprados por el usuario.

Los nombres de los tickets son abreviados, a veces con erratas (sobre todo en
fotos) y no siempre coinciden con el catálogo. Para que el asistente entienda
conceptos como "yogures" o "pan" se agrupan automáticamente (aprendizaje no
supervisado) todos los productos distintos que aparecen en los tickets.

Representación de cada producto (vector disperso):
  - TF-IDF de n-gramas de caracteres (3-5) del nombre normalizado: robusto
    frente a abreviaturas y letras que faltan ("YOGR LIQUIDO" ~ "YOGUR LIQUIDO").
  - TF-IDF de palabras del nombre.
  - One-hot de las subcategorías asignadas por el categorizador.

Algoritmo: K-Means sobre los vectores normalizados (distancia ~ coseno). El
número de grupos k se elige automáticamente maximizando el coeficiente de
silueta. Cada grupo se etiqueta con su subcategoría más frecuente y sus
palabras más representativas.
"""
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from categorizador import normalizar

PESO_CHAR, PESO_PALABRA, PESO_CATEGORIA = 1.0, 0.6, 1.2


class ClusteringProductos:
    def __init__(self, semilla: int = 42):
        self.semilla = semilla
        self.entrenado = False
        self.productos: List[str] = []            # nombres normalizados únicos
        self.nombre_original: Dict[str, str] = {}
        self.cats_producto: Dict[str, List[Tuple[str, str]]] = {}
        self.etiquetas: np.ndarray = np.array([])
        self.grupos: Dict[int, dict] = {}
        self.k = 0
        self.silueta = None

    # ── Representación ──
    def _vector_categorias(self, lista_cats) -> csr_matrix:
        filas, cols = [], []
        for i, cats in enumerate(lista_cats):
            for c in cats:
                j = self._idx_cat.get(c[1] or c[0])
                if j is not None:
                    filas.append(i); cols.append(j)
        datos = np.ones(len(filas))
        m = csr_matrix((datos, (filas, cols)), shape=(len(lista_cats), max(1, len(self._idx_cat))))
        return normalize(m)

    def _vectorizar(self, nombres: List[str], lista_cats) -> csr_matrix:
        xc = self.vec_char.transform(nombres) * PESO_CHAR
        xp = self.vec_palabra.transform(nombres) * PESO_PALABRA
        xk = self._vector_categorias(lista_cats) * PESO_CATEGORIA
        return normalize(hstack([xc, xp, xk]).tocsr())

    # ── Entrenamiento ──
    def entrenar(self, items: List[dict]) -> "ClusteringProductos":
        """items: [{'descripcion': str, 'categorias': [{'categoria','subcategoria'}]}] (puede haber repetidos)."""
        cats_por_nombre: Dict[str, Counter] = {}
        for it in items:
            n = normalizar(it.get("descripcion", ""))
            if not n:
                continue
            self.nombre_original.setdefault(n, it.get("descripcion", ""))
            cnt = cats_por_nombre.setdefault(n, Counter())
            for c in it.get("categorias") or []:
                cnt[(c.get("categoria", ""), c.get("subcategoria", ""))] += 1
        self.productos = sorted(cats_por_nombre)
        self.cats_producto = {n: [c for c, _ in cats_por_nombre[n].most_common()] for n in self.productos}

        n = len(self.productos)
        if n < 6:
            self.entrenado = False
            return self

        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(self.productos)
        self.vec_palabra = TfidfVectorizer(analyzer="word", token_pattern=r"[a-zñ]{2,}", sublinear_tf=True).fit(self.productos)
        subcats = sorted({c[1] or c[0] for cats in self.cats_producto.values() for c in cats})
        self._idx_cat = {s: i for i, s in enumerate(subcats)}
        self.X = self._vectorizar(self.productos, [self.cats_producto[p] for p in self.productos])

        # Elección de k por coeficiente de silueta
        k_min, k_max = max(2, n // 12), max(3, min(n // 3, 80))
        candidatos = sorted(set(np.linspace(k_min, k_max, num=min(12, k_max - k_min + 1)).astype(int)))
        mejor = (None, -1.0, None)
        for k in candidatos:
            km = KMeans(n_clusters=k, n_init=5, random_state=self.semilla).fit(self.X)
            s = silhouette_score(self.X, km.labels_, metric="cosine")
            if s > mejor[1]:
                mejor = (k, s, km)
        self.k, self.silueta, self.modelo = int(mejor[0]), float(mejor[1]), mejor[2]
        self.etiquetas = self.modelo.labels_
        self._describir_grupos()
        self.entrenado = True
        return self

    def _describir_grupos(self):
        palabras = np.array(self.vec_palabra.get_feature_names_out())
        Xp = self.vec_palabra.transform(self.productos)
        self.grupos = {}
        for g in range(self.k):
            idx = np.where(self.etiquetas == g)[0]
            miembros = [self.productos[i] for i in idx]
            subcats = Counter(c[1] or c[0] for m in miembros for c in self.cats_producto[m][:1])
            pesos = np.asarray(Xp[idx].sum(axis=0)).ravel()
            top = [palabras[i] for i in pesos.argsort()[::-1][:3] if pesos[i] > 0]
            principal = subcats.most_common(1)[0][0] if subcats else None
            par = Counter(c for m in miembros for c in self.cats_producto[m][:1]).most_common(1)
            self.grupos[g] = {
                "id": int(g),
                "categoria_principal": list(par[0][0]) if par else None,
                "nombre": principal or " / ".join(top) or f"Grupo {g}",
                "palabras_clave": top,
                "subcategoria_principal": principal,
                "miembros": [self.nombre_original[m] for m in miembros],
            }

    # ── Consultas ──
    def grupo_de(self, descripcion: str) -> Optional[int]:
        if not self.entrenado:
            return None
        n = normalizar(descripcion)
        if n in self._pos:
            return int(self.etiquetas[self._pos[n]])
        return int(self.modelo.predict(self._vectorizar([n], [[]]))[0])

    @property
    def _pos(self):
        if not hasattr(self, "_pos_cache") or len(self._pos_cache) != len(self.productos):
            self._pos_cache = {p: i for i, p in enumerate(self.productos)}
        return self._pos_cache

    def similares(self, texto: str, umbral: float = 0.35, solo_nombre: bool = True) -> List[Tuple[str, float]]:
        """Productos del historial más parecidos a un texto (por n-gramas de caracteres y palabras)."""
        if not self.entrenado:
            return []
        q = normalizar(texto)
        vq = normalize(hstack([self.vec_char.transform([q]) * PESO_CHAR, self.vec_palabra.transform([q]) * PESO_PALABRA]).tocsr())
        Xn = normalize(hstack([self.vec_char.transform(self.productos) * PESO_CHAR,
                               self.vec_palabra.transform(self.productos) * PESO_PALABRA]).tocsr())
        sims = (Xn @ vq.T).toarray().ravel()
        orden = sims.argsort()[::-1]
        return [(self.productos[i], float(sims[i])) for i in orden if sims[i] >= umbral]

    def categorias_por_vecinos(self, descripcion: str, umbral: float = 0.6):
        """Categorías del producto más parecido del historial (útil cuando una errata impide encontrarlo en el catálogo)."""
        for nombre, sim in self.similares(descripcion, umbral=umbral)[:5]:
            if self.cats_producto.get(nombre):
                cats = [{"categoria": c, "subcategoria": s} for c, s in self.cats_producto[nombre]]
                return cats, self.nombre_original[nombre], sim
        return None

    def categoria_por_grupo(self, descripcion: str, umbral: float = 0.3):
        """Categoría de un producto sin categoría (p. ej. por una errata de impresión) a partir de su grupo:
        se le asigna la categoría principal del grupo si se parece lo suficiente a algún miembro categorizado."""
        if not self.entrenado:
            return None
        g = self.grupo_de(descripcion)
        info = self.grupos.get(g)
        if not info or not info.get("categoria_principal"):
            return None
        miembros = {normalizar(m) for m in info["miembros"]}
        parecidos = [(n, s) for n, s in self.similares(descripcion, umbral=umbral)
                     if n in miembros and self.cats_producto.get(n) and n != normalizar(descripcion)]
        if not parecidos:
            return None
        cat, sub = info["categoria_principal"]
        return [{"categoria": cat, "subcategoria": sub}], info["nombre"], parecidos[0][1]

    def inferir_categorias(self, descripcion: str, umbral_vecino: float = 0.5):
        """Para productos sin categoría: primero por su grupo (clustering) y, si no, por el vecino más parecido."""
        r = self.categoria_por_grupo(descripcion)
        if r:
            return r[0], "clustering"
        r = self.categorias_por_vecinos(descripcion, umbral=umbral_vecino)
        if r:
            return r[0], "vecino"
        return None, None

    def resumen(self) -> dict:
        return {
            "entrenado": self.entrenado,
            "productos": len(self.productos),
            "k": self.k,
            "silueta": round(self.silueta, 3) if self.silueta is not None else None,
            "grupos": sorted(self.grupos.values(), key=lambda g: -len(g["miembros"])),
        }
