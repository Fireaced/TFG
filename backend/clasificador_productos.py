"""
Clasificación de productos por similitud (respaldo del buscador del catálogo).

Resultado del experimento validacion/evaluar_clustering.py + evaluar_hibrido.py: el buscador por
palabras del catálogo es el más preciso cuando encuentra el producto (92,1 %), pero cuando no responde
o su confianza es baja (< 0,75) es mejor asignar la categoría del producto conocido MÁS PARECIDO
(1-NN) en un espacio que combina:
  - embeddings semánticos del modelo bge-m3 (Ollama), que entienden el significado, y
  - TF-IDF de n-gramas de caracteres (LSA), que entienden las abreviaturas de los tickets.
Con esa combinación, el sistema híbrido llega al 95,7 % en los productos de los tickets.

Productos de referencia: los 3.931 del catálogo de Mercadona + los de los tickets guardados (con las
categorías revisadas por el usuario). Si Ollama no está disponible se usa solo la parte TF-IDF.
"""
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from categorizador import normalizar

UMBRAL_BUSCADOR = 0.75     # por debajo de esta confianza del buscador se usa el vecino más parecido
PESO_CARACTERES = 0.7      # peso de la parte TF-IDF frente al embedding (como en el experimento)


class ClasificadorProductos:
    def __init__(self, categorizador, cliente_embeddings=None):
        self.categorizador = categorizador
        self._cliente = cliente_embeddings
        self.nombres: List[str] = []
        self.categorias: List[List[Dict]] = []
        self.origen: List[str] = []          # "catalogo" | "historial"
        self._indice: Dict[str, int] = {}
        self.X: Optional[np.ndarray] = None
        self.emb = None                      # ClienteEmbeddings si Ollama responde
        self.listo = False
        self._lock = threading.Lock()

    # ── Construcción del índice ──
    def construir(self, productos_historial: List[Tuple[str, List[Dict]]]):
        """Se llama en segundo plano al arrancar: catálogo + productos de los tickets guardados."""
        t0 = time.time()
        nombres, cats, origen = [], [], []
        for nombre, pares in self.categorizador.categorias_por_producto.items():
            nombres.append(nombre)
            cats.append([{"categoria": c, "subcategoria": s} for c, s in sorted(pares)])
            origen.append("catalogo")
        for nombre, c in productos_historial:
            if nombre and c:
                nombres.append(nombre)
                cats.append(c)
                origen.append("historial")

        textos = [normalizar(n) for n in nombres]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(textos)
        T = self.vec.transform(textos)
        self.svd = TruncatedSVD(min(256, T.shape[1] - 1), random_state=0).fit(T)
        C = normalize(self.svd.transform(T))

        E = None
        if os.environ.get("EMB_ACTIVADO", "true").lower() in ("1", "true", "si", "sí", "yes"):
            try:
                from embeddings import ClienteEmbeddings
                cliente = self._cliente or ClienteEmbeddings()
                E = cliente.vectores([n.lower() for n in nombres])
                self.emb = cliente
            except Exception as e:
                print(f"⚠️ Embeddings no disponibles ({type(e).__name__}): la clasificación por similitud usará solo TF-IDF")
        X = normalize(np.hstack([E, PESO_CARACTERES * C])) if E is not None else C
        with self._lock:
            self.nombres, self.categorias, self.origen = nombres, cats, origen
            self._indice = {normalizar(n): i for i, n in enumerate(nombres)}
            self.X = X
            self.listo = True
        modo = f"embeddings {self.emb.modelo} + caracteres" if self.emb else "solo caracteres (TF-IDF)"
        print(f"🧭 Clasificador por similitud listo: {len(nombres)} productos de referencia, {modo} ({time.time() - t0:.1f} s)")

    def _vector(self, nombres: List[str]) -> np.ndarray:
        C = normalize(self.svd.transform(self.vec.transform([normalizar(n) for n in nombres])))
        if self.emb is None:
            return C
        try:
            E = self.emb.vectores([n.lower() for n in nombres])
        except Exception:
            # Ollama se ha cerrado después de arrancar: se completa con ceros la parte semántica
            E = np.zeros((len(nombres), self.X.shape[1] - C.shape[1]), dtype=np.float32)
        return normalize(np.hstack([E, PESO_CARACTERES * C]))

    def añadir(self, productos: List[Tuple[str, List[Dict]]]):
        """Añade (o actualiza) productos del historial tras guardar un ticket revisado."""
        if not self.listo:
            return
        nuevos = []
        with self._lock:
            for nombre, cats in productos:
                if not nombre or not cats:
                    continue
                i = self._indice.get(normalizar(nombre))
                if i is not None and self.origen[i] == "historial":
                    self.categorias[i] = cats            # el usuario ha corregido su categoría
                elif i is None:
                    nuevos.append((nombre, cats))
        if not nuevos:
            return
        V = self._vector([n for n, _ in nuevos])
        with self._lock:
            for (nombre, cats), v in zip(nuevos, V):
                self._indice[normalizar(nombre)] = len(self.nombres)
                self.nombres.append(nombre)
                self.categorias.append(cats)
                self.origen.append("historial")
            self.X = np.vstack([self.X, V])

    # ── Consulta ──
    def vecino(self, descripcion: str) -> Optional[Dict]:
        """Producto conocido más parecido: {"categorias", "nombre", "similitud", "origen"} o None."""
        if not self.listo or not normalizar(descripcion):
            return None
        v = self._vector([descripcion])[0]
        with self._lock:
            sims = self.X @ v
            i = int(np.argmax(sims))
            return {"categorias": self.categorias[i], "nombre": self.nombres[i],
                    "similitud": float(sims[i]), "origen": self.origen[i]}
