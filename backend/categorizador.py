"""
Categorizador de productos a partir del catálogo de Mercadona (catalogo_mercadona.csv).

Los nombres que aparecen en los tickets están abreviados y en mayúsculas
("LECHE ENTERA P6", "Q. UNTAR SUAVE", "COLA ZERO LATA"), mientras que en el
catálogo aparecen completos ("Leche entera Hacendado", ...). Por eso no se
puede hacer una búsqueda exacta: se usa un emparejamiento por tokens que
admite abreviaturas (prefijos) y pondera cada palabra por su rareza (IDF).

Un mismo producto puede aparecer en varias filas del CSV con distintas
categorías (p. ej. "Tomate frito Hacendado" está en "Otras salsas" y en
"Tomate"), así que cada producto del catálogo guarda un conjunto de
categorías y el resultado siempre es una lista.
"""
import csv
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CSV_PATH = Path(__file__).parent / "catalogo_mercadona.csv"

# Umbral mínimo de similitud para aceptar una categoría automática
UMBRAL_CONFIANZA = 0.55

# Donut a veces genera letras griegas visualmente idénticas a las latinas
# (p. ej. "ΒΑΝΑΝΑ", "ΡΕΡΙΝΟ"). Se traducen antes de comparar.
_GRIEGO_A_LATINO = str.maketrans({
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x",
})

# Palabras que no aportan información sobre el tipo de producto
_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "con", "sin", "y", "en", "al", "a",
    "para", "por", "c", "s", "pack", "pk", "p", "uds", "ud", "uni", "unid",
    "kg", "g", "gr", "l", "ml", "cl", "hacendado", "bosque", "verde",
    "deliplus", "compy",
}

# Abreviaturas frecuentes en los tickets de Mercadona
_ABREVIATURAS = {
    "q": "queso", "j": "jamon", "t": "tomate", "z": "zumo", "gall": "galleta",
    "desn": "desnatada", "semi": "semidesnatada", "pechu": "pechuga",
    "croiss": "croissant", "salch": "salchicha", "deo": "desodorante",
    "colg": "colgador", "pat": "patatas", "energ": "energetica",
    "cong": "congelada", "rec": "recambio", "desma": "desmaquillantes",
}


def normalizar(texto: str) -> str:
    texto = (texto or "").translate(_GRIEGO_A_LATINO).lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ]+", " ", texto).strip()


def tokenizar(texto: str, expandir: bool = False) -> List[str]:
    tokens = []
    for tok in normalizar(texto).split():
        if expandir:
            tok = _ABREVIATURAS.get(tok, tok)
        # Se descartan cantidades y formatos: "p6", "4pack", "1kg", "500", "2l"...
        if any(ch.isdigit() for ch in tok):
            continue
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        tokens.append(tok)
    return tokens


def _singular(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


class Categorizador:
    def __init__(self, csv_path: Path = CSV_PATH):
        # nombre del catálogo -> {(categoria, subcategoria), ...}
        self.categorias_por_producto: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.tokens_por_producto: Dict[str, List[str]] = {}
        self.idf: Dict[str, float] = {}
        self.arbol: Dict[str, Set[str]] = defaultdict(set)  # categoría -> subcategorías
        self._cargar(csv_path)

    def _cargar(self, csv_path: Path) -> None:
        if not csv_path.exists():
            print(f"⚠️ No se encontró el catálogo en {csv_path}; la categorización quedará desactivada.")
            return
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for fila in csv.DictReader(f, delimiter=";"):
                nombre = (fila.get("Nombre") or "").strip()
                cat = (fila.get("Categoria_Superior") or "").strip()
                sub = (fila.get("Subcategoria") or "").strip()
                if not nombre or not cat:
                    continue
                self.categorias_por_producto[nombre].add((cat, sub))
                self.arbol[cat].add(sub)

        df = defaultdict(int)
        for nombre in self.categorias_por_producto:
            toks = [_singular(t) for t in tokenizar(nombre)]
            self.tokens_por_producto[nombre] = toks
            for t in set(toks):
                df[t] += 1
        n = max(len(self.tokens_por_producto), 1)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}
        print(f"🗂️ Catálogo cargado: {n} productos, {len(self.arbol)} categorías.")

    # ── Emparejamiento ──
    def _similitud_token(self, q: str, c: str) -> float:
        if q == c:
            return 1.0
        # Abreviatura del ticket: "ultracong" -> "ultracongelado", "choc" -> "chocolate".
        # Si la palabra del ticket ya existe completa en el catálogo ("agua", "pan",
        # "limon"), no se trata como abreviatura para evitar "agua" -> "aguacate".
        es_palabra = q in self.idf
        if len(q) >= 3 and c.startswith(q):
            return 0.35 if es_palabra else 0.9
        if len(c) >= 5 and q.startswith(c):
            return 0.35 if es_palabra else 0.8
        return 0.0

    def _puntuar(self, q_toks: List[str], c_toks: List[str]) -> float:
        if not q_toks or not c_toks:
            return 0.0
        peso_total = 0.0
        peso_ok = 0.0
        for i, q in enumerate(q_toks):
            # La primera palabra del ticket suele ser el "tipo" de producto
            importancia = 1.5 if i == 0 else 1.0
            mejor, idf = 0.0, self.idf.get(q, 3.0)
            for c in c_toks:
                s = self._similitud_token(q, c)
                if s > mejor:
                    mejor, idf = s, self.idf.get(c, idf)
            peso_total += importancia * self.idf.get(q, idf)
            peso_ok += importancia * mejor * idf
        puntuacion = peso_ok / peso_total if peso_total else 0.0
        # Bonus si la palabra principal coincide también en el catálogo
        if self._similitud_token(q_toks[0], c_toks[0]) > 0:
            puntuacion += 0.1
        # Ligera penalización a nombres del catálogo muy largos (más específicos)
        puntuacion -= 0.01 * max(len(c_toks) - len(q_toks), 0)
        return puntuacion

    def buscar(self, descripcion: str) -> Dict:
        """Devuelve el producto del catálogo más parecido y sus categorías."""
        q_toks = [_singular(t) for t in tokenizar(descripcion, expandir=True)]
        vacio = {"categorias": [], "producto_catalogo": None, "confianza": 0.0}
        if not q_toks or not self.tokens_por_producto:
            return vacio

        mejor_nombre: Optional[str] = None
        mejor_punt = 0.0
        for nombre, c_toks in self.tokens_por_producto.items():
            p = self._puntuar(q_toks, c_toks)
            if p > mejor_punt or (p == mejor_punt and mejor_nombre and len(nombre) < len(mejor_nombre)):
                mejor_nombre, mejor_punt = nombre, p

        if not mejor_nombre or mejor_punt < UMBRAL_CONFIANZA:
            return {**vacio, "confianza": round(mejor_punt, 2)}

        categorias = [
            {"categoria": c, "subcategoria": s}
            for c, s in sorted(self.categorias_por_producto[mejor_nombre])
        ]
        return {
            "categorias": categorias,
            "producto_catalogo": mejor_nombre,
            "confianza": round(min(mejor_punt, 1.0), 2),
        }

    def listar_categorias(self) -> List[Dict]:
        return [
            {"categoria": cat, "subcategorias": sorted(subs)}
            for cat, subs in sorted(self.arbol.items())
        ]


# Instancia única que se carga al arrancar el servidor
categorizador = Categorizador()
