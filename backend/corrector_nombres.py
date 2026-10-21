"""
Corrección automática de nombres de producto mal leídos por Donut.

Donut a veces lee mal una letra o parte una palabra, sobre todo en fotos
("CROQLETA JAMON", "EDP FOSE NUDE", "PECHL GA DE POLLO 92%"). Este módulo
propone la corrección y decide si es lo bastante segura para aplicarla sin
preguntar al usuario.

Vocabulario de referencia (de más a menos fiable):
  1. Correcciones que ya hizo el usuario (leído -> correcto).
  2. Nombres de producto de tickets ya guardados (revisados por el usuario)
     y del dataset etiquetado: así es como Mercadona escribe los productos.
  3. Palabras del catálogo de Mercadona (sirven para productos nuevos).

Se compara con una distancia de edición ponderada para OCR: confundir letras
que se parecen impresas (U/L, R/F, O/Q, E/F, C/G...) o partir/juntar palabras
cuesta menos que un cambio cualquiera.

Una corrección solo se aplica sola si:
  - el nombre leído no es ya un nombre conocido,
  - la corrección está muy cerca (pocas letras distintas para su longitud),
  - no hay otra alternativa igual de cercana (sin empates),
  - y el resultado es un nombre conocido o encaja con el catálogo mejor que el
    nombre leído.
Si no se cumple todo, se devuelve como sugerencia para que decida el usuario.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Pares de caracteres que se confunden al leer tickets impresos (coste reducido)
_PARECIDOS = [
    "UL", "UV", "UO", "RF", "RP", "RB", "OQ", "OD", "O0", "OC", "EF", "EB", "CG", "CO", "IL", "I1", "L1",
    "S5", "B8", "ZS", "Z2", "NH", "NM", "MH", "A4", "TI", "TL", "G6", "KX", "PF", "ÑN", "YV",
]
COSTE_PARECIDO = 0.5
COSTE_ESPACIO = 0.5      # partir o juntar palabras ("PECHU GA")
COSTE_NORMAL = 1.0
_PAR = {(a, b) for a, b in _PARECIDOS} | {(b, a) for a, b in _PARECIDOS}


def normalizar_nombre(texto: str) -> str:
    texto = unicodedata.normalize("NFD", str(texto or "").upper())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn" or c == "̃")
    texto = unicodedata.normalize("NFC", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _sustitucion(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if (a, b) in _PAR:
        return COSTE_PARECIDO
    return COSTE_NORMAL


def distancia(a: str, b: str, tope: float = 9.0) -> float:
    """Levenshtein ponderada para OCR (espacios baratos). Corta pronto si supera `tope`."""
    if a == b:
        return 0.0
    if abs(len(a) - len(b)) > tope * 2 + 1:
        return tope + 1
    prev = [0.0]
    for ch in b:
        prev.append(prev[-1] + (COSTE_ESPACIO if ch == " " else COSTE_NORMAL))
    for ca in a:
        cur = [prev[0] + (COSTE_ESPACIO if ca == " " else COSTE_NORMAL)]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + (COSTE_ESPACIO if ca == " " else COSTE_NORMAL),
                cur[j - 1] + (COSTE_ESPACIO if cb == " " else COSTE_NORMAL),
                prev[j - 1] + _sustitucion(ca, cb),
            ))
        if min(cur) > tope:
            return tope + 1
        prev = cur
    return prev[-1]


def _limite(longitud: int) -> float:
    """Distancia máxima admitida según la longitud de la palabra o nombre."""
    if longitud <= 3:
        return 0.0
    if longitud <= 5:
        return 1.0
    if longitud <= 9:
        return 1.5
    return 2.0


def _es_palabra(tok: str) -> bool:
    """Palabra de letras, admitiendo dígitos que en realidad son letras mal leídas ("B4CON", "P1ZZA")
    pero no cantidades ni formatos ("P6", "4PACK", "500G", "1,5L")."""
    if re.fullmatch(r"[A-ZÑ]+", tok):
        return True
    return bool(re.fullmatch(r"[A-ZÑ]+[0-9][A-ZÑ0-9]*", tok)) and sum(c.isalpha() for c in tok) >= 3 \
        and sum(c.isdigit() for c in tok) == 1 and not re.fullmatch(r"[A-ZÑ]+[0-9]+(?:G|KG|L|ML|CL|UDS?|X)?", tok)


def _numeros(nombre: str) -> List[str]:
    """Números de verdad (cantidades, %, formatos), sin contar dígitos dentro de palabras."""
    return [n for tok in nombre.split() if not _es_palabra(tok) for n in re.findall(r"\d+", tok)]


class CorrectorNombres:
    def __init__(self, categorizador=None):
        self.categorizador = categorizador
        self.nombres: Dict[str, int] = {}          # nombre de ticket conocido -> nº de veces visto
        self.palabras: Counter = Counter()         # palabra -> peso (los tickets pesan más que el catálogo)
        self.aprendidas: Dict[str, str] = {}       # leído -> corregido (por el usuario)
        self._por_longitud: Dict[int, List[str]] = defaultdict(list)
        # Palabras de formato que aparecen en los tickets pero no en los nombres del catálogo
        for p in ("PACK", "LATA", "LATAS", "BOTELLA", "BOTELLAS", "MALLA", "BOLSA", "BANDEJA", "BRIK", "GARRAFA",
                  "UDS", "UNI", "UNID", "TARRO", "FRASCO", "BOTE", "CAJA", "PAQUETE", "FRIO", "FRIA", "FAMIL"):
            self.palabras[p] += 3
        if categorizador is not None:
            for nombre in categorizador.categorias_por_producto:
                for p in normalizar_nombre(nombre).split():
                    p = re.sub(r"[^A-ZÑ]", "", p)
                    if len(p) >= 2:
                        self.palabras[p] += 1
        self._indexar()

    # ── Vocabulario ──
    def añadir_nombres(self, nombres: Iterable[str], peso: int = 5):
        for n in nombres:
            n = normalizar_nombre(n)
            if not n:
                continue
            self.nombres[n] = self.nombres.get(n, 0) + 1
            for p in re.split(r"[\s.]+", n):
                if _es_palabra(p) and len(p) >= 2:
                    self.palabras[p] += peso
        self._indexar()

    def añadir_aprendidas(self, pares: Iterable[Tuple[str, str]]):
        for leido, correcto in pares:
            if leido and correcto and normalizar_nombre(leido) != normalizar_nombre(correcto):
                self.aprendidas[normalizar_nombre(leido)] = normalizar_nombre(correcto)
        self.añadir_nombres([c for _, c in pares])

    def _indexar(self):
        self._por_longitud = defaultdict(list)
        for p in self.palabras:
            self._por_longitud[len(p)].append(p)
        self._nombres_por_longitud = defaultdict(list)
        for n in self.nombres:
            self._nombres_por_longitud[len(n)].append(n)

    def _es_conocida(self, p: str) -> bool:
        """Palabra del vocabulario o abreviatura de una (los tickets abrevian: "PECHU", "SEMIDESN")."""
        if p in self.palabras:
            return True
        if len(p) >= 3:
            return any(w.startswith(p) for L in range(len(p) + 1, len(p) + 12) for w in self._por_longitud.get(L, ()))
        return False

    def _candidatas(self, p: str, tope: float) -> List[Tuple[float, str]]:
        res = []
        for L in range(max(1, len(p) - 2), len(p) + 3):
            for w in self._por_longitud.get(L, ()):
                d = distancia(p, w, tope)
                if d <= tope:
                    res.append((d, w))
        res.sort(key=lambda x: (x[0], -self.palabras[x[1]]))
        return res

    # ── Corrección ──
    def corregir(self, leido: str, nombres_extra: Iterable[str] = ()) -> Optional[dict]:
        """Devuelve None si el nombre parece correcto, o
        {"nombre", "original", "automatica": bool, "motivo", "distancia"}."""
        original = normalizar_nombre(leido)
        if not original:
            return None
        extra = {normalizar_nombre(n) for n in nombres_extra if n}
        if original in self.aprendidas:
            return self._res(original, self.aprendidas[original], True, "corrección que ya hiciste antes", 0)
        if original in self.nombres or original in extra:
            return None

        # 1) Nombre completo muy parecido a uno conocido (texto del propio PDF o historial)
        completo = self._mejor_nombre(original, extra | set(self.nombres))
        if completo:
            nombre, d, unico = completo
            return self._res(original, nombre, unico, "producto que ya has comprado" if nombre in self.nombres
                             else "nombre del propio ticket", d)

        # 2) Palabra a palabra con el vocabulario (productos nuevos)
        tokens = original.split(" ")
        nuevos, total, dudoso, cambios = [], 0.0, False, 0
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not _es_palabra(tok) or len(tok) < 3 or self._es_conocida(tok):
                # ¿Dos trozos que juntos forman una palabra? ("PECHL GA" -> "PECHUGA")
                if i + 1 < len(tokens) and _es_palabra(tok) and _es_palabra(tokens[i + 1]) and \
                        (min(len(tok), len(tokens[i + 1])) <= 3 or not self._es_conocida(tokens[i + 1])):
                    r = self._unir(tok, tokens[i + 1])
                    if r and r[1] == 0:            # "PECHU GA" -> "PECHUGA" (solo si sale una palabra exacta)
                        nuevos.append(r[0]); total += COSTE_ESPACIO; cambios += 1
                        i += 2
                        continue
                nuevos.append(tok)
                i += 1
                continue
            # Palabra desconocida: ¿juntándola con la siguiente sale una conocida?
            if i + 1 < len(tokens) and _es_palabra(tokens[i + 1]):
                r = self._unir(tok, tokens[i + 1])
                if r:
                    w, d, unico = r
                    nuevos.append(w); total += d + COSTE_ESPACIO; dudoso |= not unico; cambios += 1
                    i += 2
                    continue
            cands = self._candidatas(tok, _limite(len(tok)))
            if not cands:
                return None                       # palabra desconocida sin corrección: puede ser un producto nuevo
            d, w = cands[0]
            empate = [c for c in cands[1:] if c[0] - d < 0.5]
            if empate:
                # Desempate por contexto: el que mejor encaja con el catálogo
                w, dudoso_ctx = self._desempatar(tokens, i, [w] + [c[1] for c in empate])
                dudoso |= dudoso_ctx
            nuevos.append(w); total += d; cambios += 1
            i += 1
        if not cambios:
            return None
        corregido = " ".join(nuevos)
        if corregido == original:
            return None
        if total > _limite(len(original.replace(" ", ""))) + 0.5:
            return None
        if not self._cambio_permitido(original, corregido):
            return None
        if not (corregido in self.nombres or self._encaja_mejor(corregido, original)):
            return None                           # la corrección no se puede confirmar: mejor no proponer nada
        return self._res(original, corregido, not dudoso,
                         "palabras corregidas con el catálogo y tu historial", total)

    def _palabras(self, nombre: str) -> List[str]:
        return [p for p in re.split(r"[\s./,-]+", nombre) if p]

    def _cambio_permitido(self, original: str, candidato: str) -> bool:
        """Salvaguardas: no se tocan números ni palabras que ya son correctas."""
        if _numeros(original) != _numeros(candidato):
            return False                                   # "6 HUEVOS" no puede pasar a "12 HUEVOS"
        palabras_cand = set(self._palabras(candidato))
        conocidas = [p for p in self._palabras(original) if _es_palabra(p) and len(p) >= 2 and p in self.palabras]
        # "FUENTE" no puede pasar a "PUENTE"; sí puede unirse a otra ("CON DENSADA" -> "CONDENSADA")
        return all(p in palabras_cand or any(p in w and len(w) > len(p) for w in palabras_cand) for p in conocidas)

    def _mejor_nombre(self, original: str, conocidos: set) -> Optional[Tuple[str, float, bool]]:
        desconocidas = [p for p in self._palabras(original) if _es_palabra(p) and p not in self.palabras]
        if not desconocidas:
            return None                                    # todas sus palabras existen: no hay nada que corregir
        tope = min(_limite(len(original)), 0.5 + 0.75 * len(desconocidas))
        cands = []
        for n in conocidos:
            if abs(len(n) - len(original)) > 3:
                continue
            d = distancia(original, n, tope)
            if d <= tope and self._cambio_permitido(original, n):
                cands.append((d, n))
        if not cands:
            return None
        cands.sort()
        d, n = cands[0]
        unico = len(cands) == 1 or cands[1][0] - d >= 0.5
        return n, d, unico and d > 0

    def _unir(self, a: str, b: str) -> Optional[Tuple[str, float, bool]]:
        junto = a + b
        if junto in self.palabras:
            return junto, 0.0, True
        cands = self._candidatas(junto, min(1.0, _limite(len(junto))))
        if not cands:
            return None
        d, w = cands[0]
        unico = len(cands) == 1 or cands[1][0] - d >= 0.5
        return w, d, unico

    def _desempatar(self, tokens: List[str], i: int, opciones: List[str]) -> Tuple[str, bool]:
        if self.categorizador is None:
            return opciones[0], True
        puntos = []
        for w in opciones:
            prueba = " ".join(tokens[:i] + [w] + tokens[i + 1:])
            puntos.append((self.categorizador.buscar(prueba)["confianza"], self.palabras[w], w))
        puntos.sort(reverse=True)
        claro = len(puntos) == 1 or puntos[0][0] - puntos[1][0] >= 0.1
        return puntos[0][2], not claro

    def _encaja_mejor(self, corregido: str, original: str) -> bool:
        """El nombre corregido debe encajar con un producto del catálogo, y ese producto debe
        explicar también las palabras que no se han tocado (si no, la coincidencia puede venir
        solo de la palabra corregida: "FUENTE DEHESA" -> "PUENTE DEHESA" encaja con un licor "Puente Pazos")."""
        if self.categorizador is None:
            return False
        from categorizador import _singular, normalizar
        r = self.categorizador.buscar(corregido)
        c, o = r["confianza"], self.categorizador.buscar(original)["confianza"]
        if c < 0.6 or c < o + 0.05 or not r["producto_catalogo"]:
            return False
        intactas = [_singular(normalizar(p)) for p in self._palabras(corregido)
                    if p in self._palabras(original) and re.fullmatch(r"[A-ZÑ]{3,}", p)]
        if not intactas:
            return c >= 0.8
        toks = self.categorizador.tokens_por_producto.get(r["producto_catalogo"], [])
        return any(t.startswith(w) or w.startswith(t) for w in intactas for t in toks if len(t) >= 3)

    @staticmethod
    def _res(original, nombre, automatica, motivo, d):
        return {"original": original, "nombre": nombre, "automatica": bool(automatica),
                "motivo": motivo, "distancia": round(float(d), 2)}
