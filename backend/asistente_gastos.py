"""
Asistente conversacional de gastos (sin LLM).

Responde preguntas en lenguaje natural sobre los tickets guardados, por ejemplo:
    "¿Cuánto he gastado en lácteos este mes?"
    "¿Cuántas veces compro yogures a la semana?"
    "¿Cuál ha sido mi compra más cara del mes?"
    "¿De qué categoría compro más productos?"
    "¿Cuál es el producto que más he comprado?"

Funcionamiento (todo determinista y explicable):
  1. Intención: clasificador de vecino más cercano por centroides (TF-IDF de
     caracteres y palabras) entrenado con frases de ejemplo de cada intención,
     reforzado con patrones de palabras clave.
  2. Periodo: expresiones temporales ("este mes", "en marzo", "últimos 30 días"…).
  3. Concepto: lo que queda de la frase ("lácteos", "yogures", "pan de molde"…)
     se resuelve contra (a) un diccionario de conceptos coloquiales, (b) las
     categorías del catálogo y (c) los productos del historial agrupados por
     clustering (ver clustering_productos.py).
  4. Cálculo con pandas sobre los tickets y respuesta en texto + tabla.
"""
import re
import unicodedata
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from categorizador import normalizar
from clustering_productos import ClusteringProductos
from ajustes_consulta import ajustar_consulta
from motor_consultas import SUGERENCIAS, Consulta, EjecutorConsultas, Filtros, explicar

DIAS_SIN_TILDE = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]


def sin_tildes(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def euros(v: float) -> str:
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def fecha_es(d: date) -> str:
    return d.strftime("%d/%m/%Y")


# ════════════════════════════════════════════════════════════════════════
# 1. Intenciones
# ════════════════════════════════════════════════════════════════════════
EJEMPLOS = {
    "gasto_total": [
        "cuanto he gastado", "cuanto he gastado en lacteos este mes", "cuanto dinero me he gastado en fruta",
        "gasto total en carne", "cuanto me gaste en marzo", "cuanto llevo gastado este año",
        "dinero gastado en limpieza", "total gastado en yogures", "cuanto gasto en cerveza",
        "cuanto he pagado en total", "cuanto me he dejado en el supermercado",
    ],
    "frecuencia": [
        "cuantas veces compro yogures a la semana", "cada cuanto compro pan", "con que frecuencia compro leche",
        "cuantas veces he comprado huevos", "cuantas veces al mes compro fruta", "cada cuantos dias compro cerveza",
        "cuantas veces compro carne", "frecuencia de compra de pizza",
    ],
    "compra_mas_cara": [
        "cual ha sido mi compra mas cara", "compra mas cara del mes", "ticket mas caro",
        "cual fue la compra en la que mas gaste", "mayor compra de este año", "el ticket de mayor importe",
    ],
    "compra_mas_barata": [
        "cual ha sido mi compra mas barata", "ticket mas barato", "la compra en la que menos gaste",
        "menor compra del mes",
    ],
    "categoria_top": [
        "cual es la categoria de la que mas productos consumo", "en que categoria gasto mas",
        "que tipo de productos compro mas", "categoria con mas gasto", "en que gasto mas dinero",
        "ranking de categorias", "reparto del gasto por categorias", "de que categoria compro mas",
    ],
    "producto_top": [
        "cual es el producto que mas he comprado", "que producto compro mas", "productos mas comprados",
        "top productos", "en que producto gasto mas", "lo que mas compro", "que es lo que mas he comprado este mes",
    ],
    "producto_mas_caro": [
        "cual es el producto mas caro que he comprado", "producto mas caro", "articulo mas caro del mes",
        "que producto me ha costado mas",
    ],
    "num_compras": [
        "cuantas veces he ido a comprar", "cuantos tickets tengo", "cuantas compras he hecho este mes",
        "numero de compras", "cuantas veces fui al supermercado",
    ],
    "gasto_medio": [
        "cuanto gasto de media por compra", "gasto medio por ticket", "media de gasto semanal",
        "cuanto gasto a la semana de media", "gasto promedio al mes", "cuanto gasto al mes",
    ],
    "gasto_por_mes": [
        "como ha evolucionado mi gasto", "gasto por meses", "compara este mes con el anterior",
        "evolucion del gasto mensual", "gasto de cada mes", "he gastado mas este mes que el pasado", "gasto mensual",
    ],
    "precio_producto": [
        "cuanto cuesta la leche", "a cuanto esta el pan", "ha subido el precio de los huevos",
        "evolucion del precio del aceite", "precio del yogur", "cuanto me cuesta la cerveza",
    ],
    "ultima_compra": [
        "cuando compre leche por ultima vez", "ultima vez que compre pizza", "cuando fue mi ultima compra",
        "cuando he comprado huevos",
    ],
    "ayuda": [
        "hola", "ayuda", "que puedes hacer", "que te puedo preguntar", "buenas", "como funciona", "gracias", "vale",
        "que puedo preguntarte",
    ],
}

# Patrones que refuerzan una intención cuando aparecen (sobre texto sin tildes)
PATRONES = [
    ("frecuencia", r"\b(cuantas veces|cada cuanto|frecuencia|cada cuantos)\b", 0.35),
    ("num_compras", r"\b(cuantas (compras|veces he ido|veces fui)|cuantos tickets|numero de compras)\b", 0.45),
    ("compra_mas_cara", r"\b(compra|ticket)s? (mas (cara|caro|grande)|de mayor)|mayor compra\b", 0.5),
    ("compra_mas_barata", r"\b(compra|ticket)s? (mas (barat[oa]|pequen[oa])|de menor)|menor compra\b", 0.6),
    ("producto_mas_caro", r"\b(producto|articulo)s? mas car[oa]\b|\blo mas car[oa]\b", 0.5),
    ("categoria_top", r"\ben que (me )?gast\w* mas\b", 0.5),
    ("producto_top", r"\bproducto\w* .*mas frecuen", 0.6),
    ("num_compras", r"\bvoy (a comprar|al super)|\bvoy a mercadona\b", 0.3),
    ("categoria_top", r"\bcategoria|tipo de productos?\b", 0.3),
    ("producto_top", r"\b(producto|productos) (que )?mas\b|\bmas (he )?comprad|\blo que mas\b|\btop\b", 0.25),
    ("gasto_medio", r"\b(media|medio|promedio)\b", 0.2),
    ("gasto_por_mes", r"\b(evolucion|compara|por meses|cada mes|mensual)\b", 0.3),
    ("precio_producto", r"\b(cuanto cuesta|a cuanto|precio|subido|bajado)\b", 0.35),
    ("ultima_compra", r"\b(ultima vez|ultima compra|cuando (compre|he comprado|fue))\b", 0.45),
    ("gasto_total", r"\b(gastado|gaste|gasto|dinero|pagado|dejado)\b", 0.15),
    ("ayuda", r"^\W*(hola|buenas|buenos dias|gracias|hey|ayuda|vale|ok)\b", 0.5),
]


class ClasificadorIntenciones:
    def __init__(self):
        frases, etiquetas = [], []
        for intencion, ejemplos in EJEMPLOS.items():
            for e in ejemplos:
                frases.append(e)
                etiquetas.append(intencion)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(frases)
        X = normalize(self.vec.transform(frases))
        self.intenciones = list(EJEMPLOS)
        self.centroides = normalize(np.vstack([
            np.asarray(X[[i for i, e in enumerate(etiquetas) if e == it]].mean(axis=0)) for it in self.intenciones
        ]))

    def predecir(self, texto: str) -> Tuple[str, float, Dict[str, float]]:
        t = sin_tildes(texto)
        v = normalize(self.vec.transform([t])).toarray()
        puntuaciones = dict(zip(self.intenciones, (self.centroides @ v.T).ravel()))
        for intencion, patron, bonus in PATRONES:
            if re.search(patron, t):
                puntuaciones[intencion] += bonus
        # "¿cuánto gasto en X al mes / a la semana?" es un gasto medio, no un total
        if re.search(r"\b(a la semana|al mes|por semana|por mes)\b", t) and \
                re.search(r"\b(gast|diner|pag)", t) and not re.search(r"\bveces\b|frecuencia|cada cuant", t):
            puntuaciones["gasto_medio"] += 0.45
        mejor = max(puntuaciones, key=puntuaciones.get)
        return mejor, float(puntuaciones[mejor]), puntuaciones


# ════════════════════════════════════════════════════════════════════════
# 2. Periodos de tiempo
# ════════════════════════════════════════════════════════════════════════
def fin_de_mes(a: int, m: int) -> date:
    return (date(a + (m == 12), m % 12 + 1, 1) - timedelta(days=1))


def extraer_periodo(texto: str, hoy: date) -> Tuple[Optional[date], Optional[date], str, str]:
    """Devuelve (inicio, fin, etiqueta, texto_sin_periodo)."""
    t = sin_tildes(texto)
    meses_re = "|".join(MESES)

    def quitar(patron):
        return re.sub(patron, " ", t)

    reglas = [
        (r"\bhoy\b", lambda m: (hoy, hoy, "hoy")),
        (r"\bayer\b", lambda m: (hoy - timedelta(1), hoy - timedelta(1), "ayer")),
        # Día concreto: "el 13 de abril", "13 de abril de 2025", "13/04", "13/04/2025"
        (rf"\b(\d{{1,2}}) de ({meses_re})(?: de(?:l)? (\d{{4}}))?\b", lambda m: _dia(hoy, int(m.group(1)), MESES.index(m.group(2)) + 1, m.group(3))),
        (r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lambda m: _dia(hoy, int(m.group(1)), int(m.group(2)), m.group(3))),
        # Quincenas, estaciones y Navidad (antes que "el año pasado", que puede ir dentro)
        (rf"\b(primera|segunda) quincena de ({meses_re})(?: de(?:l)? (\d{{4}}))?\b", lambda m: _quincena(hoy, m)),
        (r"\b(este |esta |el |la |en |del |de )?(verano|invierno|primavera|otono)( pasad[oa]| anterior| del ano pasado| de (\d{4}))?\b", lambda m: _estacion(hoy, m)),
        (r"\bnavidad(?:es)?( pasadas?| del ano pasado| de (\d{4}))?\b", lambda m: _navidad(hoy, m)),
        # "en lo que va de año / lo que llevamos de mes" = desde el inicio del periodo actual hasta hoy
        (r"\b(?:en )?lo que (?:va|llevamos|llevo) de(?:l)? (ano|mes|semana)\b", lambda m: {
            "ano": (date(hoy.year, 1, 1), hoy, f"en {hoy.year}"), "mes": (hoy.replace(day=1), hoy, "este mes"),
            "semana": (hoy - timedelta(hoy.weekday()), hoy, "esta semana")}[m.group(1)]),
        (r"\b(esta semana)\b", lambda m: (hoy - timedelta(hoy.weekday()), hoy, "esta semana")),
        (r"\b(la )?semana (pasada|anterior)\b", lambda m: (hoy - timedelta(hoy.weekday() + 7), hoy - timedelta(hoy.weekday() + 1), "la semana pasada")),
        (r"\b(este|en este|del) mes( actual)?\b(?! pasado| anterior)|\beste mes\b", lambda m: (hoy.replace(day=1), hoy, "este mes")),
        (r"\b(el |del )?mes (pasado|anterior)\b", lambda m: _mes_pasado(hoy)),
        (r"\b(este|del|en este) ano\b(?! pasado| anterior)", lambda m: (date(hoy.year, 1, 1), hoy, f"en {hoy.year}")),
        (r"\b(el |del )?ano (pasado|anterior)\b", lambda m: (date(hoy.year - 1, 1, 1), date(hoy.year - 1, 12, 31), f"en {hoy.year - 1}")),
        (r"\b(?:los |las )?ultim[oa]s (\d+) (dias|semanas|meses)\b", lambda m: _ultimos(hoy, int(m.group(1)), m.group(2))),
        (r"\b(?:el |la )?ultim[oa] (semana|mes|ano)\b", lambda m: _ultimos(hoy, 1, m.group(1) + ("s" if m.group(1) != "mes" else "es"))),
        (rf"\b(?:desde|a partir de) (?:el mes de )?({meses_re})(?: de (\d{{4}}))?\b", lambda m: _desde_mes(hoy, m)),
        (rf"\b(?:en |de |del mes de |durante )?({meses_re})(?: (?:de|del) (\d{{4}}))?\b", lambda m: _mes(hoy, m)),
        (r"\b(?:en|del|de|durante) (?:el )?(?:ano )?(\d{4})\b", lambda m: (date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31), f"en {m.group(1)}")),
        (r"\b(20\d{2})\b", lambda m: (date(int(m.group(1)), 1, 1), date(int(m.group(1)), 12, 31), f"en {m.group(1)}")),
    ]
    for patron, fn in reglas:
        m = re.search(patron, t)
        if m:
            try:
                ini, fin, etiqueta = fn(m)
            except ValueError:          # fecha imposible ("31 de febrero", "13/24"): se ignora esa regla
                continue
            return ini, fin, etiqueta, quitar(patron)
    return None, None, "en todo tu historial", t


def _dia(hoy, d, m, anio=None):
    if anio:
        anio = int(anio) + (2000 if len(anio) == 2 else 0)
    else:
        anio = hoy.year if (m, d) <= (hoy.month, hoy.day) else hoy.year - 1   # la fecha pasada más reciente
    f = date(anio, m, d)
    return f, f, f"el {f.strftime('%d/%m/%Y')}"


def _quincena(hoy, m):
    mes = MESES.index(m.group(2)) + 1
    anio = int(m.group(3)) if m.group(3) else (hoy.year if mes <= hoy.month else hoy.year - 1)
    if m.group(1) == "primera":
        return date(anio, mes, 1), date(anio, mes, 15), f"en la primera quincena de {m.group(2)} de {anio}"
    return date(anio, mes, 16), fin_de_mes(anio, mes), f"en la segunda quincena de {m.group(2)} de {anio}"


# Estaciones astronómicas aproximadas: (mes, día) de inicio y de fin
ESTACIONES = {"primavera": ((3, 20), (6, 20)), "verano": ((6, 21), (9, 22)),
              "otono": ((9, 23), (12, 20)), "invierno": ((12, 21), (3, 19))}


def _estacion(hoy, m):
    nombre = m.group(2)
    (mi, di), (mf, df) = ESTACIONES[nombre]
    cruza = mf < mi                                  # el invierno empieza un año y acaba el siguiente

    def rango(a):
        return date(a, mi, di), date(a + cruza, mf, df)

    if m.group(4):
        a = int(m.group(4))
    elif (m.group(1) or "").strip() in ("este", "esta"):
        a = hoy.year - (cruza and hoy.month <= mf)
    else:
        # "el verano pasado" / "en verano": la última que ya ha terminado (o la actual si "en verano" está en curso)
        a = hoy.year
        while rango(a)[0] > hoy or (m.group(3) and rango(a)[1] >= hoy):
            a -= 1
    ini, fin = rango(a)
    nombre_txt = nombre.replace("otono", "otoño")
    return ini, min(fin, hoy), f"en {nombre_txt} de {a}" + (f"-{a + 1}" if cruza else "")


def _navidad(hoy, m):
    if m.group(2):
        a = int(m.group(2))
    else:
        a = hoy.year if hoy >= date(hoy.year, 12, 24) else hoy.year - 1
    return date(a, 12, 24), min(date(a + 1, 1, 6), hoy), f"en Navidad de {a}"


def _mes_pasado(hoy):
    a, m = (hoy.year, hoy.month - 1) if hoy.month > 1 else (hoy.year - 1, 12)
    return date(a, m, 1), fin_de_mes(a, m), f"en {MESES[m - 1]} de {a}"


def _ultimos(hoy, n, unidad):
    dias = {"dias": 1, "semanas": 7, "meses": 30, "anos": 365}.get(unidad, 30) * n
    etiqueta = f"en los últimos {n} {unidad.replace('anos', 'años').replace('dias', 'días')}"
    return hoy - timedelta(days=dias - 1), hoy, etiqueta


def _mes(hoy, m):
    mes = MESES.index(m.group(1)) + 1
    anio = int(m.group(2)) if m.group(2) else (hoy.year if mes <= hoy.month else hoy.year - 1)
    return date(anio, mes, 1), fin_de_mes(anio, mes), f"en {m.group(1)} de {anio}"


def _desde_mes(hoy, m):
    mes = MESES.index(m.group(1)) + 1
    anio = int(m.group(2)) if m.group(2) else (hoy.year if mes <= hoy.month else hoy.year - 1)
    return date(anio, mes, 1), hoy, f"desde {m.group(1)} de {anio}"


# ════════════════════════════════════════════════════════════════════════
# 3. Conceptos (qué productos)
# ════════════════════════════════════════════════════════════════════════
# Conceptos coloquiales -> categorías ("Cat") o subcategorías ("Cat/Sub") del catálogo
CONCEPTOS = {
    "lacteo": ["Huevos, leche y Mantequilla/Leche y bebidas vegetales", "Huevos, leche y Mantequilla/Mantequilla y margarina",
               "Postres y yogures/Yogures naturales y sabores", "Postres y yogures/Yogures líquidos", "Postres y yogures/Bífidus",
               "Postres y yogures/Yogures desnatados", "Postres y yogures/Postres de soja / Yogures griegos",
               "Postres y yogures/Yogures y postres infantiles", "Charcuteria y quesos/Queso curado, semicurado y tierno",
               "Charcuteria y quesos/Queso lonchas, rallado y en porciones", "Charcuteria y quesos/Queso untable, fresco y especialidades"],
    "queso": ["Charcuteria y quesos/Queso curado, semicurado y tierno", "Charcuteria y quesos/Queso lonchas, rallado y en porciones",
              "Charcuteria y quesos/Queso untable, fresco y especialidades"],
    "yogur": ["Postres y yogures/Yogures naturales y sabores", "Postres y yogures/Yogures líquidos", "Postres y yogures/Bífidus",
              "Postres y yogures/Yogures desnatados", "Postres y yogures/Postres de soja / Yogures griegos",
              "Postres y yogures/Yogures y postres infantiles"],
    "postre": ["Postres y yogures"],
    "carne": ["Carne", "Congelados/Carne"],
    "pollo": ["Carne/Aves y pollo"],
    "embutido": ["Charcuteria y quesos/Embutido curado", "Charcuteria y quesos/Jamón serrano", "Charcuteria y quesos/Aves y jamón cocido",
                 "Charcuteria y quesos/Chopped y mortadela", "Charcuteria y quesos/Bacón y salchichas", "Carne/Embutido"],
    "charcuteria": ["Charcuteria y quesos"],
    "pescado": ["Marisco y pescado", "Congelados/Pescado", "Congelados/Marisco", "Conservas, caldos y cremas/Atún y otras conservas de pescado",
                "Conservas, caldos y cremas/Berberechos y mejillones"],
    "marisco": ["Marisco y pescado/Marisco", "Congelados/Marisco"],
    "fruta": ["Fruta y verdura/Fruta"],
    "fruto seco": ["Aperitivos/Frutos secos y fruta desecada"],
    "verdura": ["Fruta y verdura/Verdura", "Fruta y verdura/Lechuga y ensalada preparada", "Congelados/Fruta y verdura"],
    "hortaliza": ["Fruta y verdura/Verdura", "Fruta y verdura/Lechuga y ensalada preparada"],
    "bebida": ["Agua y refrescos", "Zumos", "Bodega"],
    "refresco": ["Agua y refrescos"],
    "agua": ["Agua y refrescos/Agua"],
    "zumo": ["Zumos"],
    "alcohol": ["Bodega"],
    "bodega": ["Bodega"],
    "vino": ["Bodega/Vino blanco", "Bodega/Vino tinto", "Bodega/Vino rosado", "Bodega/Vino lambrusco y espumoso", "Bodega/Tinto de verano y sangría"],
    "cerveza": ["Bodega/Cerveza", "Bodega/Cerveza sin alcohol"],
    "limpieza": ["Limpieza y hogar"],
    "hogar": ["Limpieza y hogar"],
    "higiene": ["Cuidado facial y corporal", "Cuidado del cabello", "Limpieza y hogar/Papel higiénico y celulosa"],
    "aseo": ["Cuidado facial y corporal", "Cuidado del cabello"],
    "cosmetica": ["Cuidado facial y corporal", "Cuidado del cabello", "Maquillaje"],
    "dulce": ["Azucar, caramelos y chocolate", "Panaderia y Pasteleria/Bollería de horno", "Panaderia y Pasteleria/Bollería envasada",
              "Panaderia y Pasteleria/Tartas y pasteles", "Congelados/Helados", "Cereales y galletas/Galletas"],
    "golosina": ["Azucar, caramelos y chocolate/Golosinas", "Azucar, caramelos y chocolate/Chicles y caramelos"],
    "chuche": ["Azucar, caramelos y chocolate/Golosinas", "Azucar, caramelos y chocolate/Chicles y caramelos"],
    "chocolate": ["Azucar, caramelos y chocolate/Chocolate"],
    "bolleria": ["Panaderia y Pasteleria/Bollería de horno", "Panaderia y Pasteleria/Bollería envasada"],
    "snack": ["Aperitivos"],
    "aperitivo": ["Aperitivos"],
    "picoteo": ["Aperitivos"],
    "congelado": ["Congelados"],
    "helado": ["Congelados/Helados"],
    "desayuno": ["Cereales y galletas", "Cacao, cafe e infusiones", "Panaderia y Pasteleria/Bollería de horno", "Panaderia y Pasteleria/Bollería envasada"],
    "cafe": ["Cacao, cafe e infusiones"],
    "pan": ["Panaderia y Pasteleria/Pan de horno", "Panaderia y Pasteleria/Pan de molde y otras especialidades",
            "Panaderia y Pasteleria/Pan tostado y rallado", "Panaderia y Pasteleria/Picos, rosquilletas y picatostes"],
    "panaderia": ["Panaderia y Pasteleria"],
    "pasta": ["Arroz, legumbres y pasta/Pasta y fideos", "Congelados/Arroz y pasta"],
    "legumbre": ["Arroz, legumbres y pasta/Legumbres"],
    "conserva": ["Conservas, caldos y cremas"],
    "salsa": ["Aceites, especias y salsas/Mayonesa, ketchup y mostaza", "Aceites, especias y salsas/Otras salsas"],
    "aceite": ["Aceites, especias y salsas/ACEITES"],
    "especia": ["Aceites, especias y salsas/ESPECIAS"],
    "huevo": ["Huevos, leche y Mantequilla/Huevos"],
    "leche": ["Huevos, leche y Mantequilla/Leche y bebidas vegetales"],
    "precocinado": ["Pizzas y platos preparados", "Congelados/Pizzas", "Congelados/Rebozados"],
    "plato preparado": ["Pizzas y platos preparados"],
    "pizza": ["Pizzas y platos preparados/Pizzas", "Congelados/Pizzas"],
    "mascota": ["Mascotas"],
    "bebe": ["Bebe"],
    "parafarmacia": ["Fisioterapia y Parafarmacia"],
    "maquillaje": ["Maquillaje"],
}

# Sinónimos: palabra del usuario -> palabra con la que aparece el producto en el ticket
SINONIMOS = {"platano": "banana", "platanito": "banana", "papa": "patata", "gamba": "langostino",
             "lechuga": "ensalada", "birra": "cerveza"}

# Conceptos cuya subcategoría incluye otros productos: se exige además que el nombre contenga la palabra
CONCEPTOS_LITERALES = {"leche": "leche", "huevo": "huevo"}

STOPWORDS = set("""
a al algo ante con cual cuales cuando cuanta cuantas cuanto cuantos de del desde dime donde e el en entre es esta este esto
fue ha han he hemos la las le lo los me mi mis muy mas menos o para pero por que quien se sea ser si sin sobre son su sus
te tengo tu un una uno unos unas y ya yo has hay tiene tienen sido ido hecho podrias puedes favor gracias quiero saber
gastado gaste gasto gastar gastos gasta dinero pagado pague dejado llevo total cuesta costado coste precio precios subido bajado
compro compre comprado compra compras comprar comprando veces vez frecuencia cada dias semana semanas mes meses ano anos
semanal mensual media medio promedio cual cuales producto productos articulo articulos categoria categorias tipo tipos
caro cara caros caras barato barata baratos baratas mayor menor ultima ultimo top ranking reparto consumo consumir consumido
evolucion evolucionado compara comparar anterior pasado pasada actual ticket tickets supermercado numero hecho ido fui
cosas cosa cuantos lo mercadona mucho poco tanto todo todos toda todas hoy ayer esta estas estos
como tal hola buenas evolucionado ha sido costado semanalmente mensualmente voy vas ir vamos
lunes martes miercoles jueves viernes sabado domingo sabados domingos euro euros ensena ensenarme muestra muestrame lista dame
hecho hice hago dinero devuelto han ensename muestrame dime quiero ver saber mostrar
pequena pequeno grande grandes gracias vale ok dia dias
""".split())


def singular(p: str) -> str:
    if len(p) > 4 and p.endswith("es") and p[-3] not in "aeiou":
        return p[:-2]
    if len(p) > 3 and p.endswith("s"):
        return p[:-1]
    return p


class ResolutorConceptos:
    def __init__(self, arbol_categorias: List[dict]):
        self.categorias = {sin_tildes(g["categoria"]): g["categoria"] for g in arbol_categorias}
        self.subcategorias = {}
        for g in arbol_categorias:
            for s in g["subcategorias"]:
                self.subcategorias.setdefault(sin_tildes(s), []).append((g["categoria"], s))

    @staticmethod
    def texto_concepto(texto_sin_periodo: str) -> str:
        palabras = [p for p in re.findall(r"[a-zñ0-9]+", texto_sin_periodo) if p not in STOPWORDS and not p.isdigit()]
        return " ".join(palabras)

    def resolver(self, concepto: str, df: pd.DataFrame, clusters: ClusteringProductos) -> Tuple[Optional[pd.Series], str, dict]:
        """Devuelve (máscara de filas de df, etiqueta legible, detalle)."""
        if not concepto:
            return None, "", {}
        palabras = [SINONIMOS.get(singular(p), singular(p)) for p in concepto.split()]
        clave = " ".join(palabras)

        def por_diccionario(k):
            mascara = df.apply(lambda r: self._coincide_destinos(r, CONCEPTOS[k]), axis=1)
            if k in CONCEPTOS_LITERALES:
                # "leche" no debe incluir la horchata ni las bebidas vegetales de la misma subcategoría
                patron = rf"\b{CONCEPTOS_LITERALES[k]}\b(?!\s+(?:condensad|evaporad|en polvo))"
                literal = (df["nombre_norm"].fillna("").str.contains(patron, case=False, regex=True) |
                           df["catalogo_norm"].fillna("").astype(str).str.contains(patron, case=False, regex=True))
                if (mascara & literal).any():
                    mascara = mascara & literal
            return mascara, concepto, {"tipo": "concepto", "destinos": CONCEPTOS[k]}

        def forma(nombre):  # "Frutos secos y fruta desecada" -> "fruto seco y fruta desecada"
            return " ".join(singular(w) for w in re.findall(r"[a-z]+", sin_tildes(nombre)))

        # (a) Concepto coloquial completo ("lácteos", "plato preparado"…)
        if clave in CONCEPTOS:
            return por_diccionario(clave)

        # (b) Nombre exacto de una categoría o subcategoría del catálogo
        for nombre_norm, nombre in self.categorias.items():
            if clave in (forma(nombre), forma(nombre.split(",")[0])):
                return df["categorias_lista"].apply(lambda cs: any(c[0] == nombre for c in cs)), nombre, {"tipo": "categoria"}
        for nombre_norm, pares in self.subcategorias.items():
            if clave == forma(pares[0][1]):
                destinos = [f"{c}/{s}" for c, s in pares]
                mascara = df.apply(lambda r: self._coincide_destinos(r, destinos), axis=1)
                return mascara, pares[0][1], {"tipo": "subcategoria"}

        # (c) Varias palabras ("yogur líquido"): primero productos concretos
        if len(palabras) > 1:
            r = self._por_productos(palabras, df, clusters, concepto)
            if r[0] is not None and r[0].any():
                return r
        # (d) Alguna palabra suelta es un concepto conocido
        for k in palabras:
            if k in CONCEPTOS:
                return por_diccionario(k)

        # (e) Productos del historial
        return self._por_productos(palabras, df, clusters, concepto)

    @staticmethod
    def _coincide_destinos(fila, destinos) -> bool:
        for d in destinos:
            cat, _, sub = d.partition("/")
            for c in fila["categorias_lista"]:
                if c[0] == cat and (not sub or c[1] == sub):
                    return True
        return False

    def _por_productos(self, palabras, df, clusters, concepto):
        nombres = df["nombre_norm"]
        raiz = [p[:max(3, len(p) - 1)] if len(p) > 4 else p for p in palabras]

        def contiene(texto):
            toks = texto.split()
            return bool(toks) and all(any(t.startswith(r) for t in toks) for r in raiz)

        por_nombre = set(nombres[nombres.apply(contiene)].unique())
        # El nombre completo del catálogo ayuda con abreviaturas ("CAPRICHOS JAMÓN" = croquetas)
        por_catalogo = set(df.loc[df["catalogo_norm"].apply(contiene), "nombre_norm"].unique()) - por_nombre
        # Coincidencia aproximada por n-gramas de caracteres: erratas de impresión o lectura
        por_parecido = {n for n, _ in clusters.similares(" ".join(palabras), umbral=0.55)} - por_nombre - por_catalogo
        directos = por_nombre | por_catalogo | por_parecido
        if not directos:
            return None, concepto, {"tipo": "sin_coincidencias"}

        # Se descartan coincidencias de otra familia (p. ej. "TORTITAS ARROZ YOGUR" al buscar yogures)
        cabeza = lambda t: bool(t.split()) and any(t.split()[0].startswith(r) for r in raiz)
        cat_de = dict(zip(df["nombre_norm"], df["catalogo_norm"]))
        principales = {n for n in directos if cabeza(n) or cabeza(cat_de.get(n, ""))} or directos
        subcats = pd.Series([c[1] or c[0] for n in principales for c in clusters.cats_producto.get(n, [])[:1]])
        sub_dominantes = set(subcats.value_counts().index[:2]) if len(subcats) else set()

        def compatible(n):
            cats = clusters.cats_producto.get(n, [])
            return not sub_dominantes or not cats or any((c[1] or c[0]) in sub_dominantes for c in cats)

        incluidos = {n for n in directos if n in principales or compatible(n)}
        detalle = {"tipo": "productos", "productos": sorted(clusters.nombre_original.get(n, n) for n in incluidos),
                   "por_parecido": sorted(clusters.nombre_original.get(n, n) for n in incluidos & por_parecido)}
        return nombres.isin(incluidos), concepto, detalle


# ════════════════════════════════════════════════════════════════════════
# 4. Asistente
# ════════════════════════════════════════════════════════════════════════
class AsistenteGastos:
    UMBRAL_INTENCION = 0.30

    def __init__(self, arbol_categorias: List[dict], traductor=None):
        self.arbol = arbol_categorias
        self.clasificador = ClasificadorIntenciones()
        self.resolutor = ResolutorConceptos(arbol_categorias)
        self.clusters = ClusteringProductos()
        self.traductor = traductor          # TraductorLLM opcional
        self._firma = None
        self.df = pd.DataFrame()

    # ── Datos ──
    def preparar(self, tickets: List[dict]):
        """Convierte los tickets en una tabla de productos y reentrena el clustering si han cambiado."""
        filas = []
        for t in tickets:
            f = _fecha(t)
            if not f:
                continue
            for it in t.get("items", []):
                cats = [(c.get("categoria", ""), c.get("subcategoria", "")) for c in it.get("categorias") or []]
                cantidad = it.get("cantidad", 1) or 1
                filas.append({
                    "ticket_id": t.get("id"), "fecha": f, "total_ticket": float(t.get("precio_total") or 0),
                    "nombre_ticket": t.get("nombre_personalizado") or t.get("tienda") or "Ticket",
                    "descripcion": it.get("descripcion", ""), "nombre_norm": normalizar(it.get("descripcion", "")),
                    "cantidad": cantidad, "precio_unitario": float(it.get("precio_unitario") or 0),
                    "importe": float(it.get("precio_unitario") or 0) * cantidad,
                    "categorias_lista": cats,
                    "catalogo_norm": normalizar(it.get("producto_catalogo") or ""),
                    "categoria_inferida": False,
                    "categoria": cats[0][0] if cats else "Sin categoría",
                    "es_devolucion": float(t.get("precio_total") or 0) < 0 or cantidad < 0,
                })
        self.df = pd.DataFrame(filas)
        self.tickets_df = (self.df.groupby("ticket_id").agg(fecha=("fecha", "first"), total=("total_ticket", "first"),
                                                            nombre=("nombre_ticket", "first"), productos=("cantidad", "sum"))
                           .reset_index() if len(self.df) else pd.DataFrame())
        firma = (len(tickets), len(filas), tuple(sorted(t.get("id", "") for t in tickets)))
        if firma != self._firma:
            self.clusters = ClusteringProductos().entrenar(
                [{"descripcion": it.get("descripcion"), "categorias": it.get("categorias")} for t in tickets for it in t.get("items", [])])
            self._firma = firma
        # Productos sin categoría (erratas, nombres que no están en el catálogo): se les asigna la
        # categoría de su grupo del clustering para que cuenten en las consultas por categoría
        if len(self.df) and self.clusters.entrenado:
            sin = self.df["categorias_lista"].apply(len) == 0
            for i in self.df.index[sin]:
                cats, _ = self.clusters.inferir_categorias(self.df.at[i, "descripcion"])
                if cats:
                    self.df.at[i, "categorias_lista"] = [(c["categoria"], c["subcategoria"]) for c in cats]
                    self.df.at[i, "categoria"] = cats[0]["categoria"]
                    self.df.at[i, "categoria_inferida"] = True

    # ── Punto de entrada ──
    def responder(self, mensaje: str, tickets: List[dict], hoy: Optional[date] = None,
                  historial: Optional[List[dict]] = None, usar_llm: bool = True) -> dict:
        """1) El LLM local traduce la pregunta a una Consulta (si está disponible).
           2) Si no, el motor de reglas construye la Consulta.
           3) El motor de consultas la ejecuta sobre los datos reales."""
        hoy = hoy or date.today()
        self.preparar(tickets)
        consulta, motor = self.interpretar(mensaje, hoy, historial, usar_llm)
        if consulta is None:
            return {"respuesta": "No estoy seguro de haber entendido la pregunta. Puedo responder cosas como:",
                    "sugerencias": SUGERENCIAS, "interpretacion": {"motor": motor, "explicacion": ""}}

        ejecutor = EjecutorConsultas(self.resolutor, self.clusters, self.arbol, extraer_periodo)
        resultado = ejecutor.ejecutar(consulta.model_copy(deep=True), self.df, hoy)
        _, _, periodo = ejecutor._periodo(consulta.filtros, hoy)
        resultado["interpretacion"] = {"motor": motor, "explicacion": explicar(consulta, periodo),
                                       "consulta": consulta.model_dump(mode="json", exclude_defaults=True),
                                       "ajustes": getattr(self, "ultimos_ajustes", [])}
        return resultado

    def interpretar(self, mensaje: str, hoy: date, historial: Optional[List[dict]] = None,
                    usar_llm: bool = True, excluir_ejemplo: Optional[str] = None) -> Tuple[Optional[Consulta], str]:
        """Pregunta -> Consulta. Con LLM si está disponible; si no (o si falla), con el motor de reglas.
        Devuelve (consulta o None si no se entiende, motor usado)."""
        self.ultimos_ajustes = []
        if usar_llm and self.traductor is not None and self.traductor.disponible:
            # RAG: ejemplos validados parecidos a la pregunta (ver rag_chat.py)
            ejemplos = self.rag.recuperar(mensaje, excluir=excluir_ejemplo) if getattr(self, "rag", None) else None
            self.ultimos_ejemplos = ejemplos or []
            consulta = self.traductor.traducir(mensaje, hoy, [g["categoria"] for g in self.arbol], historial, ejemplos)
            if consulta is not None:
                # Se contrasta la consulta del LLM con el texto de la pregunta (ver ajustes_consulta.py)
                previas = [h.get("pregunta", "") for h in (historial or [])[-3:]]
                consulta, self.ultimos_ajustes = ajustar_consulta(consulta, mensaje, previas, hoy)
                self._recuperar_concepto(consulta, mensaje, hoy)
                return consulta, "llm"
        consulta = self.continuar_consulta(mensaje, hoy, historial) or self.reglas_a_consulta(mensaje, hoy)
        return consulta, "reglas"

    def _recuperar_concepto(self, consulta: Consulta, mensaje: str, hoy: date):
        """Si el LLM ha olvidado el producto ("¿cuánto gasté los lunes en carne?" sin "carne"), se recupera
        del texto, pero solo con palabras que son con seguridad un producto o tipo de producto."""
        if consulta is None or consulta.tipo != "consulta" or consulta.filtros.concepto or consulta.filtros.categoria:
            return
        _, _, _, resto = extraer_periodo(mensaje, hoy)
        resto = re.sub(r"\b(devoluci\w*|devuelt\w*|reembols\w*|fines?|semana|" + "|".join(DIAS_SIN_TILDE) + r")s?\b", " ", resto)
        candidato = self.resolutor.texto_concepto(resto)
        # Palabra principal de cada producto comprado ("pizza", "bacon", "leche"...)
        cabezas = {n.split()[0] for n in self.df["nombre_norm"] if n} if len(self.df) else set()
        t = sin_tildes(mensaje)
        validas = [p for p in candidato.split() if len(p) >= 3
                   and not re.search(rf"\b(cuanta|mucha|poca|tanta|la|mas|menos|algo de) {p}\b", t)   # "cuánta / más pasta" = dinero
                   and (singular(p) in CONCEPTOS or singular(p) in SINONIMOS or p in self.resolutor.categorias
                        or p in cabezas or singular(p) in cabezas)]
        if validas:
            consulta.filtros.concepto = " ".join(validas)
            self.ultimos_ajustes.append(f"concepto «{consulta.filtros.concepto}» recuperado del texto")

    # ── Preguntas de continuación sin LLM: "¿y el mes pasado?", "¿y de fruta?" ──
    def continuar_consulta(self, mensaje: str, hoy: date, historial: Optional[List[dict]]) -> Optional[Consulta]:
        t = sin_tildes(mensaje).strip(" ¿?¡!.")
        if not historial or not re.match(r"^(y|e|tambien|y si)\b", t) or len(t.split()) > 8:
            return None
        try:
            anterior = Consulta.model_validate(historial[-1].get("consulta") or {})
        except Exception:
            return None
        if anterior.tipo != "consulta":
            return None
        ini, _, _, resto = extraer_periodo(mensaje, hoy)
        nueva = anterior.model_copy(deep=True)
        if ini:
            nueva.filtros.periodo, nueva.filtros.desde, nueva.filtros.hasta = mensaje, None, None
        resto = re.sub(r"^\W*(y|e|tambien|y si)\b", " ", sin_tildes(resto))
        concepto = self.resolutor.texto_concepto(resto)
        if concepto:
            nueva.filtros.concepto, nueva.filtros.categoria = concepto, None
        return nueva if (ini or concepto) else None

    # ── Motor de reglas (sin LLM): intención + periodo + concepto -> Consulta ──
    def reglas_a_consulta(self, mensaje: str, hoy: date) -> Optional[Consulta]:
        t = sin_tildes(mensaje)
        intencion, confianza, _ = self.clasificador.predecir(mensaje)
        ini, _, _, resto = extraer_periodo(mensaje, hoy)
        filtros = Filtros(periodo=mensaje if ini else None)
        por_dinero = bool(re.search(r"gast|diner|pag|euro", t))

        # Devoluciones
        if re.search(r"devol|devuelt|reembols", t):
            filtros.tipo_ticket = "devoluciones"
            if intencion not in ("compra_mas_cara", "compra_mas_barata", "ultima_compra", "gasto_por_mes", "num_compras"):
                intencion = "gasto_total" if re.search(r"cuanto (dinero|me han|he)|importe|devuelto", t) else "num_compras"
            confianza = max(confianza, 1.0)
        # Días de la semana
        dias = [i for i, d in enumerate(DIAS_SIN_TILDE) if re.search(rf"\b{d}s?\b", t)]
        if dias and not re.search(r"dia de la semana", t):
            filtros.dias_semana = dias
        # Importe de los tickets ("compras de más de 50 €")
        m = re.search(r"\b(mas|menos) de (\d+(?:[.,]\d+)?) ?(?:€|euros?)", t)
        if m:
            v = float(m.group(2).replace(",", "."))
            if m.group(1) == "mas":
                filtros.importe_min = v
            else:
                filtros.importe_max = v
            if re.search(r"ensena|muestra|lista|cuales|que (compras|tickets)", t):
                intencion, confianza = "lista_tickets", 1.0

        if intencion == "ayuda" or confianza < self.UMBRAL_INTENCION:
            return Consulta(tipo="saludo") if intencion == "ayuda" and confianza >= self.UMBRAL_INTENCION else None

        concepto = self.resolutor.texto_concepto(re.sub(r"\b(devoluci\w*|devuelt\w*|reembols\w*)\b", " ", resto))
        originales = {sin_tildes(p): p for p in re.findall(r"\w+", mensaje.lower())}
        filtros.concepto = " ".join(originales.get(p, p) for p in concepto.split()) or None

        # "¿Qué día de la semana gasto más / voy más a comprar?"
        if re.search(r"dia de la semana", t):
            return Consulta(medida="gasto" if por_dinero else "compras", operacion="ranking",
                            agrupar_por="dia_semana", filtros=filtros)

        def por_texto(defecto):
            if "semana" in t:
                return "semana"
            if re.search(r"\bmes\b", t):
                return "mes"
            if re.search(r"al dia|diario", t):
                return "dia"
            return defecto

        mapa = {
            "gasto_total": lambda: Consulta(medida="gasto", operacion="total", filtros=filtros),
            "frecuencia": lambda: Consulta(medida="compras", operacion="media", por=por_texto("semana"), filtros=filtros),
            "compra_mas_cara": lambda: Consulta(medida="gasto", operacion="maximo", agrupar_por="ticket", filtros=filtros),
            "compra_mas_barata": lambda: Consulta(medida="gasto", operacion="minimo", agrupar_por="ticket", filtros=filtros),
            "categoria_top": lambda: Consulta(medida="gasto" if por_dinero else "unidades", operacion="ranking",
                                              agrupar_por="categoria", filtros=filtros),
            "producto_top": lambda: Consulta(medida="gasto" if por_dinero else "unidades", operacion="ranking",
                                             agrupar_por="producto", filtros=filtros),
            "producto_mas_caro": lambda: Consulta(medida="precio", operacion="maximo", agrupar_por="producto", filtros=filtros),
            "num_compras": lambda: Consulta(medida="compras", operacion="total", filtros=filtros),
            "gasto_medio": lambda: Consulta(medida="gasto", operacion="media",
                                            por="compra" if "por compra" in t else por_texto("compra"), filtros=filtros),
            "gasto_por_mes": lambda: Consulta(medida="gasto", operacion="evolucion", agrupar_por="mes", filtros=filtros),
            "precio_producto": lambda: Consulta(medida="precio", operacion="evolucion", filtros=filtros),
            "ultima_compra": lambda: Consulta(medida="compras", operacion="ultimo", filtros=filtros),
            "lista_tickets": lambda: Consulta(medida="compras", operacion="lista", agrupar_por="ticket", filtros=filtros),
        }
        if re.search(r"(producto|articulo|cosa)s? mas barat", t):
            intencion = "producto_mas_barato"
        mapa["producto_mas_barato"] = lambda: Consulta(medida="precio", operacion="minimo", agrupar_por="producto", filtros=filtros)
        consulta = mapa[intencion]()
        if consulta.operacion == "ranking":
            if re.search(r"\bmenos\b", t) and not m:
                consulta.orden = "asc"
            n = re.search(r"\b(\d{1,2}) (productos|categorias|subcategorias|cosas|articulos)", t)
            if n:
                consulta.limite = max(1, min(50, int(n.group(1))))
        return consulta


def _fecha(t) -> Optional[date]:
    iso = t.get("fecha_iso")
    if not iso:
        m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(t.get("fecha_compra") or ""))
        if not m:
            return None
        iso = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    try:
        return date.fromisoformat(iso)
    except ValueError:
        return None
