"""
Ajuste de la consulta generada por el LLM contra el texto de la pregunta ("grounding").

El LLM entiende bien la intención de la pregunta, pero a veces se inventa o confunde detalles:
filtros que nadie ha pedido (importe_min = 0), días mal numerados (fin de semana = [6, 0]),
periodos recortados ("agosto de 2025" -> "en 2025"), una media donde se pedía un total o el
nombre de una categoría en lugar de la palabra que dijo el usuario ("fruta" -> "fruta y verdura").

Este módulo revisa cada campo de la consulta y lo contrasta con lo que dice la pregunta:
  - Lo que se puede extraer de forma fiable del texto con reglas (importes, días de la semana,
    periodo, número de resultados) se toma del texto.
  - Lo que el LLM ha puesto sin que haya nada en el texto que lo respalde, se quita.
  - El resto (qué medir, qué operación, cómo agrupar) se respeta, salvo incoherencias claras.

Se aplica también a las preguntas de continuación: el "texto de apoyo" incluye las preguntas
anteriores del historial, de las que el LLM puede heredar campos.
"""
import re
from datetime import date
from typing import List, Optional, Tuple

from motor_consultas import Consulta

PALABRAS_DINERO = {"pasta", "pelas", "guita", "lana", "dinero", "perras", "money"}
IDIOMA_DINERO = r"cuanta|cuanto|mucha|mucho|poca|poco|tanta|tanto|la|el|las|mas|menos|algo de|mi|tu"
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10, "quince": 15, "veinte": 20}


def _sin_tildes(texto: str) -> str:
    from asistente_gastos import sin_tildes
    return sin_tildes(texto or "")


# ── Extracción de datos fiables del texto ──
def dias_en_texto(t: str) -> Optional[List[int]]:
    if re.search(r"\bfin(es)? de semana\b", t):
        dias = [5, 6]
    elif re.search(r"\bentre semana\b|\bdias? laborables?\b|\bde lunes a viernes\b", t):
        dias = [0, 1, 2, 3, 4]
    else:
        dias = []
    sin_dia_semana = re.sub(r"\bdias? de la semana\b", " ", t)
    for i, d in enumerate(DIAS):
        if re.search(rf"\b{d}s?\b", sin_dia_semana) and i not in dias:
            dias.append(i)
    return sorted(dias) or None


def _numero(txt: str) -> float:
    return float(txt.replace(".", "").replace(",", ".")) if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", txt) \
        else float(txt.replace(",", "."))


def importes_en_texto(t: str) -> Tuple[Optional[float], Optional[float]]:
    num = r"(\d+(?:[.,]\d+)?)"
    euro = r"\s*(?:€|euros?|eur)?"
    m = re.search(rf"\bentre {num}{euro} y {num}{euro}", t)
    if m:
        a, b = sorted((_numero(m.group(1)), _numero(m.group(2))))
        return a, b
    mn = re.search(rf"\b(?:mas de|mayor(?:es)? de|superior(?:es)? a|por encima de|de al menos|minimo de) {num}{euro}", t)
    mx = re.search(rf"\b(?:menos de|menor(?:es)? de|inferior(?:es)? a|por debajo de|de hasta|maximo de|que no (?:llegan|pasan) de) {num}{euro}", t)
    return (_numero(mn.group(1)) if mn else None), (_numero(mx.group(1)) if mx else None)


def limite_en_texto(t: str) -> Optional[int]:
    t = re.sub(r"\b(los |las )?ultim[oa]s \d+ (dias|semanas|meses|anos)\b", " ", t)   # "los últimos 3 meses" es un periodo
    m = re.search(r"\b(?:los|las|top|primer[oa]s)\s+(\d{1,2}|" + "|".join(NUMEROS) + r")\b(?! (?:dias|semanas|meses|anos|euros|€))", t) or \
        re.search(r"\b(\d{1,2}|" + "|".join(k for k in NUMEROS if k not in ("un", "una", "uno")) + r")\s+"
                  r"(?:productos|categorias|subcategorias|articulos|cosas|tickets|compras|primer[oa]s)\b", t)
    if not m:
        return None
    v = m.group(1)
    return int(v) if v.isdigit() else NUMEROS[v]


_PATRON_MEDIA = (r"\bmedia\b|\bpromedio\b|\bde normal\b|\bnormalmente\b|\bsuel[oe]\b|\bfrecuencia\b|\bcada cuanto\b|"
                 r"\bhabitual(es|mente)?\b|\btermino medio\b|\btipic[oa]\b|\bal dia\b|\bal mes\b|\ba la semana\b|\bpor (compra|visita|ticket|semana|mes|dia)\b|"
                 r"\bcada (vez|visita|compra|semana|mes|dia)\b|\bdiari[oa]s?\b|\bsemanal(mente)?\b|\bmensual(mente)?\b|\bmedio\b")


def por_en_texto(t: str) -> Optional[str]:
    if re.search(r"\bpor (compra|visita|ticket)\b|\bcada (vez|visita|compra)\b|\bcuando (compro|voy|hago la compra)\b|\ben cada (compra|ticket)\b", t):
        return "compra"
    if re.search(r"\bal dia\b|\bdiari[oa]s?\b|\bcada dia\b|\bpor dia\b", t):
        return "dia"
    if re.search(r"\ba la semana\b|\bsemanal(mente)?\b|\bcada semana\b|\bpor semana\b", t):
        return "semana"
    if re.search(r"\bal mes\b|\bmensual(mente)?\b|\bcada mes\b|\bpor mes\b|\bmes a mes\b", t):
        return "mes"
    return None


def _quitar_frases_semana(t: str) -> str:
    return re.sub(r"\bfin(es)? de semana\b|\bentre semana\b|\ba la semana\b|\bdias? de la semana\b|\besta semana\b|"
                  r"\b(la )?semana (pasada|anterior)\b", " ", t)


def agrupacion_respaldada(agrupar: str, t: str) -> bool:
    if agrupar == "mes":
        return bool(re.search(r"\b(que|cual|cada|por|el|al) mes\b(?! pasado| anterior)|\bmes a mes\b|\bmeses\b|\bmensual", t))
    if agrupar == "semana":
        return bool(re.search(r"\bsemanas?\b", _quitar_frases_semana(t)))
    if agrupar == "dia_semana":
        return bool(re.search(r"\bdias? de la semana\b|\bque dia\b|\ben que dias?\b|\bque dias\b", t))
    return True


# ── Ajuste ──
def ajustar_consulta(c: Consulta, pregunta: str, historial: Optional[List[str]] = None,
                     hoy: Optional[date] = None) -> Tuple[Consulta, List[str]]:
    """Devuelve la consulta ajustada y la lista de cambios aplicados (para registro y para la memoria)."""
    if c is None or c.tipo != "consulta":
        return c, []
    from asistente_gastos import extraer_periodo, singular

    hoy = hoy or date.today()
    c = c.model_copy(deep=True)
    f = c.filtros
    t = _sin_tildes(pregunta).lower()
    previas = " ".join(_sin_tildes(h).lower() for h in (historial or []))
    apoyo = f"{t} {previas}"
    cambios = []

    # 1) Importes del ticket: del texto; si el texto no los menciona, fuera
    imin, imax = importes_en_texto(t)
    if imin is not None or imax is not None:
        if (f.importe_min, f.importe_max) != (imin, imax):
            cambios.append(f"importes {f.importe_min}/{f.importe_max} -> {imin}/{imax} (del texto)")
        f.importe_min, f.importe_max = imin, imax
    else:
        nums = {_numero(n) for n in re.findall(r"\d+(?:[.,]\d+)?", apoyo)}
        for campo in ("importe_min", "importe_max"):
            v = getattr(f, campo)
            if v is not None and (v <= 0 or v >= 1e5 or v not in nums):
                cambios.append(f"{campo}={v} quitado (no aparece en la pregunta)")
                setattr(f, campo, None)
        if f.importe_min is not None and f.importe_min == f.importe_max:
            cambios.append("importe_min = importe_max: se deja solo el mínimo")
            f.importe_max = None

    # 2) Días de la semana: del texto (0 = lunes); si no se nombra ninguno, fuera
    dias = dias_en_texto(t)
    if dias:
        if sorted(f.dias_semana) != dias:
            cambios.append(f"dias_semana {f.dias_semana} -> {dias} (del texto)")
        f.dias_semana = dias
    elif f.dias_semana and not dias_en_texto(previas):
        cambios.append(f"dias_semana {f.dias_semana} quitado (no se nombra ningún día)")
        f.dias_semana = []

    # 3) Periodo: si la pregunta tiene una expresión temporal que el motor sabe interpretar, manda el texto
    ini, fin, _, _ = extraer_periodo(pregunta, hoy)
    if ini or fin:
        if f.periodo != pregunta or f.desde or f.hasta:
            cambios.append(f"periodo «{f.periodo or ''}» -> el de la pregunta")
        f.periodo, f.desde, f.hasta = pregunta, None, None
    elif f.periodo and not (f.desde or f.hasta):
        p_ini, p_fin, _, _ = extraer_periodo(f.periodo, hoy)
        if not (p_ini or p_fin):
            h_ini, _, _, _ = extraer_periodo(previas, hoy) if previas else (None, None, "", "")
            if not h_ini:
                cambios.append(f"periodo «{f.periodo}» quitado (no se entiende ni aparece en la pregunta)")
                f.periodo = None

    # 4) Concepto: con las palabras del usuario. Si el LLM lo ha "ampliado" (p. ej. al nombre de una
    #    categoría), se recorta a las palabras que sí dijo el usuario.
    # 4a) Expresiones coloquiales de dinero ("¿cuánta pasta me he dejado?", "la guita", "las pelas"):
    #     no son un producto aunque "pasta" también lo sea
    if f.concepto and c.medida == "gasto" and not re.search(r"\bcompr", t):
        palabras = re.findall(r"[a-zñ]+", _sin_tildes(f.concepto).lower())
        dinero = {w for w in palabras if w in PALABRAS_DINERO and re.search(
            rf"(?<!\ben )(?<!\bde )\b({IDIOMA_DINERO}) {w}\b|\b{w} (me|te|nos|se) (he |has |hemos |ha )?(dejad|gastad|fundid|pulid)", t)}
        if dinero:
            resto = " ".join(w for w in palabras if w not in dinero)
            cambios.append(f"concepto «{f.concepto}» -> «{resto or '-'}» (expresión de dinero)")
            f.concepto = resto or None

    if f.concepto:
        palabras_apoyo = set(re.findall(r"[a-zñ]+", apoyo))
        raices_apoyo = {singular(p) for p in palabras_apoyo}
        propias, sustituidas = [], []
        for w in re.findall(r"[a-zñ]+", _sin_tildes(f.concepto).lower()):
            if len(w) < 3 or w in ("del", "las", "los", "con", "sin", "para"):
                continue
            if w in palabras_apoyo or singular(w) in raices_apoyo:
                propias.append(w)
            else:
                # palabra del usuario de la que el LLM ha puesto una forma recortada o parecida ("bebe" <- "bebida")
                largas = [p for p in palabras_apoyo if len(w) >= 4 and len(p) > len(w) and
                          (p.startswith(w) or (p[:len(w) - 1] == w[:-1] and len(w) - 1 >= 3))]
                if largas:
                    sustituidas.append(max(largas, key=len))
        nuevo = " ".join(propias + sustituidas)
        if nuevo and nuevo != _sin_tildes(f.concepto).lower():
            cambios.append(f"concepto «{f.concepto}» -> «{nuevo}» (palabras del usuario)")
            # se conservan las tildes originales del usuario si las había
            originales = {_sin_tildes(p).lower(): p for p in re.findall(r"\w+", (pregunta + " " + " ".join(historial or [])).lower())}
            f.concepto = " ".join(originales.get(p, p) for p in nuevo.split())

    # 5) "¿Qué compré el 13 de abril?" / "productos distintos" -> listado de productos
    if re.search(r"\bproductos? (distint|diferent)", t) or \
            (re.search(r"\bque (compre|he comprado|productos compre|cosas compre)\b", t) and not re.search(r"\bmas\b|\bmenos\b", t)):
        if (c.operacion, c.agrupar_por) != ("lista", "producto"):
            cambios.append(f"{c.operacion}/{c.agrupar_por} -> lista de productos")
        c.operacion, c.agrupar_por, c.medida = "lista", "producto", "unidades"

    # 6) Una sola compra / ticket concreto
    if re.search(r"\ben una (sola )?compra\b|\ben un (solo )?ticket\b", t) and c.operacion in ("maximo", "minimo", "ranking") \
            and c.agrupar_por != "ticket":
        cambios.append(f"agrupar_por {c.agrupar_por} -> ticket («en una compra»)")
        c.agrupar_por = "ticket"

    # 7) Medida: "cuántas veces / compras" = compras; "cuántos <producto>" = unidades
    if re.search(r"\bcuant[oa]s (veces|ocasiones|compras|tickets|visitas|devoluciones)\b", t):
        if c.medida != "compras" and c.operacion != "lista":
            cambios.append(f"medida {c.medida} -> compras («cuántas veces»)")
            c.medida = "compras"
    elif re.search(r"\bcuant[oa]s (?!productos distint|productos diferent|euros|dias|semanas|meses)[a-zñ]{3,}", t) \
            and c.medida == "compras" and c.operacion not in ("lista", "ultimo"):
        cambios.append("medida compras -> unidades («cuántos X»)")
        c.medida = "unidades"

    # 8) Media solo si la pregunta la pide (o una frecuencia); y "por" según el texto
    if c.operacion == "media" and not re.search(_PATRON_MEDIA, apoyo):
        cambios.append("media -> total (la pregunta no pide una media)")
        c.operacion, c.por = "total", None
    if c.operacion == "total" and c.por and re.search(_PATRON_MEDIA, apoyo):
        cambios.append(f"total con por={c.por} -> media")
        c.operacion = "media"
    if c.operacion == "media":
        por = por_en_texto(t) or por_en_texto(previas)
        if not por:
            por = "semana" if c.medida == "compras" else "compra"
        if c.por != por:
            cambios.append(f"por {c.por} -> {por}")
            c.por = por
    elif c.por:
        c.por = None

    # 9) Agrupaciones temporales sin respaldo en el texto
    if c.agrupar_por in ("mes", "semana", "dia_semana") and not agrupacion_respaldada(c.agrupar_por, apoyo):
        if c.operacion in ("total", "media"):
            cambios.append(f"agrupar_por {c.agrupar_por} quitado (la pregunta no lo pide)")
            c.agrupar_por = None
    # Un total "agrupado" es en realidad una evolución (por tiempo) o un ranking (por producto/categoría)
    if c.operacion == "total" and c.agrupar_por:
        nueva = "evolucion" if c.agrupar_por in ("mes", "semana", "dia_semana") else \
            "ranking" if c.agrupar_por in ("producto", "categoria", "subcategoria") else None
        if nueva:
            cambios.append(f"total agrupado por {c.agrupar_por} -> {nueva}")
            c.operacion = nueva

    # 10) Orden y número de resultados
    if c.orden == "asc" and not re.search(r"\bmenos\b|\bmenor\b|\bpeor|\bmas barat|\bminim|\bde menos a mas\b|\bascendente\b", apoyo):
        cambios.append("orden asc -> desc (la pregunta no pide «menos»)")
        c.orden = "desc"
    limite = limite_en_texto(t)
    if limite and c.operacion in ("ranking", "maximo", "minimo") and c.limite != limite:
        cambios.append(f"limite {c.limite} -> {limite} (del texto)")
        c.limite = limite
        if c.operacion in ("maximo", "minimo") and limite > 1:
            c.orden = "desc" if c.operacion == "maximo" else "asc"
            c.operacion = "ranking"
    elif not limite and c.limite != 5 and not limite_en_texto(previas):
        if not ("limite" in c.model_fields_set and c.limite == 1):
            cambios.append(f"limite {c.limite} -> 5 (la pregunta no da un número)")
            c.limite = 5

    if f.tipo_ticket == "devoluciones" and not re.search(r"devol|devuelt|reembols", apoyo):
        cambios.append("tipo_ticket devoluciones quitado (no se mencionan)")
        f.tipo_ticket = "todos"
    return c, cambios
