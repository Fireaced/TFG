"""
Comprobación de la fecha leída por Donut.

En las fotos, Donut a veces confunde o intercambia dígitos de la fecha
(p. ej. lee 02/01/2026 cuando el ticket es del 02/10/2026). Para detectarlo se
compara la fecha leída con una fecha de referencia:

  1. La fecha en la que se hizo la foto (metadatos EXIF de la imagen): el ticket
     no puede ser posterior y normalmente tampoco muy anterior (45 días).
  2. Si no hay EXIF, la fecha de última modificación del archivo que envía el
     navegador. En los PDF solo se usa como límite superior (se pueden descargar meses después).

Un ticket se fotografía o se descarga el mismo día de la compra o algo después,
nunca antes. Si la fecha leída es posterior a la referencia (o al día de hoy), o
muy anterior, se genera un aviso y se busca una corrección plausible probando los
errores típicos de lectura: dígitos intercambiados, día y mes intercambiados y
dígitos que se parecen (0/8, 1/7, 3/8, 5/6…).
"""
import io
import re
from datetime import date, datetime, timedelta
from itertools import product
from typing import Optional

from PIL import Image

# Dígitos que Donut confunde con facilidad en fotos
PARECIDOS = {
    "0": "86", "1": "74", "2": "7", "3": "85", "4": "1", "5": "63",
    "6": "580", "7": "12", "8": "3069", "9": "8",
}
MARGEN_ANTERIOR = 45   # días: más antigua que esto respecto a la referencia se considera sospechosa
MARGEN_POSTERIOR = 1   # días: el ticket no puede ser posterior a la foto (margen por zonas horarias)

PATRON = re.compile(r"\s*(\d{1,2})/(\d{1,2})/(\d{2,4})(.*)$", re.S)


def fecha_exif(contenido: bytes) -> Optional[date]:
    """Fecha en la que se tomó la foto (EXIF DateTimeOriginal o DateTime), si existe."""
    if contenido.startswith(b"%PDF"):
        return None
    try:
        exif = Image.open(io.BytesIO(contenido)).getexif()
        valor = exif.get_ifd(0x8769).get(36867) or exif.get(306)    # DateTimeOriginal / DateTime
        if valor:
            return datetime.strptime(str(valor).strip()[:19], "%Y:%m:%d %H:%M:%S").date()
    except Exception:
        pass
    return None


def _a_fecha(d: str, m: str, a: str) -> Optional[date]:
    try:
        anio = int(a) + (2000 if len(a) == 2 else 0)
        return date(anio, int(m), int(d))
    except ValueError:
        return None


def candidatas(d: str, m: str, a: str):
    """Fechas alternativas según los errores de lectura habituales (con su explicación)."""
    d, m = d.zfill(2), m.zfill(2)
    vistas = set()

    def emitir(dd, mm, aa, motivo):
        f = _a_fecha(dd, mm, aa)
        if f and f not in vistas:
            vistas.add(f)
            yield f, motivo

    yield from emitir(d, m[::-1], a, "dígitos del mes intercambiados")
    yield from emitir(d[::-1], m, a, "dígitos del día intercambiados")
    yield from emitir(m, d, a, "día y mes intercambiados")
    for anio in (str(int(a) - 1), str(int(a) + 1)):
        yield from emitir(d, m, anio, "año mal leído")
    texto = d + m
    for i, j in product(range(4), repeat=2):
        if i > j:
            continue
        for ci in PARECIDOS[texto[i]]:
            for cj in (PARECIDOS[texto[j]] if j != i else [ci]):
                nuevo = list(texto)
                nuevo[i] = ci
                if j != i:
                    nuevo[j] = cj
                nuevo = "".join(nuevo)
                yield from emitir(nuevo[:2], nuevo[2:], a, "dígito parecido mal leído")


def _distancia(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def por_digitos(fecha_leida: str, desde: date, hasta: date) -> Optional[date]:
    """Fecha del intervalo [desde, hasta] cuyos dígitos (ddmmaaaa o ddmmaa) se parecen más a los leídos.
    Recupera lecturas con barras perdidas o dígitos de más/de menos ("2907/20", "0705/2025", "3007026")
    y meses mal leídos que no están en la tabla de dígitos parecidos ("07/09/2025" -> 07/05/2025).
    Solo se usa con una fecha de referencia (intervalo de unas semanas), nunca a ciegas."""
    sin_hora = re.sub(r"\s+\d{1,2}:\d{2}.*$", "", str(fecha_leida or "").strip())
    digitos = re.sub(r"\D", "", sin_hora)
    if len(digitos) < 4 or (hasta - desde).days > 120:
        return None
    puntuadas = []
    f = desde
    while f <= hasta:
        d = min(_distancia(digitos, f.strftime("%d%m%Y")), _distancia(digitos, f.strftime("%d%m%y")))
        puntuadas.append((d, (hasta - f).days, f))
        f += timedelta(days=1)
    puntuadas.sort()                      # menos dígitos distintos y, a igualdad, más cerca de la referencia
    mejor = puntuadas[0]
    if mejor[0] > 2:
        return None                       # demasiado distinta: no se adivina
    if mejor[0] == 2 and len(puntuadas) > 1 and puntuadas[1][0] == 2:
        return None                       # empate con 2 dígitos distintos: demasiado ambiguo
    return mejor[2]                       # con 1 dígito distinto, ante empate, la más cercana a la referencia


def revisar_fecha(fecha_leida: str, contenido: bytes = b"", ultima_modificacion_ms: Optional[float] = None,
                  hoy: Optional[date] = None) -> dict:
    """Devuelve {fecha_referencia, origen_referencia, aviso, fecha_sugerida, motivo_sugerencia}."""
    hoy = hoy or date.today()
    res = {"fecha_referencia": None, "origen_referencia": None, "aviso": None,
           "fecha_sugerida": None, "motivo_sugerencia": None}

    es_imagen = bool(contenido) and not contenido.startswith(b"%PDF")
    referencia, origen = fecha_exif(contenido), "foto"
    if referencia is None and ultima_modificacion_ms:
        try:
            referencia = datetime.fromtimestamp(float(ultima_modificacion_ms) / 1000).date()
            origen = "imagen" if es_imagen else "descarga del archivo"
        except (ValueError, OSError, OverflowError):
            referencia = None
    if referencia and referencia > hoy:
        referencia = None
    if referencia:
        res["fecha_referencia"], res["origen_referencia"] = referencia.strftime("%d/%m/%Y"), origen

    m = PATRON.match(str(fecha_leida or ""))
    if not m:
        res["aviso"] = "No se ha podido leer la fecha del ticket. Revísala."
        recuperada = por_digitos(fecha_leida, referencia - timedelta(days=MARGEN_ANTERIOR),
                                 min(hoy, referencia + timedelta(days=MARGEN_POSTERIOR))) if referencia and es_imagen else None
        if recuperada:
            res["fecha_sugerida"] = recuperada.strftime("%d/%m/%Y")
            res["motivo_sugerencia"] = "dígitos de la fecha leída (faltan o sobran barras o dígitos)"
        elif referencia:
            res["fecha_sugerida"] = referencia.strftime("%d/%m/%Y")
            res["motivo_sugerencia"] = f"fecha de la {origen}"
        return res
    d, mes, a, resto = m.groups()
    leida = _a_fecha(d, mes, a)
    limite_sup = min(hoy, referencia + timedelta(days=MARGEN_POSTERIOR)) if referencia else hoy

    def plausible(f: date) -> bool:
        if f > limite_sup:
            return False
        # Solo en imágenes se exige cercanía por abajo: un PDF puede descargarse meses después de la compra
        if referencia and es_imagen and f < referencia - timedelta(days=MARGEN_ANTERIOR):
            return False
        return f >= hoy - timedelta(days=365 * 5)

    if leida and plausible(leida):
        return res

    # La fecha leída no encaja: se explica por qué y se busca la corrección más cercana a la referencia
    if leida is None:
        res["aviso"] = f"La fecha leída ({fecha_leida.strip()}) no es una fecha válida."
    elif leida > hoy:
        res["aviso"] = f"La fecha leída ({leida.strftime('%d/%m/%Y')}) es posterior a hoy."
    elif referencia and leida > limite_sup:
        res["aviso"] = (f"La fecha leída ({leida.strftime('%d/%m/%Y')}) es posterior a la fecha de la {origen} "
                        f"({referencia.strftime('%d/%m/%Y')}).")
    elif referencia and es_imagen:
        res["aviso"] = (f"La fecha leída ({leida.strftime('%d/%m/%Y')}) es muy anterior a la fecha de la {origen} "
                        f"({referencia.strftime('%d/%m/%Y')}).")
    else:
        res["aviso"] = f"La fecha leída ({leida.strftime('%d/%m/%Y')}) parece demasiado antigua."

    objetivo = referencia or hoy
    opciones = [(f, motivo) for f, motivo in candidatas(d, mes, a) if plausible(f)]
    if opciones:
        f, motivo = min(opciones, key=lambda x: abs((objetivo - x[0]).days))
        res["fecha_sugerida"] = f.strftime("%d/%m/%Y") + resto.rstrip()
        res["motivo_sugerencia"] = motivo
    elif referencia and es_imagen and (recuperada := por_digitos(
            fecha_leida, referencia - timedelta(days=MARGEN_ANTERIOR), limite_sup)):
        res["fecha_sugerida"] = recuperada.strftime("%d/%m/%Y") + resto.rstrip()
        res["motivo_sugerencia"] = "fecha cercana a la de la foto con los dígitos más parecidos"
    elif referencia:
        res["fecha_sugerida"] = referencia.strftime("%d/%m/%Y") + resto.rstrip()
        res["motivo_sugerencia"] = f"fecha de la {origen}"
    return res
