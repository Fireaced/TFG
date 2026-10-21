"""
Detección del tipo de documento y de tickets repetidos (usado por server.py y por la validación).

  - es_extracto_pdf: rechaza los PDF que no son tickets (extractos bancarios), ya que la app solo admite tickets.
  - numero_factura: nº de factura simplificada de un ticket PDF (identifica el ticket sin ambigüedad).
  - huella_ticket: huella del CONTENIDO de un ticket (fecha, total y productos) para detectar el mismo
    ticket subido dos veces desde archivos distintos.
"""
import hashlib
import io
import json
import re

import pdfplumber

from categorizador import normalizar


def fecha_a_iso(fecha) -> str:
    """'dd/mm/yyyy[ HH:MM]' -> 'yyyy-mm-dd' (cadena vacía si no se reconoce)."""
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(fecha or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


# La aplicación solo trabaja con tickets. Para no pasar por Donut un PDF que no lo es (p. ej. un extracto
# del banco, donde «MERCADONA» aparece como concepto), se rechazan los PDF con marcas de extracto y sin
# ninguna marca exclusiva de ticket. Los devueltos no llevan «FACTURA SIMPLIFICADA» pero sí el CIF.
# OJO: «TARJETA BANCARIA» o «IMPORTE:» también salen en extractos, así que no sirven como marca de ticket.
MARCAS_TICKET = ("FACTURA SIMPLIFICADA", "A-46103834", "SE ADMITEN DEVOLUCIONES")
MARCAS_BANCO = ("SALDO", "IBAN", "EXTRACTO", "FECHA VALOR", "F. VALOR", "MOVIMIENTOS")


def es_extracto(texto: str) -> bool:
    t = texto.upper()
    return any(m in t for m in MARCAS_BANCO) and not any(m in t for m in MARCAS_TICKET)


def es_extracto_pdf(pdf_bytes: bytes) -> bool:
    """True si el PDF es un extracto bancario (u otro documento con sus marcas), no un ticket."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            texto = "\n".join((page.extract_text() or "") for page in pdf.pages[:2])
    except Exception:
        return False
    return es_extracto(texto)


def huella_ticket(fecha_compra, precio_total, items) -> str:
    """Huella del contenido del ticket: misma fecha, mismo total y exactamente los mismos productos
    (descripción, cantidad y precio unitario), sin importar el orden ni mayúsculas/espacios."""
    productos = sorted(
        f"{normalizar(str(i.get('descripcion', '')))}|{int(i.get('cantidad', 1) or 1)}|{float(i.get('precio_unitario', 0) or 0):.2f}"
        for i in items
    )
    base = json.dumps([fecha_a_iso(fecha_compra), f"{float(precio_total or 0):.2f}", productos], ensure_ascii=False)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


PATRON_FACTURA = re.compile(r"FACTURA SIMPLIFICADA:?\s*([0-9][0-9-]{6,})", re.I)


def numero_factura(contenido: bytes):
    """Nº de factura simplificada del texto de un PDF de ticket (identifica el ticket sin ambigüedad).
    Las fotos no tienen texto y los tickets de devolución no llevan este número: devuelve None."""
    if not contenido.startswith(b"%PDF"):
        return None
    try:
        with pdfplumber.open(io.BytesIO(contenido)) as pdf:
            texto = "\n".join((p.extract_text() or "") for p in pdf.pages[:1])
    except Exception:
        return None
    m = PATRON_FACTURA.search(texto)
    return m.group(1) if m else None
