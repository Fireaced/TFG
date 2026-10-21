"""
Extractos bancarios en PDF SIMULADOS para comprobar que la app (que solo admite tickets) los rechaza
en vez de pasarlos por Donut como si fueran tickets.

No hay extractos reales en el proyecto (ni conviene usarlos: tienen datos bancarios), así que se generan
varios formatos genéricos de banco español, sin marcas reales, con movimientos inventados (algunos con el
concepto «MERCADONA», que es lo que puede confundir). Cada PDF lleva en el nombre el nº de movimientos.

Formatos:
  clasico          «EXTRACTO DE CUENTA», IBAN, tabla con rejilla Fecha | Concepto | F. valor | Importe | Saldo
  movimientos      «Movimientos de la cuenta», sin la palabra extracto, fechas dd-mm-aaaa
  con_importe      resumen con «Importe: …» (texto que también aparece en los tickets)
  tarjeta          extracto de tarjeta: «TARJETA BANCARIA ****», sin IBAN
  sin_rejilla      tabla sin líneas (columnas alineadas con espacios)
  varias_paginas   45 movimientos en 2 páginas, fechas dd/mm sin año

Uso (desde backend/):  python -m validacion.generar_extractos_sinteticos
"""
import random
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SALIDA = Path(__file__).resolve().parent / "extractos_sinteticos"
CONCEPTOS = ["COMPRA TARJ. MERCADONA MADRID", "RECIBO LUZ ENERGIA SA", "TRANSFERENCIA DE J. PEREZ",
             "BIZUM ENVIADO", "COMPRA TARJ. FARMACIA CENTRAL", "NOMINA EMPRESA SL", "COMPRA TARJ. MERCADONA 28018",
             "CAJERO RETIRADA", "SUSCRIPCION STREAMING", "COMPRA ONLINE TIENDA"]


def euros(v):
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def movimientos(n, rnd, formato_fecha="%d/%m/%Y"):
    f, saldo, filas = date(2026, 3, 1), 1500.0, []
    for _ in range(n):
        f += timedelta(days=rnd.randint(0, 3))
        concepto = rnd.choice(CONCEPTOS)
        imp = round(rnd.uniform(800, 1600), 2) if "NOMINA" in concepto else -round(rnd.uniform(1, 90), 2)
        saldo += imp
        filas.append([f.strftime(formato_fecha), concepto, f.strftime(formato_fecha), euros(imp), euros(saldo)])
    return filas


def documento(nombre, cabecera, filas, cabecera_tabla=("Fecha", "Concepto", "F. valor", "Importe", "Saldo"),
              rejilla=True, pie=()):
    est = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(SALIDA / nombre), pagesize=A4)
    partes = [Paragraph(l, est["Normal"]) for l in cabecera] + [Spacer(1, 12)]
    if rejilla:
        t = Table([list(cabecera_tabla)] + filas, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        partes.append(t)
    else:
        for fila in [list(cabecera_tabla)] + filas:
            partes.append(Paragraph("&nbsp;&nbsp;&nbsp;".join(fila), est["Code"]))
    partes += [Spacer(1, 12)] + [Paragraph(l, est["Normal"]) for l in pie]
    doc.build(partes)


def main():
    SALIDA.mkdir(exist_ok=True)
    rnd = random.Random(3)
    titular = ["Banco de Ejemplo, S.A.", "Titular: CLIENTE DE PRUEBA"]
    f = movimientos(12, rnd)
    documento("clasico_12mov.pdf", titular + ["EXTRACTO DE CUENTA", "IBAN ES00 0000 0000 0000 0000 0000",
                                              "Periodo: 01/03/2026 - 31/03/2026"], f,
              pie=[f"Saldo final: {f[-1][4]} €"])
    f = movimientos(10, rnd, "%d-%m-%Y")
    documento("movimientos_10mov.pdf", titular + ["Movimientos de la cuenta", "Cuenta: ES00 **** **** 1234"], f,
              pie=[f"Saldo disponible: {f[-1][4]} €"])
    f = movimientos(8, rnd)
    documento("con_importe_8mov.pdf", titular + ["EXTRACTO DE CUENTA", "IBAN ES00 0000 0000 0000 0000 0000"], f,
              pie=["Resumen del periodo", f"Importe: {euros(sum(float(x[3].replace('.', '').replace(',', '.')) for x in f))} €",
                   f"Saldo final: {f[-1][4]} €"])
    f = [r[:2] + r[3:4] for r in movimientos(9, rnd)]
    documento("tarjeta_9mov.pdf", titular + ["Movimientos de tarjeta", "TARJETA BANCARIA **** **** **** 4321",
                                             "Fecha valor de cargo: 05/04/2026"], f, cabecera_tabla=("Fecha", "Comercio", "Importe"))
    f = movimientos(7, rnd)
    documento("sin_rejilla_7mov.pdf", titular + ["EXTRACTO DE CUENTA", "IBAN ES00 0000 0000 0000 0000 0000"], f,
              rejilla=False, pie=[f"Saldo final: {f[-1][4]} €"])
    f = movimientos(45, rnd, "%d/%m")
    documento("varias_paginas_45mov.pdf", titular + ["EXTRACTO DE CUENTA", "IBAN ES00 0000 0000 0000 0000 0000"], f,
              pie=[f"Saldo final: {f[-1][4]} €"])
    print(f"✅ Extractos simulados en {SALIDA}")


if __name__ == "__main__":
    main()
