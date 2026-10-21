# Validación: solo tickets y tickets repetidos — 13/10/2026 16:11

## 1) Solo tickets: ¿se rechazan los PDF que no lo son?

- Tickets PDF de Mercadona (formatos de 2024, desde 08/2025 con tabla y desde 10/2026 con «Cnt.»): aceptados 100.0 % (108/108)
- Extractos bancarios simulados (6 formatos): rechazados 100.0 % (6/6)

## 2) Tickets repetidos

108 tickets distintos (PDF con su ticket correcto).

### Falsos positivos (tickets distintos que la regla tomaría por el mismo)

- Misma huella (fecha + total + productos): 0 casos
- Mismo nº de factura: 0 casos (106 de 108 PDF tienen nº de factura; los de devolución no)
- Misma fecha y total (solo aviso): 0 casos

### Detección al volver a subir un ticket ya guardado (otro archivo, mismo ticket)

El mismo archivo exacto se detecta siempre por su hash (100 %). Aquí el archivo es distinto.

| Se vuelve a subir como | Por huella (antes) | + nº de factura | + aviso fecha y total | Total detectado |
|---|---|---|---|---|
| PDF | 87.1 % (27/31) | 100.0 % (31/31) | 96.8 % (30/31) | 100.0 % (31/31) |
| foto simulada | 31.2 % (29/93) | — | 72.0 % (67/93) | 72.0 % (67/93) |

Nota: al guardar, la app vuelve a comprobar la huella con los datos ya revisados por el usuario; si los deja iguales que los del ticket guardado, el repetido se detecta entonces aunque Donut se haya equivocado.

