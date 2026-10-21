"""
Preparación del dataset de tickets de Mercadona para Donut (v3).

Qué hace:
  1. Lee todos los PDF de la carpeta de origen y elimina duplicados exactos
     (mismo contenido aunque tengan distinto nombre, p. ej. "... (1).pdf").
  2. Extrae el Ground Truth directamente del texto del PDF con un parser
     línea a línea específico del formato de Mercadona:
        - cantidad, descripción, precio unitario (si lo hay) e importe
        - productos a peso (línea extra "0,806 kg 1,60 €/kg 1,29")
        - fecha y total
  3. VALIDA cada ticket: la suma de importes debe coincidir con el TOTAL.
     Si no cuadra, el ticket se descarta y se informa (no entra ruido).
  4. Renderiza la imagen de cada ticket y la guarda con un nombre simple
     (ticket_001.jpg...), sin espacios ni "€" (evita problemas al descomprimir).
  5. Divide en train / validation (85 % / 15 %) de forma reproducible y escribe
     un metadata.jsonl por carpeta, en el formato que espera `imagefolder`.

Uso:
    python preparar_dataset.py <carpeta_con_pdfs> <carpeta_salida>
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pdfplumber

DPI = 200            # 200 ppp es más que suficiente: Donut reescala a ~960 px de ancho
VAL_RATIO = 0.15
SEED = 42

PRECIO = r"-?\d+,\d{2}"
RE_ITEM = re.compile(r"^(-?\d+)\s+(.+)$")
RE_PESO = re.compile(r"^(\d+,\d{3})\s*kg\s+(\d+,\d{2})\s*€/kg\s+(" + PRECIO + r")$")
RE_FECHA = re.compile(r"(\d{2}/\d{2}/\d{4})")
RE_TOTAL = re.compile(r"^TOTAL \(€\)\s+(" + PRECIO + r")$")


def a_centimos(valor: str) -> int:
    return int(round(float(valor.replace(",", ".")) * 100))


def parsear_ticket(texto: str) -> dict:
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    fecha, total, items, ignoradas = None, None, [], []
    en_productos = False
    i = 0
    while i < len(lineas):
        linea = lineas[i]
        if fecha is None:
            m = RE_FECHA.search(linea)
            if m:
                fecha = m.group(1)
        if linea.startswith("Descripción"):
            en_productos = True
            i += 1
            continue
        m_total = RE_TOTAL.match(linea)
        if m_total:
            total = m_total.group(1)
            break
        if en_productos:
            m = RE_ITEM.match(linea)
            if not m:
                ignoradas.append(linea)          # p. ej. "ENTRADA 18:38 SALIDA 19:19"
                i += 1
                continue
            cnt, resto = m.group(1), m.group(2)
            tokens = resto.split(" ")
            precios = []
            while tokens and len(precios) < 2 and re.fullmatch(PRECIO, tokens[-1]):
                precios.insert(0, tokens.pop())
            nm = " ".join(tokens).strip()
            item = {"cnt": cnt, "nm": nm}
            if not precios:
                # Producto a peso: el precio está en la línea siguiente
                m_peso = RE_PESO.match(lineas[i + 1]) if i + 1 < len(lineas) else None
                if not m_peso:
                    raise ValueError(f"Línea de producto sin precio: {linea!r}")
                item["weight"] = f"{m_peso.group(1)} kg"
                item["unitprice"] = f"{m_peso.group(2)} €/kg"
                item["price"] = m_peso.group(3)
                i += 1
            elif len(precios) == 2:
                item["unitprice"], item["price"] = precios
            else:
                item["price"] = precios[0]
            items.append(item)
        i += 1

    if fecha is None or total is None or not items:
        raise ValueError("No se encontró fecha, total o productos")
    suma = sum(a_centimos(it["price"]) for it in items)
    if suma != a_centimos(total):
        raise ValueError(f"La suma de importes ({suma/100:.2f}) no cuadra con el TOTAL ({total})")
    return {"date": fecha, "menu": items, "total": {"total_price": total}}, ignoradas


def main(origen: Path, destino: Path):
    # Se procesan primero los nombres "limpios" para que, ante duplicados, se descarte la copia "(1)"
    pdfs = sorted(origen.glob("*.pdf"), key=lambda p: ("(" in p.stem, p.name))
    vistos, unicos, duplicados = {}, [], []
    for pdf in pdfs:
        h = hashlib.md5(pdf.read_bytes()).hexdigest()
        if h in vistos:
            duplicados.append((pdf.name, vistos[h]))
        else:
            vistos[h] = pdf.name
            unicos.append(pdf)

    registros, errores = [], []
    for pdf in unicos:
        try:
            with pdfplumber.open(pdf) as doc:
                texto = "\n".join(p.extract_text() or "" for p in doc.pages)
                gt, ignoradas = parsear_ticket(texto)
                imagen = doc.pages[0].to_image(resolution=DPI).original.convert("RGB")
            registros.append((pdf.name, gt, imagen, ignoradas))
        except Exception as e:
            errores.append((pdf.name, str(e)))

    random.Random(SEED).shuffle(registros)
    n_val = max(1, round(len(registros) * VAL_RATIO))
    splits = {"validation": registros[:n_val], "train": registros[n_val:]}

    informe = {"pdfs": len(pdfs), "duplicados": duplicados, "errores": errores, "splits": {}}
    contador = 1
    for split, regs in splits.items():
        carpeta = destino / split
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / "metadata.jsonl", "w", encoding="utf-8") as f:
            for nombre_pdf, gt, imagen, ignoradas in regs:
                nombre = f"ticket_{contador:03d}.jpg"
                contador += 1
                imagen.save(carpeta / nombre, "JPEG", quality=92)
                f.write(json.dumps({
                    "file_name": nombre,
                    "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False),
                    "origen": nombre_pdf,
                }, ensure_ascii=False) + "\n")
                if ignoradas:
                    informe.setdefault("lineas_ignoradas", {})[nombre_pdf] = ignoradas
        informe["splits"][split] = len(regs)

    (destino / "informe_dataset.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in informe.items() if k != "lineas_ignoradas"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
