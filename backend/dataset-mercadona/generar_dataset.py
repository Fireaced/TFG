import os
import json
import re
import pdfplumber
import pytesseract
from datetime import datetime, timezone

# --- FUNCIONES DE AYUDA ---
def parse_spanish_float(val: str) -> float:
    if not val:
        return 0.0
    val = val.strip().replace('€', '').replace('\u20ac', '').strip()
    val = val.replace('.', '').replace(',', '.')
    try:
        return float(val)
    except ValueError:
        return 0.0

def extraer_datos_ticket(texto: str) -> dict:
    texto = texto.replace('@', '0').replace('O,', '0,')
    lines = [line.strip() for line in texto.split('\n') if line.strip()]

    store = 'Desconocida'
    fecha_compra = None
    items = []
    precio_total = 0.0
    
    # --- BANDERAS DE ESTADO ---
    leyendo_productos = False
    buffer = []
    
    precio_regex = re.compile(r'^\d+[,\.]\d{2}$')

    def parse_buffer(buf):
        texto_unido = " ".join(buf)
        
        texto_unido = re.sub(r'\d+[,\.]\d+\s*kg', '', texto_unido, flags=re.IGNORECASE)
        texto_unido = re.sub(r'\d+[,\.]\d+\s*€/kg', '', texto_unido, flags=re.IGNORECASE)
        texto_unido = texto_unido.replace('€', '').strip()
        
        precios = re.findall(r'\b\d+[,\.]\d{2}\b', texto_unido)
        if not precios:
            return None
            
        importe = parse_spanish_float(precios[-1])
        precio_unitario = importe
        if len(precios) >= 2:
            precio_unitario = parse_spanish_float(precios[-2])
            
        texto_sin_precios = texto_unido
        for p in precios:
            partes = texto_sin_precios.rsplit(p, 1)
            texto_sin_precios = "".join(partes)
        texto_sin_precios = texto_sin_precios.strip()
        
        tokens = texto_sin_precios.split()
        if not tokens:
            return None
            
        cantidad = 1
        if tokens[-1].isdigit() and len(tokens) > 1:
            cantidad = int(tokens[-1])
            descripcion = " ".join(tokens[:-1])
        elif tokens[0].isdigit() and len(tokens) > 1:
            cantidad = int(tokens[0])
            descripcion = " ".join(tokens[1:])
        else:
            descripcion = texto_sin_precios
            
        descripcion = re.sub(r'\s+', ' ', descripcion).strip()
        
        if descripcion:
            return {
                "descripcion": descripcion,
                "cantidad": cantidad,
                "precio_unitario": precio_unitario,
                "importe": importe
            }
        return None

    # --- BUCLE PRINCIPAL ---
    i = 0
    while i < len(lines):
        line = lines[i]
        upper_line = line.upper()

        # 1. Buscar Supermercado
        if store == 'Desconocida':
            if 'MERCADONA' in upper_line:
                store = 'MERCADONA, S.A.'
            elif 'LIDL' in upper_line:
                store = 'Lidl'

        # 2. Buscar Fecha
        if not fecha_compra:
            m = re.search(r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})', line)
            if m:
                d, mo, y = m.group(1), m.group(2), m.group(3)
                if len(y) == 2: y = '20' + y
                fecha_compra = f"{d.zfill(2)}/{mo.zfill(2)}/{y}"

        # 3. Detectar INICIO de productos
        if "DESCRIPCIÓN" in upper_line or "P. UNIT" in upper_line or "IMPORTE" in upper_line:
            leyendo_productos = True
            i += 1
            continue
            
        # 4. Detectar FIN de productos (El Cierre de Seguridad)
        if leyendo_productos and (upper_line.startswith("TOTAL") or "IVA" in upper_line or "TARJETA" in upper_line or "BANCARIA" in upper_line):
            leyendo_productos = False
            if buffer:
                item = parse_buffer(buffer)
                if item: items.append(item)
                buffer = []
        
        # 5. Buscar el TOTAL (Se busca SIEMPRE, incluso si ya no estamos en la zona de productos)
        if "TOTAL" in upper_line:
            m = re.search(r'(\d+[,\.]\d{2})', upper_line)
            if m:
                total_encontrado = parse_spanish_float(m.group(1))
                if total_encontrado > precio_total:
                    precio_total = total_encontrado
            elif i + 1 < len(lines):
                m2 = re.search(r'(\d+[,\.]\d{2})', lines[i+1])
                if m2:
                    total_encontrado = parse_spanish_float(m2.group(1))
                    if total_encontrado > precio_total:
                        precio_total = total_encontrado

        # 6. Procesar líneas de productos (SOLO si estamos en la zona permitida)
        if leyendo_productos:
            m_precio_final = re.search(r'\b\d+[,\.]\d{2}$', line)
            if m_precio_final:
                buffer.append(line)
                if i + 1 < len(lines) and precio_regex.match(lines[i+1]):
                    buffer.append(lines[i+1])
                    i += 1
                    
                item = parse_buffer(buffer)
                if item:
                    items.append(item)
                buffer = []
            else:
                buffer.append(line)
        
        i += 1

    if precio_total == 0.0 and items:
        precio_total = sum(i['importe'] for i in items)

    return {
        'tienda': store,
        'fecha_compra': fecha_compra or datetime.now(timezone.utc).strftime('%d/%m/%Y'),
        'items': items,
        'precio_total': round(precio_total, 2)
    }

# --- SCRIPT PRINCIPAL ---
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_metadata = os.path.join(carpeta_actual, 'metadata.jsonl')

print("🚀 Iniciando extracción masiva HERMÉTICA...\n")

with open(ruta_metadata, 'w', encoding='utf-8') as f_meta:
    for archivo in os.listdir(carpeta_actual):
        if archivo.lower().endswith('.pdf'):
            ruta_pdf = os.path.join(carpeta_actual, archivo)
            nombre_sin_ext = os.path.splitext(archivo)[0]
            nombre_jpg = f"{nombre_sin_ext}.jpg"
            ruta_jpg = os.path.join(carpeta_actual, nombre_jpg)

            print(f"Procesando: {archivo}...")

            try:
                with pdfplumber.open(ruta_pdf) as pdf:
                    if len(pdf.pages) == 0:
                        continue
                    
                    imagen = pdf.pages[0].to_image(resolution=300).original
                    imagen_rgb = imagen.convert("RGB")
                    imagen_rgb.save(ruta_jpg, "JPEG", quality=95)

                    texto = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: texto += page_text + "\n"
                    
                    if not texto.strip():
                        texto = pytesseract.image_to_string(imagen_rgb, lang='spa')

                    datos_extraidos = extraer_datos_ticket(texto)

                    menu_items = []
                    for item in datos_extraidos.get('items', []):
                        precio_str = f"{item['precio_unitario']:.2f}".replace('.', ',')
                        menu_items.append({
                            "nm": item['descripcion'],
                            "cnt": str(item['cantidad']),
                            "price": precio_str
                        })

                    total_str = f"{datos_extraidos.get('precio_total', 0):.2f}".replace('.', ',')
                    fecha_limpia = datos_extraidos.get('fecha_compra', '').split()[0]
                    nombre_tienda = datos_extraidos.get('tienda', 'MERCADONA, S.A.')
                    if nombre_tienda.upper() == "MERCADONA":
                        nombre_tienda = "MERCADONA, S.A."

                    donut_json = {
                        "gt_parse": {
                            "store_info": {
                                "name": nombre_tienda,
                                "date": fecha_limpia
                            },
                            "menu": menu_items,
                            "total": {
                                "total_price": total_str
                            }
                        }
                    }

                    linea = {
                        "file_name": nombre_jpg,
                        "ground_truth": json.dumps(donut_json, ensure_ascii=False)
                    }
                    
                    f_meta.write(json.dumps(linea, ensure_ascii=False) + '\n')
                    print(f"  ✅ Extraído sin IVA.")

            except Exception as e:
                print(f"  ❌ Error procesando el archivo: {e}")

print("\n🎉 ¡Proceso completado! Revisa el JSONL, ahora debería estar impecable.")