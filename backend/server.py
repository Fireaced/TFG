import os
import io
import re
import csv
import json
import uuid
import logging
import asyncio
import pdfplumber
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
import hashlib
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from categorizador import categorizador, normalizar
from asistente_gastos import AsistenteGastos
from traductor_llm import TraductorLLM
from fechas_ticket import revisar_fecha

# ─── Configuración Inicial e Inicialización de BD ───
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'fintrack')]

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Inicialización Global del Modelo Donut ───
def localizar_modelo(carpeta: str) -> Optional[str]:
    """Devuelve la carpeta que contiene config.json, aunque al descomprimir haya quedado anidada
    (p. ej. mi_modelo_mercadona_v3/mi_modelo_mercadona_v3/config.json)."""
    if not os.path.isdir(carpeta):
        return None
    candidatos = sorted(Path(carpeta).rglob("config.json"), key=lambda p: len(p.parts))
    return str(candidatos[0].parent) if candidatos else None

# Se usa el modelo v3 si existe; si no, el antiguo.
MODEL_PATH = os.environ.get("DONUT_MODEL_PATH") or next(
    (ruta for ruta in (localizar_modelo(os.path.join(ROOT_DIR, d))
                       for d in ("mi_modelo_mercadona_v3", "mi_modelo_mercadona")) if ruta),
    os.path.join(ROOT_DIR, "mi_modelo_mercadona"),
)
print(f"Cargando modelo Donut desde {MODEL_PATH}...")

# Configuración específica del modelo v3 (token de tarea y caracteres remapeados, p. ej. ñ -> Ñ).
# Los modelos antiguos no tienen este fichero y siguen usando <s_cord-v2>.
DONUT_CFG = {"task_token": "<s_cord-v2>", "mapeo_inverso": {}}
_cfg_path = os.path.join(MODEL_PATH, "donut_mercadona.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, encoding="utf-8") as _f:
        DONUT_CFG.update(json.load(_f))

try:
    processor = DonutProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✅ Modelo cargado y listo en: {device} (token de tarea {DONUT_CFG['task_token']})")
except Exception as e:
    print(f"⚠️ Error cargando el modelo: {e}")

# ─── Helpers ───
def parse_spanish_float(val: str) -> float:
    if not val: return 0.0
    val = val.strip().replace('€', '').replace('\u20ac', '').strip()
    val = val.replace('.', '').replace(',', '.')
    try:
        return float(val)
    except ValueError:
        return 0.0

def detect_file_type(filename: str, content_type: str) -> str:
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'): return 'image'
    if ext == 'csv': return 'csv'
    if ext == 'pdf': return 'pdf'
    if 'image' in content_type: return 'image'
    return 'unknown'

# ─── Pipelines de Procesamiento ───

def process_receipt_donut(file_bytes: bytes) -> dict:
    """Extrae la información usando Donut (Operación Síncrona Pesada)."""
    if file_bytes.startswith(b'%PDF'):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) > 0:
                image = pdf.pages[0].to_image(resolution=200).original.convert("RGB")  # misma resolución que en el entrenamiento
            else:
                return {}
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    task_prompt = DONUT_CFG["task_token"]
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            # Sin repetition_penalty: en un ticket se repiten legítimamente precios y productos
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.replace(task_prompt, "", 1).strip()
    sequence = sequence.translate(str.maketrans(DONUT_CFG["mapeo_inverso"]))

    return token2json(sequence)


def token2json(tokens: str, interno: bool = False):
    """Convierte la secuencia <s_clave>valor</s_clave> de Donut en dict (adaptado de clovaai/donut, MIT)."""
    salida = {}
    while tokens:
        inicio = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if inicio is None:
            break
        clave = inicio.group(1)
        fin = re.search(rf"</s_{re.escape(clave)}>", tokens, re.IGNORECASE)
        inicio = inicio.group()
        if fin is None:
            tokens = tokens.replace(inicio, "")
            continue
        fin = fin.group()
        contenido = re.search(f"{re.escape(inicio)}(.*?){re.escape(fin)}", tokens, re.IGNORECASE | re.DOTALL)
        if contenido is not None:
            contenido = contenido.group(1).strip()
            if "<s_" in contenido and "</s_" in contenido:
                valor = token2json(contenido, interno=True)
                if valor:
                    salida[clave] = valor[0] if len(valor) == 1 else valor
            else:
                hojas = [h.strip() for h in contenido.split("<sep/>")]
                salida[clave] = hojas[0] if len(hojas) == 1 else hojas
        tokens = tokens[tokens.find(fin) + len(fin):].strip()
        if tokens[:6] == "<sep/>":
            return [salida] + token2json(tokens[6:], interno=True)
    if salida:
        return [salida] if interno else salida
    return [] if interno else {"text_sequence": tokens}


def rescatar_json_roto(texto_crudo: str) -> dict:
    """Parser Defensivo: Rescata datos con Regex si la IA alucina o rompe el JSON."""
    tienda = "Desconocida"
    m_tienda = re.search(r'[\'"]name[\'"]:\s*[\'"]([^\'"]+)[\'"]', texto_crudo)
    if m_tienda: tienda = m_tienda.group(1)

    fecha = datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')
    m_fecha = re.search(r'[\'"]date[\'"]:\s*[\'"]([^\'"]+)[\'"]', texto_crudo)
    if m_fecha: fecha = m_fecha.group(1)

    total = 0.0
    m_total = re.search(r'[\'"]total_price[\'"]:\s*[\'"]([^\'"]+)[\'"]', texto_crudo)
    if m_total:
        total = parse_spanish_float(m_total.group(1))

    items = []
    patron_productos = re.finditer(r'[\'"]nm[\'"]:\s*[\'"]([^\'"]+)[\'"].*?(?:[\'"]cnt[\'"]|[\'"]price[\'"]):\s*[\'"]([\d,\.\-]+)[\'"]', texto_crudo)
    for match in patron_productos:
        nombre = match.group(1).strip()
        if len(nombre) < 3: continue
        items.append({
            "descripcion": nombre,
            "cantidad": 1,
            "precio_unitario": parse_spanish_float(match.group(2))
        })

    return {"tienda": tienda.title(), "fecha_compra": fecha, "items": items, "precio_total": total}


def parse_donut_output(donut_data: dict) -> dict:
    """Combina el JSON limpio (Camino Feliz) con el Parser Defensivo (Plan B)."""
    print("🤖 JSON devuelto por Donut:", donut_data)
    texto_crudo = donut_data.get("text_sequence", str(donut_data))
    datos_extraidos = donut_data.get("gt_parse", donut_data)

    if not datos_extraidos.get("menu") and "{" in texto_crudo:
        try:
            json_str = texto_crudo[texto_crudo.find("{"):]
            parsed = json.loads(json_str)
            datos_extraidos = parsed.get("gt_parse", parsed)
        except json.JSONDecodeError:
            pass

    items = []
    precio_total = 0.0
    tienda = "Desconocida"
    fecha_compra = datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')

    es_v3 = DONUT_CFG["task_token"] != "<s_cord-v2>"
    if isinstance(datos_extraidos, dict) and ("menu" in datos_extraidos or "store_info" in datos_extraidos):
        store_info = datos_extraidos.get("store_info", {})
        if isinstance(store_info, dict):
            tienda = store_info.get("name", "Desconocida")
            fecha = store_info.get("date", "")
            if fecha: fecha_compra = fecha
        if datos_extraidos.get("date"):               # formato v3
            fecha_compra = str(datos_extraidos["date"])
            tienda = "Mercadona"

        raw_items = datos_extraidos.get("menu", [])
        if isinstance(raw_items, dict): raw_items = [raw_items]
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                importe = parse_spanish_float(str(item.get("price", "0")))
                if es_v3:
                    # v3: price = importe de la línea, unitprice = precio unitario (o €/kg en productos a peso)
                    try:
                        cantidad = int(str(item.get("cnt", "1")).strip())
                    except ValueError:
                        cantidad = 1
                    if "weight" in item or cantidad == 0:
                        cantidad, precio_unitario = 1, importe
                    elif "unitprice" in item:
                        precio_unitario = parse_spanish_float(str(item["unitprice"]))
                    else:
                        precio_unitario = round(importe / cantidad, 2)
                else:
                    cantidad, precio_unitario = 1, importe   # modelo antiguo: la cantidad no era fiable
                items.append({
                    "descripcion": str(item.get("nm", "Producto sin nombre")),
                    "cantidad": cantidad,
                    "precio_unitario": precio_unitario,
                })

        total_data = datos_extraidos.get("total", {})
        if isinstance(total_data, dict):
            precio_total = parse_spanish_float(str(total_data.get("total_price", "0")))

    # Si la estructura estaba vacía o rota, entra el Parser Defensivo
    if not items:
        print("🛡️ Activando Parser Defensivo...")
        rescate = rescatar_json_roto(texto_crudo)
        tienda = rescate["tienda"] if rescate["tienda"] != "Desconocida" else tienda
        fecha_compra = rescate["fecha_compra"]
        items = rescate["items"]
        precio_total = rescate["precio_total"]

    # Corrección de seguridad final
    if precio_total == 0.0 and items:
        precio_total = sum(i['precio_unitario'] * i['cantidad'] for i in items)

    return {
        'tienda': tienda.title(),
        'fecha_compra': fecha_compra,
        'items': items,
        'precio_total': round(precio_total, 2),
        'texto_ocr': texto_crudo
    }


def process_bank_csv(csv_bytes: bytes) -> list:
    # (Código intacto, asume tu procesamiento CSV previo)
    text = csv_bytes.decode('utf-8', errors='replace')
    transactions = []
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text[:2000])
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ';' if ';' in text else ','
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows: return []
    # Simplified mapping based on typical usage
    header = [h.strip().lower() for h in rows[0]]
    for row in rows[1:]:
        if not row or all(not c.strip() for c in row): continue
        transactions.append({
            'fecha_oper': row[0] if len(row) > 0 else '',
            'concepto': row[1] if len(row) > 1 else '',
            'fecha_valor': row[0] if len(row) > 0 else '',
            'importe': row[2] if len(row) > 2 else '0',
            'saldo': row[3] if len(row) > 3 else '0',
        })
    return transactions

# Textos propios de un ticket (no de un extracto, donde "MERCADONA" puede aparecer como concepto)
MARCAS_TICKET = ("FACTURA SIMPLIFICADA", "A-46103834", "TOTAL (€)", "TARJETA BANCARIA", "IMPORTE: ")
MARCAS_BANCO = ("SALDO", "IBAN", "EXTRACTO", "FECHA VALOR", "F. VALOR", "MOVIMIENTOS")
PATRON_FECHA_CELDA = re.compile(r"^\s*\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?\s*$")


def process_bank_pdf(pdf_bytes: bytes) -> list:
    """Extrae movimientos de un extracto bancario en PDF. Devuelve [] si el PDF no es un extracto.

    Desde agosto de 2025 los tickets PDF de Mercadona se maquetan con una tabla, así que
    "tener una tabla" ya no basta: antes se comprueba que no sea un ticket de supermercado y
    que el documento tenga pinta de extracto (saldo, IBAN…), y solo se aceptan filas que
    empiezan por una fecha."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        texto = "\n".join((page.extract_text() or "") for page in pdf.pages[:2]).upper()
        if any(m in texto for m in MARCAS_TICKET) or not any(m in texto for m in MARCAS_BANCO):
            return []
        transactions = []
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    if len(row) >= 3 and PATRON_FECHA_CELDA.match(row[0] or ""):
                        transactions.append({
                            'fecha_oper': row[0],
                            'concepto': row[1] if len(row) > 1 else '',
                            'fecha_valor': row[2] if len(row) > 2 else row[0],
                            'importe': row[3] if len(row) > 3 else '0',
                            'saldo': row[4] if len(row) > 4 else '0',
                        })
    return transactions


@app.on_event("startup")
async def limpiar_tickets_tomados_por_banco():
    """Los tickets PDF nuevos de Mercadona se guardaban por error como movimientos bancarios.
    Se borran esos movimientos falsos y la subida se marca como error para poder volver a subirla."""
    # La cabecera del ticket ("MERCADONA, S.A. A-46103834") acababa en la columna de fecha
    falsos = await db.bank_transactions.distinct("upload_id", {"fecha_oper": {"$regex": "A-46103834|MERCADONA, S\\.A\\.", "$options": "i"}})
    if falsos:
        borrados = await db.bank_transactions.delete_many({"upload_id": {"$in": falsos}})
        await db.uploads_history.update_many(
            {"id": {"$in": falsos}},
            {"$set": {"status": "error", "result_type": "ticket leído como extracto bancario (corregido: vuelve a subirlo)"}})
        print(f"🧹 {len(falsos)} tickets se habían guardado como movimientos bancarios: "
              f"{borrados.deleted_count} movimientos falsos eliminados. Vuelve a subir esos tickets.")

# ─── Endpoints de la API ───

# ─── Detección de tickets duplicados ───

def huella_ticket(fecha_compra, precio_total, items) -> str:
    """Huella del contenido del ticket: misma fecha, mismo total y exactamente los mismos productos
    (descripción, cantidad y precio unitario), sin importar el orden ni mayúsculas/espacios."""
    productos = sorted(
        f"{normalizar(str(i.get('descripcion', '')))}|{int(i.get('cantidad', 1) or 1)}|{float(i.get('precio_unitario', 0) or 0):.2f}"
        for i in items
    )
    base = json.dumps([fecha_a_iso(fecha_compra), f"{float(precio_total or 0):.2f}", productos], ensure_ascii=False)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


async def buscar_duplicado(huella: Optional[str] = None, archivo_hash: Optional[str] = None):
    condiciones = []
    if huella:
        condiciones.append({"huella": huella})
    if archivo_hash:
        condiciones.append({"archivo_hash": archivo_hash})
    if not condiciones:
        return None
    return await db.tickets.find_one({"$or": condiciones}, {"_id": 0, "id": 1, "fecha_compra": 1,
                                                            "precio_total": 1, "nombre_personalizado": 1, "tienda": 1})


def euros_es(valor) -> str:
    return f"{float(valor or 0):.2f}".replace(".", ",")


def respuesta_duplicado(existente: dict, upload_record: dict) -> dict:
    nombre = existente.get("nombre_personalizado") or existente.get("tienda") or "ticket"
    return {
        "id": upload_record["id"],
        "filename": upload_record["filename"],
        "status": "duplicado",
        "duplicado": True,
        "mensaje": (f"Este ticket ya está guardado ({nombre} del {existente.get('fecha_compra')}, "
                    f"{euros_es(existente.get('precio_total'))} €). Se ha omitido."),
        "ticket_existente_id": existente.get("id"),
    }


@app.on_event("startup")
async def rellenar_huellas():
    """Calcula la huella de los tickets guardados antes de que existiera la detección de duplicados."""
    async for t in db.tickets.find({"huella": {"$exists": False}}, {"_id": 0}):
        await db.tickets.update_one(
            {"id": t["id"]},
            {"$set": {"huella": huella_ticket(t.get("fecha_compra"), t.get("precio_total"), t.get("items", [])),
                      "fecha_iso": fecha_a_iso(t.get("fecha_compra"))}},
        )


@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...), ultima_modificacion: Optional[str] = Form(None)):
    """ultima_modificacion: fecha de modificación del archivo según el navegador (ms), para comprobar la fecha leída."""
    content = await file.read()
    archivo_hash = hashlib.sha256(content).hexdigest()
    file_type = detect_file_type(file.filename, file.content_type or '')

    upload_record = {
        'id': str(uuid.uuid4()),
        'filename': file.filename,
        'file_type': file_type,
        'upload_date': datetime.now(timezone.utc).isoformat(),
        'status': 'procesando',
        'result_type': '',
        'result_count': 0
    }

    try:
        if file_type == 'image' or file_type == 'pdf':
            # Diferenciamos si es PDF bancario
            transactions = []
            if file_type == 'pdf':
                transactions = process_bank_pdf(content)

            if transactions:
                # Flujo PDF Bancario (Se guarda directo, no requiere revisión humana de momento)
                docs = [{'id': str(uuid.uuid4()), 'upload_id': upload_record['id'], **t, 'created_at': datetime.now(timezone.utc).isoformat()} for t in transactions]
                await db.bank_transactions.insert_many(docs)
                upload_record['status'] = 'procesado'
                upload_record['result_type'] = 'banco'
                upload_record['result_count'] = len(transactions)
                
                await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
                return {
                    'id': upload_record['id'],
                    'filename': upload_record['filename'],
                    'file_type': upload_record['file_type'],
                    'status': upload_record['status'],
                    'result_type': upload_record['result_type'],
                    'result_count': upload_record['result_count']
                }
            else:
                # Flujo Ticket (IA Donut) - REVISIÓN HUMANA ACTIVADA
                # 1) Mismo archivo ya subido: se detecta al instante, sin ejecutar el modelo
                existente = await buscar_duplicado(archivo_hash=archivo_hash)
                if existente:
                    upload_record.update({'status': 'duplicado', 'result_type': 'ticket'})
                    await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
                    return respuesta_duplicado(existente, upload_record)

                raw_donut_data = await asyncio.to_thread(process_receipt_donut, content)
                result = parse_donut_output(raw_donut_data)

                # 2) Archivo distinto (otra foto, otro PDF) pero mismo ticket: misma fecha, total y productos
                existente = await buscar_duplicado(
                    huella=huella_ticket(result['fecha_compra'], result['precio_total'], result.get('items', [])))
                if existente:
                    upload_record.update({'status': 'duplicado', 'result_type': 'ticket'})
                    await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
                    return respuesta_duplicado(existente, upload_record)

                result['items'] = await categorizar_items(result.get('items', []))
                # 3) Comprobación de la fecha leída (en fotos Donut a veces intercambia o confunde dígitos)
                revision_fecha = revisar_fecha(result['fecha_compra'], content, ultima_modificacion)
                if revision_fecha.get('aviso'):
                    print(f"📅 {revision_fecha['aviso']} Sugerencia: {revision_fecha.get('fecha_sugerida')}")
                
                # Ya NO guardamos en db.tickets aquí. 
                # Solo registramos la subida como pendiente de revisión.
                upload_record['status'] = 'pendiente_revision'
                upload_record['result_type'] = 'ticket'
                upload_record['result_count'] = len(result.get('items', []))
                await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict

                # Devolvemos los datos de la IA crudos al Frontend para que el usuario los revise
                return {
                    'id': upload_record['id'],
                    'filename': upload_record['filename'],
                    'status': upload_record['status'],
                    'tienda': result['tienda'],
                    'fecha_compra': result['fecha_compra'],
                    'items': result['items'],
                    'precio_total': result['precio_total'],
                    'upload_id': upload_record['id'], # Pasamos el ID para enlazarlo luego
                    'archivo_hash': archivo_hash,
                    'revision_fecha': revision_fecha,
                }

        elif file_type == 'csv':
             # (Flujo CSV intacto)
            transactions = process_bank_csv(content)
            if transactions:
                docs = [{'id': str(uuid.uuid4()), 'upload_id': upload_record['id'], **t, 'created_at': datetime.now(timezone.utc).isoformat()} for t in transactions]
                await db.bank_transactions.insert_many(docs)
                upload_record['status'] = 'procesado'
                upload_record['result_type'] = 'banco'
                upload_record['result_count'] = len(transactions)
            else:
                upload_record['status'] = 'sin_datos'
                
            await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
            return upload_record
            
        else:
            upload_record['status'] = 'error'
            upload_record['result_type'] = 'formato_no_soportado'
            await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
            return upload_record

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        upload_record['status'] = 'error'
        upload_record['result_type'] = str(e)[:200]
        await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
        return upload_record

# ─── Categorización de productos (catálogo CSV + correcciones del usuario) ───

class CategoriaModel(BaseModel):
    categoria: str
    subcategoria: Optional[str] = ""

async def categorizar_descripcion(descripcion: str) -> dict:
    """
    1º) Si el usuario ya corrigió antes este mismo producto, se reutiliza su elección.
    2º) Si no, se busca en el catálogo de Mercadona (catalogo_mercadona.csv).
    Un producto puede devolver varias categorías.
    """
    clave = normalizar(descripcion)
    if clave:
        aprendida = await db.categorias_aprendidas.find_one({"clave": clave}, {"_id": 0})
        if aprendida:
            return {
                "categorias": aprendida.get("categorias", []),
                "producto_catalogo": aprendida.get("producto_catalogo"),
                "confianza": 1.0,
                "origen": "usuario",
            }
    resultado = await asyncio.to_thread(categorizador.buscar, descripcion)
    resultado["origen"] = "catalogo" if resultado["categorias"] else "sin_categoria"
    if resultado["categorias"]:
        return resultado

    # 3º) Sin coincidencia en el catálogo (erratas de impresión, abreviaturas raras): se usa el historial
    #     de compras agrupado por clustering. Si hay un producto ya comprado casi igual, se propone su nombre.
    clusters = await clusters_actualizados()
    vecino = clusters.categorias_por_vecinos(descripcion, umbral=0.5)
    if vecino:
        cats, nombre_parecido, similitud = vecino
        resultado.update({"categorias": cats, "origen": "historial", "confianza": round(similitud, 2)})
        if normalizar(nombre_parecido) != clave:
            resultado["sugerencia_nombre"] = nombre_parecido
        return resultado
    por_grupo = clusters.categoria_por_grupo(descripcion)
    if por_grupo:
        cats, nombre_grupo, similitud = por_grupo
        resultado.update({"categorias": cats, "origen": "clustering", "confianza": round(similitud, 2), "grupo": nombre_grupo})
    return resultado

async def categorizar_items(items: list) -> list:
    for item in items:
        info = await categorizar_descripcion(item.get("descripcion", ""))
        item["categorias"] = info["categorias"]
        item["producto_catalogo"] = info["producto_catalogo"]
        item["confianza"] = info["confianza"]
        item["origen_categoria"] = info["origen"]
        if info.get("sugerencia_nombre"):
            item["sugerencia_nombre"] = info["sugerencia_nombre"]
    return items


# ─── Asistente de gastos (chat) y clustering de productos ───

# El LLM local (Ollama / LM Studio) solo traduce la pregunta a una consulta estructurada;
# si no está disponible, el asistente usa el motor de reglas.
traductor = TraductorLLM()
asistente = AsistenteGastos(categorizador.listar_categorias(), traductor=traductor)


async def clusters_actualizados():
    """Reentrena el clustering solo si los tickets guardados han cambiado desde la última vez."""
    tickets = await db.tickets.find({}, {"_id": 0}).to_list(20000)
    await asyncio.to_thread(asistente.preparar, tickets)
    return asistente.clusters


class ChatRequest(BaseModel):
    mensaje: str
    historial: List[dict] = []   # últimas preguntas y sus consultas, para entender "¿y el mes pasado?"
    motor: str = "llm"           # "llm" (IA local, con reglas de respaldo) o "reglas" (solo motor propio)


@api_router.post("/chat")
async def chat(req: ChatRequest):
    """Responde preguntas en lenguaje natural sobre los gastos guardados."""
    mensaje = (req.mensaje or "").strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje está vacío")
    tickets = await db.tickets.find({}, {"_id": 0}).to_list(20000)
    return a_json(await asyncio.to_thread(asistente.responder, mensaje[:500], tickets, None, req.historial[-3:],
                                         req.motor != "reglas"))


def a_json(obj):
    """Convierte tipos de numpy/pandas a tipos nativos de Python para poder enviarlos como JSON."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@app.on_event("startup")
async def calentar_llm():
    # En segundo plano: el servidor arranca sin esperar a que el modelo se cargue en la GPU
    asyncio.get_running_loop().run_in_executor(None, traductor.calentar)


@api_router.get("/chat/estado")
async def estado_chat():
    """Indica si el LLM local está conectado y qué modelo usa."""
    return await asyncio.to_thread(traductor.estado)


@api_router.get("/chat/clusters")
async def ver_clusters():
    """Grupos de productos encontrados por el clustering (para inspección y para la memoria)."""
    clusters = await clusters_actualizados()
    return a_json(clusters.resumen())

class CategorizarRequest(BaseModel):
    descripcion: str

@api_router.get("/categorias")
async def get_categorias():
    """Árbol de categorías/subcategorías del catálogo, para los desplegables del frontend."""
    return categorizador.listar_categorias()

@api_router.post("/categorizar")
async def categorizar(req: CategorizarRequest):
    """Categoriza una descripción suelta (producto añadido o editado a mano en la revisión)."""
    return await categorizar_descripcion(req.descripcion)

# ─── NUEVO ENDPOINT: Guardado definitivo tras Revisión Humana ───

# Definimos el esquema de Pydantic para validar los datos que llegarán del Frontend
class ItemModel(BaseModel):
    descripcion: str
    cantidad: int
    precio_unitario: float
    categorias: List[CategoriaModel] = []
    producto_catalogo: Optional[str] = None
    categoria_manual: bool = False   # True si el usuario cambió las categorías a mano
    añadido_manual: bool = False     # True si el producto no lo leyó la IA y lo añadió el usuario

class DraftTicketModel(BaseModel):
    nombre_personalizado: Optional[str] = ""
    tienda: str
    fecha_compra: str
    items: List[ItemModel]
    precio_total: float
    upload_id: Optional[str] = None # Para enlazar con el uploads_history
    archivo_hash: Optional[str] = None

@api_router.post("/save-ticket")
async def save_ticket(ticket_data: DraftTicketModel):
    """Guarda el ticket definitivo en MongoDB tras la revisión del usuario."""
    
    # 0. Última comprobación de duplicados, ya con los datos revisados por el usuario
    items_dict = [item.model_dump() for item in ticket_data.items]
    huella = huella_ticket(ticket_data.fecha_compra, ticket_data.precio_total, items_dict)
    existente = await buscar_duplicado(huella=huella, archivo_hash=ticket_data.archivo_hash)
    if existente:
        if ticket_data.upload_id:
            await db.uploads_history.update_one({"id": ticket_data.upload_id}, {"$set": {"status": "duplicado"}})
        nombre = existente.get("nombre_personalizado") or existente.get("tienda") or "ticket"
        raise HTTPException(status_code=409, detail=(
            f"Este ticket ya está guardado ({nombre} del {existente.get('fecha_compra')}, "
            f"{euros_es(existente.get('precio_total'))} €). No se ha vuelto a guardar."))

    # 1. Empaquetamos los datos verificados
    ticket_final = {
        'id': str(uuid.uuid4()),
        'upload_id': ticket_data.upload_id or str(uuid.uuid4()),
        'nombre_personalizado': ticket_data.nombre_personalizado,
        'tienda': ticket_data.tienda,
        'fecha_compra': ticket_data.fecha_compra,
        'fecha_iso': fecha_a_iso(ticket_data.fecha_compra),
        'items': items_dict,
        'precio_total': ticket_data.precio_total,
        'huella': huella,
        'archivo_hash': ticket_data.archivo_hash,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    # 2. Inserción oficial en la Base de Datos
    await db.tickets.insert_one(ticket_final)

    # 2.b Aprendizaje: si el usuario corrigió categorías, se recuerdan para próximos tickets
    for item in ticket_data.items:
        clave = normalizar(item.descripcion)
        if item.categoria_manual and clave and item.categorias:
            await db.categorias_aprendidas.update_one(
                {"clave": clave},
                {"$set": {
                    "clave": clave,
                    "descripcion": item.descripcion,
                    "categorias": [c.model_dump() for c in item.categorias],
                    "producto_catalogo": item.producto_catalogo,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }},
                upsert=True,
            )
    
    # 3. Actualizamos el historial para marcarlo como procesado definitivamente
    if ticket_data.upload_id:
        await db.uploads_history.update_one(
            {"id": ticket_data.upload_id},
            {"$set": {"status": "procesado"}}
        )
        
    return {"message": "Ticket verificado y guardado con éxito", "id": ticket_final["id"]}

@api_router.get("/uploads-history")
async def get_uploads_history():
    return await db.uploads_history.find({}, {"_id": 0}).sort("upload_date", -1).to_list(500)

@api_router.get("/tickets")
async def get_tickets(tienda: Optional[str] = None, categoria: Optional[str] = None):
    query = {}
    if tienda: query['tienda'] = tienda
    if categoria: query['items.categorias.categoria'] = categoria
    tickets = await db.tickets.find(query, {"_id": 0}).to_list(5000)
    # Orden por la fecha del ticket (dd/mm/yyyy), no por la de subida: más reciente primero
    tickets.sort(key=lambda t: fecha_a_iso(t.get("fecha_compra")), reverse=True)
    return tickets


def fecha_a_iso(fecha) -> str:
    """'dd/mm/yyyy[ HH:MM]' -> 'yyyy-mm-dd' (cadena vacía si no se reconoce)."""
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(fecha or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""

@api_router.get("/tickets/analytics")
async def get_tickets_analytics():
    tickets = await db.tickets.find({}, {"_id": 0}).to_list(5000)

    # Gasto por categoría. Como un producto puede estar en varias categorías,
    # su importe cuenta en cada una de ellas: la suma de categorías puede superar
    # el gasto total, que se calcula siempre a partir de los tickets.
    por_categoria = {}
    for t in tickets:
        for item in t.get('items', []):
            importe = item.get('precio_unitario', 0) * item.get('cantidad', 1)
            cats = item.get('categorias') or [{"categoria": "Sin categoría", "subcategoria": ""}]
            for c in {cat.get('categoria', 'Sin categoría') for cat in cats}:
                por_categoria[c] = por_categoria.get(c, 0) + importe

    return {
        'total_gasto': round(sum(t.get('precio_total', 0) for t in tickets), 2),
        'num_tickets': len(tickets),
        'gasto_por_categoria': sorted(
            [{'categoria': k, 'gasto': round(v, 2)} for k, v in por_categoria.items()],
            key=lambda x: -x['gasto']
        ),
    }

@api_router.delete("/tickets/{ticket_id}")
async def delete_ticket(ticket_id: str):
    """Elimina un ticket guardado (y marca su subida como eliminada en el historial)."""
    ticket = await db.tickets.find_one({"id": ticket_id}, {"_id": 0, "upload_id": 1})
    if not ticket:
        raise HTTPException(status_code=404, detail="El ticket no existe")
    await db.tickets.delete_one({"id": ticket_id})
    if ticket.get("upload_id"):
        await db.uploads_history.update_one({"id": ticket["upload_id"]}, {"$set": {"status": "eliminado"}})
    return {"status": "eliminado", "id": ticket_id}

@api_router.delete("/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    await db.tickets.delete_many({"upload_id": upload_id})
    await db.bank_transactions.delete_many({"upload_id": upload_id})
    await db.uploads_history.delete_one({"id": upload_id})
    return {"status": "eliminado"}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()