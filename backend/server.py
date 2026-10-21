import os
import io
import re
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
from corrector_nombres import CorrectorNombres, distancia, normalizar_nombre
from clasificador_productos import ClasificadorProductos, UMBRAL_BUSCADOR
from embeddings import ClienteEmbeddings
from rag_chat import BaseEjemplos, MemoriaChat
from lectura_donut import normalizar_prediccion
from deteccion_documentos import es_extracto_pdf, fecha_a_iso, huella_ticket, numero_factura

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
    donut_data = normalizar_prediccion(donut_data)   # estructura rota (ticket muy largo, foto difícil) -> dict limpio
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

    # Corrección de seguridad final (si no se leyó el total, la lectura queda marcada como incompleta)
    sin_total = precio_total == 0.0 and bool(items)
    if sin_total:
        precio_total = sum(i['precio_unitario'] * i['cantidad'] for i in items)

    return {
        'tienda': tienda.title(),
        'fecha_compra': fecha_compra,
        'items': items,
        'precio_total': round(precio_total, 2),
        'texto_ocr': texto_crudo,
        # Ticket demasiado largo para Donut (se corta antes del TOTAL) o campos ilegibles: siempre a revisión
        'lectura_incompleta': bool(donut_data.get("truncada") or donut_data.get("campos_rotos") or sin_total),
    }


# ─── Endpoints de la API ───

# ─── Detección de tickets duplicados ───

async def buscar_duplicado(huella: Optional[str] = None, archivo_hash: Optional[str] = None,
                           factura: Optional[str] = None):
    condiciones = []
    if factura:
        condiciones.append({"factura": factura})
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
            # La aplicación solo trabaja con tickets: un extracto bancario en PDF se rechaza sin pasar por Donut
            if file_type == 'pdf' and es_extracto_pdf(content):
                upload_record['status'] = 'error'
                upload_record['result_type'] = 'no_es_ticket'
                await db.uploads_history.insert_one(dict(upload_record))  # copia: Mongo añade _id al dict
                return {k: upload_record[k] for k in ('id', 'filename', 'file_type', 'status', 'result_type')}
            # Flujo Ticket (IA Donut) - REVISIÓN HUMANA ACTIVADA
            # 1) Mismo archivo ya subido: se detecta al instante, sin ejecutar el modelo
            # 1.b) Otro PDF del mismo ticket: mismo nº de factura simplificada (texto del PDF)
            factura = numero_factura(content)
            existente = await buscar_duplicado(archivo_hash=archivo_hash, factura=factura)
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

            # 2.b) Misma fecha y total que un ticket guardado pero productos leídos distintos: puede ser el
            #      mismo ticket mal leído (p. ej. una foto). No se omite: se avisa y decide el usuario.
            posible = await db.tickets.find_one(
                {"fecha_iso": fecha_a_iso(result['fecha_compra']),
                 "precio_total": {"$gte": result['precio_total'] - 0.005, "$lte": result['precio_total'] + 0.005}},
                {"_id": 0, "id": 1, "fecha_compra": 1, "precio_total": 1, "nombre_personalizado": 1, "tienda": 1}
            ) if fecha_a_iso(result['fecha_compra']) and result['precio_total'] else None

            result['items'] = corregir_nombres(result.get('items', []), nombres_del_pdf(content))
            result['items'] = await categorizar_items(result['items'])
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
                'factura': factura,
                'revision_fecha': revision_fecha,
                'lectura_incompleta': result.get('lectura_incompleta', False),
                'posible_duplicado': ({'id': posible.get('id'), 'mensaje': (
                    f"Ya hay un ticket guardado del {posible.get('fecha_compra')} con el mismo total "
                    f"({euros_es(posible.get('precio_total'))} €). Comprueba que no sea el mismo.")} if posible else None),
            }

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

# ─── Corrección automática de nombres mal leídos ───

corrector = CorrectorNombres(categorizador)


def nombres_dataset() -> list:
    """Nombres de producto del dataset etiquetado a mano (dataset-mercadona/metadata.jsonl), si existe."""
    ruta = ROOT_DIR / "dataset-mercadona" / "metadata.jsonl"
    if not ruta.exists():
        return []
    texto, dec, i, nombres = ruta.read_text(encoding="utf-8"), json.JSONDecoder(), 0, []
    while i < len(texto):
        while i < len(texto) and texto[i].isspace():
            i += 1
        if i >= len(texto):
            break
        try:
            obj, i = dec.raw_decode(texto, i)
            gt = json.loads(obj["ground_truth"])["gt_parse"]
            menu = gt.get("menu", [])
            nombres += [m.get("nm", "") for m in ([menu] if isinstance(menu, dict) else menu)]
        except Exception:
            break
    return nombres


@app.on_event("startup")
async def cargar_vocabulario_corrector():
    """Vocabulario del corrector: nombres del dataset, de los tickets guardados y correcciones del usuario."""
    corrector.añadir_nombres(nombres_dataset())
    async for t in db.tickets.find({}, {"_id": 0, "items.descripcion": 1}):
        corrector.añadir_nombres(i.get("descripcion", "") for i in t.get("items", []))
    pares = [(c["leido"], c["correcto"]) async for c in db.correcciones_nombres.find({}, {"_id": 0})]
    corrector.añadir_aprendidas(pares)
    print(f"🔤 Corrector de nombres listo: {len(corrector.nombres)} nombres conocidos, "
          f"{len(corrector.aprendidas)} correcciones aprendidas.")


# ─── Clasificación por similitud (respaldo del buscador; ver clasificador_productos.py) ───

# Un único cliente de embeddings (bge-m3 en Ollama) para el clasificador y el RAG del chat
cliente_embeddings = ClienteEmbeddings() if os.environ.get("EMB_ACTIVADO", "true").lower() in ("1", "true", "si", "sí", "yes") else None
clasificador = ClasificadorProductos(categorizador, cliente_embeddings)


async def productos_historial() -> list:
    """(nombre, categorías) de los productos de los tickets guardados (categorías revisadas por el usuario)."""
    from collections import Counter, defaultdict
    vistos = defaultdict(Counter)
    async for t in db.tickets.find({}, {"_id": 0, "items.descripcion": 1, "items.categorias": 1}):
        for it in t.get("items", []):
            cats = it.get("categorias") or []
            if it.get("descripcion") and cats:
                vistos[it["descripcion"]][json.dumps(cats, sort_keys=True, ensure_ascii=False)] += 1
    return [(n, json.loads(c.most_common(1)[0][0])) for n, c in vistos.items()]


@app.on_event("startup")
async def preparar_clasificador():
    # En segundo plano: calcular los embeddings del catálogo puede tardar la primera vez (luego hay caché)
    historial = await productos_historial()
    asyncio.get_running_loop().run_in_executor(None, clasificador.construir, historial)


PATRON_LINEA_PDF = re.compile(r"^-?\d+\s+(.+?)(?:\s+-?\d+,\d{2}){0,2}\s*$")


def nombres_del_pdf(contenido: bytes) -> list:
    """Nombres de producto exactos del texto del PDF (el PDF de Mercadona lleva el texto real)."""
    if not contenido.startswith(b"%PDF"):
        return []
    try:
        with pdfplumber.open(io.BytesIO(contenido)) as pdf:
            texto = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return []
    nombres, dentro = [], False
    for linea in texto.splitlines():
        if "Descripción" in linea and "Importe" in linea:   # "Descripción ..." o, desde 2026, "Cnt. Descripción ..."
            dentro = True
            continue
        if linea.startswith("TOTAL"):
            break
        m = PATRON_LINEA_PDF.match(linea.strip()) if dentro else None
        if m and re.search(r"[A-ZÑ]{2}", m.group(1)):
            nombres.append(m.group(1).strip())
    return nombres


def corregir_nombres(items: list, nombres_extra: list = ()) -> list:
    """Corrige los nombres mal leídos. Si la corrección es segura se aplica directamente
    (guardando el nombre leído para poder deshacerla); si no, queda como sugerencia."""
    for item in items:
        leido = item.get("descripcion", "")
        item["descripcion_leida"] = leido
        r = corrector.corregir(leido, nombres_extra)
        if not r:
            continue
        if r["automatica"]:
            item["descripcion"] = r["nombre"]
            item["correccion_auto"] = {"original": leido, "motivo": r["motivo"]}
            print(f"✏️  {leido} -> {r['nombre']} ({r['motivo']})")
        else:
            item["sugerencia_nombre"] = r["nombre"]
    return items


async def categorizar_descripcion(descripcion: str) -> dict:
    """
    1º) Si el usuario ya corrigió antes este mismo producto, se reutiliza su elección.
    2º) Buscador por palabras en el catálogo de Mercadona: si responde con confianza >= 0,75, se usa.
    3º) Si no responde o duda: categoría del producto conocido más parecido (catálogo + historial) en el
        espacio embeddings bge-m3 + caracteres (95,7 % frente a 92,1 % del buscador solo, ver validación).
    4º) Sin clasificador disponible: vecinos y grupos del clustering TF-IDF del historial.
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
    if resultado["categorias"] and resultado["confianza"] >= UMBRAL_BUSCADOR:
        return resultado

    vecino = await asyncio.to_thread(clasificador.vecino, descripcion) if clasificador.listo else None
    if vecino:
        resultado.update({"categorias": vecino["categorias"], "confianza": round(vecino["similitud"], 2),
                          "origen": "similitud_" + vecino["origen"]})
        if vecino["origen"] == "catalogo":
            resultado["producto_catalogo"] = vecino["nombre"]
        else:
            a, b = normalizar_nombre(descripcion), normalizar_nombre(vecino["nombre"])
            if normalizar(vecino["nombre"]) != clave and distancia(a, b, 3) <= max(1.0, 0.15 * len(a)):
                resultado["sugerencia_nombre"] = vecino["nombre"]
        return resultado
    if resultado["categorias"]:
        return resultado

    # 3º) Sin coincidencia en el catálogo (erratas de impresión, abreviaturas raras): se usa el historial
    #     de compras agrupado por clustering. Si hay un producto ya comprado casi igual, se propone su nombre.
    clusters = await clusters_actualizados()
    vecino = clusters.categorias_por_vecinos(descripcion, umbral=0.5)
    if vecino:
        cats, nombre_parecido, similitud = vecino
        resultado.update({"categorias": cats, "origen": "historial", "confianza": round(similitud, 2)})
        a, b = normalizar_nombre(descripcion), normalizar_nombre(nombre_parecido)
        if normalizar(nombre_parecido) != clave and distancia(a, b, 3) <= max(1.0, 0.15 * len(a)):
            resultado["sugerencia_nombre"] = nombre_parecido
        return resultado
    por_grupo = clusters.categoria_por_grupo(descripcion)
    if por_grupo:
        cats, nombre_grupo, similitud = por_grupo
        resultado.update({"categorias": cats, "origen": "clustering", "confianza": round(similitud, 2), "grupo": nombre_grupo})
    return resultado

async def categorizar_items(items: list) -> list:
    for item in items:
        if item.get("sugerencia_nombre"):      # ya hay una sugerencia del corrector: no se sustituye
            info = await categorizar_descripcion(item.get("descripcion", ""))
            info.pop("sugerencia_nombre", None)
        else:
            info = None
        info = info or await categorizar_descripcion(item.get("descripcion", ""))
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

# RAG: base de ejemplos validados + historial persistente del chat (ver rag_chat.py)
base_ejemplos = BaseEjemplos(cliente_embeddings)
memoria_chat = MemoriaChat(db, base_ejemplos)
asistente.rag = base_ejemplos


@app.on_event("startup")
async def preparar_rag():
    try:
        await memoria_chat.iniciar()
    except Exception as e:
        print(f"⚠️ No se pudo preparar el RAG del chat: {e}")


async def clusters_actualizados():
    """Reentrena el clustering solo si los tickets guardados han cambiado desde la última vez."""
    tickets = await db.tickets.find({}, {"_id": 0}).to_list(20000)
    await asyncio.to_thread(asistente.preparar, tickets)
    return asistente.clusters


class ChatRequest(BaseModel):
    mensaje: str
    historial: List[dict] = []   # últimas preguntas y sus consultas, para entender "¿y el mes pasado?"
    motor: str = "llm"           # "llm" (IA local, con reglas de respaldo) o "reglas" (solo motor propio)
    conversacion: str = "general"  # identificador de la conversación (historial persistente)


class ValoracionRequest(BaseModel):
    id: str
    util: bool


@api_router.post("/chat")
async def chat(req: ChatRequest):
    """Responde preguntas en lenguaje natural sobre los gastos guardados."""
    mensaje = (req.mensaje or "").strip()
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje está vacío")
    tickets = await db.tickets.find({}, {"_id": 0}).to_list(20000)
    # Historial para las continuaciones: el que manda la web o, si no (p. ej. tras recargar), el guardado
    historial = req.historial[-3:] or await memoria_chat.ultimas_consultas(req.conversacion)
    resultado = a_json(await asyncio.to_thread(asistente.responder, mensaje[:500], tickets, None, historial,
                                               req.motor != "reglas"))
    resultado["id"] = await memoria_chat.guardar(req.conversacion, mensaje[:500], resultado)
    return resultado


@api_router.get("/chat/historial")
async def historial_chat(conversacion: str = "general", limite: int = 50):
    """Mensajes guardados de una conversación (para recuperarla al recargar la página)."""
    return a_json(await memoria_chat.historial(conversacion, min(limite, 200)))


@api_router.post("/chat/valoracion")
async def valorar_respuesta(req: ValoracionRequest):
    """👍 / 👎 de una respuesta. Las útiles se añaden a la base de ejemplos del RAG."""
    if not await memoria_chat.valorar(req.id, req.util):
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    return {"ok": True, "ejemplos_rag": len(base_ejemplos.preguntas)}


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
    descripcion_leida: Optional[str] = None   # nombre tal y como lo leyó Donut (antes de corregirlo)
    correccion_auto: Optional[dict] = None    # {"original", "motivo"} si se corrigió automáticamente

class DraftTicketModel(BaseModel):
    nombre_personalizado: Optional[str] = ""
    tienda: str
    fecha_compra: str
    items: List[ItemModel]
    precio_total: float
    upload_id: Optional[str] = None # Para enlazar con el uploads_history
    archivo_hash: Optional[str] = None
    factura: Optional[str] = None     # nº de factura simplificada (solo tickets PDF)

@api_router.post("/save-ticket")
async def save_ticket(ticket_data: DraftTicketModel):
    """Guarda el ticket definitivo en MongoDB tras la revisión del usuario."""
    
    # 0. Última comprobación de duplicados, ya con los datos revisados por el usuario
    items_dict = [item.model_dump() for item in ticket_data.items]
    huella = huella_ticket(ticket_data.fecha_compra, ticket_data.precio_total, items_dict)
    existente = await buscar_duplicado(huella=huella, archivo_hash=ticket_data.archivo_hash, factura=ticket_data.factura)
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
        'factura': ticket_data.factura,
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
    
    # 2.c Aprendizaje de nombres: lo guardado es correcto (revisado por el usuario) y, si cambió
    #     el nombre leído por Donut, se recuerda la corrección para aplicarla sola la próxima vez
    pares = []
    for item in ticket_data.items:
        leido = item.descripcion_leida
        if leido and not item.añadido_manual and normalizar_nombre(leido) != normalizar_nombre(item.descripcion):
            pares.append((leido, item.descripcion))
            await db.correcciones_nombres.update_one(
                {"leido": normalizar_nombre(leido)},
                {"$set": {"leido": normalizar_nombre(leido), "correcto": item.descripcion,
                          "updated_at": datetime.now(timezone.utc).isoformat()}},
                upsert=True)
        elif leido and normalizar_nombre(leido) in corrector.aprendidas and \
                normalizar_nombre(leido) == normalizar_nombre(item.descripcion):
            # El usuario deshizo una corrección aprendida: se olvida
            corrector.aprendidas.pop(normalizar_nombre(leido), None)
            await db.correcciones_nombres.delete_one({"leido": normalizar_nombre(leido)})
    corrector.añadir_nombres(i.descripcion for i in ticket_data.items)
    asyncio.get_running_loop().run_in_executor(
        None, clasificador.añadir, [(i.descripcion, [c.model_dump() for c in i.categorias]) for i in ticket_data.items])
    corrector.añadir_aprendidas(pares)

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