# FinTrack Pro — Analizador de tickets con IA

Aplicación web de código abierto para digitalizar los tickets de compra de Mercadona y llevar el control de los gastos personales. Es el Trabajo de Fin de Grado en Ingeniería de Computadores (ETSISI, UPM).

## Cómo funciona

1. El usuario sube un ticket (PDF o foto). Si es un archivo, un ticket o un nº de factura ya guardado, se omite.
2. **Donut** (*Document Understanding Transformer*), ajustado a los tickets de Mercadona, lee la fecha, los productos, las cantidades, los precios y el total.
3. Se limpia la lectura, se corrigen nombres mal leídos, se revisa la fecha y se avisa si el ticket puede estar repetido.
4. Cada producto se clasifica en categorías: buscador sobre el catálogo de Mercadona + vecino más parecido con *embeddings* (bge-m3) cuando el buscador duda. El usuario revisa y guarda.
5. Un **asistente** responde preguntas sobre los gastos: un LLM local (Qwen2.5 7B en Ollama) o un motor de reglas traduce la pregunta a una consulta y un motor determinista la calcula con los tickets reales. Incluye historial y RAG.

## Mapa del proyecto

| Carpeta | Qué hay |
|---|---|
| `backend/` | Servidor FastAPI (`server.py`) y módulos de la aplicación (ver tabla de abajo) |
| `backend/mi_modelo_mercadona_v3/` | Modelo Donut que usa la aplicación (no está en git) |
| `backend/catalogo_mercadona.csv` | Catálogo de Mercadona: 3.931 productos con categoría y subcategoría |
| `backend/dataset-mercadona/` | PDF originales de los 91 tickets del dataset y `metadata.jsonl` (nombres de productos que usa el corrector) |
| `backend/herramientas/` | Scripts sueltos: descargar el catálogo, aplicar etiquetas revisadas, probar el asistente |
| `backend/validacion/` | Todas las pruebas de la memoria, sus datos y sus resultados (ver `validacion/LEEME.md`) |
| `backend/cache_embeddings/` | Caché de vectores de bge-m3 (se regenera sola) |
| `frontend/` | Interfaz web en React + Tailwind (`src/App.js`, `src/components/ChatAsistente.js`) |
| `entrenamiento/` | Cuadernos de Google Colab (entrenamiento v3 y comparación v2/v3) y `datos_donut_v3/` (PDF, `preparar_dataset.py`, `generar_sinteticos.py`) |
| `Memoria/` | Memoria en LaTeX (la versión buena está en Overleaf); `borrador_v1/` = borradores de Claude; `plantilla_tutor/`; `reuniones_tutor/` |
| `archivo/` | Lo antiguo que ya no se usa (modelos v1 y v2, dataset v1, resultados obsoletos). No está en git |

### Módulos del backend

| Fichero | Función |
|---|---|
| `server.py` | API: subida de tickets, guardado, historial, chat |
| `lectura_donut.py` | Limpia la salida de Donut (lecturas cortadas o con campos rotos) |
| `corrector_nombres.py` | Corrige nombres de producto mal leídos |
| `fechas_ticket.py` | Revisa la fecha leída con la fecha de la foto o del archivo |
| `deteccion_documentos.py` | Huella de tickets repetidos, nº de factura y rechazo de PDF que no son tickets |
| `categorizador.py` | Buscador de productos en el catálogo |
| `clasificador_productos.py` | Vecino más parecido (bge-m3 + TF-IDF) cuando el buscador duda |
| `clustering_productos.py` | Agrupa los productos comprados en familias (K-Means) |
| `embeddings.py` | Cliente de *embeddings* de Ollama con caché |
| `asistente_gastos.py` | Asistente: reglas, periodos, conceptos y orquestación |
| `traductor_llm.py` | Instrucciones y llamada al LLM local |
| `ajustes_consulta.py` | Contrasta la consulta del LLM con la pregunta |
| `motor_consultas.py` | Ejecuta la consulta sobre los tickets (pandas) |
| `rag_chat.py` | Historial del chat en MongoDB y recuperación de ejemplos (RAG) |

## Puesta en marcha

**Backend** (Windows)

```bash
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8001
```

Hace falta MongoDB en local, Ollama con `qwen2.5:7b` y `bge-m3` (`ollama pull qwen2.5:7b`, `ollama pull bge-m3`) y un fichero `backend/.env` con `MONGO_URL`, `DB_NAME` y, opcionalmente, `LLM_URL`, `LLM_MODELO` y `EMB_MODELO`. El modelo entrenado se descomprime en `backend/mi_modelo_mercadona_v3/`.

**Frontend**

```bash
cd frontend
yarn install
yarn start
```

Necesita un fichero `frontend/.env` con `REACT_APP_BACKEND_URL=http://localhost:8001`.

## Privacidad

Los tickets, sus PDF y los resultados de validación en bruto contienen compras reales del autor, por lo que **no se publican** (están en `.gitignore`): `backend/dataset-mercadona/`, `entrenamiento/datos_donut_v3/pdfs/`, `backend/validacion/tickets_nuevos/`, `backend/validacion/fotos/`, `backend/validacion/tickets_prueba.json` y los `.csv`/`.json`/`.jsonl` de `backend/validacion/resultados/`. Sí se publican los resúmenes de cada validación (`resultados/resumen_*.md`).

Para reproducir las validaciones o reentrenar el modelo hay que usar tickets propios (PDF de la app de Mercadona) colocados en esas carpetas.
