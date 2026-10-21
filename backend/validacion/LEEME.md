# backend/validacion — pruebas de la memoria

Todos los programas se ejecutan desde `backend/` con `python -m validacion.<programa>`. Los resultados (CSV y resúmenes en Markdown) se guardan en `resultados/`.

## Programas

| Programa | Qué prueba | Capítulo de la memoria |
|---|---|---|
| `evaluar_donut.py` | Lectura de tickets en PDF y fotos (por campo, por línea, detección) | Lectura de tickets |
| `generar_fotos_sinteticas.py` | Crea fotos simuladas (leve/media/fuerte) a partir de los PDF | Lectura de tickets |
| `evaluar_corrector.py` | Corrector de nombres (errores reales y simulados) | Corrector y fechas |
| `evaluar_fechas.py` | Revisión de fechas (fechas reales y errores simulados) | Corrector y fechas |
| `evaluar_documentos.py` | Tickets repetidos y rechazo de PDF que no son tickets | Repetidos |
| `generar_extractos_sinteticos.py` | Crea extractos bancarios simulados (para comprobar que se rechazan) | Repetidos |
| `exportar_etiquetas.py` | Exporta los productos de los tickets para etiquetarlos a mano | Categorización |
| `evaluar_clustering.py` | Representaciones, clustering y vecinos más cercanos | Categorización |
| `evaluar_hibrido.py` | Buscador + vecino más cercano o clustering según el umbral | Categorización |
| `generar_banco.py` | Crea los bancos de preguntas del chat | Asistente |
| `evaluar_chat.py` | Asistente con reglas, LLM y RAG (`--banco desarrollo|prueba|rag`) | Asistente |
| `generar_semillas_rag.py` | Semillas y paráfrasis del experimento de adaptación del RAG | Asistente |
| `generar_figuras.py` | Gráficas de la memoria (PDF) en `Memoria/figuras` | Todos |

## Datos

| Fichero o carpeta | Qué es |
|---|---|
| `tickets_prueba.json` | Copia fija de los 91 tickets usada por las pruebas del chat (hoy = 20/04/2026) |
| `banco_preguntas.json` / `banco_prueba.json` | Bancos de preguntas de desarrollo (116) y de prueba (40) |
| `semillas_rag.json`, `texto_para_otra_ia.txt`, `parafrasis_rag.txt`, `banco_rag.json` | Experimento de adaptación del RAG |
| `etiquetas_revisadas.csv`, `categorias_validas.csv` | Categorías correctas de los 280 productos de los tickets |
| `split_entrenamiento.json` | Qué tickets fueron de entrenamiento (77) y de validación (14) en Donut |
| `tickets_nuevos/` | 17 PDF de compras posteriores al entrenamiento (medida limpia de Donut) |
| `fotos_sinteticas/` | Fotos simuladas generadas (se pueden borrar y regenerar) |
| `fotos/` | Para fotos reales de tickets (mismo nombre que su PDF) |
| `extractos_sinteticos/` | Extractos simulados generados (se pueden borrar y regenerar) |
| `resultados/` | Salidas de todas las pruebas; `donut_lecturas.jsonl` es la caché de lecturas de Donut |

## Resultados que cita la memoria

| Prueba | Fichero |
|---|---|
| Chat antes de los ajustes | `resumen_chat_20261008_1506.md` |
| Chat final (desarrollo / prueba, con y sin RAG) | `resumen_chat_desarrollo_20261013_1224.md`, `resumen_chat_prueba_20261013_1156.md` |
| Experimento de adaptación del RAG | `resumen_chat_rag_20261013_1300.md` |
| Clustering y categorización | `resumen_clustering_20261012_1431.md`, `resumen_hibrido.md` |
| Lectura de tickets | `resumen_donut_20261013_1532.md` |
| Corrector y fechas | `resumen_corrector_20261013_1554.md`, `resumen_fechas_20261013_1554.md` |
| Repetidos | `resumen_documentos_20261013_1612.md` |
