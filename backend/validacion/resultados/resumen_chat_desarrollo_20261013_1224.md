# Evaluación del asistente (banco de desarrollo) — 13/10/2026 12:24

Banco: 116 preguntas · datos de prueba con hoy = 20/04/2026

## Acierto por categoría (respuesta equivalente a la esperada)

| Categoría | n | llm (qwen2.5:7b) | llm+rag (qwen2.5:7b) |
|---|---|---|---|
| basica | 20 | 100.0 % | 100.0 % |
| coloquial | 20 | 90.0 % | 80.0 % |
| combinada | 20 | 100.0 % | 95.0 % |
| filtros | 20 | 100.0 % | 100.0 % |
| periodos | 15 | 100.0 % | 100.0 % |
| continuacion | 10 | 100.0 % | 100.0 % |
| fuera_de_ambito | 7 | 100.0 % | 100.0 % |
| limites | 4 | 100.0 % | 100.0 % |
| **Total (sin «limites»)** | 112 | **98.2 %** | **95.5 %** |

## Coincidencia exacta de la consulta y latencia

| Motor | Coincidencia exacta | Latencia media | Mediana | p95 | Respaldo a reglas |
|---|---|---|---|---|---|
| llm (qwen2.5:7b) | 91.1 % | 3.423 s | 3.333 s | 3.732 s | 0 |
| llm+rag (qwen2.5:7b) | 90.2 % | 5.606 s | 5.427 s | 7.742 s | 0 |

## Fallos — llm (qwen2.5:7b) (2)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 22 | coloquial | ¿Qué es lo que más pillo en el Mercadona? | `{"operacion": "ranking", "agrupar_por": "categoria"}` |
| 37 | coloquial | Hazme un resumen de cuánto gasto cada mes | `{"operacion": "ranking", "agrupar_por": "mes"}` |

## Fallos — llm+rag (qwen2.5:7b) (5)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 21 | coloquial | ¿Cuánta pasta me he dejado en el súper este año? | `{"filtros": {"concepto": "pasta", "periodo": "¿Cuánta pasta me he dejado en el súper este año?"}}` |
| 22 | coloquial | ¿Qué es lo que más pillo en el Mercadona? | `{"operacion": "ranking", "agrupar_por": "producto"}` |
| 34 | coloquial | Oye, ¿en limpieza cuánto llevo? | `{"operacion": "ranking", "agrupar_por": "categoria", "filtros": {"concepto": "limpieza"}}` |
| 37 | coloquial | Hazme un resumen de cuánto gasto cada mes | `{"operacion": "media", "por": "mes"}` |
| 57 | combinada | ¿Qué semana de 2025 hice más compras? | `{"medida": "compras", "operacion": "ultimo", "agrupar_por": "semana", "filtros": {"periodo": "¿Qué semana de 2025 hice más compras?"}}` |
