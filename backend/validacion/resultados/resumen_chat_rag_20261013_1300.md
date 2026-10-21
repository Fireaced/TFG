# Evaluación del asistente (banco de rag) — 13/10/2026 13:00

Banco: 37 preguntas · datos de prueba con hoy = 20/04/2026

## Acierto por categoría (respuesta equivalente a la esperada)

| Categoría | n | llm (qwen2.5:7b) | llm+rag (qwen2.5:7b) | llm+rag+semillas (qwen2.5:7b) |
|---|---|---|---|---|
| parafrasis | 37 | 78.4 % | 86.5 % | 86.5 % |
| **Total (sin «limites»)** | 37 | **78.4 %** | **86.5 %** | **86.5 %** |

## Coincidencia exacta de la consulta y latencia

| Motor | Coincidencia exacta | Latencia media | Mediana | p95 | Respaldo a reglas |
|---|---|---|---|---|---|
| llm (qwen2.5:7b) | 70.3 % | 3.659 s | 3.457 s | 4.001 s | 0 |
| llm+rag (qwen2.5:7b) | 78.4 % | 7.954 s | 7.891 s | 8.835 s | 0 |
| llm+rag+semillas (qwen2.5:7b) | 83.8 % | 7.933 s | 7.922 s | 8.799 s | 0 |

## Fallos — llm (qwen2.5:7b) (8)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| S02b | parafrasis | ¿Cuánto sale mes a mes ir al supermercado por término medio? | `{"operacion": "media", "por": "compra"}` |
| S05b | parafrasis | ¿En cuántas ocasiones pasé por el supermercado el mes anterior? | `{"medida": "unidades", "filtros": {"periodo": "¿En cuántas ocasiones pasé por el supermercado el mes anterior?"}}` |
| S07a | parafrasis | ¿Qué día de la semana suelo ir habitualmente a hacer la compra? | `{"operacion": "ranking", "agrupar_por": "dia_semana"}` |
| S07b | parafrasis | ¿En qué día de la semana recaen normalmente mis visitas al supermercado? | `{"operacion": "ranking", "agrupar_por": "dia_semana"}` |
| S08b | parafrasis | ¿Qué llevo invertido en artículos de limpieza doméstica en lo que va de año? | `{"filtros": {"concepto": "limpieza"}}` |
| S10a | parafrasis | ¿Cuál es el importe habitual de mis tickets de compra? | `{}` |
| S15b | parafrasis | ¿De qué manera ha ido cambiando lo que pago por yogures cada mes? | `{"medida": "precio", "operacion": "evolucion", "agrupar_por": "mes", "filtros": {"concepto": "yogures"}}` |
| S18b | parafrasis | ¿Qué artículo de menor precio he metido en el carro en lo que va de año? | `{"medida": "precio", "operacion": "minimo", "agrupar_por": "producto", "orden": "asc", "limite": 1}` |

## Fallos — llm+rag (qwen2.5:7b) (5)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| S02b | parafrasis | ¿Cuánto sale mes a mes ir al supermercado por término medio? | `{"operacion": "media", "por": "compra"}` |
| S05b | parafrasis | ¿En cuántas ocasiones pasé por el supermercado el mes anterior? | `{"medida": "unidades", "filtros": {"periodo": "¿En cuántas ocasiones pasé por el supermercado el mes anterior?"}}` |
| S08b | parafrasis | ¿Qué llevo invertido en artículos de limpieza doméstica en lo que va de año? | `{"filtros": {"concepto": "limpieza"}}` |
| S10a | parafrasis | ¿Cuál es el importe habitual de mis tickets de compra? | `{}` |
| S15b | parafrasis | ¿De qué manera ha ido cambiando lo que pago por yogures cada mes? | `{"medida": "precio", "operacion": "evolucion", "agrupar_por": "mes", "filtros": {"concepto": "yogures"}}` |

## Fallos — llm+rag+semillas (qwen2.5:7b) (5)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| S02b | parafrasis | ¿Cuánto sale mes a mes ir al supermercado por término medio? | `{"operacion": "media", "por": "compra"}` |
| S05b | parafrasis | ¿En cuántas ocasiones pasé por el supermercado el mes anterior? | `{"medida": "unidades", "filtros": {"periodo": "¿En cuántas ocasiones pasé por el supermercado el mes anterior?"}}` |
| S08b | parafrasis | ¿Qué llevo invertido en artículos de limpieza doméstica en lo que va de año? | `{"filtros": {"concepto": "limpieza"}}` |
| S10a | parafrasis | ¿Cuál es el importe habitual de mis tickets de compra? | `{}` |
| S18b | parafrasis | ¿Qué artículo de menor precio he metido en el carro en lo que va de año? | `{"medida": "precio", "operacion": "minimo", "agrupar_por": "producto"}` |
