# Evaluación del asistente (banco de prueba) — 13/10/2026 11:56

Banco: 40 preguntas · datos de prueba con hoy = 20/04/2026

## Acierto por categoría (respuesta equivalente a la esperada)

| Categoría | n | llm (qwen2.5:7b) | llm+rag (qwen2.5:7b) |
|---|---|---|---|
| basica | 9 | 66.7 % | 88.9 % |
| combinada | 8 | 100.0 % | 100.0 % |
| filtros | 10 | 100.0 % | 100.0 % |
| periodos | 6 | 100.0 % | 100.0 % |
| continuacion | 4 | 100.0 % | 100.0 % |
| fuera_de_ambito | 3 | 100.0 % | 100.0 % |
| **Total (sin «limites»)** | 40 | **92.5 %** | **97.5 %** |

## Coincidencia exacta de la consulta y latencia

| Motor | Coincidencia exacta | Latencia media | Mediana | p95 | Respaldo a reglas |
|---|---|---|---|---|---|
| llm (qwen2.5:7b) | 85.0 % | 3.664 s | 3.412 s | 4.624 s | 0 |
| llm+rag (qwen2.5:7b) | 92.5 % | 7.834 s | 7.851 s | 8.522 s | 0 |

## Fallos — llm (qwen2.5:7b) (3)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 119 | basica | ¿Qué ticket tuvo el importe más alto en 2025? | `{"medida": "compras", "operacion": "ultimo", "filtros": {"periodo": "¿Qué ticket tuvo el importe más alto en 2025?"}}` |
| 121 | basica | ¿Cuántos paquetes de bacon he comprado? | `{"medida": "unidades", "filtros": {"concepto": "paquetes bacon"}}` |
| 123 | basica | ¿Cuánto me dejo al mes en el súper? | `{"filtros": {"periodo": "este mes"}}` |

## Fallos — llm+rag (qwen2.5:7b) (1)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 123 | basica | ¿Cuánto me dejo al mes en el súper? | `{"filtros": {"periodo": "este mes"}}` |
