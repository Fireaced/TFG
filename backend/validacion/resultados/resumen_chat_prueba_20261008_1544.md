# Evaluación del asistente (banco de prueba) — 08/10/2026 15:44

Banco: 40 preguntas · datos de prueba con hoy = 20/04/2026

## Acierto por categoría (respuesta equivalente a la esperada)

| Categoría | n | reglas | llm (qwen2.5:7b) |
|---|---|---|---|
| basica | 9 | 55.6 % | 66.7 % |
| combinada | 8 | 12.5 % | 100.0 % |
| filtros | 10 | 20.0 % | 100.0 % |
| periodos | 6 | 83.3 % | 100.0 % |
| continuacion | 4 | 25.0 % | 100.0 % |
| fuera_de_ambito | 3 | 66.7 % | 100.0 % |
| **Total (sin «limites»)** | 40 | **40.0 %** | **92.5 %** |

## Coincidencia exacta de la consulta y latencia

| Motor | Coincidencia exacta | Latencia media | Mediana | p95 | Respaldo a reglas |
|---|---|---|---|---|---|
| reglas | 35.0 % | 0.001 s | 0.001 s | 0.002 s | 0 |
| llm (qwen2.5:7b) | 85.0 % | 3.662 s | 3.442 s | 4.764 s | 0 |

## Fallos — reglas (24)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 119 | basica | ¿Qué ticket tuvo el importe más alto en 2025? | `{"operacion": "maximo", "agrupar_por": "ticket", "filtros": {"concepto": "tuvo importe alto", "periodo": "¿Qué ticket tuvo el importe más alto en 2025?"}}` |
| 121 | basica | ¿Cuántos paquetes de bacon he comprado? | `{"medida": "compras", "operacion": "media", "por": "semana", "filtros": {"concepto": "paquetes bacon"}}` |
| 123 | basica | ¿Cuánto me dejo al mes en el súper? | `{"filtros": {"concepto": "dejo súper"}}` |
| 124 | basica | ¿En qué categoría me dejo más pasta? | `{"medida": "unidades", "operacion": "ranking", "agrupar_por": "categoria", "filtros": {"concepto": "dejo pasta"}}` |
| 126 | combinada | ¿En qué mes de 2024 gasté menos? | `{"operacion": "minimo", "agrupar_por": "ticket", "filtros": {"periodo": "¿En qué mes de 2024 gasté menos?"}}` |
| 127 | combinada | Dime las 3 categorías en las que más unidades compro | `{"medida": "unidades", "operacion": "ranking", "agrupar_por": "categoria", "limite": 3, "filtros": {"concepto": "unidades"}}` |
| 128 | combinada | ¿Cuál es el mes en el que más gasté en carne? | `{"filtros": {"concepto": "carne"}}` |
| 130 | combinada | ¿Cuánto pago de media por un yogur? | `{"operacion": "media", "por": "compra", "filtros": {"concepto": "pago yogur"}}` |
| 131 | combinada | ¿Qué semana gasté más en 2026? | `{"filtros": {"periodo": "¿Qué semana gasté más en 2026?"}}` |
| 132 | combinada | Los 4 productos que más dinero me cuestan | `{"operacion": "ranking", "agrupar_por": "producto", "limite": 4, "filtros": {"concepto": "cuestan"}}` |
| 133 | combinada | ¿Cuántos plátanos compro al mes? | `{"medida": "compras", "operacion": "media", "por": "mes", "filtros": {"concepto": "plátanos"}}` |
| 135 | filtros | ¿Cuánto gasto de media cada martes que voy al súper? | `{"operacion": "media", "por": "compra", "filtros": {"concepto": "súper", "dias_semana": [1]}}` |
| 137 | filtros | ¿Cuántos tickets por debajo de 3 euros tengo? | `{"medida": "compras", "filtros": {"concepto": "debajo"}}` |
| 138 | filtros | ¿Cuál fue la devolución más reciente? | `{"medida": "compras", "filtros": {"concepto": "reciente", "tipo_ticket": "devoluciones"}}` |
| 139 | filtros | ¿Cuánto dinero me devolvieron en 2024? | `{"filtros": {"concepto": "devolvieron", "periodo": "¿Cuánto dinero me devolvieron en 2024?", "tipo_ticket": "devoluciones"}}` |
| 140 | filtros | ¿Cuál es mi compra más pequeña sin contar las devoluciones? | `{"operacion": "minimo", "agrupar_por": "ticket", "filtros": {"concepto": "contar", "tipo_ticket": "devoluciones"}}` |
| 141 | filtros | ¿Cuántas veces compré entre semana en 2025? | `{"medida": "compras", "operacion": "media", "por": "semana", "filtros": {"periodo": "¿Cuántas veces compré entre semana en 2025?"}}` |
| 142 | filtros | ¿Qué frutas distintas he comprado? | `{"medida": "compras", "operacion": "media", "por": "semana", "filtros": {"concepto": "frutas distintas"}}` |
| 143 | filtros | ¿Cuánto me dejé en carne los fines de semana? | `{"filtros": {"concepto": "dejé carne fines"}}` |
| 146 | periodos | ¿Qué compré el 29 de enero? | `no entendida` |
| 151 | continuacion | ¿y los jueves? | `no entendida` |
| 152 | continuacion | ¿y cuántas unidades son? | `{"filtros": {"concepto": "unidades"}}` |
| 153 | continuacion | ¿y en cuál menos? | `{"operacion": "minimo", "agrupar_por": "ticket"}` |
| 155 | fuera_de_ambito | ¿Me ayudas a planificar un viaje? | `{"tipo": "saludo"}` |

## Fallos — llm (qwen2.5:7b) (3)

| # | Categoría | Pregunta | Consulta generada |
|---|---|---|---|
| 119 | basica | ¿Qué ticket tuvo el importe más alto en 2025? | `{"medida": "compras", "operacion": "ultimo", "filtros": {"periodo": "¿Qué ticket tuvo el importe más alto en 2025?"}}` |
| 121 | basica | ¿Cuántos paquetes de bacon he comprado? | `{"medida": "unidades", "filtros": {"concepto": "paquetes bacon"}}` |
| 123 | basica | ¿Cuánto me dejo al mes en el súper? | `{"filtros": {"periodo": "este mes"}}` |
