"""
Genera banco_preguntas.json: preguntas de prueba del asistente con su consulta correcta ("gold").

Cada pregunta tiene:
  - id, categoria (tipo de dificultad), pregunta
  - historial (opcional): pregunta anterior con su consulta correcta, para las continuaciones
  - esperado: la Consulta correcta escrita a mano
Las preguntas NO coinciden con los ejemplos del prompt del LLM ni con las frases de ejemplo del
clasificador de intenciones, para no evaluar con datos de entrenamiento.

Fecha de referencia ("hoy") de todo el banco: 20/04/2026 (datos: 21/08/2024 - 15/04/2026).
"""
import json
from pathlib import Path


def C(medida="gasto", operacion="total", agrupar=None, por=None, orden=None, limite=None, tipo=None, **filtros):
    c = {}
    if tipo:
        return {"tipo": tipo}
    if medida != "gasto":
        c["medida"] = medida
    if operacion != "total":
        c["operacion"] = operacion
    if agrupar:
        c["agrupar_por"] = agrupar
    if por:
        c["por"] = por
    if orden:
        c["orden"] = orden
    if limite:
        c["limite"] = limite
    if filtros:
        c["filtros"] = filtros
    return c


FUERA, SALUDO = C(tipo="fuera_de_ambito"), C(tipo="saludo")

BANCO = {
    "basica": [
        ("¿Cuánto me he gastado este mes?", C(periodo="este mes")),
        ("¿Cuánto llevo gastado en 2025?", C(periodo="en 2025")),
        ("¿Cuántas veces he ido al súper este año?", C("compras", periodo="este año")),
        ("¿Cuál fue mi ticket más caro?", C(operacion="maximo", agrupar="ticket")),
        ("¿Cuál fue mi compra más económica de 2025?", C(operacion="minimo", agrupar="ticket", periodo="en 2025")),
        ("¿En qué categoría gasto más dinero?", C(operacion="ranking", agrupar="categoria")),
        ("¿Cuánto he gastado en fruta?", C(concepto="fruta")),
        ("¿Qué gasto medio tengo en cada compra?", C(operacion="media", por="compra")),
        ("¿Qué día hice la compra más reciente?", C("compras", "ultimo")),
        ("¿Cuánto me gasto al mes de media?", C(operacion="media", por="mes")),
        ("¿Cuántas compras hice en marzo?", C("compras", periodo="en marzo")),
        ("¿Qué es lo más caro que he comprado nunca?", C("precio", "maximo", "producto")),
        ("¿Cuánto he gastado en carne este año?", C(concepto="carne", periodo="este año")),
        ("¿Cuántos huevos he comprado?", C("unidades", concepto="huevos")),
        ("¿Cuánto he pagado en total por el jamón?", C(concepto="jamón")),
        ("¿Cuál es el producto en el que más dinero me gasto?", C(operacion="ranking", agrupar="producto")),
        ("¿Cada cuánto voy a comprar?", C("compras", "media", por="semana")),
        ("¿Cuánto gasté en enero de 2026?", C(periodo="en enero de 2026")),
        ("¿Cuánto he gastado en bacon?", C(concepto="bacon")),
        ("¿Cuántas unidades de pizza he comprado?", C("unidades", concepto="pizza")),
    ],
    "coloquial": [
        ("¿Cuánta pasta me he dejado en el súper este año?", C(periodo="este año")),
        ("¿Qué es lo que más pillo en el Mercadona?", C("unidades", "ranking", "producto")),
        ("¿En qué se me va el dinero?", C(operacion="ranking", agrupar="categoria")),
        ("¿Me dejo mucho en congelados?", C(concepto="congelados")),
        ("¿Qué día de la semana me fundo más dinero?", C(operacion="ranking", agrupar="dia_semana")),
        ("¿Cuánto me sale de media cada visita al súper?", C(operacion="media", por="compra")),
        ("¿Cuántas veces me he pasado por el súper en 2025?", C("compras", periodo="en 2025")),
        ("Dime cuánto he soltado en yogures", C(concepto="yogures")),
        ("¿Cuál fue el día que más me dejé en una sola compra?", C(operacion="maximo", agrupar="ticket")),
        ("¿Cuánto me gasto de normal a la semana?", C(operacion="media", por="semana")),
        ("¿Gasto mucho en bebida?", C(concepto="bebidas")),
        ("¿Con qué frecuencia compro pan?", C("compras", "media", por="semana", concepto="pan")),
        ("¿Cuánto me costaron los helados el año pasado?", C(concepto="helados", periodo="el año pasado")),
        ("Oye, ¿en limpieza cuánto llevo?", C(concepto="limpieza")),
        ("¿Qué categoría me sale más cara?", C(operacion="ranking", agrupar="categoria")),
        ("¿Cuál es mi producto estrella?", C("unidades", "ranking", "producto")),
        ("Hazme un resumen de cuánto gasto cada mes", C(operacion="evolucion", agrupar="mes")),
        ("¿Cuánto me cuesta de media la cesta de la compra?", C(operacion="media", por="compra")),
        ("¿Cuánto llevo gastado desde septiembre?", C(periodo="desde septiembre")),
        ("¿Me estoy pasando con el embutido?", C(concepto="embutido")),
    ],
    "combinada": [
        ("¿En qué mes gasté más?", C(operacion="maximo", agrupar="mes")),
        ("¿Cuál fue la semana en la que menos gasté?", C(operacion="minimo", agrupar="semana")),
        ("Las 5 subcategorías en las que más gasto", C(operacion="ranking", agrupar="subcategoria", limite=5)),
        ("¿Cuántos yogures compro de media cada vez que voy?", C("unidades", "media", por="compra", concepto="yogures")),
        ("¿Qué mes compré más fruta?", C("unidades", "maximo", "mes", concepto="fruta")),
        ("Top 3 de categorías por número de productos", C("unidades", "ranking", "categoria", limite=3)),
        ("¿Qué día de la semana voy más a comprar?", C("compras", "ranking", "dia_semana")),
        ("¿Cuál es el producto de charcutería que más compro?", C("unidades", "ranking", "producto", concepto="charcutería")),
        ("¿En qué mes de 2025 gasté más en carne?", C(operacion="maximo", agrupar="mes", concepto="carne", periodo="en 2025")),
        ("¿Cuánto gasto de media al día?", C(operacion="media", por="dia")),
        ("Los 10 productos en los que más dinero me he gastado", C(operacion="ranking", agrupar="producto", limite=10)),
        ("¿Cuáles son las 3 categorías en las que menos gasto?", C(operacion="ranking", agrupar="categoria", orden="asc", limite=3)),
        ("¿Cuánto me gasto de media en carne cada vez que la compro?", C(operacion="media", por="compra", concepto="carne")),
        ("¿Cómo ha ido mi gasto en lácteos mes a mes?", C(operacion="evolucion", agrupar="mes", concepto="lácteos")),
        ("¿Cuántas unidades de leche compro al mes?", C("unidades", "media", por="mes", concepto="leche")),
        ("¿Cuál es el producto más barato que he comprado este año?", C("precio", "minimo", "producto", periodo="este año")),
        ("¿Qué semana de 2025 hice más compras?", C("compras", "maximo", "semana", periodo="en 2025")),
        ("¿Cuál es la subcategoría de la que más unidades compro?", C("unidades", "ranking", "subcategoria")),
        ("¿Qué producto de limpieza compro más?", C("unidades", "ranking", "producto", concepto="limpieza")),
        ("¿Cuál es el precio medio de los yogures que compro?", C("precio", "media", concepto="yogures")),
    ],
    "filtros": [
        ("¿Cuánto me han devuelto en total?", C(tipo_ticket="devoluciones")),
        ("¿Cuántos tickets de devolución tengo?", C("compras", tipo_ticket="devoluciones")),
        ("Enséñame los productos que devolví", C("unidades", "lista", "producto", tipo_ticket="devoluciones")),
        ("¿Cuánto gasto de media los lunes?", C(operacion="media", por="compra", dias_semana=[0])),
        ("¿Cuántas veces he comprado en fin de semana?", C("compras", dias_semana=[5, 6])),
        ("Muéstrame los tickets de menos de 5 euros", C("compras", "lista", "ticket", importe_max=5)),
        ("¿Cuántas compras de más de 30 € he hecho?", C("compras", importe_min=30)),
        ("Tickets entre 20 y 40 euros", C("compras", "lista", "ticket", importe_min=20, importe_max=40)),
        ("¿Cuál fue mi compra más barata sin contar devoluciones?", C(operacion="minimo", agrupar="ticket", tipo_ticket="compras")),
        ("¿Cuánto gasté los viernes de 2025?", C(dias_semana=[4], periodo="en 2025")),
        ("¿Cuánto gasto de media entre semana?", C(operacion="media", por="compra", dias_semana=[0, 1, 2, 3, 4])),
        ("¿Cuánto gasto de media cuando compro en fin de semana?", C(operacion="media", por="compra", dias_semana=[5, 6])),
        ("Lista de compras de más de 60 euros este año", C("compras", "lista", "ticket", importe_min=60, periodo="este año")),
        ("¿Cuánto me dejé en total en los tickets de más de 50 €?", C(importe_min=50)),
        ("¿Cuál es la compra más cara que he hecho un sábado?", C(operacion="maximo", agrupar="ticket", dias_semana=[5])),
        ("¿Cuántas devoluciones hice en 2025?", C("compras", tipo_ticket="devoluciones", periodo="en 2025")),
        ("¿Qué día devolví algo por última vez?", C("compras", "ultimo", tipo_ticket="devoluciones")),
        ("¿Cuántos productos distintos de carne he comprado?", C("unidades", "lista", "producto", concepto="carne")),
        ("¿Cuánto gasté los lunes en carne?", C(concepto="carne", dias_semana=[0])),
        ("¿Cuántas compras pequeñas, de menos de 10 €, he hecho este año?", C("compras", importe_max=10, periodo="este año")),
    ],
    "periodos": [
        ("¿Cuánto gasté el mes pasado?", C(periodo="el mes pasado")),
        ("¿Cuánto gasté la semana pasada?", C(periodo="la semana pasada")),
        ("¿Cuánto gasté en los últimos 3 meses?", C(periodo="los últimos 3 meses")),
        ("¿Cuántas compras hice el año pasado?", C("compras", periodo="el año pasado")),
        ("¿Cuánto me gasté en agosto de 2025?", C(periodo="en agosto de 2025")),
        ("¿Cuál fue mi compra más cara de 2024?", C(operacion="maximo", agrupar="ticket", periodo="en 2024")),
        ("¿Cuánto gasté en lácteos en los últimos 6 meses?", C(concepto="lácteos", periodo="los últimos 6 meses")),
        ("¿Cuánto he gastado desde enero?", C(periodo="desde enero")),
        ("¿Cuánto gasté en diciembre?", C(periodo="en diciembre")),
        ("¿Cuántas veces compré en los últimos 30 días?", C("compras", periodo="los últimos 30 días")),
        ("¿Cuál fue el producto más caro del mes pasado?", C("precio", "maximo", "producto", periodo="el mes pasado")),
        ("Gasto total de 2024", C(periodo="en 2024")),
        ("¿Cuánto gasté en fruta en agosto de 2025?", C(concepto="fruta", periodo="en agosto de 2025")),
        ("¿Qué categoría fue la de más gasto el año pasado?", C(operacion="ranking", agrupar="categoria", periodo="el año pasado")),
        ("¿Cuántas compras hice en abril?", C("compras", periodo="en abril")),
    ],
    "continuacion": [
        ("¿y el mes pasado?", C(concepto="lácteos", periodo="el mes pasado"),
         ("¿Cuánto he gastado en lácteos este mes?", C(concepto="lácteos", periodo="este mes"))),
        ("¿y en carne?", C(concepto="carne", periodo="en 2025"),
         ("¿Cuánto gasté en fruta en 2025?", C(concepto="fruta", periodo="en 2025"))),
        ("¿y en 2025?", C(operacion="ranking", agrupar="categoria", periodo="en 2025"),
         ("¿En qué categoría gasto más?", C(operacion="ranking", agrupar="categoria"))),
        ("lo mismo pero en febrero", C("compras", periodo="en febrero"),
         ("¿Cuántas compras hice en marzo?", C("compras", periodo="en marzo"))),
        ("¿y la más barata?", C(operacion="minimo", agrupar="ticket", periodo="en 2025"),
         ("¿Cuál fue mi compra más cara de 2025?", C(operacion="maximo", agrupar="ticket", periodo="en 2025"))),
        ("¿y los viernes?", C(operacion="media", por="compra", dias_semana=[4]),
         ("¿Cuánto gasto de media los sábados?", C(operacion="media", por="compra", dias_semana=[5]))),
        ("ahora dime los 5 primeros", C("unidades", "ranking", "producto", limite=5),
         ("Los 3 productos que más compro", C("unidades", "ranking", "producto", limite=3))),
        ("¿y cuántas veces la he comprado?", C("compras", concepto="carne"),
         ("¿Cuánto he gastado en carne?", C(concepto="carne"))),
        ("¿y en febrero?", C(periodo="en febrero de 2026"),
         ("¿Cuánto gasté en enero de 2026?", C(periodo="en enero de 2026"))),
        ("¿y solo este año?", C(concepto="pizza", periodo="este año"),
         ("¿Cuánto he gastado en pizza?", C(concepto="pizza"))),
    ],
    "fuera_de_ambito": [
        ("¿Va a llover el fin de semana?", FUERA),
        ("Recomiéndame una receta con pollo", FUERA),
        ("¿Quién ganó el último mundial?", FUERA),
        ("Escríbeme un poema", FUERA),
        ("Hola, buenas", SALUDO),
        ("Muchas gracias por la ayuda", SALUDO),
        ("¿Qué cosas me puedes responder?", SALUDO),
    ],
    # Límites conocidos del motor: la consulta correcta existe, pero el texto del periodo o el
    # concepto no lo sabe interpretar el motor. Se evalúan aparte y se documentan en la memoria.
    "limites": [
        ("¿Cuánto gasté el verano pasado?", C(desde="2025-06-21", hasta="2025-09-22")),
        ("¿Qué compré el 13 de abril?", C("unidades", "lista", "producto", desde="2026-04-13", hasta="2026-04-13")),
        ("¿Cuánto he gastado en plátanos?", C(concepto="banana")),
        ("¿Cuánto gasté en la primera quincena de agosto de 2025?", C(desde="2025-08-01", hasta="2025-08-15")),
    ],
}


# Banco de PRUEBA (reservado): preguntas nuevas escritas después de ajustar el sistema con el banco de
# desarrollo. Las cifras finales de la memoria se dan con este banco, que no se ha usado para ajustar nada.
BANCO_PRUEBA = {
    "basica": [
        ("¿Cuánto llevo gastado este año en total?", C(periodo="este año")),
        ("¿Cuántas veces fui a comprar en 2024?", C("compras", periodo="en 2024")),
        ("¿Qué ticket tuvo el importe más alto en 2025?", C(operacion="maximo", agrupar="ticket", periodo="en 2025")),
        ("¿Cuánto he gastado en queso?", C(concepto="queso")),
        ("¿Cuántos paquetes de bacon he comprado?", C("unidades", concepto="bacon")),
        ("¿Cuánto me cuesta de media ir a comprar?", C(operacion="media", por="compra")),
        ("¿Cuánto me dejo al mes en el súper?", C(operacion="media", por="mes")),
        ("¿En qué categoría me dejo más pasta?", C(operacion="ranking", agrupar="categoria")),
        ("¿Cuánto gasté en marzo de 2025?", C(periodo="en marzo de 2025")),
    ],
    "combinada": [
        ("¿En qué mes de 2024 gasté menos?", C(operacion="minimo", agrupar="mes", periodo="en 2024")),
        ("Dime las 3 categorías en las que más unidades compro", C("unidades", "ranking", "categoria", limite=3)),
        ("¿Cuál es el mes en el que más gasté en carne?", C(operacion="maximo", agrupar="mes", concepto="carne")),
        ("¿Cómo evoluciona lo que gasto en carne?", C(operacion="evolucion", agrupar="mes", concepto="carne")),
        ("¿Cuánto pago de media por un yogur?", C("precio", "media", concepto="yogur")),
        ("¿Qué semana gasté más en 2026?", C(operacion="maximo", agrupar="semana", periodo="en 2026")),
        ("Los 4 productos que más dinero me cuestan", C(operacion="ranking", agrupar="producto", limite=4)),
        ("¿Cuántos plátanos compro al mes?", C("unidades", "media", por="mes", concepto="banana")),
    ],
    "filtros": [
        ("¿Cuánto gasté en total los sábados?", C(dias_semana=[5])),
        ("¿Cuánto gasto de media cada martes que voy al súper?", C(operacion="media", por="compra", dias_semana=[1])),
        ("Enséñame los tickets de más de 40 euros de 2025", C("compras", "lista", "ticket", importe_min=40, periodo="en 2025")),
        ("¿Cuántos tickets por debajo de 3 euros tengo?", C("compras", importe_max=3)),
        ("¿Cuál fue la devolución más reciente?", C("compras", "ultimo", tipo_ticket="devoluciones")),
        ("¿Cuánto dinero me devolvieron en 2024?", C(tipo_ticket="devoluciones", periodo="en 2024")),
        ("¿Cuál es mi compra más pequeña sin contar las devoluciones?", C(operacion="minimo", agrupar="ticket", tipo_ticket="compras")),
        ("¿Cuántas veces compré entre semana en 2025?", C("compras", dias_semana=[0, 1, 2, 3, 4], periodo="en 2025")),
        ("¿Qué frutas distintas he comprado?", C("unidades", "lista", "producto", concepto="fruta")),
        ("¿Cuánto me dejé en carne los fines de semana?", C(concepto="carne", dias_semana=[5, 6])),
    ],
    "periodos": [
        ("¿Cuánto gasté en la segunda quincena de julio de 2025?", C(desde="2025-07-16", hasta="2025-07-31")),
        ("¿Cuánto gasté el invierno pasado?", C(desde="2025-12-21", hasta="2026-03-19")),
        ("¿Qué compré el 29 de enero?", C("unidades", "lista", "producto", desde="2026-01-29", hasta="2026-01-29")),
        ("¿Cuánto gasté en la primavera de 2025?", C(desde="2025-03-20", hasta="2025-06-20")),
        ("¿Cuántas compras hice en los últimos 2 meses?", C("compras", periodo="los últimos 2 meses")),
        ("¿Cuánto he gastado desde marzo de 2025?", C(periodo="desde marzo de 2025")),
    ],
    "continuacion": [
        ("¿y en 2024?", C(concepto="carne", periodo="en 2024"),
         ("¿Cuánto gasté en carne en 2025?", C(concepto="carne", periodo="en 2025"))),
        ("¿y los jueves?", C(operacion="media", por="compra", dias_semana=[3]),
         ("¿Cuánto gasto de media por compra los lunes?", C(operacion="media", por="compra", dias_semana=[0]))),
        ("¿y cuántas unidades son?", C("unidades", concepto="yogures"),
         ("¿Cuánto he gastado en yogures?", C(concepto="yogures"))),
        ("¿y en cuál menos?", C(operacion="minimo", agrupar="mes"),
         ("¿En qué mes gasté más?", C(operacion="maximo", agrupar="mes"))),
    ],
    "fuera_de_ambito": [
        ("¿Cuál es la capital de Francia?", FUERA),
        ("¿Me ayudas a planificar un viaje?", FUERA),
        ("Buenos días", SALUDO),
    ],
}


def main():
    banco, n = [], 0
    for categoria, preguntas in BANCO.items():
        for item in preguntas:
            n += 1
            p = {"id": n, "categoria": categoria, "pregunta": item[0], "esperado": item[1]}
            if len(item) > 2:
                p["historial"] = [{"pregunta": item[2][0], "consulta": item[2][1]}]
            banco.append(p)
    ruta = Path(__file__).parent / "banco_preguntas.json"
    ruta.write_text(json.dumps(banco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(banco)} preguntas -> {ruta}")
    prueba = []
    for categoria, preguntas in BANCO_PRUEBA.items():
        for item in preguntas:
            n += 1
            p = {"id": n, "categoria": categoria, "pregunta": item[0], "esperado": item[1]}
            if len(item) > 2:
                p["historial"] = [{"pregunta": item[2][0], "consulta": item[2][1]}]
            prueba.append(p)
    ruta = Path(__file__).parent / "banco_prueba.json"
    ruta.write_text(json.dumps(prueba, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(prueba)} preguntas (prueba) -> {ruta}")


if __name__ == "__main__":
    main()
