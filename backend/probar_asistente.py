"""
Prueba rápida del asistente de gastos (sin levantar el servidor).

Uso (con el entorno virtual del backend activado y MongoDB en marcha):
    python probar_asistente.py                      # batería de preguntas de ejemplo
    python probar_asistente.py "¿Qué productos me han devuelto?"

Muestra qué motor ha interpretado cada pregunta (IA local o reglas), la consulta
estructurada que ha generado y la respuesta calculada con tus tickets reales.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(Path(__file__).parent / ".env")

from categorizador import categorizador          # noqa: E402
from asistente_gastos import AsistenteGastos     # noqa: E402
from traductor_llm import TraductorLLM           # noqa: E402

PREGUNTAS = [
    "¿Cuánto he gastado en lácteos este mes?",
    "¿y el mes pasado?",
    "¿Qué productos me han devuelto?",
    "¿Cuál es la semana en la que más gasté en fruta?",
    "¿Qué día de la semana suelo gastar más?",
    "¿Cuánto me dejo de media cuando voy un sábado?",
    "Enséñame las compras de más de 50 euros",
    "¿Cuáles son los 3 productos en los que menos gasto?",
    "¿Ha subido el precio de la leche?",
    "¿Qué tiempo hace mañana?",
]


async def cargar_tickets():
    cliente = AsyncIOMotorClient(os.environ["MONGO_URL"])
    return await cliente[os.environ["DB_NAME"]].tickets.find({}, {"_id": 0}).to_list(20000)


def main():
    tickets = asyncio.run(cargar_tickets())
    traductor = TraductorLLM()
    print("Tickets:", len(tickets))
    print("LLM:", json.dumps(traductor.estado(), ensure_ascii=False))
    asistente = AsistenteGastos(categorizador.listar_categorias(), traductor=traductor)

    historial = []
    for pregunta in sys.argv[1:] or PREGUNTAS:
        t0 = time.time()
        r = asistente.responder(pregunta, tickets, historial=historial)
        i = r.get("interpretacion", {})
        print(f"\n> {pregunta}   ({time.time() - t0:.1f} s)")
        print(f"  motor: {i.get('motor')} | entendido: {i.get('explicacion')}")
        print(f"  consulta: {json.dumps(i.get('consulta'), ensure_ascii=False)}")
        print(f"  {r['respuesta']}")
        if r.get("tabla"):
            print("  ", r["tabla"]["columnas"], r["tabla"]["filas"][:3])
        if i.get("consulta"):
            historial = (historial + [{"pregunta": pregunta, "consulta": i["consulta"]}])[-3:]


if __name__ == "__main__":
    main()
