"""
Aplica a la base de datos las categorías revisadas a mano (validacion/etiquetas_revisadas.csv).

Muchos productos de los tickets guardados tienen la categoría que les puso el buscador del catálogo,
y algunas eran incorrectas (p. ej. BOLAS CEREAL CHOCOLA como queso). Como el clasificador por similitud
usa el historial como referencia, conviene corregirlas. El script:
  - actualiza las categorías de esos productos en todos los tickets guardados, y
  - las guarda como "categorías aprendidas" para que los próximos tickets las usen directamente.

Uso (desde backend/, con MongoDB en marcha):
    python herramientas/aplicar_etiquetas_revisadas.py            # solo muestra los cambios
    python herramientas/aplicar_etiquetas_revisadas.py --aplicar  # los aplica
"""
import asyncio
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

RAIZ = Path(__file__).resolve().parent.parent      # backend/
sys.path.insert(0, str(RAIZ))                      # los módulos de la aplicación están en backend/
load_dotenv(RAIZ / ".env")
from categorizador import normalizar  # noqa: E402


def leer_etiquetas():
    etiquetas = {}
    with open(RAIZ / "validacion" / "etiquetas_revisadas.csv", encoding="utf-8-sig", newline="") as fh:
        for f in csv.DictReader(fh, delimiter=";"):
            if (f.get("revisado") or "").strip().lower() not in ("si", "sí"):
                continue
            cats = [{"categoria": f["categoria"], "subcategoria": f["subcategoria"]}]
            if f.get("categoria_2"):
                cats.append({"categoria": f["categoria_2"], "subcategoria": f["subcategoria_2"]})
            etiquetas[f["nombre"]] = cats
    return etiquetas


async def main(aplicar: bool):
    db = AsyncIOMotorClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    etiquetas = leer_etiquetas()
    cambios, productos = 0, set()
    async for t in db.tickets.find({}, {"_id": 0}):
        items, tocado = t.get("items", []), False
        for it in items:
            nuevas = etiquetas.get(it.get("descripcion"))
            clave = lambda cs: sorted((c["categoria"], c["subcategoria"]) for c in cs or [])  # noqa: E731
            if nuevas and clave(nuevas) != clave(it.get("categorias")):
                antes = ", ".join(f"{c['categoria']}/{c['subcategoria']}" for c in it.get("categorias") or []) or "sin categoría"
                if it["descripcion"] not in productos:
                    print(f"  {it['descripcion']}: {antes}  ->  " + ", ".join(f"{c['categoria']}/{c['subcategoria']}" for c in nuevas))
                productos.add(it["descripcion"])
                it["categorias"], it["categoria_manual"], tocado = nuevas, True, True
                cambios += 1
        if tocado and aplicar:
            await db.tickets.update_one({"id": t["id"]}, {"$set": {"items": items}})
    if aplicar:
        for nombre in productos:
            await db.categorias_aprendidas.update_one(
                {"clave": normalizar(nombre)},
                {"$set": {"clave": normalizar(nombre), "descripcion": nombre, "categorias": etiquetas[nombre],
                          "producto_catalogo": None, "updated_at": datetime.now(timezone.utc).isoformat()}},
                upsert=True)
    print(f"\n{len(productos)} productos, {cambios} líneas de ticket " + ("actualizadas." if aplicar else
          "por actualizar. Ejecuta con --aplicar para guardar los cambios."))


if __name__ == "__main__":
    asyncio.run(main("--aplicar" in sys.argv))
