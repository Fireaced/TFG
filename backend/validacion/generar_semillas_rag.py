"""
Experimento de adaptación del RAG: ¿aprende el asistente de las respuestas que el usuario valida (👍)?

1. SEMILLAS (este fichero -> semillas_rag.json): 20 preguntas con su consulta correcta, escritas con
   expresiones y estructuras que NO aparecen en el prompt del LLM ni en los bancos de desarrollo/prueba
   ("compra gorda", "pavos", "echar al carro", "pisar el súper"...). Simulan preguntas que el usuario
   hizo en la web y marcó como útiles: el RAG las añade a su base de ejemplos.
2. PARÁFRASIS (parafrasis_rag.txt): 2 formas distintas de preguntar lo mismo por cada semilla, escritas
   por OTRA IA (no por quien escribe las semillas ni el sistema) para no inflar el parecido.
   construir_banco_rag() las convierte en banco_rag.json (cada paráfrasis hereda la consulta de su semilla).
3. EVALUACIÓN: python -m validacion.evaluar_chat --banco rag
   compara LLM sin RAG / LLM + RAG (base = banco de desarrollo) / LLM + RAG + semillas.
   La diferencia entre las dos últimas es lo que aporta "aprender de los 👍".

Uso:
    python -m validacion.generar_semillas_rag            # escribe semillas_rag.json y el texto para la otra IA
    python -m validacion.generar_semillas_rag --banco    # lee parafrasis_rag.txt y escribe banco_rag.json
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))
from validacion.generar_banco import C  # noqa: E402

SEMILLAS = [
    ("¿Cuál ha sido mi compra gorda de este año?", C(operacion="maximo", agrupar="ticket", periodo="este año")),
    ("¿Cuánto me sale el súper cada mes, de media?", C(operacion="media", por="mes")),
    ("Hazme un desglose mensual de lo que llevo gastado en 2025", C(operacion="evolucion", agrupar="mes", periodo="en 2025")),
    ("¿Qué es lo que más echo al carro?", C("unidades", operacion="ranking", agrupar="producto")),
    ("¿Cuántas veces he pisado el Mercadona el mes pasado?", C("compras", periodo="el mes pasado")),
    ("¿Cuál fue la semana de 2025 en la que más veces fui a comprar?", C("compras", operacion="maximo", agrupar="semana", periodo="en 2025")),
    ("¿Qué día de la semana suelo hacer la compra?", C("compras", operacion="ranking", agrupar="dia_semana")),
    ("¿Qué me ha costado la limpieza de casa este año?", C(concepto="limpieza", periodo="este año")),
    ("¿Cuál es el ticket más gordo que he pagado un sábado?", C(operacion="maximo", agrupar="ticket", dias_semana=[5])),
    ("¿Cuánto suelo gastarme por ticket?", C(operacion="media", por="compra")),
    ("¿Cuánto he invertido en carne desde septiembre?", C(concepto="carne", periodo="desde septiembre")),
    ("¿En qué mes he ido menos veces a comprar?", C("compras", operacion="minimo", agrupar="mes")),
    ("¿Cuánto dinero se me ha ido en tickets de más de 50 pavos?", C(importe_min=50)),
    ("¿Cuántas compras rápidas, de menos de 5 euros, he hecho?", C("compras", importe_max=5)),
    ("¿Cómo ha variado lo que me gasto en yogures cada mes?", C(operacion="evolucion", agrupar="mes", concepto="yogures")),
    ("¿Qué subcategoría se lleva más dinero?", C(operacion="ranking", agrupar="subcategoria")),
    ("¿Qué precio medio pago por la leche?", C("precio", operacion="media", concepto="leche")),
    ("¿Cuál es la cosa más barata que he metido en el carro este año?", C("precio", operacion="minimo", agrupar="producto", periodo="este año")),
    ("¿Cuánto me gasté en el súper entre semana en 2025?", C(dias_semana=[0, 1, 2, 3, 4], periodo="en 2025")),
    ("Dime las 3 cosas que más compro", C("unidades", operacion="ranking", agrupar="producto", limite=3)),
]

TEXTO_OTRA_IA = """Te paso {n} preguntas que un usuario hace a una aplicación que analiza sus tickets del supermercado
(Mercadona). Para CADA una, escribe 2 formas distintas de preguntar EXACTAMENTE lo mismo, como lo diría
una persona normal en español de España, de forma natural y algo coloquial.

Reglas:
- Debe pedirse la misma información: mismo producto, mismo periodo de tiempo ("este año", "el mes pasado",
  "en 2025", "desde septiembre"...), mismos días, importes y número de resultados.
- Cambia el vocabulario y la estructura: no repitas más de 2-3 palabras de la pregunta original
  (salvo el producto, las fechas y las cifras).
- Las 2 versiones deben ser diferentes entre sí.
- Responde SOLO con líneas en este formato exacto, sin nada más:
S01a | primera versión
S01b | segunda versión

Preguntas:
{lista}
"""


def escribir_semillas():
    semillas = [{"id": f"S{i:02d}", "pregunta": p, "esperado": c} for i, (p, c) in enumerate(SEMILLAS, 1)]
    (RAIZ / "semillas_rag.json").write_text(json.dumps(semillas, ensure_ascii=False, indent=1), encoding="utf-8")
    texto = TEXTO_OTRA_IA.format(n=len(semillas), lista="\n".join(f"{s['id']}: {s['pregunta']}" for s in semillas))
    (RAIZ / "texto_para_otra_ia.txt").write_text(texto, encoding="utf-8")
    print(f"✅ semillas_rag.json ({len(semillas)}) y texto_para_otra_ia.txt escritos")


def construir_banco_rag():
    semillas = {s["id"]: s for s in json.loads((RAIZ / "semillas_rag.json").read_text(encoding="utf-8"))}
    banco, vistos = [], set()
    for linea in (RAIZ / "parafrasis_rag.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*\**(S\d{2})([ab])\**\s*[|:]\s*(.+?)\s*$", linea)
        if not m or m.group(1) not in semillas or m.group(3) in vistos:
            continue
        vistos.add(m.group(3))
        banco.append({"id": f"{m.group(1)}{m.group(2)}", "categoria": "parafrasis", "semilla": m.group(1),
                      "pregunta": m.group(3), "esperado": semillas[m.group(1)]["esperado"]})
    (RAIZ / "banco_rag.json").write_text(json.dumps(banco, ensure_ascii=False, indent=1), encoding="utf-8")
    faltan = sorted(set(semillas) - {b["semilla"] for b in banco})
    print(f"✅ banco_rag.json: {len(banco)} paráfrasis" + (f" (sin paráfrasis: {', '.join(faltan)})" if faltan else ""))


if __name__ == "__main__":
    construir_banco_rag() if "--banco" in sys.argv else escribir_semillas()
