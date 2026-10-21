"""
Evaluación del asistente de gastos con el banco de preguntas (banco_preguntas.json).

Para cada pregunta se obtiene la consulta que genera el sistema (motor de reglas o LLM local, por el
mismo camino que usa la web) y se compara con la consulta correcta escrita a mano:

  - Acierto (principal): ejecutar ambas consultas sobre los mismos tickets da la MISMA respuesta
    (mismo número, mismo primer elemento del ranking, mismo conjunto de tickets...). Así se dan por
    buenas consultas escritas de otra forma pero equivalentes (p. ej. "máximo por mes" y
    "ranking por mes" responden lo mismo a "¿en qué mes gasté más?").
  - Coincidencia exacta: la consulta generada es idéntica campo a campo a la esperada.
  - Latencia de la interpretación.

Uso (desde backend/, con el venv activado):
    python -m validacion.evaluar_chat --motor reglas
    python -m validacion.evaluar_chat --motor llm                       # usa LLM_URL / LLM_MODELO del .env
    python -m validacion.evaluar_chat --motor llm --modelo qwen2.5:3b
    python -m validacion.evaluar_chat --motor ambos

Los resultados se guardan en validacion/resultados/ (CSV con el detalle y Markdown con el resumen).
Datos fijos: validacion/tickets_prueba.json, con "hoy" = 20/04/2026, para que sea reproducible.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ.parent / ".env")

from categorizador import categorizador  # noqa: E402
from asistente_gastos import AsistenteGastos, extraer_periodo, sin_tildes  # noqa: E402
from motor_consultas import Consulta, EjecutorConsultas  # noqa: E402

HOY = date(2026, 4, 20)
CATEGORIAS = ["basica", "coloquial", "combinada", "filtros", "periodos", "continuacion", "fuera_de_ambito", "limites", "parafrasis"]


# ── Comparación de resultados ──
def ejecutar(ejecutor, asistente, consulta: Consulta) -> dict:
    if consulta is None:
        return {"tipo": "no_entendida"}
    try:
        r = ejecutor.ejecutar(consulta.model_copy(deep=True), asistente.df, HOY)
        return r.get("valor") or {"tipo": "sin_valor"}
    except Exception as e:  # una consulta mal formada no debe parar la evaluación
        return {"tipo": "error", "datos": str(e)[:100]}


def _clase_top(clase: str) -> str:
    return {"ticket": "ticket:gasto"}.get(clase, clase)


def equivalentes(esperada: Consulta, v_esp: dict, generada, v_gen: dict) -> bool:
    if esperada.tipo != "consulta":
        if generada is None:
            return esperada.tipo == "fuera_de_ambito"   # "no te entiendo" es válido para algo fuera de ámbito
        return generada.tipo == esperada.tipo
    if generada is None or generada.tipo != "consulta" or v_gen.get("tipo") != v_esp.get("tipo"):
        return False
    tipo = v_esp["tipo"]
    if tipo == "escalar":
        return v_gen["clase"] == v_esp["clase"] and abs(v_gen["datos"] - v_esp["datos"]) <= 0.011
    if tipo == "top":
        k = esperada.limite if "limite" in esperada.model_fields_set else 1
        a, b = v_esp["datos"][:k], v_gen["datos"][:k]
        if [x[0] for x in a] != [x[0] for x in b]:
            return False
        return _clase_top(v_esp["clase"]) != _clase_top(v_gen["clase"]) or \
            all(abs(x[1] - y[1]) <= 0.011 for x, y in zip(a, b))
    return v_gen.get("datos") == v_esp.get("datos")


def normalizar_consulta(c: Consulta):
    """Forma canónica para la coincidencia exacta (periodo como fechas, concepto sin tildes)."""
    if c is None:
        return None
    if c.tipo != "consulta":
        return (c.tipo,)
    f = c.filtros
    if f.desde or f.hasta:
        ini, fin = f.desde, f.hasta
    elif f.periodo:
        ini, fin, _, _ = extraer_periodo(f.periodo, HOY)
    else:
        ini = fin = None
    concepto = sin_tildes(f.concepto or "").strip().rstrip("s") or None
    return (c.medida, c.operacion, c.agrupar_por, c.por if c.operacion == "media" else None,
            c.orden if c.operacion == "ranking" else None,
            c.limite if c.operacion == "ranking" and c.limite != 5 else None,
            concepto, (f.categoria or None), str(ini), str(fin), f.tipo_ticket,
            tuple(sorted(f.dias_semana)), f.importe_min, f.importe_max)


# ── Evaluación ──
def evaluar(motor: str, banco: list, tickets: list, modelo: str = None, rag: bool = False,
            semillas: bool = False, rag_k: int = None, rag_umbral: float = None) -> dict:
    traductor = None
    if motor == "llm":
        if modelo:
            os.environ["LLM_MODELO"] = modelo
        from traductor_llm import TraductorLLM
        traductor = TraductorLLM()
        estado = traductor.estado()
        if not estado.get("conectado"):
            sys.exit(f"❌ No hay conexión con el LLM ({traductor.url}): {estado.get('error')}")
        print(f"🤖 LLM: {traductor.modelo} en {traductor.url}")
        traductor.calentar()
    asistente = AsistenteGastos(categorizador.listar_categorias(), traductor=traductor)
    asistente.preparar(tickets)
    if rag:
        # RAG: base de ejemplos = banco de DESARROLLO (gold). Al evaluar ese mismo banco se excluye la
        # propia pregunta (leave-one-out); el banco de prueba no está en la base.
        from embeddings import ClienteEmbeddings
        from rag_chat import BaseEjemplos
        cliente = ClienteEmbeddings()
        base = BaseEjemplos(cliente if cliente.disponible() else None)
        if rag_k:
            base.k = rag_k
        if rag_umbral is not None:
            base.umbral = rag_umbral
        n = base.cargar_banco()
        if semillas:
            # Experimento de adaptación: preguntas "validadas por el usuario" (👍) de las que salen las paráfrasis
            n += base.cargar_semillas(RAIZ / "semillas_rag.json")
        print(f"📚 RAG: {n} ejemplos, k = {base.k}, umbral = {base.umbral} "
              f"({'embeddings ' + cliente.modelo if base.emb else 'TF-IDF'})")
        asistente.rag = base
    ejecutor = EjecutorConsultas(asistente.resolutor, asistente.clusters, asistente.arbol, extraer_periodo)

    filas = []
    for i, p in enumerate(banco, 1):
        esperada = Consulta.model_validate(p["esperado"])
        historial = p.get("historial")
        t0 = time.perf_counter()
        generada, motor_usado = asistente.interpretar(p["pregunta"], HOY, historial, usar_llm=(motor == "llm"),
                                                      excluir_ejemplo=p["pregunta"])
        latencia = time.perf_counter() - t0
        v_esp, v_gen = ejecutar(ejecutor, asistente, esperada), ejecutar(ejecutor, asistente, generada)
        ok = equivalentes(esperada, v_esp, generada, v_gen)
        exacta = normalizar_consulta(generada) == normalizar_consulta(esperada)
        filas.append({
            "id": p["id"], "categoria": p["categoria"], "pregunta": p["pregunta"],
            "acierto": int(ok), "exacta": int(exacta), "latencia_s": round(latencia, 3),
            "motor_usado": motor_usado,
            "respaldo_reglas": int(motor == "llm" and motor_usado != "llm"),
            "consulta_esperada": json.dumps(p["esperado"], ensure_ascii=False),
            "consulta_generada": json.dumps(generada.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False) if generada else "",
            "respuesta_esperada": json.dumps(_resumen(v_esp), ensure_ascii=False),
            "respuesta_generada": json.dumps(_resumen(v_gen), ensure_ascii=False),
        })
        print(f"[{i:3d}/{len(banco)}] {'✅' if ok else '❌'} {latencia:5.2f}s  {p['pregunta']}")
    return {"motor": motor + ("+rag" if rag else "") + ("+semillas" if semillas else ""), "modelo": traductor.modelo if traductor else "-", "filas": filas}


def _resumen(v: dict):
    if v.get("tipo") in ("top", "serie", "conjunto"):
        return {**v, "datos": v.get("datos", [])[:5]}
    return v


# ── Informe ──
def resumen(res: dict) -> dict:
    por_cat = defaultdict(list)
    for f in res["filas"]:
        por_cat[f["categoria"]].append(f)
    principal = [f for f in res["filas"] if f["categoria"] != "limites"]
    lat = [f["latencia_s"] for f in principal]
    return {
        "por_categoria": {c: (sum(f["acierto"] for f in fs), sum(f["exacta"] for f in fs), len(fs))
                          for c, fs in por_cat.items()},
        "total": (sum(f["acierto"] for f in principal), sum(f["exacta"] for f in principal), len(principal)),
        "latencia_media": statistics.mean(lat), "latencia_mediana": statistics.median(lat),
        "latencia_p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
        "respaldo": sum(f["respaldo_reglas"] for f in res["filas"]),
    }


def pct(a, n):
    return f"{100 * a / n:.1f} %" if n else "—"


def informe_md(resultados: list) -> str:
    r = [(x, resumen(x)) for x in resultados]
    nombres = [f"{x['motor']} ({x['modelo']})" if x["motor"].startswith("llm") else "reglas" for x, _ in r]
    lineas = [f"# Evaluación del asistente — {datetime.now():%d/%m/%Y %H:%M}", "",
              f"Banco: {len(resultados[0]['filas'])} preguntas · datos de prueba con hoy = {HOY:%d/%m/%Y}", "",
              "## Acierto por categoría (respuesta equivalente a la esperada)", "",
              "| Categoría | n | " + " | ".join(nombres) + " |", "|---|---|" + "---|" * len(nombres)]
    for c in CATEGORIAS:
        celdas = []
        n = 0
        for _, s in r:
            a, _, n = s["por_categoria"].get(c, (0, 0, 0))
            celdas.append(pct(a, n))
        if n:
            lineas.append(f"| {c} | {n} | " + " | ".join(celdas) + " |")
    lineas.append(f"| **Total (sin «limites»)** | {r[0][1]['total'][2]} | " +
                  " | ".join(f"**{pct(s['total'][0], s['total'][2])}**" for _, s in r) + " |")
    lineas += ["", "## Coincidencia exacta de la consulta y latencia", "",
               "| Motor | Coincidencia exacta | Latencia media | Mediana | p95 | Respaldo a reglas |", "|---|---|---|---|---|---|"]
    for (x, s), nom in zip(r, nombres):
        lineas.append(f"| {nom} | {pct(s['total'][1], s['total'][2])} | {s['latencia_media']:.3f} s | "
                      f"{s['latencia_mediana']:.3f} s | {s['latencia_p95']:.3f} s | {s['respaldo']} |")
    for (x, s), nom in zip(r, nombres):
        fallos = [f for f in x["filas"] if not f["acierto"]]
        lineas += ["", f"## Fallos — {nom} ({len(fallos)})", "", "| # | Categoría | Pregunta | Consulta generada |", "|---|---|---|---|"]
        for f in fallos:
            lineas.append(f"| {f['id']} | {f['categoria']} | {f['pregunta']} | `{f['consulta_generada'] or 'no entendida'}` |")
    return "\n".join(lineas) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", choices=["reglas", "llm", "ambos"], default="reglas")
    ap.add_argument("--modelo", help="modelo del LLM (por defecto, LLM_MODELO del .env)")
    ap.add_argument("--categorias", nargs="*", help="evaluar solo estas categorías")
    ap.add_argument("--rag", action="store_true",
                    help="evalúa además el LLM con RAG (ejemplos recuperados del banco de desarrollo)")
    ap.add_argument("--banco", choices=["desarrollo", "prueba", "rag"], default="desarrollo",
                    help="desarrollo: banco usado para ajustar el sistema; prueba: banco reservado para las cifras finales; "
                         "rag: paráfrasis de las semillas (experimento de adaptación, ver generar_semillas_rag.py)")
    ap.add_argument("--rag-k", type=int, help="nº de ejemplos recuperados por el RAG (por defecto 8)")
    ap.add_argument("--rag-umbral", type=float, help="similitud mínima de los ejemplos recuperados (por defecto 0.35)")
    args = ap.parse_args()

    fichero = {"desarrollo": "banco_preguntas.json", "prueba": "banco_prueba.json", "rag": "banco_rag.json"}[args.banco]
    banco = json.loads((RAIZ / fichero).read_text(encoding="utf-8"))
    if args.categorias:
        banco = [p for p in banco if p["categoria"] in args.categorias]
    tickets = json.loads((RAIZ / "tickets_prueba.json").read_text(encoding="utf-8"))
    motores = ["reglas", "llm"] if args.motor == "ambos" else [args.motor]
    opciones_rag = {"rag_k": args.rag_k, "rag_umbral": args.rag_umbral}
    if args.banco == "rag":
        # LLM sin RAG / con RAG (base = banco de desarrollo) / con RAG + semillas (lo aprendido de los 👍)
        resultados = [evaluar("llm", banco, tickets, args.modelo),
                      evaluar("llm", banco, tickets, args.modelo, rag=True, **opciones_rag),
                      evaluar("llm", banco, tickets, args.modelo, rag=True, semillas=True, **opciones_rag)]
    else:
        resultados = [evaluar(m, banco, tickets, args.modelo) for m in motores]
        if args.rag:
            resultados.append(evaluar("llm", banco, tickets, args.modelo, rag=True, **opciones_rag))

    salida = RAIZ / "resultados"
    salida.mkdir(exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M")
    for res in resultados:
        nombre = res["motor"] if res["motor"] == "reglas" else f"{res['motor'].replace('+', '_')}_{res['modelo'].replace(':', '-')}"
        with open(salida / f"chat_{args.banco}_{nombre}_{marca}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res["filas"][0]), delimiter=";")
            w.writeheader()
            w.writerows(res["filas"])
    md = informe_md(resultados)
    md = md.replace("# Evaluación del asistente", f"# Evaluación del asistente (banco de {args.banco})", 1)
    (salida / f"resumen_chat_{args.banco}_{marca}.md").write_text(md, encoding="utf-8")
    print("\n" + md.split("## Fallos")[0])
    print(f"📄 Resultados en {salida}")


if __name__ == "__main__":
    main()
