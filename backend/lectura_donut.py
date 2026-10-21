"""
Limpieza de la salida estructurada de Donut antes de usarla (la usan server.py y validacion/evaluar_donut.py).

Donut genera una secuencia de etiquetas <s_campo>valor</s_campo> que se convierte en dict. A veces la
estructura sale rota:
  - en tickets muy largos la secuencia llega al máximo de tokens y no se cierran <s_menu> ni <s_total>,
    con lo que el resultado es una LISTA de productos (con la fecha metida en el primero) en vez de un dict;
  - con imágenes difíciles (fotos) algún campo sale anidado ({"cnt": {"price": "2,80"}}) o como lista.
Se normaliza a {"date", "menu": [dicts de textos], "total": {"total_price"}} y se descartan los campos
que no son texto, para que la aplicación no falle y los datos dudosos vayan a revisión.
"""
from typing import Any

CAMPOS_LINEA = ("cnt", "nm", "weight", "unitprice", "price")


def _texto(v: Any):
    return v.strip() if isinstance(v, str) else None


def normalizar_prediccion(pred: Any) -> dict:
    if isinstance(pred, list):                       # secuencia truncada: lista de productos sin envoltorio
        fecha = next((p.get("date") for p in pred if isinstance(p, dict) and isinstance(p.get("date"), str)), None)
        pred = {"date": fecha, "menu": pred, "truncada": True}
    if not isinstance(pred, dict):
        return {"menu": [], "rota": True}
    if "text_sequence" in pred and "menu" not in pred:
        return pred                                  # sin estructura: lo resuelve el parser defensivo
    menu = pred.get("menu", [])
    menu = [menu] if isinstance(menu, dict) else (menu if isinstance(menu, list) else [])
    lineas, rotos = [], 0
    for it in menu:
        if not isinstance(it, dict):
            rotos += 1
            continue
        limpio = {k: _texto(it.get(k)) for k in CAMPOS_LINEA if _texto(it.get(k))}
        rotos += sum(1 for k in CAMPOS_LINEA if k in it and not _texto(it.get(k)))
        if limpio:
            lineas.append(limpio)
    total = pred.get("total")
    total = {"total_price": _texto(total.get("total_price"))} if isinstance(total, dict) and _texto(total.get("total_price")) \
        else ({"total_price": _texto(total)} if _texto(total) else {})
    salida = {"date": _texto(pred.get("date")), "menu": lineas, "total": total}
    if rotos:
        salida["campos_rotos"] = rotos
    if pred.get("truncada"):
        salida["truncada"] = True
    for k, v in pred.items():                        # se conservan otros campos (p. ej. store_info del modelo antiguo)
        salida.setdefault(k, v)
    return salida
