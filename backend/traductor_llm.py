"""
Traductor de lenguaje natural a `Consulta` mediante un LLM local.

El modelo de lenguaje NO responde a la pregunta ni calcula nada: solo convierte
la pregunta del usuario en una consulta estructurada (JSON) que después ejecuta
el motor de consultas con los datos reales. Así se aprovecha la flexibilidad del
LLM para entender el lenguaje sin arriesgarse a que invente cifras.

Funciona con cualquier servidor compatible con la API de OpenAI que se ejecute en
local, por ejemplo Ollama (http://localhost:11434/v1) o LM Studio
(http://localhost:1234/v1). Se configura con variables de entorno (backend/.env):

    LLM_URL=http://localhost:11434/v1
    LLM_MODELO=qwen2.5:7b
    LLM_ACTIVADO=true
"""
import json
import logging
import os
import re
import time
from datetime import date
from typing import List, Optional

import httpx
from pydantic import ValidationError

from motor_consultas import Consulta

logging.getLogger("httpx").setLevel(logging.WARNING)   # no llenar la consola con cada petición al LLM

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

EJEMPLOS = [
    ("¿Cuánto he gastado en lácteos este mes?",
     {"medida": "gasto", "operacion": "total", "filtros": {"concepto": "lácteos", "periodo": "este mes"}}),
    ("¿Cuántas veces compro yogures a la semana?",
     {"medida": "compras", "operacion": "media", "por": "semana", "filtros": {"concepto": "yogures"}}),
    ("¿Cuál ha sido mi compra más cara del mes?",
     {"medida": "gasto", "operacion": "maximo", "agrupar_por": "ticket", "filtros": {"periodo": "este mes"}}),
    ("¿De qué categoría compro más productos?",
     {"medida": "unidades", "operacion": "ranking", "agrupar_por": "categoria"}),
    ("¿En qué me gasto más el dinero?",
     {"medida": "gasto", "operacion": "ranking", "agrupar_por": "categoria"}),
    ("¿Cuál es el producto que más he comprado?",
     {"medida": "unidades", "operacion": "ranking", "agrupar_por": "producto"}),
    ("¿Cuántas devoluciones he hecho?",
     {"medida": "compras", "operacion": "total", "filtros": {"tipo_ticket": "devoluciones"}}),
    ("¿Qué productos me han devuelto?",
     {"medida": "gasto", "operacion": "lista", "agrupar_por": "producto", "filtros": {"tipo_ticket": "devoluciones"}}),
    ("¿Cuánto dinero me han devuelto este año?",
     {"medida": "gasto", "operacion": "total", "filtros": {"tipo_ticket": "devoluciones", "periodo": "este año"}}),
    ("¿Qué día de la semana gasto más?",
     {"medida": "gasto", "operacion": "ranking", "agrupar_por": "dia_semana"}),
    ("¿Cuánto gasto de media cuando compro los sábados?",
     {"medida": "gasto", "operacion": "media", "por": "compra", "filtros": {"dias_semana": [5]}}),
    ("¿Cuánto gasto en carne al mes?",
     {"medida": "gasto", "operacion": "media", "por": "mes", "filtros": {"concepto": "carne"}}),
    ("¿Ha subido el precio de la leche?",
     {"medida": "precio", "operacion": "evolucion", "filtros": {"concepto": "leche"}}),
    ("Enséñame las compras de más de 50 euros del año pasado",
     {"medida": "compras", "operacion": "lista", "agrupar_por": "ticket", "filtros": {"importe_min": 50, "periodo": "el año pasado"}}),
    ("¿Cuándo compré pizza por última vez?",
     {"medida": "compras", "operacion": "ultimo", "filtros": {"concepto": "pizza"}}),
    ("¿Cómo ha evolucionado mi gasto en fruta?",
     {"medida": "gasto", "operacion": "evolucion", "agrupar_por": "mes", "filtros": {"concepto": "fruta"}}),
    ("¿Cuál es el producto más caro que he comprado?",
     {"medida": "precio", "operacion": "maximo", "agrupar_por": "producto"}),
    ("Los 3 productos en los que menos gasto",
     {"medida": "gasto", "operacion": "ranking", "agrupar_por": "producto", "orden": "asc", "limite": 3}),
    ("¿Qué tiempo hace mañana?", {"tipo": "fuera_de_ambito"}),
    ("¡Gracias!", {"tipo": "saludo"}),
]


def _prompt_sistema(hoy: date, categorias: List[str]) -> str:
    ejemplos = "\n".join(f'"{p}" -> {json.dumps(c, ensure_ascii=False)}' for p, c in EJEMPLOS)
    return f"""Eres el traductor de un asistente de gastos de supermercado. Conviertes la pregunta del usuario en una consulta JSON.
NO respondes a la pregunta ni calculas nada: devuelves solo el JSON.

Hoy es {DIAS[hoy.weekday()]} {hoy.strftime('%d/%m/%Y')}.

Campos (omite los que tengan su valor por defecto):
- tipo: "consulta" (por defecto) | "saludo" (saludos, agradecimientos, qué puedes hacer) | "fuera_de_ambito" (nada que ver con compras o gastos)
- medida: "gasto" (euros, por defecto) | "unidades" (cantidad de productos) | "compras" (número de tickets o veces que se compra) | "precio" (precio por unidad)
- operacion: "total" (por defecto) | "media" | "maximo" | "minimo" | "ranking" | "evolucion" | "ultimo" (última vez) | "lista" (mostrar tickets o productos)
- agrupar_por: "producto" | "categoria" | "subcategoria" | "ticket" | "mes" | "semana" | "dia_semana"
- por (solo con "media"): "compra" | "dia" | "semana" | "mes"
- orden: "desc" (los que más, por defecto) | "asc" (los que menos); limite: número de resultados (5 por defecto)
- filtros:
  - concepto: producto o tipo de producto con las palabras del usuario ("lácteos", "yogures", "pan de molde", "carne")
  - categoria: solo si nombra exactamente una de estas categorías: {"; ".join(categorias)}
  - periodo: la expresión temporal copiada de la pregunta ("este mes", "el mes pasado", "en marzo", "en 2025", "los últimos 3 meses")
  - tipo_ticket: "devoluciones" si pregunta por devoluciones o reembolsos; "compras" si las excluye; "todos" por defecto
  - dias_semana: días concretos (0=lunes ... 6=domingo), p. ej. "los sábados" -> [5]
  - importe_min / importe_max: importe del ticket en euros ("compras de más de 50 €" -> importe_min 50)
Si la pregunta continúa la anterior ("¿y el mes pasado?", "¿y de fruta?"), repite la consulta anterior cambiando solo lo que se pide.

Ejemplos:
{ejemplos}"""


def _extraer_json(texto: str) -> dict:
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio < 0 or fin < 0:
        raise ValueError("La respuesta no contiene JSON")
    return json.loads(texto[inicio:fin + 1])


class TraductorLLM:
    def __init__(self):
        self.url = os.environ.get("LLM_URL", "http://localhost:11434/v1").rstrip("/")
        self.modelo = os.environ.get("LLM_MODELO", "qwen2.5:7b")
        self.activado = os.environ.get("LLM_ACTIVADO", "true").lower() in ("1", "true", "si", "sí", "yes")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "90"))
        self._caido_hasta = 0.0          # si el servidor no responde, no se reintenta durante un rato
        self._usar_esquema = True        # algunos servidores no aceptan json_schema: se pasa a json_object

    @property
    def disponible(self) -> bool:
        return self.activado and time.time() >= self._caido_hasta

    def estado(self) -> dict:
        try:
            r = httpx.get(f"{self.url}/models", timeout=3)
            modelos = [m.get("id") for m in r.json().get("data", [])]
            return {"activado": self.activado, "conectado": True, "modelo": self.modelo,
                    "modelo_instalado": self.modelo in modelos or any(m and m.startswith(self.modelo) for m in modelos),
                    "modelos": modelos}
        except Exception as e:
            return {"activado": self.activado, "conectado": False, "modelo": self.modelo, "error": str(e)[:200]}

    def calentar(self):
        """Carga el modelo en memoria al arrancar el servidor para que la primera pregunta no tarde."""
        if not self.activado:
            return
        try:
            httpx.post(f"{self.url}/chat/completions", timeout=self.timeout,
                       json={"model": self.modelo, "messages": [{"role": "user", "content": "hola"}], "max_tokens": 1})
            print(f"🤖 LLM local listo ({self.modelo})")
        except Exception as e:
            print(f"⚠️ LLM local no disponible ({type(e).__name__}): el chat usará el motor de reglas")

    def traducir(self, pregunta: str, hoy: date, categorias: List[str], historial: Optional[List[dict]] = None) -> Optional[Consulta]:
        """Devuelve la Consulta o None si el LLM no está disponible o no produce una consulta válida."""
        if not self.disponible:
            return None
        mensajes = [{"role": "system", "content": _prompt_sistema(hoy, categorias)}]
        for h in (historial or [])[-3:]:
            if h.get("pregunta") and h.get("consulta"):
                mensajes.append({"role": "user", "content": h["pregunta"]})
                mensajes.append({"role": "assistant", "content": json.dumps(h["consulta"], ensure_ascii=False)})
        mensajes.append({"role": "user", "content": pregunta})

        for intento in range(2):
            try:
                texto = self._llamar(mensajes)
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                print(f"⚠️ LLM no disponible ({type(e).__name__}); se usa el motor de reglas durante 60 s")
                self._caido_hasta = time.time() + 60
                return None
            except Exception as e:
                print(f"⚠️ Error llamando al LLM: {e}")
                return None
            try:
                return Consulta.model_validate(_extraer_json(texto))
            except (ValueError, ValidationError) as e:
                # Segundo intento: se le devuelve el error para que lo corrija
                mensajes += [{"role": "assistant", "content": texto},
                             {"role": "user", "content": f"Ese JSON no es válido ({str(e)[:300]}). Devuelve solo el JSON corregido."}]
        return None

    def _llamar(self, mensajes) -> str:
        cuerpo = {"model": self.modelo, "messages": mensajes, "temperature": 0, "max_tokens": 400}
        if self._usar_esquema:
            cuerpo["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "consulta", "schema": Consulta.model_json_schema()}}
        else:
            cuerpo["response_format"] = {"type": "json_object"}
        r = httpx.post(f"{self.url}/chat/completions", json=cuerpo, timeout=self.timeout)
        if r.status_code == 400 and self._usar_esquema:
            self._usar_esquema = False
            return self._llamar(mensajes)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
