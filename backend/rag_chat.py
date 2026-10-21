"""
RAG (Retrieval-Augmented Generation) para el chat de gastos.

El LLM local no "sabe" nada de cómo se formulan las preguntas en esta aplicación ni de lo que el usuario
ha preguntado antes. Con RAG, antes de llamar al LLM se RECUPERA información relevante y se AÑADE al
prompt para que genere mejor la consulta:

  1. Ejemplos validados parecidos a la pregunta (few-shot dinámico). La base de conocimiento son pares
     pregunta -> consulta correcta: el banco de desarrollo de la validación (116, revisados a mano) y
     las respuestas que el usuario marca como útiles (👍) en la web, que se van añadiendo solas.
     Se buscan los k más parecidos por similitud de embeddings (bge-m3) o, si Ollama no está, TF-IDF.
  2. Historial persistente de la conversación: cada pregunta, su consulta y la respuesta se guardan en
     MongoDB (colección chat_historial), así el chat no se pierde al recargar la página y las preguntas
     de continuación ("¿y el mes pasado?") funcionan con el historial guardado.

Lo que NO hace (a propósito): pasarle al LLM los tickets para que sume importes. El LLM solo traduce la
pregunta; las cifras las sigue calculando el motor de consultas sobre los datos reales.
"""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RUTA_BANCO = Path(__file__).parent / "validacion" / "banco_preguntas.json"
K_EJEMPLOS = 8
SIMILITUD_MINIMA = 0.35


def _texto(t: str) -> str:
    from asistente_gastos import sin_tildes
    return sin_tildes(t or "").strip(" ¿?¡!.")


class BaseEjemplos:
    """Índice en memoria de pares (pregunta, consulta) validados, con búsqueda por similitud."""

    def __init__(self, cliente_embeddings=None, k: int = K_EJEMPLOS, umbral: float = SIMILITUD_MINIMA):
        self.emb = cliente_embeddings
        self.k, self.umbral = k, umbral
        self.preguntas: List[str] = []
        self.consultas: List[dict] = []
        self.origen: List[str] = []
        self.X = None
        self._vec = None
        self._lock = threading.Lock()

    def cargar_semillas(self, ruta: Path) -> int:
        """Pares pregunta -> consulta que simulan respuestas marcadas como útiles (👍) por el usuario."""
        semillas = json.loads(ruta.read_text(encoding="utf-8"))
        self.añadir([(s["pregunta"], s["esperado"]) for s in semillas], "usuario")
        return len(semillas)

    def cargar_banco(self, ruta: Path = RUTA_BANCO):
        if not ruta.exists():
            return 0
        banco = json.loads(ruta.read_text(encoding="utf-8"))
        # Las continuaciones dependen de la pregunta anterior: no sirven como ejemplo suelto
        pares = [(p["pregunta"], p["esperado"]) for p in banco if not p.get("historial")]
        self.añadir(pares, "banco")
        return len(pares)

    def añadir(self, pares: List[Tuple[str, dict]], origen: str):
        with self._lock:
            for p, c in pares:
                if p and c and _texto(p) not in {_texto(x) for x in self.preguntas}:
                    self.preguntas.append(p)
                    self.consultas.append(c)
                    self.origen.append(origen)
            self._reindexar()

    def quitar(self, pregunta: str, origen: str):
        with self._lock:
            idx = [i for i, p in enumerate(self.preguntas) if _texto(p) == _texto(pregunta) and self.origen[i] == origen]
            for i in reversed(idx):
                del self.preguntas[i], self.consultas[i], self.origen[i]
            if idx:
                self._reindexar()

    def _vectores(self, textos: List[str]) -> np.ndarray:
        if self.emb is not None:
            try:
                return self.emb.vectores([_texto(t).lower() for t in textos])
            except Exception:
                self.emb = None
                print("⚠️ RAG: embeddings no disponibles, se usa TF-IDF para recuperar ejemplos")
        return normalize(self._vec.transform([_texto(t) for t in textos])).toarray()

    def _reindexar(self):
        if not self.preguntas:
            self.X = None
            return
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(
            [_texto(p) for p in self.preguntas])
        self.X = self._vectores(self.preguntas)

    def recuperar(self, pregunta: str, k: Optional[int] = None, excluir: Optional[str] = None) -> List[Tuple[str, dict]]:
        """Los k ejemplos más parecidos a la pregunta (sin la propia pregunta si está en la base)."""
        with self._lock:
            if self.X is None:
                return []
            v = self._vectores([pregunta])[0]
            if v.shape[0] != self.X.shape[1]:          # se ha pasado de embeddings a TF-IDF
                self.X = self._vectores(self.preguntas)
                v = self._vectores([pregunta])[0]
            sims = self.X @ v
            propia = _texto(excluir or pregunta)
            orden = [i for i in np.argsort(-sims) if _texto(self.preguntas[i]) != propia and sims[i] >= self.umbral]
            return [(self.preguntas[i], self.consultas[i]) for i in orden[:k or self.k]]


class MemoriaChat:
    """Historial persistente del chat en MongoDB + base de ejemplos para el RAG."""

    def __init__(self, db, base: BaseEjemplos):
        self.db = db
        self.base = base

    async def iniciar(self):
        n_banco = self.base.cargar_banco()
        utiles = [(d["pregunta"], d["consulta"]) async for d in
                  self.db.chat_historial.find({"util": True, "consulta": {"$ne": None}}, {"_id": 0, "pregunta": 1, "consulta": 1})]
        self.base.añadir(utiles, "usuario")
        modo = f"embeddings {self.base.emb.modelo}" if self.base.emb else "TF-IDF"
        print(f"📚 RAG listo: {n_banco} ejemplos del banco validado + {len(utiles)} valorados por el usuario ({modo})")

    async def guardar(self, conversacion: str, pregunta: str, resultado: dict) -> str:
        interp = resultado.get("interpretacion") or {}
        doc = {"id": str(uuid.uuid4()), "conversacion": conversacion or "general",
               "fecha": datetime.now(timezone.utc).isoformat(), "pregunta": pregunta,
               "consulta": interp.get("consulta"), "motor": interp.get("motor"),
               "respuesta": resultado.get("respuesta", ""), "util": None,
               "resultado": {k: resultado.get(k) for k in ("respuesta", "tabla", "nota", "sugerencias",
                                                            "interpretacion", "filtro_tickets") if resultado.get(k) is not None}}
        await self.db.chat_historial.insert_one(dict(doc))
        return doc["id"]

    async def historial(self, conversacion: str, limite: int = 50) -> List[dict]:
        docs = await self.db.chat_historial.find({"conversacion": conversacion}, {"_id": 0}) \
            .sort("fecha", -1).to_list(limite)
        return list(reversed(docs))

    async def ultimas_consultas(self, conversacion: str, n: int = 3) -> List[dict]:
        """Las últimas preguntas con su consulta, para las preguntas de continuación."""
        docs = await self.db.chat_historial.find(
            {"conversacion": conversacion, "consulta": {"$ne": None}}, {"_id": 0, "pregunta": 1, "consulta": 1}
        ).sort("fecha", -1).to_list(n)
        return list(reversed(docs))

    async def valorar(self, id_mensaje: str, util: bool) -> bool:
        doc = await self.db.chat_historial.find_one({"id": id_mensaje}, {"_id": 0})
        if not doc:
            return False
        await self.db.chat_historial.update_one({"id": id_mensaje}, {"$set": {"util": util}})
        if util and doc.get("consulta"):
            # La respuesta era correcta: el par pregunta -> consulta pasa a la base de ejemplos del RAG.
            # El motor de reglas guarda como periodo la pregunta entera: se deja solo la expresión temporal.
            consulta = json.loads(json.dumps(doc["consulta"]))
            filtros = consulta.get("filtros") or {}
            if filtros.get("periodo") == doc["pregunta"]:
                from asistente_gastos import extraer_periodo
                _, _, etiqueta, _ = extraer_periodo(doc["pregunta"], datetime.now().date())
                filtros["periodo"] = etiqueta
            self.base.añadir([(doc["pregunta"], consulta)], "usuario")
        elif not util:
            self.base.quitar(doc["pregunta"], "usuario")
        return True
