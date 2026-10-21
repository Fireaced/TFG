"""
Cliente de embeddings de texto con un modelo local (Ollama o LM Studio, API compatible con OpenAI).

Un embedding es un vector numérico que representa el significado de un texto: nombres de producto
parecidos en significado ("YOGUR GRIEGO" y "BÍFIDUS NATURAL") quedan cerca aunque no compartan letras,
algo que TF-IDF de caracteres no puede captar.

Configuración (backend/.env):
    EMB_URL=http://localhost:11434/v1      (por defecto, la misma que LLM_URL)
    EMB_MODELO=bge-m3                      (multilingüe; alternativa: nomic-embed-text)

Instalación del modelo en Ollama:  ollama pull bge-m3

Los vectores se guardan en una caché en disco (por modelo y texto) para no recalcularlos.
"""
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np

# Algunos modelos esperan un prefijo que indica la tarea
PREFIJOS = {"nomic-embed-text": "clustering: "}

CACHE_DIR = Path(__file__).parent / "cache_embeddings"


class ClienteEmbeddings:
    def __init__(self, modelo: Optional[str] = None, url: Optional[str] = None, timeout: float = 120):
        self.url = (url or os.environ.get("EMB_URL") or os.environ.get("LLM_URL") or "http://localhost:11434/v1").rstrip("/")
        self.modelo = modelo or os.environ.get("EMB_MODELO", "bge-m3")
        self.timeout = timeout
        self.prefijo = next((p for m, p in PREFIJOS.items() if self.modelo.startswith(m)), "")
        CACHE_DIR.mkdir(exist_ok=True)
        self._ruta_cache = CACHE_DIR / f"{self.modelo.replace(':', '_').replace('/', '_')}.json"
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._sin_guardar = 0
        if self._ruta_cache.exists():
            try:
                self._cache = json.loads(self._ruta_cache.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    @staticmethod
    def _clave(texto: str) -> str:
        return hashlib.sha1(texto.encode("utf-8")).hexdigest()

    def disponible(self) -> bool:
        try:
            self.vectores(["prueba"], guardar=False)
            return True
        except Exception:
            return False

    def vectores(self, textos: List[str], lote: int = 64, guardar: bool = True) -> np.ndarray:
        """Devuelve una matriz (n, d) con los embeddings normalizados (norma 1)."""
        with self._lock:
            return self._vectores(textos, lote, guardar)

    def _vectores(self, textos: List[str], lote: int, guardar: bool) -> np.ndarray:
        pendientes = [t for t in dict.fromkeys(textos) if self._clave(t) not in self._cache]
        for i in range(0, len(pendientes), lote):
            trozo = pendientes[i:i + lote]
            r = httpx.post(f"{self.url}/embeddings", timeout=self.timeout,
                           json={"model": self.modelo, "input": [self.prefijo + t for t in trozo]})
            r.raise_for_status()
            datos = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            for t, d in zip(trozo, datos):
                self._cache[self._clave(t)] = d["embedding"]
        # La caché ocupa decenas de MB: se escribe en disco solo cuando hay bastantes vectores nuevos
        self._sin_guardar += len(pendientes)
        if guardar and (self._sin_guardar >= 50 or len(pendientes) >= 20):
            self._ruta_cache.write_text(json.dumps(self._cache), encoding="utf-8")
            self._sin_guardar = 0
        m = np.array([self._cache[self._clave(t)] for t in textos], dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
