"""Recuperação semântica com modelos de linguagem (Aulas 09 e 10) e reranking (Aula 11).

Cada letra é dividida em trechos de ~80 palavras (com sobreposição) e cada
trecho vira um vetor (embedding) de 384 dimensões. A consulta também vira um
vetor e a nota de cada música é a MAIOR similaridade de cosseno entre a
consulta e os trechos dela. Assim, "música triste sobre saudade de casa" acha
músicas que falam disso mesmo sem usar essas palavras, e em qualquer idioma
(o modelo é multilíngue).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import config

PALAVRAS_POR_TRECHO = 80
SOBREPOSICAO = 20
MAX_TRECHOS = 8


def dividir_em_trechos(texto: str, palavras_por_trecho: int = PALAVRAS_POR_TRECHO,
                       sobreposicao: int = SOBREPOSICAO, max_trechos: int = MAX_TRECHOS) -> list[str]:
    palavras = texto.split()
    if not palavras:
        return []
    passo = palavras_por_trecho - sobreposicao
    trechos = []
    for inicio in range(0, len(palavras), passo):
        trechos.append(" ".join(palavras[inicio:inicio + palavras_por_trecho]))
        if inicio + palavras_por_trecho >= len(palavras) or len(trechos) >= max_trechos:
            break
    return trechos


class BuscadorSemantico:
    def __init__(self, embeddings: np.ndarray, inicio_doc: np.ndarray,
                 nome_modelo: str = config.MODELO_EMBEDDINGS):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.inicio_doc = np.asarray(inicio_doc, dtype=np.int64)  # trechos do doc d: [inicio[d], inicio[d+1])
        self.nome_modelo = nome_modelo
        self.n_docs = len(self.inicio_doc) - 1
        self._modelo = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def modelo(self):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer
            self._modelo = SentenceTransformer(self.nome_modelo, device="cpu")
        return self._modelo

    @staticmethod
    def preparar_trechos(textos: Sequence[str]) -> tuple[list[str], np.ndarray]:
        trechos: list[str] = []
        inicio = np.zeros(len(textos) + 1, dtype=np.int64)
        for d, texto in enumerate(textos):
            partes = dividir_em_trechos(texto) or [texto.strip() or "-"]
            trechos.extend(partes)
            inicio[d + 1] = len(trechos)
        return trechos, inicio

    @classmethod
    def construir(cls, textos: Sequence[str], nome_modelo: str = config.MODELO_EMBEDDINGS,
                  tamanho_lote: int = 64, mostrar_progresso: bool = True) -> "BuscadorSemantico":
        trechos, inicio = cls.preparar_trechos(textos)
        buscador = cls(np.zeros((0, 1), dtype=np.float32), inicio, nome_modelo)
        emb = buscador.modelo.encode(trechos, batch_size=tamanho_lote, normalize_embeddings=True,
                                     show_progress_bar=mostrar_progresso, convert_to_numpy=True)
        buscador.embeddings = np.ascontiguousarray(emb, dtype=np.float32)
        return buscador

    def vetor_consulta(self, consulta: str) -> np.ndarray:
        if consulta not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            v = self.modelo.encode([consulta], normalize_embeddings=True, convert_to_numpy=True)[0]
            self._cache[consulta] = v.astype(np.float32)
        return self._cache[consulta]

    def pontuar(self, consulta: str) -> np.ndarray:
        """Similaridade de cosseno de cada música (máximo entre os seus trechos)."""
        sims = self.embeddings @ self.vetor_consulta(consulta)
        return np.maximum.reduceat(sims, self.inicio_doc[:-1])

    def melhor_trecho(self, consulta: str, doc: int) -> int:
        a, b = self.inicio_doc[doc], self.inicio_doc[doc + 1]
        return int(np.argmax(self.embeddings[a:b] @ self.vetor_consulta(consulta)))

    def salvar(self, caminho: str | Path) -> None:
        np.savez(caminho, embeddings=self.embeddings.astype(np.float16),
                 inicio_doc=self.inicio_doc, nome_modelo=np.array(self.nome_modelo))

    @classmethod
    def carregar(cls, caminho: str | Path) -> "BuscadorSemantico":
        with np.load(caminho) as d:
            return cls(d["embeddings"], d["inicio_doc"], str(d["nome_modelo"]))


class Reordenador:
    """Reranking com cross-encoder: lê consulta e trecho JUNTOS e dá uma nota de
    relevância. É mais preciso que comparar vetores, mas bem mais lento, por isso
    só é aplicado aos ~30 primeiros resultados."""

    def __init__(self, nome_modelo: str = config.MODELO_RERANKING):
        self.nome_modelo = nome_modelo
        self._modelo = None

    def pontuar(self, consulta: str, textos: Sequence[str]) -> np.ndarray:
        if self._modelo is None:
            from sentence_transformers import CrossEncoder
            self._modelo = CrossEncoder(self.nome_modelo, device="cpu", max_length=256)
        notas = self._modelo.predict([(consulta, t) for t in textos], batch_size=16,
                                     show_progress_bar=False)
        return np.asarray(notas, dtype=np.float64)
