"""Modelo probabilístico BM25 (Aula 03).

    score(d, q) = Σ_{t ∈ q}  idf(t) · tf(t,d)·(k1 + 1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

    idf(t) = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )   (variante usada no Lucene/Elasticsearch,
                                                            sempre positiva)
* k1 controla a saturação do tf (repetir a palavra 20 vezes não vale 20 vezes mais);
* b controla a normalização pelo tamanho do documento (letras longas não são favorecidas).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .indice import IndiceInvertido


class BM25:
    def __init__(self, indice: IndiceInvertido, k1: float = 1.2, b: float = 0.75):
        self.indice = indice
        self.k1 = k1
        self.b = b
        n, df = indice.n_docs, indice.df.astype(np.float64)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        media = max(indice.comprimento_medio, 1e-9)
        # parte do denominador que só depende do documento: pré-calculada
        self._norma_doc = (k1 * (1 - b + b * indice.comprimentos / media)).astype(np.float32)

    def pontuar(self, termos: Iterable[str]) -> np.ndarray:
        """Score BM25 de TODOS os documentos (0 para quem não tem nenhum termo).

        Termos repetidos na consulta contam uma vez só (ex.: "yeah yeah yeah").
        """
        scores = np.zeros(self.indice.n_docs, dtype=np.float32)
        for termo in set(termos):
            tid = self.indice.vocabulario.get(termo)
            if tid is None:
                continue
            docs, tf = self.indice.postings_por_id(tid)
            tf = tf.astype(np.float32)
            scores[docs] += self.idf[tid] * tf * (self.k1 + 1) / (tf + self._norma_doc[docs])
        return scores
