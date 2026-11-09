"""Modelo vetorial: TF-IDF com similaridade do cosseno (Aula 04).

    peso(t, d) = (1 + log tf(t,d)) · idf(t),      idf(t) = log(N / df(t))
    sim(q, d)  = (q · d) / (|q| · |d|)

As normas |d| de todos os documentos são calculadas uma única vez na construção.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .indice import IndiceInvertido


class ModeloVetorial:
    def __init__(self, indice: IndiceInvertido):
        self.indice = indice
        n = max(indice.n_docs, 1)
        self.idf = np.log(n / np.maximum(indice.df, 1)).astype(np.float64)
        # peso de cada posting e norma de cada documento
        idf_por_posting = np.repeat(self.idf, indice.df)
        pesos = (1.0 + np.log(np.maximum(indice.freqs, 1))) * idf_por_posting
        normas2 = np.bincount(indice.docs, weights=pesos ** 2, minlength=indice.n_docs)
        normas = np.sqrt(normas2)
        normas[normas == 0] = 1.0
        self.normas = normas.astype(np.float32)

    def pontuar(self, termos: Iterable[str]) -> np.ndarray:
        scores = np.zeros(self.indice.n_docs, dtype=np.float32)
        norma_q2 = 0.0
        for termo, f in Counter(termos).items():
            tid = self.indice.vocabulario.get(termo)
            if tid is None:
                continue
            w_q = (1.0 + np.log(f)) * self.idf[tid]
            if w_q == 0:
                continue
            docs, tf = self.indice.postings_por_id(tid)
            w_d = (1.0 + np.log(tf)) * self.idf[tid]
            scores[docs] += (w_q * w_d).astype(np.float32)
            norma_q2 += w_q ** 2
        if norma_q2 == 0:
            return scores
        return scores / (self.normas * np.float32(np.sqrt(norma_q2)))
