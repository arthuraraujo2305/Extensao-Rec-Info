"""Busca híbrida (Aula 11): combina os rankings do BM25 e do modelo semântico.

Dois métodos de fusão:

* **CombSUM** (padrão): soma ponderada das notas normalizadas (min-max) de cada lista,
      nota(d) = α · bm25_norm(d) + (1 − α) · semantico_norm(d)
  Preserva a *margem* entre as notas: quando o BM25 tem certeza (a música certa
  com nota muito acima das outras), ela continua no topo.

* **RRF** (Reciprocal Rank Fusion): nota(d) = Σ 1 / (k + posição(d))
  Usa só as posições. É simples e não precisa normalizar, mas trata o 1º lugar de
  um ranking "confiante" igual ao 1º lugar de um ranking "em dúvida".

Na nossa avaliação (busca por trecho), o RRF deixou o híbrido bem pior que o BM25
sozinho; o CombSUM com α = 0,7 (escolhido num conjunto de desenvolvimento separado)
fica praticamente empatado com o BM25 e ainda aproveita o sinal semântico.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

PESO_LEXICO = 0.7  # α do CombSUM


def fusao_rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    notas: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for posicao, doc in enumerate(ranking, start=1):
            notas[int(doc)] += 1.0 / (k + posicao)
    return sorted(notas.items(), key=lambda x: (-x[1], x[0]))


def fusao_combsum(rankings: Sequence[Sequence[tuple[int, float]]],
                  pesos: Sequence[float]) -> list[tuple[int, float]]:
    """Cada ranking é [(doc, nota)]. Documento ausente de uma lista conta 0 nela."""
    notas: dict[int, float] = defaultdict(float)
    for ranking, peso in zip(rankings, pesos):
        if not ranking:
            continue
        normalizadas = normalizar_minmax([s for _, s in ranking])
        for (doc, _), nota in zip(ranking, normalizadas):
            notas[int(doc)] += peso * float(nota)
    return sorted(notas.items(), key=lambda x: (-x[1], x[0]))


def normalizar_minmax(valores) -> np.ndarray:
    """Leva os valores para [0, 1]. Se forem todos iguais, todos viram 1."""
    x = np.asarray(valores, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)
