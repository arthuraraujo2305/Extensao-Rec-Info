"""Dicionários e tolerância a erros (Aula 08).

* distância de edição (Levenshtein + transposição de letras vizinhas);
* índice de k-gramas (bigramas) do vocabulário, para achar rapidamente
  candidatos parecidos sem comparar com o vocabulário inteiro;
* corretor ortográfico: "coracao partdo" -> "coracao partido";
* consultas com curinga: "apaixon*" -> apaixonado, apaixonada, apaixonei…
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np

from .texto import OPERADORES, ErroConsulta, normalizar

_VAZIO = np.zeros(0, dtype=np.int32)


def distancia_edicao(a: str, b: str, limite: int | None = None) -> int:
    """Distância de Damerau-Levenshtein (variante "optimal string alignment").

    Conta inserções, remoções, substituições e trocas de letras vizinhas
    ("aomr" -> "amor" custa 1). Com `limite`, para cedo e devolve limite+1.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if limite is not None and abs(la - lb) > limite:
        return limite + 1
    if la == 0 or lb == 0:
        return max(la, lb)
    anterior2: list[int] = []
    anterior = list(range(lb + 1))
    for i in range(1, la + 1):
        atual = [i] + [0] * lb
        for j in range(1, lb + 1):
            custo = 0 if a[i - 1] == b[j - 1] else 1
            atual[j] = min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + custo)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                atual[j] = min(atual[j], anterior2[j - 2] + 1)
        if limite is not None and min(atual) > limite and min(anterior) > limite:
            return limite + 1
        anterior2, anterior = anterior, atual
    return anterior[lb]


def kgramas(termo: str, k: int = 2) -> set[str]:
    """Bigramas com marcadores de início/fim: 'amor' -> {$a, am, mo, or, r$}."""
    t = f"${termo}$"
    return {t[i:i + k] for i in range(len(t) - k + 1)}


class IndiceKGramas:
    """k-grama -> lista dos termos do vocabulário que o contêm."""

    def __init__(self, termos: Sequence[str], k: int = 2):
        self.k = k
        self.termos = list(termos)
        listas: dict[str, list[int]] = defaultdict(list)
        self.n_gramas = np.empty(len(self.termos), dtype=np.int32)
        for i, termo in enumerate(self.termos):
            gramas = kgramas(termo, k)
            self.n_gramas[i] = len(gramas)
            for g in gramas:
                listas[g].append(i)
        self._listas = {g: np.asarray(v, dtype=np.int32) for g, v in listas.items()}

    def __contains__(self, grama: str) -> bool:
        return grama in self._listas

    def contar_compartilhados(self, gramas: set[str]) -> tuple[np.ndarray, np.ndarray]:
        """(ids dos termos, quantos dos `gramas` cada um contém)."""
        arrays = [self._listas[g] for g in gramas if g in self._listas]
        if not arrays:
            return _VAZIO, _VAZIO
        return np.unique(np.concatenate(arrays), return_counts=True)


# --------------------------------------------------------------------------
# Corretor ortográfico
# --------------------------------------------------------------------------
_PALAVRA = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


class CorretorOrtografico:
    """Sugere a palavra do vocabulário mais parecida com um termo desconhecido.

    1. candidatos = termos que compartilham bigramas (coeficiente de Jaccard);
    2. filtra por distância de edição (≤1 para palavras de até 4 letras, ≤2 acima)
       e fica só com os de MENOR distância;
    3. correção sensível ao contexto: se a consulta tem outras palavras, escolhe o
       candidato que mais aparece junto com elas nas letras
       ("coracao partdo" -> "partido", e não "parado");
    4. desempate final: frequência no corpus + bigramas em comum.

    `associacao(candidato, contexto) -> float` é fornecida pelo motor, que tem
    acesso às listas de postings.
    """

    def __init__(self, frequencias: Mapping[str, int], min_frequencia: int = 2, k: int = 2,
                 associacao: Callable[[str, list[str]], float] | None = None):
        self.frequencias = dict(frequencias)
        termos = [t for t, f in self.frequencias.items()
                  if f >= min_frequencia and len(t) >= 2 and not t.isdigit()]
        self._freq = np.array([self.frequencias[t] for t in termos], dtype=np.int64)
        self.kgramas = IndiceKGramas(termos, k)
        self.associacao = associacao

    @staticmethod
    def distancia_maxima(termo: str) -> int:
        if len(termo) <= 2:
            return 0
        return 1 if len(termo) <= 4 else 2

    def conhecido(self, termo: str) -> bool:
        return self.frequencias.get(termo, 0) >= 3

    def candidatos(self, termo: str) -> list[tuple[str, int, float, int]]:
        """[(candidato, distância, jaccard, frequência)] com a menor distância encontrada."""
        dmax = self.distancia_maxima(termo)
        if dmax == 0 or termo.isdigit():
            return []
        gramas = kgramas(termo, self.kgramas.k)
        ids, compartilhados = self.kgramas.contar_compartilhados(gramas)
        if ids.size == 0:
            return []
        jaccard = compartilhados / (len(gramas) + self.kgramas.n_gramas[ids] - compartilhados)
        mantidos = jaccard >= 0.2
        ids, jaccard = ids[mantidos], jaccard[mantidos]
        ordem = np.argsort(-jaccard, kind="stable")[:300]
        encontrados = []
        for i, jac in zip(ids[ordem], jaccard[ordem]):
            candidato = self.kgramas.termos[i]
            if candidato == termo or abs(len(candidato) - len(termo)) > dmax:
                continue
            d = distancia_edicao(termo, candidato, dmax)
            if d <= dmax:
                encontrados.append((candidato, d, float(jac), int(self._freq[i])))
        if not encontrados:
            return []
        menor = min(d for _, d, _, _ in encontrados)
        return [c for c in encontrados if c[1] == menor]

    def sugerir(self, termo: str, contexto: Sequence[str] = ()) -> str | None:
        """Correção para `termo` (já normalizado) ou None se não precisar/não achar."""
        freq_atual = self.frequencias.get(termo, 0)
        if freq_atual >= 3:
            return None
        candidatos = self.candidatos(termo)
        if not candidatos:
            return None

        def prioridade(c):  # frequência (prior) + semelhança de bigramas
            return math.log(c[3]) + 4 * c[2]

        contexto = [t for t in contexto if t != termo]
        notas = {}
        if self.associacao is not None and contexto and len(candidatos) > 1:
            notas = {c[0]: self.associacao(c[0], contexto) for c in candidatos}
        melhor = max(candidatos, key=lambda c: (notas.get(c[0], 0.0), prioridade(c)))
        # palavra existe mas é raríssima: só troca se a sugestão for MUITO mais comum
        if freq_atual > 0 and (melhor[1] > 1 or melhor[3] < 50 * freq_atual):
            return None
        return melhor[0]

    def corrigir_consulta(self, consulta: str) -> tuple[str, bool]:
        """Corrige palavra por palavra, preservando operadores, campos e curingas."""
        palavras = []  # (início, fim, palavra, pode_corrigir)
        for m in _PALAVRA.finditer(consulta):
            ini, fim = m.span()
            protegida = (m.group(0) in OPERADORES
                         or (fim < len(consulta) and consulta[fim] in ":*")
                         or (ini > 0 and consulta[ini - 1] == "*"))
            palavras.append((ini, fim, m.group(0), not protegida))
        contexto = [normalizar(p) for _, _, p, ok in palavras if ok and self.conhecido(normalizar(p))]

        partes, ultimo, mudou = [], 0, False
        for ini, fim, palavra, ok in palavras:
            sugestao = self.sugerir(normalizar(palavra), contexto) if ok else None
            partes.append(consulta[ultimo:ini])
            partes.append(sugestao if sugestao is not None else palavra)
            mudou |= sugestao is not None
            ultimo = fim
        partes.append(consulta[ultimo:])
        return "".join(partes), mudou


# --------------------------------------------------------------------------
# Curingas
# --------------------------------------------------------------------------
class ExpansorCuringa:
    """Expande 'amo*' para os termos do vocabulário que casam com o padrão.

    Usa o índice de bigramas: '$amo*r$' -> bigramas {$a, am, mo, r$}; os
    candidatos precisam ter todos eles e depois passam por um filtro exato.
    """

    def __init__(self, termos: Sequence[str], frequencias: np.ndarray | None = None, k: int = 2):
        self.kgramas = IndiceKGramas(termos, k)
        self.frequencias = (np.asarray(frequencias) if frequencias is not None
                            else np.ones(len(termos), dtype=np.int64))

    def expandir(self, padrao: str, limite: int = 200) -> list[str]:
        padrao = re.sub(r"[^a-z0-9*]", "", normalizar(padrao))
        if "*" not in padrao:
            return [padrao] if padrao else []
        k = self.kgramas.k
        gramas: set[str] = set()
        for parte in f"${padrao}$".split("*"):
            gramas |= {parte[i:i + k] for i in range(len(parte) - k + 1)}
        if not gramas:
            raise ErroConsulta(f"O curinga '{padrao}' é genérico demais: escreva pelo menos uma letra.")
        if any(g not in self.kgramas for g in gramas):
            return []
        ids, cont = self.kgramas.contar_compartilhados(gramas)
        ids = ids[cont == len(gramas)]
        regex = re.compile("^" + ".*".join(re.escape(p) for p in padrao.split("*")) + "$")
        ids = np.array([i for i in ids if regex.match(self.kgramas.termos[i])], dtype=np.int64)
        if ids.size == 0:
            return []
        ids = ids[np.argsort(-self.frequencias[ids], kind="stable")[:limite]]
        return [self.kgramas.termos[i] for i in ids]
