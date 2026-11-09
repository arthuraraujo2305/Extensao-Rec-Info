"""Pré-processamento de texto (Aula 07 – Indexação).

Decisões de projeto:

* tudo em minúsculas e sem acentos: "Coração" e "coracao" viram o mesmo termo,
  o que já resolve boa parte das buscas digitadas sem acento;
* apóstrofos são removidos ("don't" -> "dont", "d'água" -> "dagua");
* NÃO removemos stopwords nem aplicamos stemming. Em letra de música as palavras
  "vazias" fazem parte do trecho que a pessoa lembra ("let it be", "eu sei que
  vou te amar"). BM25 e TF-IDF já dão peso baixo a termos muito frequentes
  através do IDF, então não precisamos jogá-los fora.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np

_APOSTROFOS = str.maketrans("", "", "'’‘`´")
_TOKEN = re.compile(r"[a-z0-9]+")

# Operadores da consulta booleana (só valem em MAIÚSCULAS, para não confundir
# com as palavras "e"/"ou" que aparecem nas letras).
OPERADORES_E = {"AND", "E", "&&"}
OPERADORES_OU = {"OR", "OU", "||"}
OPERADORES_NAO = {"NOT", "NAO", "NÃO"}
OPERADORES = OPERADORES_E | OPERADORES_OU | OPERADORES_NAO

# Campos que podem ser usados na consulta (ex.: artista:"legião urbana")
CAMPOS_ALIASES = {
    "artista": "artista", "artist": "artista", "banda": "artista", "cantor": "artista",
    "cantora": "artista", "por": "artista", "by": "artista",
    "titulo": "titulo", "title": "titulo", "musica": "titulo", "song": "titulo", "nome": "titulo",
    "letra": "letra", "lyrics": "letra", "lyric": "letra", "trecho": "letra",
}


class ErroConsulta(ValueError):
    """Consulta mal formada (mensagem amigável para mostrar ao usuário)."""


def remover_acentos(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    return remover_acentos(texto.lower().translate(_APOSTROFOS))


def tokenizar(texto: str | None) -> list[str]:
    """Quebra o texto em termos normalizados: 'Não é Fácil!' -> ['nao', 'e', 'facil']."""
    if not texto:
        return []
    return _TOKEN.findall(normalizar(texto))


def contem_sequencia(tokens: Sequence[str], sequencia: Sequence[str]) -> bool:
    """Verdadeiro se `sequencia` aparece em `tokens` com as palavras adjacentes (busca por frase)."""
    n = len(sequencia)
    if n == 0:
        return True
    sequencia = list(sequencia)
    primeiro = sequencia[0]
    for i in range(len(tokens) - n + 1):
        if tokens[i] == primeiro and list(tokens[i:i + n]) == sequencia:
            return True
    return False


# --------------------------------------------------------------------------
# Trechos (snippets) para exibir nos resultados
# --------------------------------------------------------------------------
def _janela_relevante(texto: str, termos: set[str], largura: int):
    """Acha a janela de `largura` palavras com mais termos da consulta."""
    palavras: list[str] = []
    fim_linha: list[bool] = []
    for linha in texto.splitlines():
        ps = linha.split()
        if ps:
            palavras.extend(ps)
            fim_linha.extend([False] * (len(ps) - 1) + [True])
    if not palavras:
        return [], [], [], 0, 0, 0
    acertos = np.array([any(t in termos for t in tokenizar(p)) for p in palavras], dtype=np.int32)
    n = len(palavras)
    if n <= largura:
        inicio = 0
    else:
        somas = np.convolve(acertos, np.ones(largura, dtype=np.int32), mode="valid")
        inicio = int(np.argmax(somas))
    fim = min(n, inicio + largura)
    return palavras[inicio:fim], acertos[inicio:fim], fim_linha[inicio:fim], inicio, n - fim, int(acertos[inicio:fim].sum())


def trecho_relevante(texto: str, termos: Iterable[str], largura: int = 60) -> str:
    """Versão em texto puro do trecho mais relevante (usada no reranking)."""
    palavras, *_ = _janela_relevante(texto, set(termos), largura)
    return " ".join(palavras)


def gerar_trecho_html(texto: str, termos: Iterable[str], largura: int = 32) -> tuple[str, int]:
    """Trecho em HTML com os termos da consulta destacados em <mark>.

    Retorna (html, número de palavras destacadas). Quebras de linha viram " / ".
    """
    palavras, acertos, fim_linha, cortado_ini, cortado_fim, n_acertos = _janela_relevante(
        texto, set(termos), largura)
    if not palavras:
        return "", 0
    partes = ["… "] if cortado_ini else []
    for i, palavra in enumerate(palavras):
        p = html.escape(palavra)
        partes.append(f"<mark>{p}</mark>" if acertos[i] else p)
        if i < len(palavras) - 1:
            partes.append(" / " if fim_linha[i] else " ")
    if cortado_fim:
        partes.append(" …")
    return "".join(partes), n_acertos
