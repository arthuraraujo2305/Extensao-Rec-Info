"""Avaliação de RI (Aula 05).

Dois tipos de avaliação:

1. **Busca por item conhecido** (automática): sorteamos músicas, pegamos um
   pedaço da letra e simulamos como uma pessoa buscaria por ela (trecho exato,
   trecho com erros de digitação, palavras soltas, artista + trecho). A música
   de origem é a resposta certa, então não precisamos de julgamento humano.
   Métricas: MRR, Sucesso@1, Sucesso@10, nDCG@10.

2. **Julgamentos de usuários** (feedback 👍/👎 coletado no app): vira um
   conjunto de relevância (qrels) para consultas reais, com P@5, P@10 e nDCG@10.
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .motor import MODOS_SEMANTICOS, MotorDeBusca
from .texto import ErroConsulta, normalizar, tokenizar


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def precisao_em_k(ranking: Sequence[int], relevantes: Iterable[int], k: int) -> float:
    relevantes = set(relevantes)
    return sum(1 for d in ranking[:k] if d in relevantes) / k


def revocacao_em_k(ranking: Sequence[int], relevantes: Iterable[int], k: int) -> float:
    relevantes = set(relevantes)
    if not relevantes:
        return 0.0
    return sum(1 for d in ranking[:k] if d in relevantes) / len(relevantes)


def sucesso_em_k(ranking: Sequence[int], relevantes: Iterable[int], k: int) -> float:
    """1 se pelo menos um documento relevante aparece no top-k."""
    relevantes = set(relevantes)
    return float(any(d in relevantes for d in ranking[:k]))


def reciprocal_rank(ranking: Sequence[int], relevantes: Iterable[int]) -> float:
    relevantes = set(relevantes)
    for i, d in enumerate(ranking, start=1):
        if d in relevantes:
            return 1.0 / i
    return 0.0


def precisao_media(ranking: Sequence[int], relevantes: Iterable[int]) -> float:
    """Average Precision (a média dela sobre as consultas é o MAP)."""
    relevantes = set(relevantes)
    if not relevantes:
        return 0.0
    acertos, soma = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in relevantes:
            acertos += 1
            soma += acertos / i
    return soma / len(relevantes)


def ndcg_em_k(ranking: Sequence[int], relevancia: Mapping[int, float] | Iterable[int], k: int) -> float:
    """nDCG@k. `relevancia` pode ser {doc: grau} ou um conjunto (grau 1)."""
    if not isinstance(relevancia, Mapping):
        relevancia = {d: 1.0 for d in relevancia}
    dcg = sum((2 ** relevancia.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideais = sorted(relevancia.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideais))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# Geração de consultas (item conhecido)
# --------------------------------------------------------------------------
_VIZINHOS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok",
    "o": "ipl", "p": "ol", "a": "qsz", "s": "adwx", "d": "sfec", "f": "dgrv", "g": "fhtb",
    "h": "gjyn", "j": "hkun", "k": "jlim", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def inserir_erro(palavra: str, rng: random.Random) -> str:
    """Um erro de digitação típico: troca, apaga, duplica ou erra a tecla vizinha."""
    p = normalizar(palavra)  # quem digita rápido costuma esquecer os acentos
    if len(p) < 3:
        return p
    i = rng.randrange(1, len(p) - 1)
    operacao = rng.choice(["trocar", "apagar", "duplicar", "vizinha"])
    if operacao == "trocar":
        return p[:i] + p[i + 1] + p[i] + p[i + 2:]
    if operacao == "apagar":
        return p[:i] + p[i + 1:]
    if operacao == "duplicar":
        return p[:i] + p[i] + p[i:]
    vizinhas = _VIZINHOS.get(p[i])
    return p[:i] + (rng.choice(vizinhas) if vizinhas else p[i]) + p[i + 1:]


def com_erros(palavras: Sequence[str], rng: random.Random, prob: float = 0.4) -> list[str]:
    saida = [inserir_erro(p, rng) if len(p) >= 4 and rng.random() < prob else p for p in palavras]
    if saida == list(palavras):  # garante pelo menos um erro
        maior = max(range(len(palavras)), key=lambda i: len(palavras[i]))
        saida[maior] = inserir_erro(palavras[maior], rng)
    return saida


def _limpar(palavra: str) -> str:
    return re.sub(r"[^\w'’-]", "", palavra).strip("-")


def gerar_consultas_item_conhecido(motor: MotorDeBusca, n_por_idioma: int = 100, semente: int = 42,
                                   idiomas: Sequence[str] = ("pt", "en")) -> pd.DataFrame:
    """Para cada música sorteada gera 4 consultas, todas com a mesma resposta certa.

    São relevantes: a música de origem e qualquer outra cuja letra contenha
    exatamente o mesmo trecho (versões duplicadas da mesma música).
    """
    rng = random.Random(semente)
    comprimentos = motor.indices["letra"].comprimentos
    linhas = []
    for idioma in idiomas:
        docs = [int(d) for d in np.flatnonzero(motor.idiomas == idioma) if comprimentos[d] >= 40]
        for d in rng.sample(docs, min(n_por_idioma, len(docs))):
            palavras = [w for w in (_limpar(p) for p in motor.textos["letra"][d].split()) if tokenizar(w)]
            if len(palavras) < 20:
                continue
            ini = rng.randrange(0, len(palavras) - 14)
            janela = palavras[ini:ini + 14]
            trecho = janela[:rng.randint(6, 8)]
            relevantes = set(motor.docs_com_frase(" ".join(trecho)).tolist()) | {d}
            base = {"idioma": idioma, "doc_alvo": d, "relevantes": sorted(relevantes)}
            conteudo = [w for w in janela if len(w) >= 4]   # quem lembra "solto" lembra palavras de conteúdo
            fonte = conteudo if len(conteudo) >= 4 else janela
            soltas = rng.sample(fonte, min(5, len(fonte)))
            ini2 = rng.randrange(0, len(janela) - 4)
            linhas += [
                {**base, "tipo": "trecho exato", "consulta": " ".join(trecho)},
                {**base, "tipo": "trecho com erros", "consulta": " ".join(com_erros(trecho, rng))},
                {**base, "tipo": "palavras soltas", "consulta": " ".join(soltas)},
                {**base, "tipo": "artista + trecho",
                 "consulta": motor.textos["artista"][d] + " " + " ".join(janela[ini2:ini2 + 4])},
            ]
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Execução da avaliação
# --------------------------------------------------------------------------
CONFIGURACOES_PADRAO = [
    {"nome": "Booleano (AND)", "modo": "booleano"},
    {"nome": "Booleano + correção", "modo": "booleano", "corrigir": True},
    {"nome": "TF-IDF", "modo": "tfidf"},
    {"nome": "BM25", "modo": "bm25"},
    {"nome": "BM25 + correção", "modo": "bm25", "corrigir": True},
    {"nome": "Semântico", "modo": "semantico"},
    {"nome": "Híbrido RRF + correção", "modo": "hibrido", "corrigir": True, "fusao": "rrf"},
    {"nome": "Híbrido CombSUM + correção", "modo": "hibrido", "corrigir": True},
]


def avaliar_configuracoes(motor: MotorDeBusca, consultas: pd.DataFrame,
                          configuracoes: Sequence[dict] = CONFIGURACOES_PADRAO, k: int = 10,
                          progresso: Callable[[str], None] | None = print) -> pd.DataFrame:
    """Roda cada configuração em todas as consultas e devolve as métricas por consulta."""
    linhas = []
    for cfg in configuracoes:
        if cfg["modo"] in MODOS_SEMANTICOS and not motor.tem_semantico:
            if progresso:
                progresso(f"  (pulando {cfg['nome']}: embeddings não gerados)")
            continue
        if progresso:
            progresso(f"  avaliando {cfg['nome']}…")
        for linha in consultas.itertuples(index=False):
            consulta = linha.consulta
            if cfg.get("corrigir"):
                consulta, _ = motor.corretor.corrigir_consulta(consulta)
            try:
                ranking, _ = motor.ranquear(consulta, cfg["modo"], k=k, reordenar=cfg.get("reordenar", False),
                                            fusao=cfg.get("fusao", "combsum"))
            except ErroConsulta:
                ranking = []
            docs = [d for d, _ in ranking]
            relevantes = set(linha.relevantes)
            pos = next((i for i, d in enumerate(docs, start=1) if d in relevantes), None)
            linhas.append({
                "configuracao": cfg["nome"], "tipo": linha.tipo, "idioma": linha.idioma,
                "consulta": linha.consulta, "posicao_acerto": pos,
                "rr": reciprocal_rank(docs, relevantes),
                "sucesso@1": sucesso_em_k(docs, relevantes, 1),
                "sucesso@10": sucesso_em_k(docs, relevantes, 10),
                "ndcg@10": ndcg_em_k(docs, relevantes, 10),
            })
    return pd.DataFrame(linhas)


def resumir(por_consulta: pd.DataFrame, por: Sequence[str] = ("configuracao",)) -> pd.DataFrame:
    metricas = ["rr", "sucesso@1", "sucesso@10", "ndcg@10"]
    resumo = por_consulta.groupby(list(por), sort=False)[metricas].mean()
    return resumo.rename(columns={"rr": "MRR"}).round(3)


# --------------------------------------------------------------------------
# Avaliação com o feedback dos usuários
# --------------------------------------------------------------------------
def consultas_do_feedback(motor: MotorDeBusca, caminho: str | Path) -> pd.DataFrame:
    """Transforma o feedback 👍/👎 em consultas com conjunto de relevantes (qrels).

    Se a mesma música recebeu votos diferentes para a mesma consulta, vale o último.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return pd.DataFrame(columns=["consulta", "relevantes", "julgados"])
    fb = pd.read_csv(caminho)
    doc_por_link = {link: i for i, link in enumerate(motor.corpus["link_musica"])}
    fb["doc_id"] = fb["link_musica"].map(doc_por_link)
    fb = fb.dropna(subset=["doc_id"])
    fb["chave"] = fb["consulta"].astype(str).map(lambda c: " ".join(tokenizar(c)))
    fb = fb.drop_duplicates(["chave", "doc_id"], keep="last")
    linhas = []
    for _, grupo in fb.groupby("chave"):
        linhas.append({
            "consulta": grupo["consulta"].iloc[-1],
            "relevantes": sorted(int(d) for d in grupo.loc[grupo["relevante"] == 1, "doc_id"]),
            "julgados": sorted(int(d) for d in grupo["doc_id"]),
        })
    return pd.DataFrame(linhas)


def avaliar_feedback(motor: MotorDeBusca, caminho: str | Path,
                     configuracoes: Sequence[dict] = CONFIGURACOES_PADRAO, k: int = 10) -> pd.DataFrame:
    """P@5, P@10, nDCG@10 e MAP por configuração (documento não julgado = não relevante)."""
    qrels = consultas_do_feedback(motor, caminho)
    qrels = qrels[qrels["relevantes"].map(len) > 0]
    linhas = []
    for cfg in configuracoes:
        if cfg["modo"] in MODOS_SEMANTICOS and not motor.tem_semantico:
            continue
        for linha in qrels.itertuples(index=False):
            try:
                ranking, _ = motor.ranquear(linha.consulta, cfg["modo"], k=k, fusao=cfg.get("fusao", "combsum"))
            except ErroConsulta:
                ranking = []
            docs = [d for d, _ in ranking]
            linhas.append({"configuracao": cfg["nome"], "consulta": linha.consulta,
                           "P@5": precisao_em_k(docs, linha.relevantes, 5),
                           "P@10": precisao_em_k(docs, linha.relevantes, 10),
                           "nDCG@10": ndcg_em_k(docs, linha.relevantes, 10),
                           "AP": precisao_media(docs, linha.relevantes)})
    if not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas)
    return df.groupby("configuracao", sort=False)[["P@5", "P@10", "nDCG@10", "AP"]].mean().rename(
        columns={"AP": "MAP"}).round(3)
