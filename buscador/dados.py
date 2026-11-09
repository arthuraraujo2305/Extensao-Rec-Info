"""Leitura e preparação da base de letras (Vagalume / Kaggle).

O arquivo de letras tem ~430 MB e ~380 mil músicas. Para caber em pouca RAM,
ele é lido em blocos e em duas passadas:

1. lê só as colunas leves (artista e idioma) e escolhe quais músicas entram no
   corpus: as dos artistas mais populares, no máximo `max_por_artista` por
   artista (para ter variedade) e `por_idioma` músicas por idioma;
2. lê de novo, agora com as letras, guardando apenas as linhas escolhidas.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .texto import tokenizar

IDIOMAS_SUPORTADOS = {"pt": "Português", "en": "Inglês"}
_MAPA_IDIOMA = {"pt": "pt", "portuguese": "pt", "português": "pt", "en": "en", "english": "en"}
COLUNAS_CORPUS = ["titulo", "artista", "letra", "idioma", "generos", "popularidade",
                  "link_musica", "link_artista"]


def localizar_csvs(pasta: str | Path = config.DADOS_BRUTOS) -> tuple[Path, Path]:
    pasta = Path(pasta)
    artistas, letras = pasta / "artists-data.csv", pasta / "lyrics-data.csv"
    faltando = [p.name for p in (artistas, letras) if not p.exists()]
    if faltando:
        raise FileNotFoundError(
            f"Arquivos não encontrados em {pasta}: {', '.join(faltando)}. "
            "Rode antes: python scripts/baixar_dados.py")
    return artistas, letras


def carregar_artistas(caminho: str | Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    col_generos = "Genres" if "Genres" in df.columns else "Genre"
    df = df.rename(columns={"Artist": "artista", col_generos: "generos",
                            "Popularity": "popularidade", "Link": "link_artista"})
    df["popularidade"] = pd.to_numeric(df["popularidade"], errors="coerce").fillna(0.0)
    df["generos"] = df["generos"].fillna("")
    return df.drop_duplicates("link_artista")[["link_artista", "artista", "generos", "popularidade"]]


def _normalizar_idioma(serie: pd.Series) -> pd.Series:
    return serie.astype("string").str.strip().str.lower().map(_MAPA_IDIOMA)


def preparar_corpus(pasta: str | Path = config.DADOS_BRUTOS, por_idioma: int = 10_000,
                    max_por_artista: int = 40, idiomas: tuple[str, ...] = ("pt", "en"),
                    tamanho_bloco: int = 50_000,
                    progresso: Callable[[str], None] = print) -> pd.DataFrame:
    """Monta o corpus (um DataFrame com uma linha por música)."""
    csv_artistas, csv_letras = localizar_csvs(pasta)
    artistas = carregar_artistas(csv_artistas)
    popularidade = dict(zip(artistas["link_artista"], artistas["popularidade"]))

    colunas = pd.read_csv(csv_letras, nrows=0).columns
    col_idioma = next(c for c in ("language", "Idiom", "idiom") if c in colunas)

    # ---- passada 1: só metadados ---------------------------------------------
    progresso("Passada 1/2: escolhendo as músicas…")
    partes = []
    for bloco in pd.read_csv(csv_letras, usecols=["ALink", col_idioma], chunksize=tamanho_bloco):
        bloco["idioma"] = _normalizar_idioma(bloco[col_idioma])
        partes.append(bloco.loc[bloco["idioma"].isin(idiomas), ["ALink", "idioma"]])
    meta = pd.concat(partes)
    meta["linha"] = meta.index
    meta["pop"] = meta["ALink"].map(popularidade).fillna(0.0)
    meta = meta.sort_values(["pop", "linha"], ascending=[False, True], kind="stable")
    meta = meta[meta.groupby("ALink").cumcount() < max_por_artista]
    # pega uma folga de 30% porque parte das letras será descartada (vazias/duplicadas)
    escolhidas = meta.groupby("idioma").head(int(por_idioma * 1.3))
    linhas = set(escolhidas["linha"])
    progresso(f"  {len(meta):,} candidatas; {len(linhas):,} pré-selecionadas")

    # ---- passada 2: letras das músicas escolhidas ----------------------------
    progresso("Passada 2/2: lendo as letras…")
    partes = []
    for bloco in pd.read_csv(csv_letras, chunksize=tamanho_bloco):
        partes.append(bloco[bloco.index.isin(linhas)])
    letras = pd.concat(partes)
    letras["idioma"] = _normalizar_idioma(letras[col_idioma])
    letras = letras.rename(columns={"SName": "titulo", "SLink": "link_musica",
                                    "Lyric": "letra", "ALink": "link_artista"})
    letras = letras.merge(artistas, on="link_artista", how="left")
    letras["artista"] = letras["artista"].fillna(
        letras["link_artista"].str.strip("/").str.replace("-", " ").str.title())
    letras["generos"] = letras["generos"].fillna("")
    letras["popularidade"] = letras["popularidade"].fillna(0.0)
    letras["titulo"] = letras["titulo"].fillna("").astype(str).str.strip()
    letras["letra"] = letras["letra"].fillna("").astype(str).str.replace("\r", "").str.strip()

    # ---- limpeza ---------------------------------------------------------------
    antes = len(letras)
    letras = letras[letras["letra"].str.len() >= 80]
    # letras idênticas (mesma música cadastrada duas vezes, versões "ao vivo", etc.)
    assinatura = letras["letra"].map(lambda t: hashlib.md5(" ".join(tokenizar(t)).encode()).hexdigest())
    letras = letras[~assinatura.duplicated()]
    chave = letras["artista"].str.lower() + "|" + letras["titulo"].str.lower()
    letras = letras[~chave.duplicated()]
    progresso(f"  {antes - len(letras):,} letras vazias/duplicadas removidas")

    letras = letras.sort_values(["idioma", "popularidade"], ascending=[False, False], kind="stable")
    corpus = letras.groupby("idioma").head(por_idioma)
    corpus = corpus[COLUNAS_CORPUS].reset_index(drop=True)
    for idioma, n in corpus["idioma"].value_counts().items():
        progresso(f"  {IDIOMAS_SUPORTADOS.get(idioma, idioma)}: {n:,} músicas")
    return corpus
