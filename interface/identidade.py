"""Marca e ícones originais, com enquadramento feito na apresentação.

As coordenadas são da arte fornecida pelo grupo. Os SVGs são apenas janelas
para os PNGs: não redesenham nem alteram os arquivos originais.
"""
from __future__ import annotations

import base64
import html
from functools import lru_cache

import streamlit as st

from buscador.config import RAIZ

PASTA_VISUAL = RAIZ / "static"
MARCA = "brand/buscador-de-letras.png"
JANELA_MARCA = (52, 44, 1670, 298)
JANELA_SIMBOLO = (640, 452, 448, 448)

# Janelas quadradas com respiro; os PNGs dos ícones têm 1254 × 1254 px.
ICONES = {
    "buscar": ("buscar.png", (125, 150, 1000, 1000)),
    "avaliacao": ("avaliacao.png", (135, 185, 980, 980)),
    "como-funciona": ("como-funciona.png", (45, 90, 1160, 1160)),
    "configuracoes": ("configuracoes.png", (110, 130, 1040, 1040)),
    "informacao": ("informacao.png", (125, 150, 1000, 1000)),
    "confirmacao": ("confirmacao.png", (125, 150, 1000, 1000)),
    "voto-positivo": ("voto-positivo.png", (125, 145, 980, 980)),
    "voto-negativo": ("voto-negativo.png", (125, 155, 980, 980)),
}


def _url(caminho: str) -> str:
    # URL relativa: também funciona quando o Streamlit tem server.baseUrlPath.
    return f"app/static/{caminho}"


def _svg(caminho: str, janela: tuple[int, int, int, int],
         tamanho: tuple[int, int], *, incorporar: bool = False) -> str:
    origem = _url(caminho)
    if incorporar:
        dados = base64.b64encode((PASTA_VISUAL / caminho).read_bytes()).decode("ascii")
        origem = f"data:image/png;base64,{dados}"
    x, y, largura, altura = janela
    w, h = tamanho
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {largura} {altura}" '
        f'width="{largura}" height="{altura}" aria-hidden="true" focusable="false">'
        f'<image href="{origem}" width="{w}" height="{h}" /></svg>'
    )


@lru_cache(maxsize=1)
def favicon() -> str:
    # Favicons SVG não podem carregar imagens externas: incorporar só aqui.
    return _svg(MARCA, JANELA_SIMBOLO, (1774, 887), incorporar=True)


def icone_markdown(nome: str) -> str:
    arquivo, _ = ICONES[nome]
    return f"![]({_url('icons/' + arquivo)})"


def cabecalho() -> None:
    marca = _svg(MARCA, JANELA_MARCA, (1774, 887))
    simbolo = _svg(MARCA, JANELA_SIMBOLO, (1774, 887))
    st.markdown(
        '<header class="marca-cabecalho">'
        '<div class="marca-titulo" role="heading" aria-level="1"><span class="marca-extensa" aria-hidden="true">' + marca + '</span>'
        '<span class="marca-compacta" aria-hidden="true">' + simbolo +
        '<span>Buscador<br>de Letras</span></span>'
        '<span class="somente-leitor">Buscador de Letras</span></div>'
        '<p>Lembra só <strong>um pedaço da letra</strong>? Escreveu <strong>errado</strong>? '
        'Só sabe <strong>do que a música fala</strong>?<br>'
        'Busque em português ou inglês e reencontre aquela música.</p></header>',
        unsafe_allow_html=True,
    )


def titulo_secao(texto: str, icone: str, *, nivel: int = 2) -> None:
    arquivo, janela = ICONES[icone]
    imagem = _svg(f"icons/{arquivo}", janela, (1254, 1254))
    nivel = max(2, min(nivel, 3))
    st.markdown(
        f'<h{nivel} class="titulo-secao">{imagem}<span>{html.escape(texto)}</span></h{nivel}>',
        unsafe_allow_html=True,
    )


def aplicar_estilo() -> None:
    css = (PASTA_VISUAL / "styles" / "app.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
