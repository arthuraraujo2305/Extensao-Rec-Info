"""Contraste das paletas configuradas para os elementos nativos do aplicativo."""
from pathlib import Path
import tomllib

import pytest


def luminancia(cor):
    rgb = [int(cor[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return sum(v * peso for v, peso in zip(linear, (0.2126, 0.7152, 0.0722)))


def contraste(a, b):
    menor, maior = sorted((luminancia(a), luminancia(b)))
    return (maior + 0.05) / (menor + 0.05)


@pytest.fixture(params=["light", "dark"])
def paleta(request):
    caminho = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    return tomllib.loads(caminho.read_text())["theme"][request.param]


def test_textos_e_links_tem_contraste_para_leitura(paleta):
    for fundo in ["backgroundColor", "secondaryBackgroundColor", "dataframeHeaderBackgroundColor"]:
        for texto in ["textColor", "linkColor"]:
            assert contraste(paleta[texto], paleta[fundo]) >= 4.5, (texto, fundo)
    # Botões primários do Streamlit usam texto branco.
    assert contraste("#ffffff", paleta["primaryColor"]) >= 4.5


def test_controles_se_distinguem_do_fundo(paleta):
    for fundo in ["backgroundColor", "secondaryBackgroundColor"]:
        assert contraste(paleta["borderColor"], paleta[fundo]) >= 3
        assert contraste(paleta["primaryColor"], paleta[fundo]) >= 3
