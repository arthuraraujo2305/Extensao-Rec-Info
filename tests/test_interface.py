"""Fluxos da interface que usam a identidade visual, sem rede nem modelos."""
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from buscador import config
from buscador.motor import MotorDeBusca

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_local(monkeypatch, tmp_path):
    corpus = pd.DataFrame([{
        "titulo": "Tempo Perdido", "artista": "Legião Urbana", "idioma": "pt",
        "generos": "Rock", "popularidade": 10.0,
        "letra": "Não tenho mais o tempo que passou mas tenho muito tempo",
        "link_musica": "/legiao-urbana/tempo-perdido.html", "link_artista": "/legiao-urbana/",
    }])
    motor = MotorDeBusca.construir(corpus)
    monkeypatch.setattr(MotorDeBusca, "indice_existe", staticmethod(lambda: True))
    monkeypatch.setattr(MotorDeBusca, "carregar", classmethod(lambda cls: motor))
    monkeypatch.setattr(config, "ARQUIVO_FEEDBACK", tmp_path / "feedback.csv")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file(APP, default_timeout=15).run()
    st.cache_resource.clear()
    st.cache_data.clear()


def test_busca_e_votos_com_icones_preservam_feedback(app_local):
    app = app_local
    assert not app.exception
    app.text_input(key="consulta").set_value("tempo que passou").run()
    assert not app.exception

    sim = next(botao.key for botao in app.button if botao.label.endswith(" Sim"))
    nao = sim.rsplit("|", 1)[0] + "|0"
    app.button(key=sim).click().run()
    assert not app.exception
    assert pd.read_csv(config.ARQUIVO_FEEDBACK)["relevante"].tolist() == [1]
    assert app.button(key=sim).proto.type == "primary"

    app.button(key=nao).click().run()
    assert not app.exception
    votos = pd.read_csv(config.ARQUIVO_FEEDBACK)
    assert votos["relevante"].tolist() == [1, 0]
    assert votos["titulo"].tolist() == ["Tempo Perdido", "Tempo Perdido"]
    assert app.button(key=sim).proto.type == "secondary"
    assert app.button(key=nao).proto.type == "primary"

    # Clicar novamente retira a seleção, sem duplicar o registro do voto.
    app.button(key=nao).click().run()
    assert not app.exception
    assert len(pd.read_csv(config.ARQUIVO_FEEDBACK)) == 2
    assert app.button(key=nao).proto.type == "secondary"


def test_sem_indice_mostra_marca_e_orientacao(monkeypatch):
    monkeypatch.setattr(MotorDeBusca, "indice_existe", staticmethod(lambda: False))
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert "índice ainda não foi construído" in app.error[0].value
    assert "construir_indice.py" in app.code[0].value
    assert not app.text_input
