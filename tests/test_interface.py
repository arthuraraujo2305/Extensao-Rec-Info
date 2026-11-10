"""Fluxos da interface que usam a identidade visual, sem rede nem modelos."""
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from buscador import config
from buscador.motor import MotorDeBusca
from buscador.semantico import BuscadorSemantico

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_local(monkeypatch, tmp_path):
    corpus = pd.DataFrame([{
        "titulo": "Tempo Perdido", "artista": "Legião Urbana", "idioma": "pt",
        "generos": "Rock", "popularidade": 10.0,
        "letra": "Não tenho mais o tempo que passou mas tenho muito tempo",
        "link_musica": "/legiao-urbana/tempo-perdido.html", "link_artista": "/legiao-urbana/",
    }])
    outras = [dict(corpus.iloc[0], titulo=f"Outra canção {i}", letra="tempo chuva",
                    link_musica=f"/outra/{i}.html") for i in range(6)]
    corpus = pd.concat([corpus, pd.DataFrame(outras)], ignore_index=True)
    motor = MotorDeBusca.construir(corpus)
    monkeypatch.setattr(MotorDeBusca, "indice_existe", staticmethod(lambda: True))
    monkeypatch.setattr(MotorDeBusca, "carregar", classmethod(lambda cls: motor))
    monkeypatch.setattr(config, "ARQUIVO_FEEDBACK", tmp_path / "feedback.csv")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file(APP, default_timeout=60).run()
    st.cache_resource.clear()
    st.cache_data.clear()


def test_busca_e_votos_com_icones_preservam_feedback(app_local):
    app = app_local
    assert not app.exception
    app.text_input(key="consulta").set_value("tempo que passou")
    next(b for b in app.button if b.label == "Buscar música").click().run()
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
    next(b for b in app.button if b.label == "Começar outra busca").click().run()
    buscar_no_app(app, "tempo que passou")
    assert app.button(key=sim).proto.type == "secondary"
    assert app.button(key=nao).proto.type == "secondary"
    assert len(pd.read_csv(config.ARQUIVO_FEEDBACK)) == 2


def test_sem_indice_mostra_marca_e_orientacao(monkeypatch):
    monkeypatch.setattr(MotorDeBusca, "indice_existe", staticmethod(lambda: False))
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert "coleção de letras ainda não está pronta" in app.error[0].value
    assert "construir_indice.py" in app.code[0].value
    assert not app.text_input


def buscar_no_app(app, consulta):
    app.text_input(key="consulta").set_value(consulta)
    next(b for b in app.button if b.label == "Buscar música").click().run()
    assert not app.exception


def test_investigar_e_montar_regras_sem_perder_a_busca(app_local):
    app = app_local
    buscar_no_app(app, "tempo que passou")
    app.button(key="abrir_0").click().run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["1 · Sua pista", "2 · Onde procuramos", "3 · Por que apareceu?"]
    assert app.text_input(key="consulta").value == "tempo que passou"
    app.toggle(key="oficina").set_value(True).run()
    assert not app.exception
    assert "atende às regras" in app.success[-1].value
    app.text_input(key="bool_nenhuma").set_value("tempo")
    next(b for b in app.button if b.label == "Buscar com estas regras").click().run()
    assert not app.exception
    assert any("fica de fora" in m.value for m in app.info)
    assert 'NOT "tempo"' in app.code[-1].value
    next(b for b in app.button if b.label == "Voltar aos resultados").click().run()
    assert not app.exception and not app.tabs
    assert app.text_input(key="consulta").value == "tempo que passou"


def test_paginacao_e_reinicio_limpam_a_visita(app_local):
    app = app_local
    buscar_no_app(app, "tempo")
    assert len([b for b in app.button if str(b.key).startswith("abrir_")]) == 5
    next(b for b in app.button if b.label == "Ver mais músicas").click().run()
    assert len([b for b in app.button if str(b.key).startswith("abrir_")]) == 7
    app.button(key="abrir_6").click().run()
    next(b for b in app.button if b.label == "Voltar aos resultados").click().run()
    assert not app.exception
    assert len([b for b in app.button if str(b.key).startswith("abrir_")]) == 7
    next(b for b in app.button if b.label == "Começar outra busca").click().run()
    assert not app.exception
    assert app.text_input(key="consulta").value == ""
    assert not any(str(b.key).startswith("abrir_") for b in app.button)


def test_filtros_vazio_exemplo_e_semantico_indisponivel(app_local):
    app = app_local
    app.multiselect(key="idiomas").set_value(["en"])
    buscar_no_app(app, "tempo")
    assert not any(str(b.key).startswith("abrir_") for b in app.button)
    assert any("coleção não reúne todas" in i.value for i in app.info)
    app.toggle(key="mostrar_exemplos_apoio").set_value(True).run()
    app.button(key="ex_apoio_0").click().run()
    assert not app.exception
    assert app.multiselect(key="idiomas").value == []
    app.radio(key="intencao").set_value("assunto").run()
    assert not app.exception
    assert any("indisponível" in i.value for i in app.info)
    assert next(b for b in app.button if b.label == "Buscar música").disabled


def test_descricao_original_e_percurso_semantico_sem_download(app_local, monkeypatch):
    app = app_local
    motor = MotorDeBusca.carregar()
    sem = BuscadorSemantico(np.tile([1., 0.], (motor.n_docs, 1)), np.arange(motor.n_docs + 1))
    descricao = "uma descriçao de saudade e lembranças"
    sem._cache[descricao] = np.array([1., 0.], dtype=np.float32)
    motor.semantico = sem

    def nao_corrigir(_):
        raise AssertionError("A descrição semântica não deve passar pelo corretor lexical")

    monkeypatch.setattr(motor.corretor, "corrigir_consulta", nao_corrigir)
    app.radio(key="intencao").set_value("assunto").run()
    buscar_no_app(app, descricao)
    app.button(key="abrir_0").click().run()
    assert not app.exception
    assert any("preservada" in m.value for m in app.markdown)
    assert any("vetores e comparação" in e.label for e in app.expander)
    assert sem._modelo is None
