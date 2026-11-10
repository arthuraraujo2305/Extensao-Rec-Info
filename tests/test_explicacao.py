"""Evidências e regras devem reproduzir o motor, sem modelos baixados."""
import numpy as np
import pandas as pd
import pytest

from buscador.explicacao import (explicar_palavras, explicar_significado, experimentar_regras,
                                 montar_consulta_booleana)
from buscador.motor import MotorDeBusca
from buscador.semantico import BuscadorSemantico
from buscador.texto import ErroConsulta


@pytest.fixture
def motor():
    linhas = [
        ("Chuva", "Sol", "chuva chuva saudade", "pt", "MPB"),
        ("Rain", "Someone", "rain sun", "en", "Pop"),
        ("Mar", "Lua", "mar saudade", "pt", "MPB"),
    ]
    return MotorDeBusca.construir(pd.DataFrame([
        dict(titulo=t, artista=a, letra=l, idioma=i, generos=g, popularidade=1,
             link_musica=f"/musica/{n}", link_artista=f"/artista/{n}")
        for n, (t, a, l, i, g) in enumerate(linhas)]))


def test_contribuicoes_reproduzem_nota_e_pistas_do_indice(motor):
    consulta = "chuva chuva saudade desconhecida"
    detalhe = explicar_palavras(motor, consulta, 0, idiomas=["pt"])
    linhas = detalhe["termos"]
    assert len(linhas) == 3
    assert sum(l["contribuicao"] for l in linhas) == pytest.approx(motor.pontuar_lexico(consulta)[0])
    chuva = linhas[0]
    assert chuva["ocorrencias"] == 4  # título 2x + letra 2x
    assert chuva["campos"] == ["letra", "titulo"]
    assert chuva["musicas"] == 1 and chuva["musicas_filtradas"] == 1
    assert linhas[-1]["contribuicao"] == 0 and linhas[-1]["exemplos"] == []
    assert explicar_palavras(motor, "rain", 0, idiomas=["pt"])["termos"][0]["exemplos"] == []


@pytest.mark.parametrize("todas,alguma,nenhuma,docs", [
    ("saudade", "chuva mar", "", [0, 2]),
    ("saudade", "chuva mar", "chuva", [2]),
    ("", "chuva rain", "", [0, 1]),
    ("sol", "", "saudade", []),
    ('chuva OR rain', "", "", []),  # OR digitado é uma palavra, não um operador injetado
    ("saudade", "chuva rain", "", [0]),  # precedência: saudade AND (chuva OR rain)
])
def test_montagem_respeita_grupos_e_nao_injeta_sintaxe(motor, todas, alguma, nenhuma, docs):
    expr = montar_consulta_booleana(todas, alguma, nenhuma)
    assert motor.booleano.buscar(expr).tolist() == docs


def test_oficina_mostra_exclusao_real_e_respeita_filtros(motor):
    dados = experimentar_regras(motor, "saudade", "", "chuva", 0)
    assert not dados["atende"] and dados["total"] == 1
    assert dados["exemplos"][0]["titulo"] == "Mar"
    assert dados["regras"][-1] == dict(regra="Excluída", palavra="chuva", presente=True)
    assert experimentar_regras(motor, "saudade", "", "", 0, idiomas=["en"])["total"] == 0
    with pytest.raises(ErroConsulta):
        montar_consulta_booleana("", "", "chuva")


def test_percurso_semantico_usa_trechos_e_maximo_do_ranking(motor):
    textos = ["chuva " * 150, "rain sun", "mar saudade"]
    motor.textos["letra"] = textos
    partes, inicio = BuscadorSemantico.preparar_trechos(textos)
    emb = np.zeros((len(partes), 2), dtype=np.float32)
    emb[:, 1] = 1
    emb[1] = [1, 0]
    sem = BuscadorSemantico(emb, inicio, "modelo-falso-para-teste")
    sem._cache["lembrança"] = np.array([1, 0], dtype=np.float32)
    motor.semantico = sem
    dados = explicar_significado(motor, "lembrança", 0)
    assert dados["melhor"] == 1 and dados["dimensoes"] == 2
    assert dados["trechos"][1]["semelhanca"] == sem.pontuar("lembrança")[0]
    assert len(dados["trechos"]) == inicio[1]
    assert sem._modelo is None  # não carrega nem baixa um modelo
