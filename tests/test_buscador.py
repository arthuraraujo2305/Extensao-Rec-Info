"""Testes com um corpus pequeno (não precisa baixar a base nem gerar embeddings).

    pytest
"""
import numpy as np
import pandas as pd
import pytest

from buscador.avaliacao import ndcg_em_k, precisao_em_k, precisao_media, reciprocal_rank, sucesso_em_k
from buscador.hibrido import fusao_combsum, fusao_rrf
from buscador.indice import IndiceInvertido
from buscador.motor import MotorDeBusca
from buscador.semantico import dividir_em_trechos
from buscador.texto import ErroConsulta, contem_sequencia, gerar_trecho_html, tokenizar
from buscador.tolerancia import CorretorOrtografico, ExpansorCuringa, distancia_edicao

CORPUS = pd.DataFrame([
    # 0
    {"titulo": "Tempo Perdido", "artista": "Legião Urbana", "idioma": "pt", "generos": "Rock",
     "letra": "Todos os dias quando acordo\nNão tenho mais o tempo que passou\nMas tenho muito tempo",
     "popularidade": 10.0, "link_musica": "/legiao-urbana/tempo-perdido.html", "link_artista": "/legiao-urbana/"},
    # 1
    {"titulo": "Eduardo e Mônica", "artista": "Legião Urbana", "idioma": "pt", "generos": "Rock",
     "letra": "Quem um dia irá dizer\nQue existe razão nas coisas feitas pelo coração",
     "popularidade": 10.0, "link_musica": "/legiao-urbana/eduardo-e-monica.html", "link_artista": "/legiao-urbana/"},
    # 2
    {"titulo": "Chuva de Verão", "artista": "Fulano", "idioma": "pt", "generos": "MPB; Samba",
     "letra": "A chuva cai lá fora e eu penso em você\nSaudade apertando o coração apaixonado",
     "popularidade": 2.0, "link_musica": "/fulano/chuva.html", "link_artista": "/fulano/"},
    # 3
    {"titulo": "Let It Be", "artista": "The Beatles", "idioma": "en", "generos": "Rock",
     "letra": "When I find myself in times of trouble\nMother Mary comes to me\nLet it be, let it be",
     "popularidade": 20.0, "link_musica": "/the-beatles/let-it-be.html", "link_artista": "/the-beatles/"},
    # 4
    {"titulo": "Rain Song", "artista": "Someone", "idioma": "en", "generos": "Pop",
     "letra": "The rain keeps falling down on me\nI miss you more than words can say\nmy heart in love",
     "popularidade": 1.0, "link_musica": "/someone/rain.html", "link_artista": "/someone/"},
])


@pytest.fixture(scope="module")
def motor():
    return MotorDeBusca.construir(CORPUS)


# ------------------------------------------------------------------ texto
def test_tokenizar_remove_acentos_e_apostrofos():
    assert tokenizar("Não é FÁCIL, coração!") == ["nao", "e", "facil", "coracao"]
    assert tokenizar("Don't stop d'água") == ["dont", "stop", "dagua"]


def test_contem_sequencia():
    assert contem_sequencia(["a", "b", "c", "d"], ["b", "c"])
    assert not contem_sequencia(["a", "b", "c", "d"], ["c", "b"])


def test_trecho_destaca_termos():
    html, n = gerar_trecho_html("A chuva cai\nlá fora", {"chuva"})
    assert "<mark>chuva</mark>" in html and n == 1 and " / " in html


# ------------------------------------------------------------------ índice
def test_indice_invertido(tmp_path):
    indice = IndiceInvertido.construir([["a", "b", "a"], ["b", "c"], ["a"]])
    docs, tf = indice.postings("a")
    assert docs.tolist() == [0, 2] and tf.tolist() == [2, 1]
    assert indice.frequencia_documento("b") == 2
    assert indice.comprimentos.tolist() == [3, 2, 1]
    indice.salvar(tmp_path / "i.npz")
    outro = IndiceInvertido.carregar(tmp_path / "i.npz")
    assert outro.termos == indice.termos and outro.docs.tolist() == indice.docs.tolist()


# ------------------------------------------------------------------ modelos ranqueados
def test_bm25_encontra_trecho(motor):
    ranking, total = motor.ranquear("tempo que passou", "bm25", k=3)
    assert ranking[0][0] == 0 and total >= 1


def test_tfidf_encontra_trecho(motor):
    ranking, _ = motor.ranquear("mother mary", "tfidf", k=3)
    assert ranking[0][0] == 3


def test_artista_mais_letra(motor):
    ranking, _ = motor.ranquear("legião urbana coração", "bm25", k=3)
    assert ranking[0][0] == 1  # é da Legião E fala de coração


def test_campo_na_consulta_ranqueada(motor):
    ranking, _ = motor.ranquear("artista:beatles", "bm25", k=5)
    assert [d for d, _ in ranking] == [3]


def test_filtro_idioma(motor):
    mascara = motor.mascara_filtros(idiomas=["en"])
    ranking, _ = motor.ranquear("coração rain", "bm25", k=5, mascara=mascara)
    assert {d for d, _ in ranking} == {4}


def test_filtro_genero(motor):
    mascara = motor.mascara_filtros(generos=["Samba"])
    assert np.flatnonzero(mascara).tolist() == [2]


# ------------------------------------------------------------------ booleano
@pytest.mark.parametrize("consulta,esperado", [
    ("coração", {1, 2}),
    ("coração AND chuva", {2}),
    ("coração chuva", {2}),                       # AND implícito
    ("tempo OR rain", {0, 4}),
    ("coração NOT chuva", {1}),
    ("coração -chuva", {1}),
    ("artista:legiao", {0, 1}),
    ('artista:"legião urbana" AND coração', {1}),
    ('"let it be"', {3}),
    ('"be it let"', set()),
    ("(chuva OR rain) AND NOT saudade", {4}),
    ("apaixon*", {2}),
    ("titulo:chuva", {2}),
    ("E agora", set()),                           # 'E' no início vira palavra
])
def test_booleano(motor, consulta, esperado):
    assert set(motor.booleano.buscar(consulta).tolist()) == esperado


@pytest.mark.parametrize("consulta", ["amor AND", "campo:valor", "(", ""])
def test_booleano_erros(motor, consulta):
    with pytest.raises(ErroConsulta):
        motor.booleano.buscar(consulta)


# ------------------------------------------------------------------ tolerância
def test_distancia_edicao():
    assert distancia_edicao("amor", "amor") == 0
    assert distancia_edicao("amor", "amr") == 1
    assert distancia_edicao("amor", "aomr") == 1      # transposição
    assert distancia_edicao("gato", "pato") == 1
    assert distancia_edicao("saudade", "sdd", limite=2) == 3


def test_corretor():
    freq = {"coracao": 50, "saudade": 30, "partido": 20, "chuva": 40, "coracoes": 2}
    corretor = CorretorOrtografico(freq)
    assert corretor.sugerir("coracai") == "coracao"
    assert corretor.sugerir("sadade") == "saudade"
    assert corretor.sugerir("chuva") is None           # já está certo
    nova, mudou = corretor.corrigir_consulta("Coracao partdo AND artista:legiao")
    assert mudou and nova == "Coracao partido AND artista:legiao"


def test_corretor_sensivel_ao_contexto():
    freq = {"parado": 62, "partido": 31, "coracao": 500}
    # sem contexto ganha a mais frequente; com contexto, a que mais aparece junto
    juntos = {("partido", "coracao"): 40, ("parado", "coracao"): 10}
    corretor = CorretorOrtografico(freq, associacao=lambda c, ctx: sum(juntos.get((c, t), 0) for t in ctx))
    assert corretor.sugerir("partdo") == "parado"
    assert corretor.corrigir_consulta("coracao partdo") == ("coracao partido", True)


def test_curinga():
    exp = ExpansorCuringa(["amor", "amora", "amado", "amar", "clamor"], np.array([10, 1, 5, 8, 2]))
    assert exp.expandir("amo*") == ["amor", "amora"]
    assert set(exp.expandir("am*r")) == {"amor", "amar"}
    assert exp.expandir("*mor") == ["amor", "clamor"]
    with pytest.raises(ErroConsulta):
        exp.expandir("*")


def test_busca_com_correcao(motor):
    # o corretor só sugere palavras que aparecem ≥ 2 vezes (coracao, chuva)
    resposta = motor.buscar("chuvq coracai", modo="bm25")
    assert resposta.corrigida and resposta.consulta_usada == "chuva coracao"
    assert resposta.resultados[0].doc_id == 2


# ------------------------------------------------------------------ híbrido / semântico (sem modelo)
def test_rrf():
    fundido = fusao_rrf([[1, 2, 3], [3, 1, 4]])
    assert [d for d, _ in fundido][:2] == [1, 3]


def test_combsum_preserva_margem():
    # BM25 "tem certeza" do doc 1; o semântico prefere o 2 por pouco
    lexico = [(1, 30.0), (2, 5.0), (3, 4.0)]
    semantico = [(2, 0.70), (3, 0.69), (1, 0.60)]
    assert fusao_combsum([lexico, semantico], [0.7, 0.3])[0][0] == 1
    assert fusao_rrf([[1, 2, 3], [2, 3, 1]])[0][0] == 2   # o RRF perde essa informação
    assert fusao_combsum([[(9, 1.0)], []], [0.7, 0.3]) == [(9, 0.7)]   # lista com 1 item


def test_dividir_em_trechos():
    texto = " ".join(str(i) for i in range(200))
    trechos = dividir_em_trechos(texto, palavras_por_trecho=80, sobreposicao=20)
    assert len(trechos) == 3 and trechos[1].split()[0] == "60"


def test_semantico_indisponivel_gera_erro_amigavel(motor):
    resposta = motor.buscar("qualquer coisa", modo="semantico")
    assert resposta.erro and not resposta.resultados


# ------------------------------------------------------------------ métricas
def test_metricas():
    ranking = [5, 3, 9, 1]
    assert precisao_em_k(ranking, {3, 1}, 2) == 0.5
    assert reciprocal_rank(ranking, {9}) == pytest.approx(1 / 3)
    assert sucesso_em_k(ranking, {1}, 3) == 0.0
    assert precisao_media(ranking, {3, 1}) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert ndcg_em_k([1, 2], {1}, 10) == 1.0
    assert ndcg_em_k([2, 1], {1}, 10) == pytest.approx(1 / np.log2(3))


def test_salvar_e_carregar(motor, tmp_path):
    motor.salvar(tmp_path)
    outro = MotorDeBusca.carregar(tmp_path)
    assert outro.ranquear("let it be", "bm25", k=1)[0][0][0] == 3
