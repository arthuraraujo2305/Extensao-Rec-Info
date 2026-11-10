"""Busca de músicas e percurso explorável para a feira de extensão."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from buscador import config
from buscador.motor import MotorDeBusca
from interface import estado
from interface.identidade import aplicar_estilo, cabecalho, favicon, icone_markdown, titulo_secao
from interface.percurso import mostrar_percurso
from interface.tema import seletor_tema

st.set_page_config(page_title="Buscador de Letras", page_icon=favicon(), layout="wide")
aplicar_estilo()
marca, aparencia = st.columns([4, 1])
with marca:
    cabecalho()
with aparencia:
    seletor_tema()
    st.button("Começar outra busca", on_click=estado.reiniciar, width="stretch")

IDIOMAS = {"pt": "Português", "en": "Inglês"}
INTENCOES = {"palavras": "Lembro algumas palavras", "assunto": "Lembro do assunto"}
EXEMPLOS = {
    "palavras": ["eu sei que vou te amar", "legião urbana temmpo perdido", "when I find myself in times of trouble"],
    "assunto": ["saudade de casa e vontade de voltar", "uma despedida de um grande amor", "seguir em frente apesar das dificuldades"],
}


@st.cache_resource(show_spinner="Preparando a coleção de letras…")
def carregar_motor():
    motor = MotorDeBusca.carregar()
    _ = motor.corretor
    if motor.tem_semantico:
        try:
            motor.semantico.vetor_consulta("aquecimento")
        except (OSError, RuntimeError, ValueError):
            # A busca por palavras permanece disponível se o modelo local não carregar.
            motor.semantico = None
    return motor


@st.cache_data(max_entries=128, show_spinner=False)
def buscar(consulta, intencao, idiomas, generos):
    modo = "bm25" if intencao == "palavras" else "semantico"
    return carregar_motor().buscar(consulta, modo=modo, k=20, idiomas=idiomas,
                                  generos=generos, corrigir=intencao == "palavras")


def registrar_feedback(chave, resposta, resultado, valor):
    if st.session_state.get(chave) == valor:
        st.session_state[chave] = None
        return
    st.session_state[chave] = valor
    caminho = config.ARQUIVO_FEEDBACK
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "consulta": resposta.consulta_usada, "consulta_digitada": resposta.consulta,
        "modo": resposta.modo, "posicao": resultado.posicao,
        "link_musica": resultado.link.removeprefix(config.URL_VAGALUME),
        "titulo": resultado.titulo, "artista": resultado.artista, "relevante": valor,
    }]).to_csv(caminho, mode="a", header=not caminho.exists(), index=False)
    st.toast(f"{icone_markdown('confirmacao')} Obrigado! Sua resposta ajuda a melhorar o buscador.")


def mostrar_resultado(resultado, resposta, investigar=True):
    with st.container(border=True):
        st.markdown(
            f"<div class='res-titulo'>{html.escape(resultado.titulo)} "
            f"<span class='res-artista'>— {html.escape(resultado.artista)}</span></div>",
            unsafe_allow_html=True)
        st.caption(IDIOMAS.get(resultado.idioma, resultado.idioma))
        if resultado.trecho_html:
            st.markdown(f"<div class='trecho'>{resultado.trecho_html}</div>", unsafe_allow_html=True)
        with st.expander("Ver letra completa"):
            st.text(resultado.letra)
            st.link_button("Abrir no Vagalume", resultado.link)
        if investigar:
            st.button(f"{icone_markdown('como-funciona')} Como esta música apareceu?",
                      key=f"abrir_{resultado.doc_id}", on_click=estado.abrir_percurso,
                      args=(resultado.doc_id,), width="stretch")
        with st.container(key=f"feedback_{resultado.doc_id}"):
            st.caption("É o que você procurava?")
            chave = f"voto|{resposta.modo}|{resposta.consulta_usada}|{resultado.doc_id}"
            colunas = st.columns(2)
            for coluna, valor, icone, rotulo in zip(colunas, (1, 0),
                                                   ("voto-positivo", "voto-negativo"), ("Sim", "Não")):
                with coluna:
                    st.button(f"{icone_markdown(icone)} {rotulo}", key=f"{chave}|{valor}",
                              type="primary" if st.session_state.get(chave) == valor else "secondary",
                              on_click=registrar_feedback, args=(chave, resposta, resultado, valor),
                              width="stretch")


def mostrar_exemplos(intencao, sufixo="inicio"):
    st.caption("Escolha uma pista para experimentar. O exemplo começa sem filtros.")
    with st.container(key=f"exemplos_{sufixo}"):
        colunas = st.columns(3)
        for i, exemplo in enumerate(EXEMPLOS[intencao]):
            colunas[i].button(exemplo, key=f"ex_{sufixo}_{i}", on_click=estado.usar_exemplo,
                              args=(exemplo,), width="stretch")


def apoio(intencao):
    st.info("Nossa coleção não reúne todas as músicas. Você pode dar outra pista ou experimentar um exemplo. "
            "Às vezes, os resultados também não correspondem à lembrança que você tem.")
    st.button("Vou dar outra pista", key="outra_pista", on_click=estado.voltar_resultados)
    st.caption("Edite o campo de busca acima e tente incluir um trecho ou o nome do artista, "
               "ou descreva o assunto com mais detalhes.")
    if st.toggle("Quero experimentar um exemplo", key="mostrar_exemplos_apoio"):
        mostrar_exemplos(intencao, "apoio")


if not MotorDeBusca.indice_existe():
    st.error("A coleção de letras ainda não está pronta neste computador.")
    with st.expander("Orientações para a equipe do estande"):
        st.code("python scripts/baixar_dados.py\npython scripts/construir_indice.py", language="bash")
    st.stop()

motor = carregar_motor()
intencao = st.radio("Como você se lembra da música?", list(INTENCOES), key="intencao",
                    format_func=INTENCOES.get, horizontal=True,
                    captions=["Um trecho da letra, o título ou o artista.", "Descreva o tema, mesmo sem lembrar a letra."],
                    on_change=estado.mudar_intencao)
indisponivel = intencao == "assunto" and not motor.tem_semantico
if indisponivel:
    st.info("A busca pelo assunto está indisponível neste computador. "
            "Você pode continuar escolhendo ‘Lembro algumas palavras’.")

with st.form("busca"):
    st.text_input("O que você lembra?", key="consulta", max_chars=300,
                  placeholder="Um pedaço da letra, um artista…" if intencao == "palavras"
                  else "Ex.: saudade de casa e vontade de voltar")
    # O primeiro submit do formulário é acionado pelo Enter.
    st.form_submit_button("Buscar música", type="primary", on_click=estado.iniciar_busca,
                           disabled=indisponivel, width="stretch")
    with st.expander("Refinar busca"):
        st.multiselect("Idioma", list(IDIOMAS), key="idiomas", format_func=IDIOMAS.get, placeholder="Todos")
        st.multiselect("Gênero", motor.generos_disponiveis(), key="generos", placeholder="Todos",
                       help="São os gêneros cadastrados para o artista; uma música pode ter outro estilo.")
        st.form_submit_button("Limpar filtros", on_click=estado.limpar_filtros)

pedido = st.session_state.get("pedido")
if not pedido:
    if not indisponivel:
        mostrar_exemplos(intencao)
    st.caption(f"Uma coleção de {motor.n_docs:,} músicas em português e inglês. "
               "Escolha uma pista, encontre músicas e descubra como a busca funciona.".replace(",", "."))
    st.stop()

consulta, intencao_pedido, idiomas, generos = pedido
if not consulta:
    st.info("Escreva uma pista ou escolha um dos exemplos para começar.")
    if not indisponivel:
        mostrar_exemplos(intencao)
    st.stop()
if indisponivel:
    st.stop()
with st.spinner("Procurando músicas para sua pista…"):
    resposta = buscar(*pedido)
if idiomas or generos:
    st.caption("Filtros desta busca: " + " · ".join([IDIOMAS[i] for i in idiomas] + list(generos)))
if resposta.corrigida:
    st.info(f"{icone_markdown('informacao')} Ajustamos a pista para: **{resposta.consulta_usada}**. "
            "Você poderá ver essa transformação no percurso.")
if resposta.erro or not resposta.resultados:
    st.write("Ainda não encontramos uma música para essa pista.")
    apoio(intencao_pedido)
    st.stop()

investigar = st.session_state.get("investigar")
selecionado = next((r for r in resposta.resultados if r.doc_id == investigar), None)
if selecionado:
    st.button("Voltar aos resultados", on_click=estado.voltar_resultados)
    st.caption(f"Sua busca: {resposta.consulta} · {INTENCOES[intencao_pedido]}")
    mostrar_resultado(selecionado, resposta, investigar=False)
    mostrar_percurso(motor, resposta, selecionado, idiomas, generos)
else:
    titulo_secao("Músicas para sua pista", "buscar")
    st.caption(f"Sua busca: {resposta.consulta}")
    n = st.session_state.get("quantidade", 5)
    for resultado in resposta.resultados[:n]:
        mostrar_resultado(resultado, resposta)
    if n < len(resposta.resultados):
        st.button("Ver mais músicas", on_click=estado.mais_resultados, width="stretch")
    with st.expander("Não encontrou a música?"):
        apoio(intencao_pedido)
