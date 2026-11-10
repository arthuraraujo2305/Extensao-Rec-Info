"""Estado da visita e callbacks que não alteram a preferência de tema."""
import streamlit as st


def limpar_investigacao():
    for chave in list(st.session_state):
        if chave in {"investigar", "oficina", "regras_resultado"} or chave.startswith(("bool_", "pista_")):
            del st.session_state[chave]


def iniciar_busca():
    limpar_investigacao()
    st.session_state["quantidade"] = 5
    st.session_state["pedido"] = (
        st.session_state.get("consulta", "").strip(),
        st.session_state.get("intencao", "palavras"),
        tuple(st.session_state.get("idiomas", [])),
        tuple(st.session_state.get("generos", [])),
    )


def mudar_intencao():
    limpar_investigacao()
    st.session_state.pop("pedido", None)


def usar_exemplo(consulta):
    st.session_state["consulta"] = consulta
    st.session_state["idiomas"] = []
    st.session_state["generos"] = []
    iniciar_busca()


def limpar_filtros():
    st.session_state["idiomas"] = []
    st.session_state["generos"] = []
    if st.session_state.get("pedido"):
        iniciar_busca()


def reiniciar():
    limpar_investigacao()
    st.session_state["quantidade"] = 5
    for chave in list(st.session_state):
        if chave.startswith(("voto|", "resposta_voto|")):
            del st.session_state[chave]
    st.session_state.update(consulta="", intencao="palavras", idiomas=[], generos=[])
    st.session_state.pop("pedido", None)
    st.session_state.pop("mostrar_exemplos_apoio", None)


def abrir_percurso(doc):
    limpar_investigacao()
    st.session_state["investigar"] = doc


def voltar_resultados():
    limpar_investigacao()


def mais_resultados():
    st.session_state["quantidade"] = min(20, st.session_state.get("quantidade", 5) + 5)
