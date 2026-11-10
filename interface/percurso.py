"""Percurso explorável: evidências da consulta, do índice e da ordenação."""
import altair as alt
import pandas as pd
import streamlit as st

from buscador.explicacao import (explicar_palavras, explicar_significado, experimentar_regras,
                                 palavras_da_busca)
from buscador.texto import ErroConsulta, tokenizar
from interface.identidade import titulo_secao


def oficina_booleana(motor, resposta, resultado, idiomas, generos):
    st.markdown("#### E se exigíssemos todas as palavras?")
    st.write("O modelo booleano verifica regras: uma música entra ou fica de fora. "
             "Monte uma regra para descobrir o que acontece com esta música.")
    if not st.toggle("Experimentar uma busca com regras", key="oficina"):
        return
    if "bool_todas" not in st.session_state:
        st.session_state["bool_todas"] = " ".join(palavras_da_busca(motor, resposta.consulta_usada))
    with st.form("montagem_booleana"):
        st.text_input("Todas estas palavras", key="bool_todas", max_chars=180,
                      help="A música precisa ter cada uma delas: AND (E).")
        st.text_input("Pelo menos uma destas palavras", key="bool_alguma", max_chars=180,
                      help="A música precisa ter ao menos uma delas: OR (OU).")
        st.text_input("Nenhuma destas palavras", key="bool_nenhuma", max_chars=180,
                      help="Se tiver qualquer uma delas, fica de fora: NOT (NÃO).")
        executar = st.form_submit_button("Buscar com estas regras", type="primary")
    if executar or "regras_resultado" not in st.session_state:
        try:
            st.session_state["regras_resultado"] = experimentar_regras(
                motor, st.session_state["bool_todas"], st.session_state.get("bool_alguma", ""),
                st.session_state.get("bool_nenhuma", ""), resultado.doc_id, idiomas, generos)
        except ErroConsulta as erro:
            st.session_state["regras_resultado"] = {"erro": str(erro)}
    dados = st.session_state["regras_resultado"]
    if "erro" in dados:
        st.info(dados["erro"])
        return
    st.caption("Esta é a expressão que suas escolhas produziram:")
    st.code(dados["expressao"], language=None)
    st.caption("AND exige todas; OR aceita alternativas; NOT exclui. Os grupos preenchidos "
               "precisam ser satisfeitos juntos. As palavras são normalizadas, sem nova correção automática.")
    if dados["atende"]:
        st.success(f"“{resultado.titulo}” atende às regras escolhidas.")
    else:
        st.info(f"“{resultado.titulo}” fica de fora com estas regras. Veja as pistas abaixo.")
    for linha in dados["regras"]:
        presenca = "presente" if linha["presente"] else "ausente"
        st.write(f"**{linha['palavra']}** · {linha['regra'].lower()} · {presenca} nesta música")
    st.write(f"**{dados['total']} músicas** atendem a essas condições e aos filtros atuais.")
    for item in dados["exemplos"]:
        st.write(f"{item['titulo']} — {item['artista']}")
    st.caption("Exibimos até cinco, em ordem alfabética de artista e título. "
               "Aqui não há nota para dizer qual é mais relevante.")


def mostrar_percurso(motor, resposta, resultado, idiomas=(), generos=()):
    titulo_secao("Siga as pistas desta busca", "como-funciona")
    st.write("Escolha uma etapa. Você pode voltar e explorar no seu ritmo.")
    palavras = resposta.modo == "bm25"
    detalhe = (explicar_palavras(motor, resposta.consulta_usada, resultado.doc_id, idiomas, generos)
               if palavras else explicar_significado(motor, resposta.consulta_usada, resultado.doc_id))
    sua_pista, onde, por_que = st.tabs(["1 · Sua pista", "2 · Onde procuramos", "3 · Por que apareceu?"])
    with sua_pista:
        st.markdown("### A busca começa com o que você lembra")
        st.write("**Você escreveu:**", resposta.consulta)
        if palavras:
            if resposta.corrigida:
                st.write("**Após a correção:**", resposta.consulta_usada)
            else:
                st.caption("Nenhuma correção de digitação foi aplicada.")
            st.write("**Palavras preparadas:**", " · ".join(tokenizar(resposta.consulta_usada)))
            st.write("Minúsculas e remoção de acentos permitem que ‘coração’ e ‘coracao’ "
                     "apontem para a mesma palavra. As palavras pequenas também são preservadas.")
        else:
            st.write("Sua descrição foi preservada. O modelo transforma esse texto em uma sequência "
                     "de números para compará-lo com trechos de letras.")
            st.caption(f"Nesta busca, a representação tem {detalhe['dimensoes']} números. "
                       "Cada número isolado não corresponde a uma palavra ou emoção específica.")
        st.info("As letras e seus índices já foram preparados antes da sua visita. "
                "Agora usamos sua pista para consultar essa coleção.")
    with onde:
        st.markdown("### Onde sua pista encontra a coleção")
        if palavras:
            st.write("O índice guarda uma lista de músicas para cada palavra. Clique em uma pista "
                     "para abrir uma dessas listas — sem percorrer todas as letras novamente.")
            termos = [t["termo"] for t in detalhe["termos"]]
            if termos:
                escolhido = st.radio("Qual palavra você quer investigar?", termos,
                                     horizontal=True, key=f"pista_{resultado.doc_id}")
                linha = next(t for t in detalhe["termos"] if t["termo"] == escolhido)
                st.metric("Músicas ligadas a esta palavra na coleção", linha["musicas"])
                if idiomas or generos:
                    st.caption(f"Com seus filtros: {linha['musicas_filtradas']} músicas.")
                if linha["campos"]:
                    campos = ", ".join({"titulo": "título", "artista": "artista", "letra": "letra"}[c]
                                       for c in linha["campos"])
                    st.success(f"Nesta música, a pista aparece em: {campos}.")
                else:
                    st.info("Esta pista não aparece na música escolhida. Outras palavras contribuíram.")
                st.caption("Alguns exemplos reais dessa lista, respeitando seus filtros:")
                for nome in linha["exemplos"]:
                    st.write(nome)
                if not linha["exemplos"]:
                    st.write("Nenhum exemplo encontrado com os filtros atuais.")
            st.caption(f"A coleção contém {motor.n_docs:,} músicas. "
                       f"Sua busca recuperou {resposta.total} candidatas com os filtros atuais.")
        else:
            st.write("Antes da busca, as letras foram divididas em trechos e representadas por números. "
                     "Agora comparamos sua descrição com essas representações.")
            st.metric("Trechos preparados na coleção", detalhe["total_trechos"])
            st.caption("O modelo é multilíngue: a descrição e a letra podem estar em idiomas diferentes. "
                       "Os filtros de idioma continuam valendo.")
            st.markdown("**Explore um trecho desta música:**")
            indice = st.selectbox("Trecho", range(len(detalhe["trechos"])),
                                  index=detalhe["melhor"], format_func=lambda i: f"Trecho {i + 1}")
            trecho = detalhe["trechos"][indice]
            st.write(trecho["texto"])
            st.caption(f"Semelhança calculada: {trecho['semelhanca']:.3f}. É uma medida de comparação, "
                       "não uma porcentagem de certeza.")
    with por_que:
        st.markdown("### O que colocou esta música nos resultados")
        if palavras:
            tabela = pd.DataFrame(detalhe["termos"])
            positivas = tabela[tabela["contribuicao"] > 0].sort_values("contribuicao", ascending=False)
            if not positivas.empty:
                forte = positivas.iloc[0]
                st.write(f"**{forte['termo']}** foi a palavra que mais contribuiu para a nota desta música.")
                grafico = alt.Chart(positivas.head(8)).mark_bar(color="#8541ee").encode(
                    x=alt.X("contribuicao:Q", title="Contribuição para a nota"),
                    y=alt.Y("termo:N", title="Palavra", sort="-x"),
                    tooltip=[alt.Tooltip("termo:N", title="Palavra"),
                             alt.Tooltip("contribuicao:Q", title="Contribuição", format=".3f")],
                ).properties(height=max(120, min(8, len(positivas)) * 32))
                st.altair_chart(grafico, width="stretch")
                if len(positivas) > 8:
                    st.caption("As oito maiores contribuições estão no gráfico. "
                               "Todas as palavras aparecem na tabela do aprofundamento.")
            st.write("O BM25 combina pistas. Uma palavra menos comum na coleção pode ajudar mais "
                     "a distinguir uma música de outra; não é preciso encontrar todas as palavras.")
            with st.expander("Entenda o BM25: pesos, repetições e tamanho"):
                st.write("Título, artista e letra formam o documento pesquisado. Título e artista "
                         "entram duas vezes para dar mais peso a essas informações.")
                st.write("A raridade (IDF) mede o quanto uma palavra distingue documentos. "
                         "Repetições ajudam com ganho decrescente; o tamanho do documento também é ajustado.")
                st.dataframe(tabela[["termo", "musicas", "ocorrencias", "raridade", "contribuicao"]].rename(
                    columns={"termo": "Palavra", "musicas": "Músicas na coleção", "ocorrencias": "Ocorrências ponderadas",
                             "raridade": "Raridade (IDF)", "contribuicao": "Contribuição"}),
                    hide_index=True, width="stretch")
                st.caption(f"Nota desta música: {resultado.score:.3f}. Documento: {detalhe['tamanho']} palavras "
                           f"ponderadas; média da coleção: {detalhe['media']:.1f}. "
                           f"Parâmetros: k₁ = {detalhe['k1']}; b = {detalhe['b']}.")
                st.latex(r"\mathrm{nota}(d,q)=\sum_{t\in q}\mathrm{IDF}(t)"
                         r"\frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b|d|/\mathrm{média})}")
                st.caption("A nota ordena as músicas; ela não mede a chance de esta ser a música que você tinha em mente.")
            oficina_booleana(motor, resposta, resultado, idiomas, generos)
        else:
            melhor = detalhe["trechos"][detalhe["melhor"]]
            st.write("Este foi o trecho com maior semelhança com sua descrição:")
            st.info(melhor["texto"])
            st.write("A nota da música é a maior semelhança entre sua descrição e os trechos dela. "
                     "Esse sinal ajuda a descobrir temas próximos, mas pode sugerir algo diferente do que você imaginou.")
            with st.expander("Entenda a busca por significado: vetores e comparação"):
                st.write("O modelo foi treinado para aproximar representações de textos com sentidos relacionados. "
                         "Usamos o produto entre os vetores normalizados, equivalente à semelhança do cosseno. "
                         "Não é uma leitura humana nem uma explicação palavra por palavra.")
                st.dataframe(pd.DataFrame([
                    {"Trecho": i + 1, "Semelhança": t["semelhanca"], "Maior semelhança": i == detalhe["melhor"]}
                    for i, t in enumerate(detalhe["trechos"])]), hide_index=True, width="stretch")
                st.caption(f"Modelo: {detalhe['modelo']} · Nota da música: {resultado.score:.3f}. "
                           "Os vetores dos trechos foram preparados antes da visita.")
