"""Interface do buscador de letras.

    streamlit run app.py
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from buscador import config
from buscador.avaliacao import avaliar_feedback
from buscador.motor import MODOS, MODOS_SEMANTICOS, MotorDeBusca, RespostaBusca

st.set_page_config(page_title="Buscador de Letras", page_icon="🎵", layout="wide")

st.markdown("""
<style>
mark { background: #ffd54f; color: #1a1a1a; padding: 0 3px; border-radius: 3px; }
.res-titulo { font-size: 1.15rem; font-weight: 600; line-height: 1.3; }
.res-artista { font-weight: 400; opacity: .75; }
.trecho { font-size: .98rem; line-height: 1.6; margin: .2rem 0 .4rem; }
</style>""", unsafe_allow_html=True)

DESCRICOES = {
    "bm25": "Ranking probabilístico. Ótimo para trechos de letra, título e artista.",
    "tfidf": "Modelo vetorial clássico (cosseno). Para comparar com o BM25.",
    "booleano": "Use AND, OR, NOT, \"frases\", artista:, titulo:, curinga*.",
    "semantico": "Busca pelo significado, mesmo sem palavras em comum (PT ↔ EN).",
    "hibrido": "Junta BM25 (70%) + semântico (30%). Bom para qualquer tipo de busca.",
}
IDIOMAS = {"pt": "🇧🇷 Português", "en": "🇺🇸 Inglês"}
EXEMPLOS = [
    ("eu sei que vou te amar", "bm25"),
    ("legião urbana tempo perdido", "bm25"),
    ("imagin all the peple", "bm25"),
    ("música triste sobre saudade de casa", "semantico"),
    ("song about a broken heart", "hibrido"),
    ('artista:"legião urbana" AND (chuva OR sol)', "booleano"),
]


# ---------------------------------------------------------------------------- carga
@st.cache_resource(show_spinner="Carregando índices e modelos…")
def carregar_motor() -> MotorDeBusca:
    motor = MotorDeBusca.carregar()
    if motor.tem_semantico:
        motor.semantico.vetor_consulta("aquecimento")  # carrega o modelo já na abertura do app
    _ = motor.corretor
    return motor


@st.cache_data(max_entries=300, show_spinner=False)
def buscar(consulta: str, modo: str, k: int, idiomas: tuple, generos: tuple,
           corrigir: bool, reordenar: bool, popularidade: bool) -> RespostaBusca:
    return carregar_motor().buscar(consulta, modo=modo, k=k, idiomas=idiomas, generos=generos,
                                   corrigir=corrigir, reordenar=reordenar, usar_popularidade=popularidade)


if not MotorDeBusca.indice_existe():
    st.title("🎵 Buscador de Letras")
    st.error("O índice ainda não foi construído.")
    st.code("python scripts/baixar_dados.py\npython scripts/construir_indice.py", language="bash")
    st.stop()

motor = carregar_motor()
modos_disponiveis = [m for m in MODOS if motor.tem_semantico or m not in MODOS_SEMANTICOS]


# ---------------------------------------------------------------------------- callbacks
def usar_exemplo(consulta: str, modo: str) -> None:
    st.session_state["consulta"] = consulta
    st.session_state["modo"] = modo if modo in modos_disponiveis else "bm25"


def registrar_feedback(chave: str, resposta: RespostaBusca, posicao: int, link: str,
                       titulo: str, artista: str) -> None:
    valor = st.session_state.get(chave)
    if valor is None:
        return
    linha = pd.DataFrame([{
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "consulta": resposta.consulta_usada, "consulta_digitada": resposta.consulta,
        "modo": resposta.modo, "posicao": posicao,
        "link_musica": link.removeprefix(config.URL_VAGALUME),
        "titulo": titulo, "artista": artista, "relevante": int(valor),
    }])
    caminho = config.ARQUIVO_FEEDBACK
    caminho.parent.mkdir(parents=True, exist_ok=True)
    linha.to_csv(caminho, mode="a", header=not caminho.exists(), index=False)
    st.toast("Obrigado! Seu voto ajuda a avaliar o buscador. 🙌")


# ---------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configurações")
    modo = st.radio("Modelo de busca", modos_disponiveis, key="modo", format_func=MODOS.get,
                    captions=[DESCRICOES[m] for m in modos_disponiveis])
    if not motor.tem_semantico:
        st.caption("ℹ️ Busca semântica desativada: gere os embeddings com `construir_indice.py`.")

    st.subheader("Filtros")
    idiomas = st.multiselect("Idioma", list(IDIOMAS), format_func=IDIOMAS.get, placeholder="Todos")
    generos = st.multiselect("Gênero", motor.generos_disponiveis(minimo=30), placeholder="Todos")

    st.subheader("Opções")
    corrigir = st.toggle("Corrigir erros de digitação", value=True)
    popularidade = st.toggle("Priorizar artistas populares", value=False, disabled=modo == "booleano")
    reordenar = st.toggle("Reranking com cross-encoder", value=False, disabled=modo == "booleano",
                          help="Reordena os 30 primeiros com um modelo mais preciso (~1 s por busca). "
                               "Na primeira vez baixa o modelo (~470 MB) e pode levar um minuto.")
    k = st.slider("Número de resultados", 5, 30, 10, step=5)
    st.divider()
    st.caption(f"Corpus: {motor.n_docs:,} músicas · {motor.corpus['artista'].nunique():,} artistas · "
               "letras do Vagalume (Kaggle)".replace(",", "."))


# ---------------------------------------------------------------------------- cabeçalho
st.title("🎵 Buscador de Letras")
st.markdown("Lembra só **um pedaço da letra**? Escreveu **errado**? Só sabe **do que a música fala**? "
            "Busque em português ou inglês.")
aba_busca, aba_avaliacao, aba_sobre = st.tabs(["🔎 Buscar", "📊 Avaliação", "ℹ️ Como funciona"])


# ---------------------------------------------------------------------------- busca
def mostrar_resultado(r, resposta: RespostaBusca) -> None:
    with st.container(border=True):
        col_info, col_link = st.columns([6, 1])
        with col_info:
            st.markdown(f"<div class='res-titulo'>{r.posicao}. {html.escape(r.titulo)} "
                        f"<span class='res-artista'>— {html.escape(r.artista)}</span></div>",
                        unsafe_allow_html=True)
            detalhes = [IDIOMAS.get(r.idioma, r.idioma)]
            if r.generos:
                detalhes.append(r.generos.replace(";", ","))
            if resposta.modo != "booleano":
                detalhes.append(f"nota {r.score:.3f}")
            st.caption(" · ".join(detalhes))
        with col_link:
            st.link_button("Vagalume ↗", r.link, width="stretch")
        if r.trecho_html:
            st.markdown(f"<div class='trecho'>{r.trecho_html}</div>", unsafe_allow_html=True)
        col_letra, col_voto = st.columns([5, 2])
        with col_letra:
            with st.expander("Ver letra completa"):
                st.text(r.letra)
        with col_voto:
            chave = f"voto|{resposta.modo}|{resposta.consulta_usada}|{r.doc_id}"
            st.caption("É o que você procurava?")
            st.feedback("thumbs", key=chave, on_change=registrar_feedback,
                        args=(chave, resposta, r.posicao, r.link, r.titulo, r.artista))


with aba_busca:
    consulta = st.text_input("O que você procura?", key="consulta",
                             placeholder="ex.: eu sei que vou te amar · legião urbana tempo · "
                                         "música triste sobre saudade")
    st.caption("Experimente:")
    colunas = st.columns(3)
    for i, (exemplo, modo_exemplo) in enumerate(EXEMPLOS):
        colunas[i % 3].button(exemplo, key=f"ex{i}", on_click=usar_exemplo, args=(exemplo, modo_exemplo),
                              width="stretch")

    if consulta.strip():
        with st.spinner("Buscando…"):
            resposta = buscar(consulta, modo, k, tuple(idiomas), tuple(generos), corrigir,
                              reordenar and modo != "booleano", popularidade and modo != "booleano")
        if resposta.corrigida:
            st.info(f"Mostrando resultados para **{resposta.consulta_usada}**. "
                    f"(Desligue *Corrigir erros de digitação* para buscar exatamente “{consulta}”.)")
        if resposta.erro:
            st.warning(resposta.erro)
        elif not resposta.resultados:
            st.warning("Nenhuma música encontrada. Tente outras palavras, outro modelo ou remova os filtros.")
        else:
            if resposta.total is not None:
                resumo = f"{resposta.total:,} músicas encontradas".replace(",", ".")
                if modo == "booleano":
                    resumo += " · o modelo booleano não ranqueia: ordenadas por popularidade do artista"
            else:
                resumo = f"{len(resposta.resultados)} músicas mais parecidas"
            st.caption(f"{resumo} · {resposta.tempo_ms:.0f} ms · {MODOS[modo]}")
            for resultado in resposta.resultados:
                mostrar_resultado(resultado, resposta)


# ---------------------------------------------------------------------------- avaliação
with aba_avaliacao:
    st.subheader("Comparação dos modelos")
    resumo_geral = config.PASTA_AVALIACAO / "resumo_geral.csv"
    resumo_tipo = config.PASTA_AVALIACAO / "resumo_por_tipo.csv"
    if resumo_geral.exists():
        st.markdown(
            "Avaliação automática por **item conhecido**: sorteamos músicas, pegamos um pedaço da letra e "
            "simulamos como alguém buscaria por ela. A música de origem é a resposta certa.\n\n"
            "- **MRR**: média de 1/posição da música certa (1 = sempre em 1º lugar)\n"
            "- **Sucesso@k**: fração das buscas com a música certa entre os k primeiros\n"
            "- **nDCG@10**: qualidade do ranking, valorizando acertos nas primeiras posições")
        geral = pd.read_csv(resumo_geral, index_col=0)
        st.dataframe(geral.style.format("{:.3f}").highlight_max(axis=0, color="#2e7d3240"),
                     width="stretch")
        st.bar_chart(geral["MRR"], horizontal=True, y_label="", x_label="MRR")
        if resumo_tipo.exists():
            st.markdown("**MRR por tipo de consulta**")
            por_tipo = pd.read_csv(resumo_tipo).pivot(index="configuracao", columns="tipo", values="MRR")
            por_tipo = por_tipo.reindex(geral.index)
            st.dataframe(por_tipo.style.format("{:.3f}").highlight_max(axis=0, color="#2e7d3240"),
                         width="stretch")
    else:
        st.info("Ainda não há resultados. Rode `python scripts/avaliar.py`.")

    st.subheader("Avaliação pelos usuários")
    if config.ARQUIVO_FEEDBACK.exists():
        feedback = pd.read_csv(config.ARQUIVO_FEEDBACK)
        c1, c2, c3 = st.columns(3)
        c1.metric("Votos", len(feedback))
        c2.metric("Consultas avaliadas", feedback["consulta"].nunique())
        c3.metric("Votos positivos", f"{feedback['relevante'].mean():.0%}")
        if st.button("Calcular métricas com os votos"):
            with st.spinner("Avaliando…"):
                tabela = avaliar_feedback(motor, config.ARQUIVO_FEEDBACK)
            if tabela.empty:
                st.info("Ainda não há votos positivos suficientes.")
            else:
                st.dataframe(tabela, width="stretch")
                st.caption("Músicas não avaliadas contam como não relevantes (P@k é um limite inferior).")
    else:
        st.info("Ninguém votou ainda. Use 👍/👎 nos resultados: os votos viram um conjunto de "
                "julgamentos de relevância para avaliar os modelos com consultas reais.")


# ---------------------------------------------------------------------------- sobre
with aba_sobre:
    st.markdown("""
### Como o buscador funciona

| Etapa | O que faz | Conteúdo da disciplina | Código |
|---|---|---|---|
| Pré-processamento | minúsculas, sem acentos, tokenização | Aula 07 – Indexação | `buscador/texto.py` |
| Índice invertido | termo → lista de músicas (postings) e frequências, para letra, título e artista | Aula 07 – Indexação | `buscador/indice.py` |
| Booleano | AND / OR / NOT, frases, campos, parênteses | Aula 02 – Modelo booleano | `buscador/booleano.py` |
| BM25 | ranking probabilístico com saturação de tf e normalização por tamanho | Aula 03 – Modelo probabilístico | `buscador/bm25.py` |
| TF-IDF | vetores de pesos + similaridade do cosseno | Aula 04 – Modelo vetorial | `buscador/vetorial.py` |
| Tolerância a erros | distância de edição, índice de bigramas, correção sensível ao contexto, curingas | Aula 08 – Dicionários e tolerância | `buscador/tolerancia.py` |
| Semântico | embeddings multilíngues dos trechos das letras | Aulas 09 e 10 – Recuperação semântica | `buscador/semantico.py` |
| Híbrido e reranking | fusão das notas do BM25 e do semântico (CombSUM, comparada com RRF) e cross-encoder | Aula 11 – Busca híbrida e reranking | `buscador/hibrido.py` |
| Popularidade | relevância + "autoridade" do artista | Aula 12 – Busca na Web | `buscador/motor.py` |
| Avaliação | MRR, Sucesso@k, nDCG, P@k, MAP | Aula 05 – Avaliação de RI | `buscador/avaliacao.py` |

### Sintaxe da busca booleana

| Exemplo | Significado |
|---|---|
| `chuva saudade` ou `chuva AND saudade` | as duas palavras (também vale `E`) |
| `amor OR paixão` | qualquer uma das duas (também vale `OU`) |
| `amor NOT traição` ou `amor -traição` | sem a palavra (também vale `NÃO`) |
| `"eu sei que vou te amar"` | frase exata |
| `artista:"legião urbana"` · `titulo:chuva` · `letra:mar` | só naquele campo |
| `apaixon*` | curinga: apaixonado, apaixonada, apaixonei… |
| `(chuva OR sol) AND verão` | parênteses para agrupar |

Os campos (`artista:`, `titulo:`) também funcionam nos modos ranqueados: `artista:queen love`.

---
Projeto de extensão da disciplina de Recuperação da Informação.
As letras pertencem aos seus autores e vêm da base pública do Vagalume no Kaggle.
""")
