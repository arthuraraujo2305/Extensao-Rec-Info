# 🎵 Buscador de Letras

Projeto de extensão da disciplina de **Recuperação da Informação**.

Um buscador de letras de música em **português e inglês** para quem lembra só um
pedaço da letra, lembra a letra errado, sabe só o artista ou sabe apenas **do que a
música fala**. A interface é feita em Streamlit e cada parte do motor de busca
corresponde a um conteúdo da disciplina.

```
"eu sei que vou te amar"                 → Eu Sei Que Vou Te Amar — Tom Jobim
"legião urbana tempo"                    → Tempo Perdido — Legião Urbana
"imagin all the peple"   (com erros)     → corrigido para "imagine all the people"
"música triste sobre saudade de casa"    → busca semântica, acha músicas em PT e EN
artista:"legião urbana" AND (chuva OR sol)   → busca booleana
```

## Conteúdo da disciplina → onde está no código

| Aula | Conteúdo | Implementação |
|---|---|---|
| 02 | Modelo booleano | [buscador/booleano.py](buscador/booleano.py): parser com AND/OR/NOT, frases, campos e parênteses |
| 03 | Modelo probabilístico | [buscador/bm25.py](buscador/bm25.py): BM25 implementado do zero |
| 04 | Modelo vetorial | [buscador/vetorial.py](buscador/vetorial.py): TF-IDF + cosseno |
| 05 | Avaliação de RI | [buscador/avaliacao.py](buscador/avaliacao.py): MRR, Sucesso@k, nDCG, P@k, MAP; [notebooks/02_avaliacao.ipynb](notebooks/02_avaliacao.ipynb) |
| 07 | Indexação | [buscador/texto.py](buscador/texto.py) (tokenização) e [buscador/indice.py](buscador/indice.py) (índice invertido) |
| 08 | Dicionários e tolerância | [buscador/tolerancia.py](buscador/tolerancia.py): distância de edição, índice de bigramas, correção sensível ao contexto, curingas |
| 09–10 | Recuperação semântica | [buscador/semantico.py](buscador/semantico.py): embeddings multilíngues por trecho da letra |
| 11 | Busca híbrida e reranking | [buscador/hibrido.py](buscador/hibrido.py) (CombSUM e RRF) e cross-encoder em [buscador/semantico.py](buscador/semantico.py) |
| 12 | Busca na Web | popularidade do artista como sinal de "autoridade" em [buscador/motor.py](buscador/motor.py) |

## Como rodar

Pré-requisito: [Anaconda/Miniconda](https://docs.conda.io/). Não precisa de GPU nem de conta no Kaggle.

```bash
# 1. criar e ativar o ambiente (uma vez só)
conda env create -f environment.yml
conda activate recinfo

# 2. baixar a base de letras (~130 MB, uma vez só)
python scripts/baixar_dados.py

# 3. montar o corpus, os índices e os embeddings (uma vez só)
python scripts/construir_indice.py
#    os índices ficam prontos em segundos; os embeddings da busca semântica
#    levaram ~45 min numa CPU de 16 threads sem GPU (77 mil trechos de letra).
#    Alternativas mais rápidas:
#    python scripts/construir_indice.py --sem-semantico        # segundos, sem busca semântica
#    python scripts/construir_indice.py --por-idioma 5000      # metade do corpus, ~metade do tempo

# 4. abrir o buscador no navegador
streamlit run app.py
```

Opcionais:

```bash
python scripts/avaliar.py      # compara os modelos e salva os resultados em eval/
pytest                         # testes automatizados
jupyter lab notebooks/         # notebooks de exploração e avaliação
```

Na primeira execução, o modelo de embeddings (~470 MB) é baixado do Hugging Face.
O reranking baixa outro modelo do mesmo tamanho na primeira vez que é usado.

**Dica para o grupo:** quem já gerou `data/processed/` (~100 MB) pode compartilhar a
pasta (ex.: pelo Drive). Assim os outros só baixam e pulam o passo 3.

## Estrutura

```
app.py                     interface Streamlit (busca, avaliação, explicação)
interface/                 identidade visual e componentes de apresentação
static/                    marca, ícones e estilos servidos pelo Streamlit
buscador/                  o motor de busca (um módulo por conteúdo da disciplina)
  motor.py                 junta tudo: modos, filtros, correção, reranking, trechos
  dados.py                 leitura da base e montagem do corpus
scripts/
  baixar_dados.py          baixa a base do Kaggle
  construir_indice.py      gera data/processed/ (corpus, índices, embeddings)
  avaliar.py               avaliação automática → eval/
notebooks/
  01_exploracao.ipynb      corpus, índice invertido, leis de Zipf e Heaps, modelos lado a lado
  02_avaliacao.ipynb       avaliação completa com gráficos e análise de erros
tests/                     testes com um corpus pequeno
eval/                      resultados da avaliação (CSV)
data/                      dados brutos e índices (fora do git)
```

### Identidade visual

Os arquivos visuais ficam em `static/brand/` (marca), `static/icons/` (ícones) e
`static/styles/` (CSS). A pasta `static/` é servida pelo próprio Streamlit, com
`server.enableStaticServing = true` em `.streamlit/config.toml`. Coloque nessa
pasta somente arquivos públicos da interface.

O PNG original da marca contém o logotipo com símbolo e o símbolo isolado.
`interface/identidade.py` enquadra cada versão com SVG, preservando a arte:
marca completa no cabeçalho, símbolo com nome em texto nas telas pequenas e
símbolo isolado no favicon. Se a imagem original for substituída por outra
composição, ajuste as janelas definidas nesse módulo.

O roxo aparece na navegação, nos títulos e nos exemplos; o amarelo continua
destacando as palavras encontradas. Na barra lateral, o controle **Tema escuro**
alterna entre os dois temas sem perder a busca. A preferência fica no navegador,
por aplicativo, e não muda o tema dos outros visitantes. Os textos e ícones roxos
ficam mais claros no fundo escuro.

As duas paletas estão em `.streamlit/config.toml`, usando os
[temas nativos do Streamlit](https://docs.streamlit.io/develop/concepts/configuration/theming).
`interface/tema.py` e `interface/tema.js` apresentam o controle na barra lateral
e acionam os seletores do menu nativo. Essa integração foi testada com Streamlit
1.67 e depende dos identificadores dos botões desse menu; ao atualizar o
Streamlit, confira a troca por clique e teclado, persistência e preservação da
consulta. Se os seletores mudarem, o controle orienta usar o menu original.

Após mudar a configuração de temas ou arquivos estáticos, reinicie
`streamlit run app.py`. A versão mínima do ambiente é Streamlit 1.67.

## Dados

Base [Song lyrics from 79 musical genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)
(letras do site Vagalume, publicada no Kaggle por Anderson Neisse), com ~380 mil letras.
Usamos as músicas dos artistas mais populares: **10 mil em português e 10 mil em inglês**,
com no máximo 40 por artista. Removemos as letras vazias e as duplicadas. O tamanho é
configurável com `--por-idioma` e `--max-por-artista`.

As letras têm direitos autorais. Por isso a pasta `data/` fica fora do repositório e
cada pessoa baixa a base localmente.

## Decisões de projeto

- **Sem remoção de stopwords nem stemming.** Em letra de música, as palavras
  "vazias" fazem parte do trecho que a pessoa lembra ("let it be"). O IDF do
  BM25/TF-IDF já dá peso baixo para elas.
- **Acentos removidos** na indexação e na consulta: "coracao" encontra "coração".
- **Título e artista no mesmo documento da letra.** BM25 e TF-IDF usam um índice
  em que cada música é título + artista (repetidos 2x, para valerem mais) + letra.
  Por isso "legião urbana tempo" funciona sem sintaxe especial. Há também um índice
  por campo, usado no booleano e em restrições como `artista:queen love`.
  *A avaliação nos fez mudar isso:* a primeira versão somava as notas de índices
  separados por campo. Palavras comuns nas letras ("love", "you") são raras nos
  títulos, então ganhavam IDF alto ali, e qualquer título com "Love" passava na frente
  da música que tinha o trecho inteiro na letra. O MRR do BM25 em trechos exatos
  passou de 0,39 para 0,88 com a mudança.
- **Embeddings por trecho** (janelas de 80 palavras): a nota da música é a do
  trecho mais parecido com a consulta, e esse trecho é o que aparece no resultado.
- **Modelo multilíngue** ([paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)):
  uma consulta em português encontra músicas em inglês e vice-versa.
- **Híbrido com CombSUM, não RRF.** O híbrido soma as notas normalizadas:
  70% BM25 + 30% semântico. Com RRF, que usa só as posições, o híbrido ficava muito
  pior que o BM25 nas buscas por trecho: quando o BM25 tem certeza, o semântico
  derrubava a música certa. O peso 0,7 foi escolhido num conjunto de
  desenvolvimento separado (outra semente), não no conjunto em que reportamos os resultados.
- **Correção sensível ao contexto**: entre candidatos igualmente próximos, vence o
  que mais aparece junto com as outras palavras da consulta ("coracao partdo" →
  "partido", e não "parado").

## Avaliação

Resultados em [eval/](eval/), e a análise completa em
[notebooks/02_avaliacao.ipynb](notebooks/02_avaliacao.ipynb).

**Busca por item conhecido.** 200 músicas sorteadas (100 PT + 100 EN). Para cada
uma, 4 consultas simulam como alguém buscaria por ela: trecho exato de 6 a 8 palavras,
o mesmo trecho com erros de digitação, 5 palavras soltas fora de ordem, e artista
+ 4 palavras. A música de origem é a resposta certa. Total: 800 consultas.

| Modelo | MRR | Sucesso@1 | Sucesso@10 | nDCG@10 |
|---|---|---|---|---|
| Booleano (AND) | 0,610 | 0,561 | 0,698 | 0,624 |
| Booleano + correção | 0,728 | 0,664 | 0,845 | 0,749 |
| TF-IDF | 0,613 | 0,519 | 0,816 | 0,658 |
| BM25 | 0,847 | 0,804 | 0,934 | 0,864 |
| **BM25 + correção** | **0,884** | **0,844** | **0,962** | **0,898** |
| Semântico | 0,137 | 0,102 | 0,216 | 0,153 |
| Híbrido RRF + correção | 0,477 | 0,336 | 0,859 | 0,562 |
| Híbrido CombSUM + correção | 0,871 | 0,820 | 0,961 | 0,888 |

MRR por tipo de consulta:

| Modelo | trecho exato | com erros | palavras soltas | artista + trecho |
|---|---|---|---|---|
| Booleano (AND) | 0,723 | 0,000 | 0,813 | 0,901 |
| Booleano + correção | 0,715 | 0,502 | 0,803 | 0,891 |
| TF-IDF | 0,682 | 0,387 | 0,754 | 0,630 |
| BM25 | 0,877 | 0,655 | 0,922 | 0,936 |
| BM25 + correção | 0,877 | 0,804 | 0,922 | 0,932 |
| Semântico | 0,243 | 0,084 | 0,163 | 0,058 |
| Híbrido RRF + correção | 0,536 | 0,476 | 0,489 | 0,406 |
| Híbrido CombSUM + correção | 0,872 | 0,802 | 0,905 | 0,904 |

O que os números mostram:

- **BM25 é o melhor modelo para busca por trecho** e supera com folga o TF-IDF,
  que favorece letras curtas.
- **A correção ortográfica** recupera boa parte das buscas com erros: no BM25, o
  MRR nas consultas com erros sobe de 0,66 para 0,80. No booleano, que sem correção
  acerta 0% dessas consultas, sobe para 0,50.
- **O booleano** acerta bem quando a pessoa lembra as palavras exatas, mas não
  ranqueia e falha com qualquer erro de digitação.
- **O semântico vai mal aqui, e isso é esperado:** essa avaliação mede a lembrança
  de palavras exatas. O ponto forte dele é a busca por tema ("música sobre saudade
  de casa"), que só dá para medir com julgamentos humanos (os votos 👍/👎 do app).
- **A forma de fundir importa:** o híbrido com RRF perde muito para o BM25, enquanto
  com CombSUM fica praticamente empatado e ainda aproveita o sinal semântico.

Limitação: as consultas são geradas a partir das próprias letras, o que favorece os
modelos léxicos. Para avaliar a busca por tema, use o feedback dos usuários
(`python scripts/avaliar.py` inclui essas métricas quando há votos).

## Extensão: envolvendo a comunidade

Sugestões para a parte de extensão:

1. **Coleta de julgamentos com usuários reais.** Os botões 👍/👎 do app salvam votos
   em `data/feedback.csv`. Esses votos viram um conjunto de relevância para avaliar
   os modelos com consultas reais (aba *Avaliação* do app e `scripts/avaliar.py`).
2. **Oficina/demonstração** em uma escola, grupo de música, coral ou rádio
   comunitária: pedir que as pessoas busquem músicas de que lembram só um pedaço e
   registrar o que funcionou e o que não funcionou.
3. **Publicar o app** (ex.: Streamlit Community Cloud) para quem não tem Python
   instalado. Para isso é preciso gerar o índice no servidor, porque `data/` não vai para o git.
