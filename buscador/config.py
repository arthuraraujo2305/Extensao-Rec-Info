"""Caminhos e constantes do projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

DADOS = RAIZ / "data"
DADOS_BRUTOS = DADOS / "raw"              # CSVs baixados do Kaggle
DADOS_PROCESSADOS = DADOS / "processed"   # corpus + índices gerados
ARQUIVO_FEEDBACK = DADOS / "feedback.csv"  # avaliações 👍/👎 feitas no app
PASTA_AVALIACAO = RAIZ / "eval"

# Base de letras do site Vagalume, publicada no Kaggle por Anderson Neisse.
# Contém letras em português e em inglês (entre outros idiomas).
DATASET_KAGGLE = "neisse/scrapped-lyrics-from-6-genres"
URL_VAGALUME = "https://www.vagalume.com.br"

# Modelo multilíngue: a mesma "linguagem vetorial" para português e inglês,
# o que permite buscar em português e encontrar músicas em inglês (e vice-versa).
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cross-encoder multilíngue treinado no mMARCO, usado para reranking.
MODELO_RERANKING = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
