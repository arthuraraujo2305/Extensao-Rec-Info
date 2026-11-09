"""Baixa a base de letras do Vagalume (Kaggle) para data/raw/.

    python scripts/baixar_dados.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buscador import config  # noqa: E402

ARQUIVOS = ("artists-data.csv", "lyrics-data.csv")


def main() -> None:
    destino = config.DADOS_BRUTOS
    destino.mkdir(parents=True, exist_ok=True)
    if all((destino / a).exists() for a in ARQUIVOS):
        print(f"Os dados já estão em {destino}")
        return
    try:
        import kagglehub
        print(f"Baixando {config.DATASET_KAGGLE} do Kaggle (~130 MB)…")
        origem = Path(kagglehub.dataset_download(config.DATASET_KAGGLE))
    except Exception as erro:  # sem internet, Kaggle fora do ar, exige login…
        print(f"\nNão consegui baixar automaticamente: {erro}\n")
        print("Baixe manualmente em https://www.kaggle.com/datasets/" + config.DATASET_KAGGLE)
        print(f"e coloque os arquivos {' e '.join(ARQUIVOS)} em {destino}")
        sys.exit(1)
    for nome in ARQUIVOS:
        arquivo = next(origem.rglob(nome), None)
        if arquivo is None:
            sys.exit(f"Arquivo {nome} não encontrado no download ({origem}).")
        shutil.copy2(arquivo, destino / nome)
        print(f"  {nome} -> {destino / nome}")
    print("Pronto! Agora rode: python scripts/construir_indice.py")


if __name__ == "__main__":
    main()
