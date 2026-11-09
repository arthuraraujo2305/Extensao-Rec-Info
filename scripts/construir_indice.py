"""Monta o corpus, os índices invertidos e os embeddings em data/processed/.

    python scripts/construir_indice.py                 # padrão: 10 mil músicas por idioma
    python scripts/construir_indice.py --por-idioma 3000 --sem-semantico   # rápido, para testar
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buscador import config  # noqa: E402
from buscador.dados import preparar_corpus  # noqa: E402
from buscador.motor import MotorDeBusca  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--por-idioma", type=int, default=10_000, help="músicas por idioma (padrão: 10000)")
    p.add_argument("--max-por-artista", type=int, default=40, help="máximo de músicas por artista (padrão: 40)")
    p.add_argument("--sem-semantico", action="store_true", help="não gera embeddings (bem mais rápido)")
    p.add_argument("--lote", type=int, default=64, help="tamanho do lote para gerar embeddings")
    args = p.parse_args()

    t0 = time.perf_counter()
    corpus = preparar_corpus(config.DADOS_BRUTOS, por_idioma=args.por_idioma,
                             max_por_artista=args.max_por_artista)
    print(f"Corpus pronto: {len(corpus):,} músicas ({time.perf_counter() - t0:.0f}s)")

    t1 = time.perf_counter()
    motor = MotorDeBusca.construir(corpus)
    for campo, indice in motor.indices.items():
        print(f"  índice '{campo}': {indice}")
    print(f"Índices invertidos prontos ({time.perf_counter() - t1:.0f}s)")

    if not args.sem_semantico:
        from buscador.semantico import BuscadorSemantico
        t2 = time.perf_counter()
        print(f"Gerando embeddings com {config.MODELO_EMBEDDINGS} (na CPU pode levar alguns minutos)…")
        motor.semantico = BuscadorSemantico.construir(motor.textos["letra"], tamanho_lote=args.lote)
        print(f"  {motor.semantico.embeddings.shape[0]:,} trechos -> vetores de "
              f"{motor.semantico.embeddings.shape[1]} dimensões ({time.perf_counter() - t2:.0f}s)")
    else:
        antigo = config.DADOS_PROCESSADOS / "semantico.npz"
        if antigo.exists():
            antigo.unlink()  # embeddings antigos não correspondem ao novo corpus

    motor.salvar(config.DADOS_PROCESSADOS)
    print(f"Tudo salvo em {config.DADOS_PROCESSADOS} ({time.perf_counter() - t0:.0f}s no total)")
    print("Agora rode: streamlit run app.py")


if __name__ == "__main__":
    main()
