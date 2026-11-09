"""Avalia e compara os modelos de busca (Aula 05). Resultados vão para eval/.

    python scripts/avaliar.py                  # 100 músicas por idioma x 4 tipos de consulta
    python scripts/avaliar.py --reranking      # inclui o cross-encoder (mais lento)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from buscador import config  # noqa: E402
from buscador.avaliacao import (CONFIGURACOES_PADRAO, avaliar_configuracoes, avaliar_feedback,  # noqa: E402
                                gerar_consultas_item_conhecido, resumir)
from buscador.motor import MotorDeBusca  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="músicas sorteadas por idioma (padrão: 100)")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--reranking", action="store_true", help="avalia também Híbrido + reranking")
    args = p.parse_args()

    pd.set_option("display.width", 140)
    motor = MotorDeBusca.carregar()
    print(f"Corpus: {motor.n_docs:,} músicas | semântico: {'sim' if motor.tem_semantico else 'não'}")

    consultas = gerar_consultas_item_conhecido(motor, n_por_idioma=args.n, semente=args.semente)
    print(f"{len(consultas)} consultas geradas. Exemplos:")
    for tipo, grupo in consultas.groupby("tipo", sort=False):
        print(f"  [{tipo}] {grupo['consulta'].iloc[0]}")

    configuracoes = list(CONFIGURACOES_PADRAO)
    if args.reranking:
        configuracoes.append({"nome": "Híbrido + correção + reranking", "modo": "hibrido",
                              "corrigir": True, "reordenar": True})
    t0 = time.perf_counter()
    por_consulta = avaliar_configuracoes(motor, consultas, configuracoes)
    print(f"Avaliação concluída em {time.perf_counter() - t0:.0f}s\n")

    geral = resumir(por_consulta)
    por_tipo = resumir(por_consulta, ["tipo", "configuracao"])
    print("=== Geral ===")
    print(geral.to_string())
    print("\n=== MRR por tipo de consulta ===")
    print(por_tipo["MRR"].unstack(0).to_string())

    config.PASTA_AVALIACAO.mkdir(exist_ok=True)
    consultas.to_csv(config.PASTA_AVALIACAO / "consultas_item_conhecido.csv", index=False)
    por_consulta.to_csv(config.PASTA_AVALIACAO / "resultados_por_consulta.csv", index=False)
    geral.to_csv(config.PASTA_AVALIACAO / "resumo_geral.csv")
    por_tipo.to_csv(config.PASTA_AVALIACAO / "resumo_por_tipo.csv")

    if config.ARQUIVO_FEEDBACK.exists():
        fb = avaliar_feedback(motor, config.ARQUIVO_FEEDBACK)
        if not fb.empty:
            print("\n=== Com os julgamentos dos usuários (feedback do app) ===")
            print(fb.to_string())
            fb.to_csv(config.PASTA_AVALIACAO / "resumo_feedback.csv")
    print(f"\nArquivos salvos em {config.PASTA_AVALIACAO}")


if __name__ == "__main__":
    main()
