"""Evidências reais para o percurso da busca e a oficina booleana."""
from __future__ import annotations

import numpy as np

from .motor import CAMPOS, MotorDeBusca
from .semantico import dividir_em_trechos
from .texto import ErroConsulta, tokenizar


def palavras_da_busca(motor: MotorDeBusca, consulta: str) -> list[str]:
    return list(dict.fromkeys(t for ts in motor._termos_por_campo(consulta).values() for t in ts))


def explicar_palavras(motor: MotorDeBusca, consulta: str, doc: int,
                      idiomas=(), generos=()) -> dict:
    indice = motor.indices["documento"]
    mascara = motor.mascara_filtros(idiomas, generos)
    linhas = motor.bm25.explicar(palavras_da_busca(motor, consulta), doc)
    for linha in linhas:
        termo = linha["termo"]
        linha["campos"] = [c for c in CAMPOS if doc in motor.indices[c].docs_com(termo)]
        docs = indice.docs_com(termo)
        if mascara is not None:
            docs = docs[mascara[docs]]
        linha["musicas_filtradas"] = len(docs)
        linha["exemplos"] = [
            f'{motor.corpus.iloc[d]["titulo"]} — {motor.corpus.iloc[d]["artista"]}'
            for d in docs[:3]
        ]
    return dict(termos=linhas, documentos=indice.n_docs,
                tamanho=int(indice.comprimentos[doc]), media=indice.comprimento_medio,
                k1=motor.bm25.k1, b=motor.bm25.b)


def explicar_significado(motor: MotorDeBusca, consulta: str, doc: int) -> dict:
    motor._exigir_semantico()
    s = motor.semantico
    a, b = s.inicio_doc[doc:doc + 2]
    valores = s.embeddings[a:b] @ s.vetor_consulta(consulta)
    trechos = dividir_em_trechos(motor.textos["letra"][doc]) or ["-"]
    return dict(trechos=[dict(texto=t, semelhanca=float(v)) for t, v in zip(trechos, valores)],
                melhor=int(np.argmax(valores)), dimensoes=int(s.embeddings.shape[1]),
                total_trechos=len(s.embeddings), modelo=s.nome_modelo)


def grupos_booleanos(todas: str, alguma: str, nenhuma: str) -> tuple[list[str], ...]:
    return tuple(list(dict.fromkeys(tokenizar(texto))) for texto in (todas, alguma, nenhuma))


def montar_consulta_booleana(todas: str, alguma: str = "", nenhuma: str = "") -> str:
    exigir, alternativas, excluir = grupos_booleanos(todas, alguma, nenhuma)
    if not exigir and not alternativas:
        raise ErroConsulta("Inclua uma palavra em ‘Todas’ ou ‘Pelo menos uma’ para começar.")
    # Aspas tornam palavras como AND e NOT literais. O visitante nunca injeta sintaxe.
    partes = [f'"{t}"' for t in exigir]
    if alternativas:
        partes.append("(" + " OR ".join(f'"{t}"' for t in alternativas) + ")")
    partes.extend(f'NOT "{t}"' for t in excluir)
    return " AND ".join(partes)


def experimentar_regras(motor: MotorDeBusca, todas: str, alguma: str, nenhuma: str,
                        doc: int, idiomas=(), generos=()) -> dict:
    expressao = montar_consulta_booleana(todas, alguma, nenhuma)
    docs = motor.booleano.buscar(expressao)
    mascara = motor.mascara_filtros(idiomas, generos)
    if mascara is not None:
        docs = docs[mascara[docs]]
    exemplos = motor.corpus.iloc[docs].sort_values(["artista", "titulo"]).head(5)
    evidencias = []
    for rotulo, termos in zip(("Obrigatória", "Alternativa", "Excluída"),
                               grupos_booleanos(todas, alguma, nenhuma)):
        for termo in termos:
            presente = doc in motor.booleano.buscar(f'"{termo}"')
            evidencias.append(dict(regra=rotulo, palavra=termo, presente=bool(presente)))
    return dict(expressao=expressao, total=len(docs), atende=bool(doc in docs),
                exemplos=exemplos[["titulo", "artista"]].to_dict("records"), regras=evidencias)
