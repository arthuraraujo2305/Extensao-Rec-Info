"""Índice invertido (Aula 07 – Indexação).

Para cada termo guardamos a sua *lista de postings*: os documentos em que ele
aparece (em ordem crescente) e quantas vezes aparece em cada um (tf).

Para economizar memória, todas as listas ficam concatenadas em dois vetores
NumPy (formato CSR):

    termos[i]  ->  docs[inicio[i]:inicio[i+1]]   (ids dos documentos)
                   freqs[inicio[i]:inicio[i+1]]  (tf em cada documento)

O tamanho de cada lista é a *frequência de documento* (df) do termo.
"""
from __future__ import annotations

from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

_VAZIO = np.zeros(0, dtype=np.int32)


class IndiceInvertido:
    def __init__(self, termos: list[str], inicio: np.ndarray, docs: np.ndarray,
                 freqs: np.ndarray, comprimentos: np.ndarray):
        self.termos = list(termos)                       # vocabulário (ordem alfabética)
        self.vocabulario = {t: i for i, t in enumerate(self.termos)}
        self.inicio = inicio.astype(np.int64)
        self.docs = docs.astype(np.int32)
        self.freqs = freqs.astype(np.int32)
        self.comprimentos = comprimentos.astype(np.int32)  # nº de tokens de cada documento
        self.n_docs = int(len(self.comprimentos))
        self.df = np.diff(self.inicio).astype(np.int32)
        self.comprimento_medio = float(self.comprimentos.mean()) if self.n_docs else 0.0

    # ------------------------------------------------------------------ construção
    @classmethod
    def construir(cls, documentos: Iterable[list[str]]) -> "IndiceInvertido":
        """Recebe os documentos já tokenizados (doc_id = posição na sequência)."""
        postings_docs: dict[str, array] = {}
        postings_freqs: dict[str, array] = {}
        comprimentos = array("i")
        for doc_id, tokens in enumerate(documentos):
            comprimentos.append(len(tokens))
            for termo, freq in Counter(tokens).items():
                lista = postings_docs.get(termo)
                if lista is None:
                    postings_docs[termo] = array("i", [doc_id])
                    postings_freqs[termo] = array("i", [freq])
                else:
                    lista.append(doc_id)
                    postings_freqs[termo].append(freq)

        termos = sorted(postings_docs)
        tamanhos = np.fromiter((len(postings_docs[t]) for t in termos), dtype=np.int64, count=len(termos))
        inicio = np.zeros(len(termos) + 1, dtype=np.int64)
        np.cumsum(tamanhos, out=inicio[1:])
        docs = np.empty(int(inicio[-1]), dtype=np.int32)
        freqs = np.empty(int(inicio[-1]), dtype=np.int32)
        for i, termo in enumerate(termos):
            docs[inicio[i]:inicio[i + 1]] = np.frombuffer(postings_docs[termo], dtype=np.int32)
            freqs[inicio[i]:inicio[i + 1]] = np.frombuffer(postings_freqs[termo], dtype=np.int32)
        return cls(termos, inicio, docs, freqs, np.frombuffer(comprimentos, dtype=np.int32).copy())

    # ------------------------------------------------------------------ consulta
    def __len__(self) -> int:
        return len(self.termos)

    def __contains__(self, termo: str) -> bool:
        return termo in self.vocabulario

    def postings_por_id(self, termo_id: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.inicio[termo_id], self.inicio[termo_id + 1]
        return self.docs[a:b], self.freqs[a:b]

    def postings(self, termo: str) -> tuple[np.ndarray, np.ndarray]:
        tid = self.vocabulario.get(termo)
        if tid is None:
            return _VAZIO, _VAZIO
        return self.postings_por_id(tid)

    def docs_com(self, termo: str) -> np.ndarray:
        """Lista (ordenada) dos documentos que contêm o termo."""
        return self.postings(termo)[0]

    def frequencia_documento(self, termo: str) -> int:
        tid = self.vocabulario.get(termo)
        return 0 if tid is None else int(self.df[tid])

    # ------------------------------------------------------------------ disco
    def salvar(self, caminho: str | Path) -> None:
        termos_bytes = np.frombuffer("\n".join(self.termos).encode("utf-8"), dtype=np.uint8)
        np.savez(caminho, termos=termos_bytes, inicio=self.inicio, docs=self.docs,
                 freqs=self.freqs, comprimentos=self.comprimentos)

    @classmethod
    def carregar(cls, caminho: str | Path) -> "IndiceInvertido":
        with np.load(caminho) as d:
            texto = d["termos"].tobytes().decode("utf-8")
            termos = texto.split("\n") if texto else []
            return cls(termos, d["inicio"], d["docs"], d["freqs"], d["comprimentos"])

    def __repr__(self) -> str:
        return (f"IndiceInvertido({self.n_docs} documentos, {len(self.termos)} termos, "
                f"{len(self.docs)} postings)")
