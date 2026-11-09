"""Modelo booleano (Aula 02).

Sintaxe aceita:

    chuva AND saudade            (E também funciona; termos lado a lado = AND implícito)
    amor OR paixão               (OU)
    amor NOT traição             (NÃO, ou o prefixo -traição)
    "eu sei que vou te amar"     frase exata (palavras adjacentes)
    artista:"legião urbana"      restringe ao campo: artista:, titulo:, letra:
    apaixon*                     curinga (Aula 08)
    (chuva OR sol) AND verão     parênteses

Os operadores só valem em MAIÚSCULAS. O resultado é um CONJUNTO de documentos
(sem ranking): por isso o app mostra esses resultados ordenados por popularidade.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from .indice import IndiceInvertido
from .texto import (CAMPOS_ALIASES, OPERADORES_E, OPERADORES_NAO, OPERADORES_OU, ErroConsulta,
                    contem_sequencia, normalizar, tokenizar)
from .tolerancia import ExpansorCuringa

_VAZIO = np.zeros(0, dtype=np.int32)

_LEXICO = re.compile(r'''
    \s*(?:
        (?P<abre>\()
      | (?P<fecha>\))
      | (?P<campo>[^\W\d_]+):(?=["\w*])
      | "(?P<frase>[^"]*)"?
      | (?P<menos>-)(?=["\w*(])
      | (?P<palavra>[^\s()"]+)
    )''', re.VERBOSE)


def _lexico(consulta: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    consulta = consulta.strip()
    pos = 0
    while pos < len(consulta):
        m = _LEXICO.match(consulta, pos)
        if m is None or m.end() == pos:
            break
        pos = m.end()
        tipo = m.lastgroup
        valor = m.group(tipo)
        if tipo == "palavra":
            if valor in OPERADORES_E:
                tipo = "E"
            elif valor in OPERADORES_OU:
                tipo = "OU"
            elif valor in OPERADORES_NAO:
                tipo = "NAO"
        tokens.append((tipo, valor))
    return tokens


class _Parser:
    """Analisador descendente recursivo.  ou := e (OU e)* ;  e := nao (E? nao)* ;
    nao := (NAO|-) nao | atomo ;  atomo := ( ou ) | campo: valor | valor"""

    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def _ver(self):
        return self.tokens[self.i][0] if self.i < len(self.tokens) else None

    def _pegar(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def analisar(self):
        if not self.tokens:
            raise ErroConsulta("Digite alguma coisa para buscar.")
        no = self._ou()
        if self.i < len(self.tokens):
            raise ErroConsulta(f"Não entendi '{self.tokens[self.i][1]}' nesta posição.")
        return no

    def _ou(self):
        no = self._e()
        while self._ver() == "OU":
            self._pegar()
            no = ("ou", no, self._e())
        return no

    def _e(self):
        no = self._nao()
        while self._ver() not in (None, "OU", "fecha"):
            if self._ver() == "E":
                self._pegar()
            no = ("e", no, self._nao())
        return no

    def _nao(self):
        if self._ver() in ("NAO", "menos"):
            self._pegar()
            return ("nao", self._nao())
        return self._atomo()

    def _atomo(self):
        tipo = self._ver()
        if tipo is None:
            raise ErroConsulta("A consulta terminou de repente: faltou um termo depois do operador?")
        if tipo == "abre":
            self._pegar()
            no = self._ou()
            if self._ver() == "fecha":
                self._pegar()
            return no
        if tipo == "campo":
            _, nome = self._pegar()
            campo = CAMPOS_ALIASES.get(normalizar(nome))
            if campo is None:
                raise ErroConsulta(f"Campo desconhecido '{nome}:'. Use artista:, titulo: ou letra:.")
            if self._ver() not in ("palavra", "frase", "E", "OU", "NAO"):
                raise ErroConsulta(f"Faltou o valor depois de '{nome}:'.")
            _, valor = self._pegar()
            return ("termo", campo, valor)
        if tipo in ("palavra", "frase", "E", "OU"):
            # "E"/"OU" no lugar de um termo são tratados como palavras ("E agora, José?")
            _, valor = self._pegar()
            return ("termo", None, valor)
        _, valor = self._pegar()
        raise ErroConsulta(f"Não entendi '{valor}' nesta posição.")


def analisar_consulta(consulta: str):
    """Árvore sintática da consulta (tuplas aninhadas)."""
    return _Parser(_lexico(consulta)).analisar()


def termos_da_consulta(consulta: str) -> list[str]:
    """Termos da consulta (sem operadores e nomes de campo), para destacar no trecho."""
    termos: list[str] = []
    for tipo, valor in _lexico(consulta):
        if tipo in ("palavra", "frase") and "*" not in valor:
            termos.extend(tokenizar(valor))
    return termos


class ModeloBooleano:
    def __init__(self, indices: Mapping[str, IndiceInvertido], textos: Mapping[str, Sequence[str]]):
        self.indices = dict(indices)
        self.textos = textos
        n_docs = next(iter(self.indices.values())).n_docs
        self.universo = np.arange(n_docs, dtype=np.int32)
        self._expansores: dict[str, ExpansorCuringa] = {}

    def buscar(self, consulta: str) -> np.ndarray:
        """Ids (ordenados) dos documentos que satisfazem a consulta."""
        resultado = self._avaliar(analisar_consulta(consulta))
        return _VAZIO if resultado is None else resultado

    # None = termo "neutro" (ex.: só pontuação), ignorado pelos operadores
    def _avaliar(self, no):
        tipo = no[0]
        if tipo == "termo":
            return self._docs_do_termo(no[1], no[2])
        if tipo == "nao":
            r = self._avaliar(no[1])
            return None if r is None else np.setdiff1d(self.universo, r, assume_unique=True)
        if tipo == "ou":
            a, b = self._avaliar(no[1]), self._avaliar(no[2])
            if a is None or b is None:
                return b if a is None else a
            return np.union1d(a, b)
        # tipo == "e"
        esq, dir_ = no[1], no[2]
        if esq[0] == "nao" and dir_[0] != "nao":
            esq, dir_ = dir_, esq
        if dir_[0] == "nao":                       # A AND NOT B  =  A \ B (sem montar o complemento)
            a, b = self._avaliar(esq), self._avaliar(dir_[1])
            if a is None:
                return None if b is None else np.setdiff1d(self.universo, b, assume_unique=True)
            return a if b is None else np.setdiff1d(a, b, assume_unique=True)
        a, b = self._avaliar(esq), self._avaliar(dir_)
        if a is None or b is None:
            return b if a is None else a
        return np.intersect1d(a, b, assume_unique=True)

    def _docs_do_termo(self, campo: str | None, texto: str):
        resultado = None
        for c in ([campo] if campo else list(self.indices)):
            docs = self.docs_no_campo(c, texto)
            if docs is None:
                continue
            resultado = docs if resultado is None else np.union1d(resultado, docs)
        return resultado

    def _expansor(self, campo: str) -> ExpansorCuringa:
        if campo not in self._expansores:
            indice = self.indices[campo]
            self._expansores[campo] = ExpansorCuringa(indice.termos, indice.df)
        return self._expansores[campo]

    def docs_no_campo(self, campo: str, texto: str):
        """Documentos cujo `campo` contém a palavra, frase ou curinga `texto`."""
        indice = self.indices[campo]
        if "*" in texto and len(texto.split()) == 1:
            termos = self._expansor(campo).expandir(texto)
            listas = [indice.docs_com(t) for t in termos]
            return np.unique(np.concatenate(listas)) if listas else _VAZIO
        tokens = tokenizar(texto.replace("*", " "))
        if not tokens:
            return None
        if len(tokens) == 1:
            return indice.docs_com(tokens[0])
        # frase: interseção das listas de postings + verificação da adjacência
        candidatos = indice.docs_com(tokens[0])
        for t in tokens[1:]:
            candidatos = np.intersect1d(candidatos, indice.docs_com(t), assume_unique=True)
            if candidatos.size == 0:
                return candidatos
        textos = self.textos[campo]
        return np.array([d for d in candidatos if contem_sequencia(tokenizar(textos[d]), tokens)],
                        dtype=np.int32)
