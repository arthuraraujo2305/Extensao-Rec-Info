"""Motor de busca: junta índices, modelos, tolerância a erros, filtros e reranking.

Uso básico:

    from buscador import MotorDeBusca
    motor = MotorDeBusca.carregar()
    resposta = motor.buscar("eu sei que vou te amar", modo="bm25")
    for r in resposta.resultados:
        print(r.posicao, r.titulo, "-", r.artista)
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .bm25 import BM25
from .booleano import ModeloBooleano, termos_da_consulta
from .hibrido import PESO_LEXICO, fusao_combsum, fusao_rrf, normalizar_minmax
from .indice import IndiceInvertido
from .semantico import BuscadorSemantico, Reordenador, dividir_em_trechos
from .texto import CAMPOS_ALIASES, ErroConsulta, gerar_trecho_html, normalizar, tokenizar, trecho_relevante
from .tolerancia import CorretorOrtografico
from .vetorial import ModeloVetorial

CAMPOS = ("letra", "titulo", "artista")
# Os modelos ranqueados (BM25 e TF-IDF) usam um índice "documento" que junta os três
# campos, com título e artista repetidos para valerem mais que uma palavra da letra.
# (Uma primeira versão somava as notas de índices separados por campo; a avaliação
# mostrou que isso era ruim: palavras comuns nas letras, como "love", são raras nos
# títulos, ganham IDF alto lá e qualquer título com "Love" passava na frente da
# música que tinha o trecho inteiro na letra. Veja o README.)
REPETICOES_CAMPOS = {"titulo": 2, "artista": 2, "letra": 1}

MODOS = {
    "bm25": "BM25 (probabilístico)",
    "tfidf": "TF-IDF + cosseno (vetorial)",
    "booleano": "Booleano (AND / OR / NOT)",
    "semantico": "Semântico (embeddings)",
    "hibrido": "Híbrido (BM25 + semântico)",
}
MODOS_SEMANTICOS = {"semantico", "hibrido"}

_CAMPO_EM_CONSULTA = re.compile(r'([^\W\d_]+):(?:"([^"]*)"?|(\S+))')


@dataclass
class Resultado:
    posicao: int
    doc_id: int
    score: float
    titulo: str
    artista: str
    idioma: str
    generos: str
    letra: str
    link: str
    trecho_html: str


@dataclass
class RespostaBusca:
    consulta: str              # o que a pessoa digitou
    consulta_usada: str        # depois da correção ortográfica
    modo: str
    resultados: list[Resultado]
    total: int | None          # nº de documentos recuperados (None nos modos semânticos)
    tempo_ms: float
    corrigida: bool = False
    erro: str | None = None


class MotorDeBusca:
    def __init__(self, corpus: pd.DataFrame, indices: dict[str, IndiceInvertido],
                 semantico: BuscadorSemantico | None = None):
        self.corpus = corpus.reset_index(drop=True)
        self.n_docs = len(self.corpus)
        self.textos = {c: self.corpus[c].fillna("").astype(str).tolist() for c in CAMPOS}
        self.indices = dict(indices)
        if "documento" not in self.indices:
            self.indices["documento"] = self._indice_documento(self.corpus)
        documento = self.indices["documento"]
        self.bm25 = BM25(documento)
        self.vetorial = ModeloVetorial(documento)
        self.booleano = ModeloBooleano({c: self.indices[c] for c in CAMPOS}, self.textos)
        self.semantico = semantico
        if semantico is not None and semantico.n_docs != self.n_docs:
            raise ValueError("Os embeddings não correspondem ao corpus: rode construir_indice.py de novo.")
        pop = self.corpus["popularidade"].fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        self.popularidade = normalizar_minmax(np.log1p(pop))
        self.idiomas = self.corpus["idioma"].astype(str).to_numpy()
        self._generos = [{g.strip() for g in str(x).split(";") if g.strip()}
                         for x in self.corpus["generos"].fillna("")]
        self._mascaras_genero: dict[str, np.ndarray] = {}
        self._reordenador: Reordenador | None = None

    # ------------------------------------------------------------------ construção / disco
    @staticmethod
    def _indice_documento(corpus: pd.DataFrame) -> IndiceInvertido:
        colunas = [corpus[c].fillna("").astype(str) for c in CAMPOS]
        return IndiceInvertido.construir(
            [t for campo, texto in zip(CAMPOS, valores) for t in tokenizar(texto) * REPETICOES_CAMPOS[campo]]
            for valores in zip(*colunas))

    @classmethod
    def construir(cls, corpus: pd.DataFrame, semantico: BuscadorSemantico | None = None) -> "MotorDeBusca":
        corpus = corpus.reset_index(drop=True)
        indices = {c: IndiceInvertido.construir(tokenizar(t) for t in corpus[c].fillna("").astype(str))
                   for c in CAMPOS}
        indices["documento"] = cls._indice_documento(corpus)
        return cls(corpus, indices, semantico)

    def salvar(self, pasta: str | Path = config.DADOS_PROCESSADOS) -> None:
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        self.corpus.to_parquet(pasta / "corpus.parquet", index=False)
        for campo, indice in self.indices.items():
            indice.salvar(pasta / f"indice_{campo}.npz")
        if self.semantico is not None:
            self.semantico.salvar(pasta / "semantico.npz")

    @staticmethod
    def indice_existe(pasta: str | Path = config.DADOS_PROCESSADOS) -> bool:
        pasta = Path(pasta)
        return (pasta / "corpus.parquet").exists() and all(
            (pasta / f"indice_{c}.npz").exists() for c in CAMPOS)

    @classmethod
    def carregar(cls, pasta: str | Path = config.DADOS_PROCESSADOS, semantico: bool = True) -> "MotorDeBusca":
        pasta = Path(pasta)
        if not cls.indice_existe(pasta):
            raise FileNotFoundError(f"Índice não encontrado em {pasta}. Rode: python scripts/construir_indice.py")
        corpus = pd.read_parquet(pasta / "corpus.parquet")
        # o índice "documento" é opcional no disco: se faltar, é montado na hora (segundos)
        indices = {c: IndiceInvertido.carregar(pasta / f"indice_{c}.npz")
                   for c in (*CAMPOS, "documento") if (pasta / f"indice_{c}.npz").exists()}
        sem = None
        if semantico and (pasta / "semantico.npz").exists():
            sem = BuscadorSemantico.carregar(pasta / "semantico.npz")
        return cls(corpus, indices, sem)

    # ------------------------------------------------------------------ utilidades
    @property
    def tem_semantico(self) -> bool:
        return self.semantico is not None

    @cached_property
    def corretor(self) -> CorretorOrtografico:
        frequencias: Counter = Counter()
        for campo in CAMPOS:
            indice = self.indices[campo]
            for termo, df in zip(indice.termos, indice.df.tolist()):
                frequencias[termo] += df
        return CorretorOrtografico(frequencias, associacao=self._associacao_contexto)

    @cached_property
    def _idf_letra(self) -> np.ndarray:
        indice = self.indices["letra"]
        return np.log(indice.n_docs / np.maximum(indice.df, 1))

    def _associacao_contexto(self, candidato: str, contexto: list[str]) -> float:
        """Quanto `candidato` aparece junto com as palavras do contexto nas letras.

        Soma, para cada palavra do contexto, o nº de letras que têm as duas
        (ponderado pelo idf da palavra, para "the"/"que" quase não contarem),
        dividido pela frequência do candidato (+5 de suavização).
        """
        indice, idf = self.indices["letra"], self._idf_letra
        docs = indice.docs_com(candidato)
        if docs.size == 0:
            return 0.0
        nota = 0.0
        for termo in contexto:
            tid = indice.vocabulario.get(termo)
            if tid is None:
                continue
            juntos = np.intersect1d(docs, indice.postings_por_id(tid)[0], assume_unique=True).size
            nota += float(idf[tid]) * juntos
        return nota / (docs.size + 5)

    def generos_disponiveis(self, minimo: int = 1) -> list[str]:
        contagem = Counter(g for gs in self._generos for g in gs)
        return [g for g, n in contagem.most_common() if n >= minimo]

    def mascara_filtros(self, idiomas: Iterable[str] | None = None,
                        generos: Iterable[str] | None = None) -> np.ndarray | None:
        mascara = None
        idiomas, generos = list(idiomas or []), list(generos or [])
        if idiomas:
            mascara = np.isin(self.idiomas, idiomas)
        if generos:
            m = np.zeros(self.n_docs, dtype=bool)
            for g in generos:
                if g not in self._mascaras_genero:
                    self._mascaras_genero[g] = np.array([g in gs for gs in self._generos], dtype=bool)
                m |= self._mascaras_genero[g]
            mascara = m if mascara is None else (mascara & m)
        return mascara

    def docs_com_frase(self, frase: str, campo: str = "letra") -> np.ndarray:
        r = self.booleano.docs_no_campo(campo, frase)
        return np.zeros(0, dtype=np.int32) if r is None else r

    # ------------------------------------------------------------------ pontuação
    def _termos_por_campo(self, consulta: str) -> dict[str | None, list[str]]:
        """'artista:queen love' -> {None: ['love'], 'artista': ['queen']}"""
        por_campo: dict[str | None, list[str]] = {None: []}

        def separar(m: re.Match) -> str:
            campo = CAMPOS_ALIASES.get(normalizar(m.group(1)))
            if campo is None:
                return m.group(0)
            valor = m.group(2) if m.group(2) is not None else m.group(3)
            por_campo.setdefault(campo, []).extend(tokenizar(valor))
            return " "

        resto = _CAMPO_EM_CONSULTA.sub(separar, consulta)
        por_campo[None].extend(tokenizar(resto))
        return por_campo

    def pontuar_lexico(self, consulta: str, modelo: str = "bm25") -> np.ndarray:
        """Nota BM25 ou TF-IDF de todos os documentos (0 = nenhum termo em comum).

        Termos com campo (artista:queen) funcionam como restrição: só ficam as
        músicas que têm aqueles termos naquele campo.
        """
        por_campo = self._termos_por_campo(consulta)
        termos = [t for lista in por_campo.values() for t in lista]
        scores = (self.bm25 if modelo == "bm25" else self.vetorial).pontuar(termos)
        for campo in CAMPOS:
            exigidos = por_campo.get(campo)
            if exigidos:
                permitidos = np.zeros(self.n_docs, dtype=bool)
                docs = self.indices[campo].docs_com(exigidos[0])
                for termo in exigidos[1:]:
                    docs = np.intersect1d(docs, self.indices[campo].docs_com(termo), assume_unique=True)
                permitidos[docs] = True
                scores = np.where(permitidos, scores, np.float32(0))
        return scores

    @staticmethod
    def _top_k(scores: np.ndarray, k: int, mascara: np.ndarray | None = None,
               minimo: float = 0.0) -> tuple[list[tuple[int, float]], int]:
        validos = scores > minimo
        if mascara is not None:
            validos &= mascara
        candidatos = np.flatnonzero(validos)
        total = int(candidatos.size)
        if total == 0:
            return [], 0
        if total > k:
            candidatos = candidatos[np.argpartition(-scores[candidatos], k - 1)[:k]]
        ordem = candidatos[np.lexsort((candidatos, -scores[candidatos]))]
        return [(int(d), float(scores[d])) for d in ordem], total

    def _exigir_semantico(self) -> None:
        if self.semantico is None:
            raise ErroConsulta("A busca semântica não está disponível: gere os embeddings com "
                               "python scripts/construir_indice.py (sem a opção --sem-semantico).")

    def ranquear(self, consulta: str, modo: str = "bm25", k: int = 10,
                 mascara: np.ndarray | None = None, reordenar: bool = False,
                 peso_popularidade: float = 0.0, fusao: str = "combsum",
                 profundidade_hibrida: int = 100) -> tuple[list[tuple[int, float]], int | None]:
        """Lista [(doc_id, nota)] com os k melhores e o total de documentos recuperados.

        `fusao` (só no modo híbrido): "combsum" (padrão) ou "rrf".
        """
        if modo not in MODOS:
            raise ValueError(f"Modo desconhecido: {modo}. Opções: {', '.join(MODOS)}")

        if modo == "booleano":
            docs = self.booleano.buscar(consulta)
            if mascara is not None:
                docs = docs[mascara[docs]]
            ordem = docs[np.lexsort((docs, -self.popularidade[docs]))][:k]
            return [(int(d), 1.0) for d in ordem], int(docs.size)

        n = max(k, 30) if (reordenar or peso_popularidade > 0) else k
        total: int | None
        if modo in ("bm25", "tfidf"):
            ranking, total = self._top_k(self.pontuar_lexico(consulta, modo), n, mascara)
        elif modo == "semantico":
            self._exigir_semantico()
            ranking, _ = self._top_k(self.semantico.pontuar(consulta), n, mascara, minimo=-np.inf)
            total = None
        else:  # híbrido
            self._exigir_semantico()
            lexico, _ = self._top_k(self.pontuar_lexico(consulta, "bm25"), profundidade_hibrida, mascara)
            semantico, _ = self._top_k(self.semantico.pontuar(consulta), profundidade_hibrida, mascara,
                                       minimo=-np.inf)
            if fusao == "rrf":
                ranking = fusao_rrf([[d for d, _ in lexico], [d for d, _ in semantico]])[:n]
            else:
                ranking = fusao_combsum([lexico, semantico], [PESO_LEXICO, 1 - PESO_LEXICO])[:n]
            total = None

        if reordenar and ranking:
            ranking = self._reordenar(consulta, ranking)
        if peso_popularidade > 0 and ranking:
            ranking = self._aplicar_popularidade(ranking, peso_popularidade)
        return ranking[:k], total

    def _aplicar_popularidade(self, ranking, peso: float):
        """Mistura a relevância com a popularidade do artista (ideia de "autoridade", Aula 12)."""
        docs = np.array([d for d, _ in ranking])
        notas = np.array([s for _, s in ranking])
        final = (1 - peso) * normalizar_minmax(notas) + peso * self.popularidade[docs]
        ordem = np.argsort(-final, kind="stable")
        return [(int(docs[i]), float(final[i])) for i in ordem]

    def _texto_para_reordenar(self, consulta: str, doc: int) -> str:
        letra = self.textos["letra"][doc]
        if self.semantico is not None:
            trechos = dividir_em_trechos(letra)
            trecho = trechos[self.semantico.melhor_trecho(consulta, doc)] if trechos else letra
        else:
            trecho = trecho_relevante(letra, tokenizar(consulta), 80)
        return f"{self.textos['titulo'][doc]} - {self.textos['artista'][doc]}. {trecho}"

    def _reordenar(self, consulta: str, ranking):
        if self._reordenador is None:
            self._reordenador = Reordenador()
        docs = [d for d, _ in ranking]
        notas = self._reordenador.pontuar(consulta, [self._texto_para_reordenar(consulta, d) for d in docs])
        ordem = np.argsort(-notas, kind="stable")
        return [(docs[i], float(notas[i])) for i in ordem]

    # ------------------------------------------------------------------ busca completa (app)
    def buscar(self, consulta: str, modo: str = "bm25", k: int = 10,
               idiomas: Iterable[str] | None = None, generos: Iterable[str] | None = None,
               corrigir: bool = True, reordenar: bool = False,
               usar_popularidade: bool = False) -> RespostaBusca:
        inicio = time.perf_counter()
        consulta = (consulta or "").strip()
        usada, corrigida = consulta, False
        if corrigir and consulta:
            usada, corrigida = self.corretor.corrigir_consulta(consulta)
        erro = None
        try:
            if not usada:
                raise ErroConsulta("Digite alguma coisa para buscar.")
            ranking, total = self.ranquear(
                usada, modo, k, self.mascara_filtros(idiomas, generos),
                reordenar=reordenar and modo != "booleano",
                peso_popularidade=0.15 if (usar_popularidade and modo != "booleano") else 0.0)
        except ErroConsulta as e:
            ranking, total, erro = [], 0, str(e)
        termos = set(termos_da_consulta(usada))
        resultados = [self._montar_resultado(pos, d, s, termos, usada)
                      for pos, (d, s) in enumerate(ranking, start=1)]
        return RespostaBusca(consulta, usada, modo, resultados, total,
                             (time.perf_counter() - inicio) * 1000, corrigida, erro)

    def _montar_resultado(self, posicao: int, doc: int, score: float, termos: set[str],
                          consulta: str) -> Resultado:
        linha = self.corpus.iloc[doc]
        letra = self.textos["letra"][doc]
        trecho, acertos = gerar_trecho_html(letra, termos, largura=32)
        if acertos == 0 and self.semantico is not None:
            # nenhuma palavra em comum: mostra o trecho mais parecido segundo os embeddings
            trechos = dividir_em_trechos(letra)
            if trechos:
                escolhido = trechos[self.semantico.melhor_trecho(consulta, doc)]
                trecho, _ = gerar_trecho_html(escolhido, termos, largura=len(escolhido.split()))
        return Resultado(
            posicao=posicao, doc_id=doc, score=score,
            titulo=str(linha["titulo"]), artista=str(linha["artista"]), idioma=str(linha["idioma"]),
            generos=str(linha["generos"]), letra=letra,
            link=config.URL_VAGALUME + str(linha["link_musica"]), trecho_html=trecho)
