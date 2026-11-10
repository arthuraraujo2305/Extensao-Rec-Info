"""Controle local que aciona os seletores nativos do Streamlit.

O Streamlit não oferece um setter público de tema em Python. Este componente
usa os botões nativos (testados com 1.67), preservando gráficos, tabelas,
preferência do navegador e a sessão, sem alterar configuração global.
"""
from pathlib import Path

import streamlit as st

PASTA = Path(__file__).resolve().parent

_alternador = st.components.v2.component(
    "buscador_alternador_tema",
    html='''
        <button class="alternador" type="button" role="switch"
                aria-checked="false" aria-label="Tema escuro">
            <span>Tema escuro</span>
            <span class="trilho" aria-hidden="true"><span></span></span>
        </button>
        <p class="estado" role="status" aria-live="polite"></p>
    ''',
    css='''
        .alternador {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; width: 100%; min-height: 44px; padding: 0.6rem 0.75rem;
            background: var(--st-secondary-background-color);
            color: var(--st-text-color); border: 1px solid var(--st-border-color);
            border-radius: 0.65rem; font: inherit; cursor: pointer;
        }
        .alternador:focus-visible {
            outline: 2px solid var(--st-link-color); outline-offset: 3px;
        }
        .alternador:disabled { cursor: wait; }
        .trilho {
            width: 38px; height: 22px; padding: 2px; box-sizing: border-box;
            border-radius: 12px; background: #81738d; flex-shrink: 0;
        }
        .trilho span {
            display: block; width: 18px; height: 18px;
            border-radius: 50%; background: white;
        }
        [aria-checked="true"] .trilho { background: #8541ee; }
        [aria-checked="true"] .trilho span { transform: translateX(16px); }
        .estado {
            color: var(--st-text-color); margin: 0.35rem 0 0;
            font-size: 0.8rem; line-height: 1.5;
        }
    ''',
    js=(PASTA / "tema.js").read_text(encoding="utf-8"),
)


def seletor_tema() -> None:
    st.caption("Aparência")
    _alternador(key="alternador_tema")
