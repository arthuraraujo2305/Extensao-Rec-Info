// Compatibilidade com os controles de tema nativos do Streamlit >= 1.67.
// Nenhum acesso à configuração do servidor ou ao armazenamento de outros apps.
export default function ({ parentElement }) {
    const botao = parentElement.querySelector(".alternador");
    const estado = parentElement.querySelector(".estado");
    const app = document.querySelector('[data-testid="stApp"]');
    if (!app) return;

    const escuro = () => getComputedStyle(app).colorScheme === "dark";
    const sincronizar = () => {
        const ativo = escuro();
        botao.setAttribute("aria-checked", String(ativo));
        app.dataset.modoVisual = ativo ? "dark" : "light";
        estado.textContent = ativo ? "Fundo escuro" : "Fundo claro";
    };

    const aguardar = (seletor) => new Promise((resolve, reject) => {
        const atual = document.querySelector(seletor);
        if (atual) return resolve(atual);
        const observador = new MutationObserver(() => {
            const elemento = document.querySelector(seletor);
            if (elemento) {
                observador.disconnect();
                clearTimeout(limite);
                resolve(elemento);
            }
        });
        const limite = setTimeout(() => {
            observador.disconnect();
            reject(new Error("Seletor de tema indisponível"));
        }, 1500);
        observador.observe(document.body, { childList: true, subtree: true });
    });

    botao.onclick = async () => {
        const menu = document.querySelector('[data-testid="stMainMenuButton"]');
        const destino = escuro() ? "Light" : "Dark";
        const abriuMenu = menu?.getAttribute("aria-expanded") !== "true";
        botao.disabled = true;
        try {
            if (!menu) throw new Error("Menu indisponível");
            if (abriuMenu) menu.click();
            const opcao = await aguardar(`[data-testid="stMainMenuItem-theme-${destino}"]`);
            opcao.click();
            if (abriuMenu && menu.getAttribute("aria-expanded") === "true") menu.click();
            requestAnimationFrame(sincronizar);
        } catch {
            estado.textContent = "Para trocar o tema, use Claro ou Escuro no menu do canto superior.";
        } finally {
            botao.disabled = false;
            botao.focus({ preventScroll: true });
        }
    };

    // Acompanha também mudanças feitas pelo menu original ou pelo sistema.
    const observador = new MutationObserver(sincronizar);
    observador.observe(app, { attributes: true, attributeFilter: ["class", "style"] });
    sincronizar();
    return () => {
        observador.disconnect();
        botao.onclick = null;
    };
}
