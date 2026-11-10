# Revisão da experiência para a feira

Roteiro para um integrante que não participou da implementação. O objetivo é
verificar a experiência, a fidelidade das explicações e a preparação do estande.
Os testes técnicos não substituem um piloto com pessoas de fora do grupo.

## Preparar a máquina

Use o ambiente `recinfo`, com Streamlit 1.67 ou superior. Siga o README para
baixar as letras e construir os índices e embeddings, caso ainda não existam.

```bash
conda activate recinfo
python -m pytest -q
streamlit run app.py
```

Os testes usam dados pequenos e não baixam modelos. O aplicativo completo
precisa da coleção em `data/processed/` e do modelo semântico em cache.
Abra o aplicativo uma vez com internet para preparar o modelo antes da feira.

## Conferência funcional

| Ação | Resultado esperado |
| --- | --- |
| Digitar `eu sei que vou te amar` e pressionar **Enter** | Executa a busca, sem precisar clicar no botão. |
| Buscar `legião urbana temmpo perdido` | Informa a correção para `tempo` e recupera Tempo Perdido entre as sugestões. |
| Buscar o trecho `when I find myself in times of trouble` | Recupera versões de Let It Be; conferir título e artista, sem exigir uma única versão. |
| Observar os resultados | Começa com cinco músicas; não exige escolher um nome técnico de modelo. |
| Abrir **Ver mais músicas**, investigar uma música e voltar | Mantém a consulta, os filtros e a quantidade de resultados já aberta. |
| Aplicar idioma/gênero e buscar | As músicas respeitam os filtros; os filtros ativos ficam indicados. Gênero vem do cadastro do artista. |
| Usar **Limpar filtros** | Retira as restrições e atualiza a busca existente. |
| Buscar `zzxxqqvvzzxxqqvv` por palavras | Mostra ajuda e aviso de cobertura parcial; não afirma saber que a música está ausente. |
| Acionar **Quero experimentar um exemplo** nessa ajuda | Oferece exemplos; escolher um executa uma busca sem os filtros anteriores. |
| Escolher **Lembro do assunto** | Descrição é preservada, sem correção lexical automática; o modelo usado é semântico. |
| Reiniciar após busca, filtros e oficina | Limpa a visita, volta ao caminho por palavras e mantém o tema. Não apaga votos salvos. |

Os exemplos refletem consultas verificadas na coleção usada na implementação.
Se o corpus for reconstruído ou ampliado, conferir novamente a presença das
músicas. Um exemplo por assunto pode ter várias respostas plausíveis.

## Conferir o percurso

1. Abra **Como esta música apareceu?** em um resultado reconhecido e em outro
   inesperado. Verifique se a música investigada corresponde ao cartão escolhido.
2. Em **Sua pista**, confirme que a consulta original, a correção e as palavras
   normalizadas estão coerentes. No semântico, a descrição deve ser preservada.
3. Em **Onde procuramos**, selecione palavras diferentes. Contagens e exemplos
   devem mudar de acordo com o índice; exemplos respeitam os filtros. A contagem
   global e a contagem filtrada são identificadas separadamente.
4. Em **Por que apareceu?**, confira se a maior contribuição do BM25 corresponde
   ao gráfico e à tabela. Título/artista têm peso por repetição no documento;
   a nota não é uma probabilidade de acerto.
5. No semântico, experimente `saudade de casa e vontade de voltar`. Explore os
   trechos e compare o maior valor com a nota da música no aprofundamento.
   Confira se o trecho exibido realmente se relaciona com a descrição.
6. Confira se um visitante consegue explicar, com as próprias palavras, o papel
   do índice ou da comparação por significado após a mediação.

## Oficina booleana

- Abra a oficina no percurso por palavras. A primeira consulta usa as pistas já
  preparadas; a expressão deve refletir os campos preenchidos.
- Use **Todas estas palavras** para exigir cada termo; use **Pelo menos uma**
  para alternativas; use **Nenhuma** para excluir. Quando combinar grupos,
  confira os parênteses na expressão gerada.
- Coloque em **Nenhuma** uma palavra presente na música investigada: ela deve
  ficar de fora, com evidência da presença do termo.
- Retire as pistas positivas e deixe apenas exclusões: deve aparecer orientação
  para incluir uma pista, sem travamento.
- Teste acentos, repetição de palavras e textos como `chuva OR sol`: palavras
  digitadas nos campos são literais; a lógica vem das opções do formulário.
- Confira que a lista booleana é apresentada em ordem alfabética e não é
  descrita como ranking de relevância. Novas regras não recebem correção
  ortográfica automática; palavras precisam existir na coleção para combinar.

## Feedback e alternância entre visitantes

- Vote **Sim**, troque para **Não** e volte a clicar no voto selecionado.
  Verifique o destaque visual e se mudanças efetivas acrescentam linhas ao CSV.
- Ao desmarcar, a seleção visual é removida e não há nova linha. **Isso não
  apaga o voto anterior do CSV**, comportamento preservado da versão anterior.
  Se o grupo quiser retirada definitiva de julgamentos, isso requer uma mudança
  específica no armazenamento e na avaliação, com tratamento dos dados antigos.
- Depois de **Começar outra busca**, repita a mesma consulta: os votos visuais
  do visitante anterior não devem reaparecer. O CSV deve continuar intacto.
- Os julgamentos indicam relevância dos resultados, não aprendizagem. A
  avaliação acadêmica existente ainda usa o último julgamento por consulta/música;
  não interpretar esse relatório como uma votação independente por pessoa.

## Temas, acessibilidade e preparação do estande

- Percorra formulário, filtros, resultados, abas e oficina usando Tab, Enter e
  Espaço. Observe foco visível e nomes das ações.
- Alterne claro/escuro durante a busca e o percurso. Confira leitura de legendas,
  gráfico, tabela, expressão, trechos destacados, menus e botões selecionados.
- O controle de tema aciona o menu nativo do Streamlit; mudanças de versão podem
  exigir atualizar os seletores em `interface/tema.js`. Conferir também o menu
  original e a preferência após recarregar a página.
- Teste o monitor físico, zoom de 125%/150% e leitura à distância. No celular,
  confira ausência de rolagem horizontal e acesso às três etapas.
- Após aquecer o modelo, desconecte a rede e faça buscas por palavras e assunto.
  A busca local deve funcionar; links para o Vagalume dependem de internet.
- Se não houver embeddings ou o modelo não carregar, a indisponibilidade deve
  ficar clara e o caminho por palavras deve continuar utilizável.
- Registre a demora do primeiro carregamento separadamente das buscas seguintes.

## Piloto de dois a três minutos

Faça ao menos algumas sessões com pessoas de fora do grupo, incluindo repertórios
e familiaridade digital diferentes. O integrante acompanha sem executar todas as
ações pela pessoa.

Anote, sem precisar identificar o participante:

- Tempo até iniciar uma busca, reconhecer uma música e explorar o percurso.
- Termos que exigiram tradução ou explicação adicional.
- Se a pessoa encontrou o convite para investigar e conseguiu explorar uma pista.
- Se entendeu uma relação entre a consulta e o resultado, além de apenas ler a tela.
- Se a oficina booleana despertou interesse ou alongou demais a interação.
- Se a ajuda preservou o interesse quando a música não foi encontrada.
- O que a pessoa quis tentar espontaneamente depois.

Pergunta de encerramento sugerida: **“Que pista você acha que mais ajudou o
buscador?”**. O roteiro e os exemplos podem ser ajustados conforme essas respostas.

## Limites que merecem revisão do grupo

- Busca semântica produz aproximações; o julgamento humano dos temas ainda é necessário.
- Exemplos, legibilidade e duração de dois a três minutos foram planejados, mas
  precisam ser avaliados com o público e no equipamento real.
- Os dados e modelos locais não estão no Git: cada integrante precisa prepará-los.
- Este trabalho não inclui fotos de artistas, áudio, reconhecimento de canto,
  pontuação, competição ou publicação de um serviço público na internet.
