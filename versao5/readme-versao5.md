# README - Versao 5

## Escopo

A `versao5` consolida o projeto em um desenho experimental explicitamente comparativo, orientado para redacao academica em formato de artigo cientifico. O foco deixa de ser apenas a evolucao incremental do pipeline e passa a ser a resposta a uma pergunta metodologica objetiva:

- sob o mesmo protocolo de dados, pre-processamento, janelamento e avaliacao, qual e o comportamento relativo da `LSTM pura`, do `modelo hibrido residual` e da `baseline de persistencia`?

## Estrutura

Esta versao foi organizada nos seguintes artefatos:

1. `1-visao-geral-dos-dados.ipynb`
   Apresenta a caracterizacao formal da base `3W`, a taxonomia dos eventos e o catalogo dos atributos originais.
2. `2-pre-processamento-e-engenharia-de-atributos.ipynb`
   Executa o pipeline de preparo, documenta as variaveis auxiliares selecionadas e descreve os atributos sinteticos.
3. `3-modelagem-comparativa.ipynb`
   Treina os dois modelos herdados das versoes anteriores sob o mesmo bundle comparativo.
4. `4-avaliacao-comparativa.ipynb`
   Compara `LSTM`, `hibrido residual` e `persistencia` no conjunto de teste, com tabelas globais e estratificadas.
5. `5-sumario-e-glossario.ipynb`
   Reune um glossario tecnico ampliado com metricas, conceitos de redes recorrentes, series temporais e termos importantes de PLN.
6. `pipeline_v5.py`
   Centraliza a logica reutilizavel de preparacao, treinamento, avaliacao e construcao de tabelas comparativas.
7. `generate_notebooks_v5.py`
   Gera programaticamente os notebooks da versao, preservando padronizacao e reprodutibilidade textual.

## Contribuicoes metodologicas

Em relacao as versoes anteriores, a `versao5` acrescenta:

- organizacao da narrativa em formato mais academico;
- descricao formal dos atributos originais com base no `dataset.ini` oficial do `3W`;
- catalogo explicito dos atributos sinteticos usados na engenharia de features;
- funcao de preparo comparativo para os tres metodos;
- rotina unica de treinamento para `LSTM` e `modelo hibrido residual`;
- avaliacao agregada e lado a lado contra a `persistencia`;
- glossario tecnico para suporte a redacao do artigo.

## Observacao importante

O glossario inclui tambem termos de `Processamento de Linguagem Natural` porque a literatura moderna de modelagem sequencial compartilha conceitos fundamentais entre PLN e series temporais, como `embedding`, `attention`, `self-attention` e `Transformer`. Ainda assim, o problema principal deste projeto permanece sendo previsao multivariada em series temporais industriais.

