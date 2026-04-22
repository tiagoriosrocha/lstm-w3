# Versao 10

A `versao10` representa uma mudanca de estrategia no projeto. Em vez de insistir apenas em arquiteturas recorrentes mais profundas, ela busca aproximar o modelo da estrutura real do dataset descrita no artigo [2507.01048v1.pdf](/home/tiagoriosrocha/Desktop/lstm-w3/2507.01048v1.pdf).

## O que a versao10 propoe

A arquitetura central da `versao10` e uma `LSTM` multitarefa e sensivel a fonte da amostra, com os seguintes elementos:

- uso explicito das `27` variaveis do dataset;
- mascara de `missing values`;
- mascara de `frozen variables`;
- atributos estatisticos agregados em `X_tab`;
- embedding da origem da amostra (`well`, `simulated`, `drawn`);
- cabeca principal para classificar a instancia entre `0` e `9`;
- cabeca auxiliar para predizer `class` por observacao;
- cabeca auxiliar para predizer `state` por observacao.

## Estado atual

Resultados registrados para a `versao10`:

- validacao: `accuracy = 0.9581`, `macro-F1 = 0.9646`, `balanced accuracy = 0.9599`
- teste: `accuracy = 0.9373`, `macro-F1 = 0.9409`, `balanced accuracy = 0.9572`
- `RandomForest` em teste: `accuracy = 0.9851`, `macro-F1 = 0.9811`, `balanced accuracy = 0.9744`
- `XGBoost` em teste: `accuracy = 0.9791`, `macro-F1 = 0.9775`, `balanced accuracy = 0.9714`

Leitura resumida:
- a `versao10` ganhou `0.0149` em `accuracy`, `0.0140` em `macro-F1` e `0.0157` em `balanced accuracy` sobre a `versao9`;
- em relacao a melhor `LSTM` anterior da `versao6`, perdeu `0.0060` em `accuracy`, mas ganhou `0.0223` em `macro-F1` e `0.0205` em `balanced accuracy`;
- portanto, a `versao10` se tornou a melhor rede recorrente do projeto em `macro-F1` e `balanced accuracy`, embora ainda fique abaixo das baselines tabulares.

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`

## Leitura metodologica

A ideia central desta versao e testar uma hipotese mais forte do que nas iteracoes anteriores:

- se o artigo estiver correto ao enfatizar rotulos por observacao, classes transitorias, fontes heterogeneas e imperfeicoes reais do sinal;
- entao a melhor forma de competir com as baselines nao e apenas resumir a serie, mas modelar explicitamente essa estrutura.
