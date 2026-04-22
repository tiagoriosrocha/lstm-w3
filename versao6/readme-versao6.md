# README - Versao 6

## Escopo

A `versao6` muda o foco do projeto: em vez de prever o proximo valor de variaveis continuas, ela passa a tratar o `3W` como um problema de **classificacao multiclasse de series temporais**.

Nesta formulacao:

- cada arquivo `parquet` e tratado como uma amostra;
- a sequencia temporal contida no arquivo e a entrada do modelo;
- o rotulo da pasta (`0` a `9`) e a classe alvo.

## Objetivo didatico

Esta versao foi organizada para servir tambem como material de estudo para iniciantes. Por isso, os notebooks foram escritos com:

- explicacoes mais longas;
- comentarios abundantes nas celulas de codigo;
- interpretacoes passo a passo do que esta acontecendo;
- comparacao explicita entre modelos sequenciais e baselines classicas.

## Estrutura

1. `1-visao-geral-dos-dados.ipynb`
   Introduz a formulacao de classificacao, descreve a base e mostra a distribuicao das classes.
2. `2-pre-processamento.ipynb`
   Construi o split, seleciona colunas informativas, reamostra as series para comprimento fixo e gera os arrays de treino, validacao e teste.
3. `3-classificacao-multiclasse-lstm.ipynb`
   Treina e avalia uma `LSTM` multiclasse com `accuracy`, `macro-F1`, `balanced accuracy` e matriz de confusao.
4. `4-baseline-randomforest-xgboost.ipynb`
   Treina baselines tabulares baseadas em estatisticas agregadas das series, com `RandomForest` e `XGBoost` quando disponivel.
5. `pipeline_v6.py`
   Centraliza as rotinas reutilizaveis de preparo, treino, avaliacao e exportacao de resultados.

## Observacao sobre o XGBoost

O notebook de baseline foi escrito para usar `RandomForest` sempre e `XGBoost` apenas quando a biblioteca estiver instalada no ambiente. Se `xgboost` nao estiver disponivel, o fluxo continua funcionando com `RandomForest`, e isso e comunicado explicitamente ao usuario.

## Metricas principais

As metricas enfatizadas nesta versao sao:

- `accuracy`
- `macro-F1`
- `balanced accuracy`
- `matriz de confusao`

Essa escolha foi feita porque o dataset e desbalanceado, e uma avaliacao baseada apenas em acuracia poderia esconder desempenho fraco em classes raras.

## Leitura dos resultados obtidos

Na execucao ja realizada para o experimento `classificacao_v6_artigo`, observou-se o seguinte comportamento:

- `LSTM` no teste:
  - `accuracy = 0.9433`
  - `macro-F1 = 0.9185`
  - `balanced accuracy = 0.9367`
- `RandomForest` no teste:
  - `accuracy = 0.9851`
  - `macro-F1 = 0.9811`
  - `balanced accuracy = 0.9744`

Interpretacao resumida:

- a `LSTM` apresentou desempenho forte e consistente;
- entretanto, a baseline `RandomForest` foi superior em todas as metricas principais;
- isso sugere que as estatisticas agregadas da serie carregam um sinal discriminativo muito forte para este problema;
- por essa razao, a `versao6` passa a funcionar tambem como base de referencia para a `versao7`, em que uma `LSTM` mais profunda sera proposta para tentar reduzir essa diferenca.
