# README - Versao 9

## Escopo

A `versao9` propõe uma nova tentativa de superar as baselines tabulares com uma arquitetura recorrente mais forte.

A ideia central e combinar, em um unico modelo:

- leitura sequencial por janelas;
- separacao entre sinais continuos e sinais de estado;
- uma LSTM de contexto entre janelas;
- fusao com `X_tab`.

## Motivacao

As versoes anteriores mostraram dois fatos importantes:

- a melhor `LSTM` do projeto continua sendo a da `versao6`;
- `RandomForest` e `XGBoost` permanecem a frente porque recebem uma representacao tabular muito forte.

A `versao9` nasce justamente para atacar esse descompasso informacional.

## Estrutura

1. `1-visao-geral-dos-dados.ipynb`
   Reapresenta o problema de classificacao e a motivacao da nova versao.
2. `2-pre-processamento.ipynb`
   Mostra como a serie e transformada em `X_seq`, `X_tab` e janelas temporais.
3. `3-classificacao-multiclasse-lstm-hibrida.ipynb`
   Treina e avalia a `LSTM` hibrida hierarquica.
4. `4-comparacao-lstm-hibrida-vs-baseline.ipynb`
   Compara a nova arquitetura com `RandomForest`, `XGBoost`, `versao8` e a melhor `LSTM` da `versao6`.
5. `pipeline_v9.py`
   Centraliza o preparo dos dados, a arquitetura hibrida e as rotinas de avaliacao.

## Estado da versao

A execucao registrada em `artifacts/reports_v9/classificacao_v9_lstm_hibrida/` produziu:

- `LSTM hibrida` em validacao:
  - `accuracy = 0.9551`
  - `macro-F1 = 0.9596`
  - `balanced accuracy = 0.9608`
- `LSTM hibrida` em teste:
  - `accuracy = 0.9224`
  - `macro-F1 = 0.9268`
  - `balanced accuracy = 0.9415`

## Leitura dos resultados

A `versao9` representou um ganho real em relacao as redes anteriores:

- sobre a `versao8`, houve ganho de `0.0120` em `accuracy`, `0.0304` em `macro-F1` e `0.0224` em `balanced accuracy`;
- sobre a melhor `LSTM` anterior da `versao6`, houve perda de `0.0209` em `accuracy`, mas ganho de `0.0083` em `macro-F1` e `0.0048` em `balanced accuracy`.

Isso significa que a arquitetura hibrida melhorou a separacao global entre classes, mas ainda nao venceu `RandomForest` e `XGBoost`.

## Leitura metodologica

O artigo do `3W Dataset 2.0.0` reforca que:

- o dataset possui 27 variaveis;
- ha rotulos por observacao (`class`) e por estado operacional (`state`);
- classes transitorias `101..109` sao parte importante do problema;
- real instances mantem `missing values`, `frozen variables` e `outliers`.

Por isso, a principal limitacao remanescente da `versao9` e que, apesar do ganho arquitetural, ela ainda trata o problema principalmente como classificacao de instancia. Isso motiva diretamente a `versao10`.
