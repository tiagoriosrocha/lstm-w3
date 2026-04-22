# README - Versao 7

## Escopo

A `versao7` mantem o problema de **classificacao multiclasse de series temporais** introduzido na `versao6`, mas altera a hipotese arquitetural principal do projeto.

Nesta etapa:

- cada arquivo `parquet` continua sendo tratado como uma amostra;
- a sequencia temporal contida no arquivo continua sendo a entrada;
- o rotulo da pasta (`0` a `9`) continua sendo a classe alvo;
- a principal mudanca esta no modelo sequencial, que passa a ser uma `LSTM` mais profunda.

## Motivacao experimental

Os resultados da `versao6` mostraram um padrao importante:

- a `LSTM` alcancou desempenho forte;
- porem o `RandomForest` ainda superou a rede neural em `accuracy`, `macro-F1` e `balanced accuracy`;
- isso sugere que a arquitetura recorrente anterior ainda nao estava extraindo do sinal temporal uma representacao tao discriminativa quanto a obtida pelas features agregadas.

Por isso, a `versao7` propoe uma nova investigacao: aumentar a profundidade da `LSTM` e enriquecer a etapa de leitura da sequencia para tentar reduzir a diferenca em relacao ao baseline.

## Resultados da execucao realizada

Na execucao registrada em `artifacts/reports_v7/classificacao_v7_lstm_profunda/`, os resultados principais foram:

- `LSTM profunda` em validacao:
  - `accuracy = 0.9251`
  - `macro-F1 = 0.8997`
  - `balanced accuracy = 0.9101`
- `LSTM profunda` em teste:
  - `accuracy = 0.9104`
  - `macro-F1 = 0.8964`
  - `balanced accuracy = 0.9191`
- `RandomForest` em teste:
  - `accuracy = 0.9851`
  - `macro-F1 = 0.9811`
  - `balanced accuracy = 0.9744`

Comparacao com a `versao6`:

- a `LSTM` da `versao6` havia obtido `accuracy = 0.9433`, `macro-F1 = 0.9185` e `balanced accuracy = 0.9367` no teste;
- portanto, a `LSTM profunda` da `versao7` nao melhorou o desempenho da rede anterior;
- o melhor modelo global continuou sendo o `RandomForest`.

Interpretacao tecnica:

- o aumento de profundidade nao foi suficiente para superar a baseline tabular;
- houve degradacao principalmente na classe `0`, cuja `recall` caiu de forma relevante;
- assim, a hipotese de que mais camadas recorrentes aproximariam automaticamente a rede do baseline nao foi confirmada nesta configuracao.

## O que muda na arquitetura

Em relacao a `versao6`, a proposta da `versao7` inclui:

- `hidden_size` maior;
- mais camadas recorrentes (`num_layers = 4`);
- `dropout` mais forte;
- combinacao de tres resumos temporais:
  - ultimo estado oculto;
  - media temporal;
  - pooling por atencao;
- cabeca densa mais profunda para classificacao final.

Em termos conceituais:

```text
serie temporal -> LSTM profunda -> resumos temporais -> MLP classificadora -> softmax -> classe 0..9
```

## Estrutura

1. `1-visao-geral-dos-dados.ipynb`
   Reapresenta a base sob a perspectiva da `versao7` e retoma a motivacao experimental.
2. `2-pre-processamento.ipynb`
   Reconstroi os artefatos de treino, validacao e teste para o experimento da nova versao.
3. `3-classificacao-multiclasse-lstm-profunda.ipynb`
   Treina e avalia a nova `LSTM` profunda.
4. `4-comparacao-lstm-profunda-vs-baseline.ipynb`
   Compara a arquitetura profunda com `RandomForest` e `XGBoost` quando disponivel.
5. `pipeline_v7.py`
   Centraliza o preparo dos dados, a nova arquitetura recorrente e as rotinas de avaliacao.

## Objetivo pedagogico

Esta versao foi escrita para mostrar um ponto muito importante em ciencia de dados aplicada:

- nao basta trocar modelos aleatoriamente;
- cada nova versao precisa nascer de uma leitura critica dos resultados anteriores;
- a comparacao com baselines deve continuar sendo justa e reproduzivel.

Assim, a `versao7` nao faz apenas "mais do mesmo". Ela formaliza uma nova hipotese: talvez o gargalo da `versao6` estivesse na capacidade da rede, e nao necessariamente na formulacao do problema.

Depois da execucao, a leitura mais honesta passa a ser ainda mais instrutiva: o gargalo nao parece ser resolvido apenas com profundidade adicional. Isso sugere que futuras iteracoes podem precisar de mudancas mais estruturais, como outra familia de arquitetura, outro esquema de representacao temporal ou uma estrategia diferente de engenharia de atributos.
