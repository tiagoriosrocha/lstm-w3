# Projeto 3W com Versionamento de Experimentos

## Visao geral

Este repositorio organiza uma sequencia de experimentos sobre o dataset `3W`, preservando o historico de decisao tecnica ao longo de varias versoes. O projeto comecou com tarefas de previsao temporal e, nas versoes mais recentes, passou a investigar **classificacao multiclasse de series temporais**.

A ideia central do repositorio e simples:

- cada pasta `versaoN/` registra uma iteracao metodologica especifica;
- os notebooks mostram o raciocinio experimental de cada etapa;
- os arquivos `pipeline_vN.py` concentram o codigo reutilizavel;
- os artefatos de execucao ficam em `artifacts/reports_v*`.

## Estado atual do projeto

Hoje, o foco mais relevante esta nas versoes `6`, `7`, `8`, `9`, `10` e `11`, todas voltadas para classificacao das classes `0` a `9` do `3W`.

Resumo dos resultados mais importantes em teste:

| Experimento | Modelo | Accuracy | Macro-F1 | Balanced Accuracy | Leitura |
| --- | --- | ---: | ---: | ---: | --- |
| `versao6` | `LSTM` | `0.9433` | `0.9185` | `0.9367` | Melhor rede recorrente ate aqui |
| `versao6` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | Melhor resultado global |
| `versao6` | `XGBoost` | `0.9821` | `0.9791` | `0.9733` | Segunda melhor baseline global |
| `versao7` | `LSTM profunda` | `0.9104` | `0.8964` | `0.9191` | Nao superou a `versao6` nem a baseline |
| `versao8` | `LSTM explicita` | `0.9104` | `0.8964` | `0.9191` | Reproduziu exatamente a `versao7`, com mais transparencia metodologica |
| `versao9` | `LSTM hibrida hierarquica` | `0.9224` | `0.9268` | `0.9415` | Melhorou fortemente sobre a `versao8`, mas ainda ficou abaixo das baselines |
| `versao10` | `LSTM multitarefa sensivel a fonte` | `0.9373` | `0.9409` | `0.9572` | Melhor LSTM do projeto em `macro-F1` e `balanced accuracy`, mas ainda abaixo das baselines |
| `versao11` | `LSTM multitarefa com filtro por state e remocao de features vazias` | `1.0000` | `1.0000` | `1.0000` | Execucao correta, mas em tarefa reduzida as classes `0` e `8`; nao comparavel diretamente com a `versao10` |

Conclusao atual:

- a baseline tabular com `RandomForest` continua sendo o melhor modelo do projeto;
- o `XGBoost` tambem permanece muito forte e proximo do `RandomForest`;
- a `LSTM` da `versao6` foi melhor que a `LSTM` profunda da `versao7`;
- a `versao8` confirmou que a explicitação do codigo nos notebooks nao alterou os resultados, apenas melhorou a auditabilidade e a utilidade do material para artigo e apresentacao;
- a `versao9` melhorou bastante sobre a `versao8` e superou a `versao6` em `macro-F1` e `balanced accuracy`, mas ainda nao venceu `RandomForest` e `XGBoost`;
- o experimento da `versao7` e sua reproducao na `versao8` continuam valiosos, porque mostram que **aumentar profundidade, por si so, nao garantiu melhoria**.
- a `versao10` confirmou essa direcao metodologica: ganhou `0.0149` em `accuracy`, `0.0140` em `macro-F1` e `0.0157` em `balanced accuracy` sobre a `versao9`;
- em relacao a melhor `LSTM` anterior da `versao6`, a `versao10` perdeu pouco em `accuracy`, mas virou a melhor rede recorrente do projeto em `macro-F1` e `balanced accuracy`;
- a `versao11` executou sem erro, mas o filtro atual por `state` preservou apenas as classes `0` e `8`, o que explica os resultados perfeitos e impede comparacao direta com a tarefa multiclasse completa;
- mesmo assim, o melhor modelo global continua sendo a baseline tabular, especialmente o `RandomForest`.

## Estrutura do repositorio

- `3W/`
  clone local do dataset `3W`, mantido fora do versionamento pesado do projeto.
- `artifacts/`
  saidas de execucao, arrays preprocessados, relatorios, checkpoints e metricas.
- `versao1/`
  primeira iteracao didatica do pipeline.
- `versao2/`
  segunda iteracao com refinamentos iniciais.
- `versao3/`
  consolidacao do pipeline com funcoes auxiliares reutilizaveis.
- `versao4/`
  expansao do trabalho com foco em previsao temporal multivariada.
- `versao5/`
  comparativo academico entre modelos de previsao e modelo de persistencia.
- `versao6/`
  reformulacao do problema para classificacao multiclasse com `LSTM` e baselines tabulares.
- `versao7/`
  proposta de `LSTM` mais profunda para tentar reduzir o gap para a baseline.
- `versao8/`
  reproducao didatica da `versao7`, com as classes da `LSTM`, `RandomForest` e `XGBoost` explicitamente mostradas nos notebooks.
- `versao9/`
  proposta de `LSTM` hibrida hierarquica com janelas, fusao com `X_tab` e comparacao direta com as melhores referencias anteriores.
- `versao10/`
  proposta de `LSTM` multitarefa e sensivel a fonte, incorporando `27` variaveis, rotulos por observacao e mascaras operacionais.
- `versao11/`
  ablacao de pre-processamento sobre a `versao10`, removendo `9` features totalmente vazias e filtrando o treino das classes `1..9` para manter apenas observacoes com `state` transiente ou de falha.

## Como executar

Fluxo recomendado:

1. clonar o dataset `3W` na raiz do projeto;
2. instalar as dependencias com `pip install -r requirements.txt`;
3. escolher a versao a ser estudada;
4. executar os notebooks na ordem em que aparecem em cada pasta.

Para os experimentos mais atuais:

- `versao6/1-visao-geral-dos-dados.ipynb`
- `versao6/2-pre-processamento.ipynb`
- `versao6/3-classificacao-multiclasse-lstm.ipynb`
- `versao6/4-baseline-randomforest-xgboost.ipynb`

ou

- `versao7/1-visao-geral-dos-dados.ipynb`
- `versao7/2-pre-processamento.ipynb`
- `versao7/3-classificacao-multiclasse-lstm-profunda.ipynb`
- `versao7/4-comparacao-lstm-profunda-vs-baseline.ipynb`

ou

- `versao8/1-visao-geral-dos-dados.ipynb`
- `versao8/2-pre-processamento.ipynb`
- `versao8/3-classificacao-multiclasse-lstm-profunda.ipynb`
- `versao8/4-comparacao-lstm-profunda-vs-baseline.ipynb`

ou

- `versao9/1-visao-geral-dos-dados.ipynb`
- `versao9/2-pre-processamento.ipynb`
- `versao9/3-classificacao-multiclasse-lstm-hibrida.ipynb`
- `versao9/4-comparacao-lstm-hibrida-vs-baseline.ipynb`

ou

- `versao10/1-visao-geral-dos-dados.ipynb`
- `versao10/2-pre-processamento.ipynb`
- `versao10/3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `versao10/4-comparacao-lstm-multitarefa-vs-baselines.ipynb`

ou

- `versao11/1-visao-geral-dos-dados.ipynb`
- `versao11/2-pre-processamento.ipynb`
- `versao11/3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `versao11/4-comparacao-lstm-multitarefa-vs-baselines.ipynb`

## Onde estao os resultados mais recentes

Os artefatos usados nas analises atuais estao em:

- `artifacts/reports_v6/classificacao_v6_artigo/`
- `artifacts/reports_v7/classificacao_v7_lstm_profunda/`
- `artifacts/reports_v8/classificacao_v8_explicita/`
- `artifacts/reports_v9/classificacao_v9_lstm_hibrida/`
- `artifacts/reports_v10/classificacao_v10_multitarefa/`
- `artifacts/reports_v11/classificacao_v11_segmentos_negativos/`

Esses diretorios contem:

- arrays de treino, validacao e teste;
- metricas em `json`;
- relatorios por classe em `csv`;
- checkpoints dos modelos;
- matrizes de confusao e previsoes exportadas.

## Observacoes metodologicas

Alguns pontos importantes do historico experimental:

- `versao6` mostrou que uma representacao tabular com atributos agregados pode ser extremamente competitiva para o `3W`;
- `versao7` testou a hipotese de que uma `LSTM` mais profunda reduziria o gap para o `RandomForest`;
- os resultados nao confirmaram essa hipotese na execucao atual;
- `versao8` reproduziu numericamente a `versao7` e mostrou que a abertura do codigo no notebook nao alterou as metricas do experimento;
- `versao9` foi desenhada para atacar a vantagem informacional das baselines, combinando dinamica temporal e atributos agregados no mesmo modelo;
- o artigo do dataset reforca que ainda ha informacao relevante nao explorada plenamente no projeto atual, como rotulos por observacao (`class`, `state`), classes transitorias e o conjunto completo de 27 variaveis;
- a `versao10` nasce exatamente dessa leitura do artigo e tenta transformar essas pistas em componentes concretos do modelo;
- os resultados da `versao10` reforcam que essa leitura estava correta, pois houve ganho real sobre a `versao9` e sobre as LSTMs anteriores em duas metricas centrais;
- a `versao11` foi desenhada como um experimento de ablacao: ela nao muda a arquitetura da `versao10`, mas muda o pre-processamento para medir com mais clareza o impacto de remover features vazias e filtrar o treino das falhas pelo `state` observacional;
- na execucao atual, esse filtro ficou muito severo e preservou apenas as classes `0` e `8`, o que torna a `versao11` util como ablacao, mas nao como substituta direta da `versao10`;
- portanto, o melhor modelo do projeto continua sendo a baseline classica, e isso deve ser tratado como um achado tecnico legitimo, nao como uma falha do experimento.

## Documentacao por versao

- [versao1/README.md](versao1/README.md)
- [versao2/README.md](versao2/README.md)
- [versao3/readme-versao3.md](versao3/readme-versao3.md)
- [versao4/readme-versao4.md](versao4/readme-versao4.md)
- [versao5/readme-versao5.md](versao5/readme-versao5.md)
- [versao6/readme-versao6.md](versao6/readme-versao6.md)
- [versao7/readme-versao7.md](versao7/readme-versao7.md)
- [versao8/readme-versao8.md](versao8/readme-versao8.md)
- [versao9/readme-versao9.md](versao9/readme-versao9.md)
- [versao10/readme-versao10.md](versao10/readme-versao10.md)
- [versao11/readme-versao11.md](versao11/readme-versao11.md)
- [result.md](result.md)
- [faq.md](faq.md)
