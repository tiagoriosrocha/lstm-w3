# Resultados Consolidados Do Projeto

## Como ler este arquivo

O projeto tem duas fases metodologicas diferentes:

- `versoes 1 a 5`: previsao temporal multivariada, com metricas como `MAE`, `RMSE`, `MSE` e comparacao com a `persistencia`;
- `versoes 6 a 11`: classificacao multiclasse de series temporais, com metricas como `accuracy`, `macro-F1` e `balanced accuracy`.

Por isso, os numeros das `versoes 1 a 5` nao devem ser comparados diretamente com os das `versoes 6 a 11`.

## Panorama geral

Leitura curta do historico experimental:

- `versao1` foi a primeira prova de conceito e perdeu claramente para a persistencia;
- `versao2` mostrou que uma arquitetura residual mais rica podia vencer bem a persistencia em arquivo unico;
- `versao3` levou essa ideia para multiplos arquivos e manteve ganho global sobre a persistencia;
- `versao4` mostrou que uma `LSTM` pura podia superar a persistencia no teste multi-step completo, mesmo com validacao dura;
- `versao5` reorganizou o problema em formato comparativo para artigo e mostrou a persistencia novamente como referencia fortissima;
- `versao6` redefiniu o problema como classificacao multiclasse e introduziu `RandomForest` e `XGBoost`, que passaram a liderar o projeto;
- `versoes7` e `8` mostraram que apenas aprofundar a `LSTM` nao bastava;
- `versao9` recuperou desempenho com uma `LSTM` hibrida hierarquica;
- `versao10` foi a melhor rede recorrente do projeto em `macro-F1` e `balanced accuracy`, mas ainda nao superou as baselines tabulares.
- `versao11` funcionou como uma ablacao de pre-processamento: a rodada antiga por `state` reduziu o problema a `0` e `8`, mas a execucao mais recente corrigiu isso com foco por `class` observacional e voltou a avaliar as `10` classes.

## Versoes 1 A 5: Previsao Temporal

### Tabela resumida

| Versao | Modelo principal | Configuracao principal | Resultado central | Leitura |
| --- | --- | --- | --- | --- |
| `versao1` | `LSTMForecaster` | `input = 6`, `seq = 30`, `hidden = 64`, `layers = 2`, `dropout = 0.2`, loss `MSE` | teste: `MSE = 0.488696`, `RMSE = 0.699068`, `MAE = 0.496597`; persistencia: `RMSE = 0.051761`, `MAE = 0.021016` | prova de conceito didatica; baseline venceu com ampla margem |
| `versao2` | `Hybrid Residual GRU + Attention` | `input = 24`, `seq = 60`, projecao `128`, `3` blocos conv residuais, `BiGRU hidden = 128`, `2` camadas, atencao, previsao residual sobre persistencia | teste escalado: `RMSE = 0.022600`, `MAE = 0.007636`; persistencia: `RMSE = 0.051535`, `MAE = 0.020833` | primeiro ganho forte e claro sobre a persistencia |
| `versao3` | `Hybrid Residual Forecaster v3` | `input = 36`, `targets = 6`, `seq = 60`, conv residual + `BiGRU` + atencao + embedding do poco, `894.031` parametros | teste: modelo `MAE = 703.94`, `RMSE = 4595.72`; persistencia `MAE = 1037.84`, `RMSE = 7137.56` | ganho global de cerca de `32.17%` em `MAE` sobre a persistencia |
| `versao4` | `PureLSTMForecaster` e `HybridResidualForecaster` | ambos com `input = 36`, `targets = 6`, horizonte `5`, `seq = 60`, `hidden = 128`, `layers = 2`, `well_embedding = 16`, `dropout = 0.2`; treino com janelas amostradas e perda ponderada | teste do modelo escolhido (`LSTM pura`): `MAE = 1938.90` contra `2236.45` da persistencia, melhora de `13.30%`; em escala padronizada, `MAE = 0.000419` contra `0.000464` | melhor modelo da previsao multi-step completa foi a `LSTM pura` |
| `versao5` | comparativo entre `LSTM pura`, `Hibrido residual` e `Persistencia` | `LSTM`: `hidden = 128`, `recurrent_layers = 2`, `well_embedding = 16`, `dropout = 0.2`; `Hibrido`: `model_dim = 128`, `hidden = 128`, `recurrent_layers = 2`, `well_embedding = 16`, `dropout = 0.2`; ambos com horizonte `5` | teste original: `persistencia MAE = 2150.57`, `hibrido MAE = 2570.24`, `lstm MAE = 6893.40`; teste escalado: `persistencia MAE = 0.000437`, `hibrido MAE = 0.000512`, `lstm MAE = 0.001318` | no protocolo academico comparativo da `versao5`, a `persistencia` voltou a ser o melhor metodo |

### O que cada modelo fazia

`versao1`
- Previa o proximo passo de `6` variaveis alvo a partir de uma janela com `30` instantes.
- Era a forma mais classica de `LSTM` many-to-one do projeto.

`versao2`
- Adicionou engenharia de atributos locais e trocou a `LSTM` simples por um bloco hibrido com convolucao residual, `BiGRU` e atencao.
- Aprendia um delta sobre a persistencia, o que ajudou bastante.

`versao3`
- Levou a ideia residual para a base com muitos arquivos.
- Incluiu embedding do poco e inferencia streaming em larga escala.

`versao4`
- Comparou, de forma justa, uma `LSTM` pura contra o modelo hibrido residual.
- O problema passou a ser de previsao multi-step com horizonte `5`.

`versao5`
- Nao criou uma familia nova de modelos.
- Reorganizou `LSTM`, `hibrido residual` e `persistencia` sob o mesmo protocolo para sustentar a escrita do artigo.

## Versoes 6 A 11: Classificacao Multiclasse

### Tabela consolidada

| Versao | Modelo | Accuracy | Macro-F1 | Balanced Accuracy | Leitura |
| --- | --- | ---: | ---: | ---: | --- |
| `versao6` | `LSTM multiclasse` | `0.9433` | `0.9185` | `0.9367` | primeira formulacao forte de classificacao |
| `versao6` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | melhor resultado global do projeto |
| `versao6` | `XGBoost` | `0.9821` | `0.9791` | `0.9733` | baseline muito forte, logo atras do `RandomForest` |
| `versao7` | `LSTM profunda` | `0.9104` | `0.8964` | `0.9191` | profundidade extra nao melhorou a rede |
| `versao7` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | baseline ainda dominante |
| `versao7` | `XGBoost` | `0.9821` | `0.9791` | `0.9733` | baseline ainda dominante |
| `versao8` | `LSTM explicita` | `0.9104` | `0.8964` | `0.9191` | reproduziu exatamente a `versao7` |
| `versao8` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | mesma referencia tabular da `versao7` |
| `versao8` | `XGBoost` | `0.9821` | `0.9791` | `0.9733` | mesma referencia tabular da `versao7` |
| `versao9` | `LSTM hibrida hierarquica` | `0.9224` | `0.9268` | `0.9415` | retomou desempenho com janelas e fusao `X_tab` |
| `versao9` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | continuou liderando |
| `versao9` | `XGBoost` | `0.9821` | `0.9791` | `0.9733` | continuou acima da rede |
| `versao10` | `LSTM multitarefa sensivel a fonte` | `0.9373` | `0.9409` | `0.9572` | melhor rede recorrente do projeto em `macro-F1` e `balanced accuracy` |
| `versao10` | `RandomForest` | `0.9851` | `0.9811` | `0.9744` | melhor modelo global ainda |
| `versao10` | `XGBoost` | `0.9791` | `0.9775` | `0.9714` | segunda melhor baseline global |
| `versao11` | `LSTM multitarefa com foco por class observacional` | `0.9194` | `0.9118` | `0.9238` | ablacao valida, mas abaixo da `versao10` |
| `versao11` | `RandomForest` | `0.9851` | `0.9812` | `0.9752` | melhor `accuracy` da `versao11` |
| `versao11` | `LGBM` | `0.9791` | `0.9798` | `0.9714` | baseline forte e estavel |
| `versao11` | `XGBoost` | `0.9821` | `0.9850` | `0.9814` | melhor baseline da `versao11` em `macro-F1` e `balanced accuracy` |

Observacao importante:

- os numeros acima correspondem a `artifacts/reports_v11/classificacao_v11_foco_por_class_observacional/`;
- o diretorio `classificacao_v11_segmentos_negativos/` permanece apenas como registro de uma rodada anterior, baseada em filtro por `state`, que reduziu a avaliacao pratica as classes `0` e `8`.

### Configuracao dos modelos de classificacao

`versao6 - LSTM multiclasse`
- `input_size = 18`
- `hidden_size = 128`
- `num_layers = 2`
- `dropout = 0.20`
- `bidirectional = True`
- cabeca densa simples para prever `10` classes
- treino com `learning_rate = 1e-3`
- entrada sequencial `X_seq` e baseline tabular `X_tab = 162` atributos estatisticos

`versao6 a versao9 - RandomForest`
- `RandomForestClassifier`
- `n_estimators = 400`
- `max_depth = None`
- `class_weight = balanced_subsample`
- `random_state = 42`
- recebe apenas `X_tab`

`versao6 a versao9 - XGBoost`
- `XGBClassifier`
- `n_estimators = 400`
- `max_depth = 6`
- `learning_rate = 0.05`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- `objective = multi:softmax`
- recebe apenas `X_tab`

`versao7 e versao8 - LSTM profunda`
- `input_size = 18`
- `hidden_size = 192`
- `num_layers = 4`
- `dropout = 0.30`
- `bidirectional = True`
- combina ultimo estado oculto, media temporal e pooling por atencao
- treino com `learning_rate = 7e-4`

`versao9 - LSTM hibrida hierarquica`
- `input_size = 18`
- `tabular_size = 162`
- separa variaveis continuas e variaveis de estado
- `window_size = 20`
- `continuous_hidden_size = 96`
- `state_hidden_size = 64`
- `context_hidden_size = 160`
- `context_num_layers = 2`
- `tabular_hidden_size = 128`
- `dropout = 0.25`
- usa codificacao local por janela, codificacao de contexto entre janelas e fusao final com `X_tab`
- treino com `learning_rate = 6e-4`

`versao10 - LSTM multitarefa sensivel a fonte`
- `input_size = 27`
- `tabular_size = 243`
- `num_step_classes = 18`
- `num_state_classes = 9`
- `source_vocab_size = 3`
- `hidden_size = 160`
- `num_layers = 2`
- `source_embedding_dim = 12`
- `tabular_hidden_size = 160`
- `dropout = 0.25`
- concatenacao de `X_seq`, `X_missing` e `X_frozen`
- frontend local com `Conv1D`
- codificacao temporal com `BiLSTM`
- cabeca principal para classe global
- cabeca auxiliar para `class` por observacao
- cabeca auxiliar para `state` por observacao
- treino com `AdamW`, `learning_rate = 5e-4`, `weight_decay = 1e-4`
- pesos auxiliares `lambda_step_class = 0.35` e `lambda_step_state = 0.15`

`versao11 - Ablacao de pre-processamento sobre a versao10`
- mesma arquitetura central da `versao10`, mas com pre-processamento diferente
- remocao de `9` features totalmente vazias: `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`
- `input_size = 18`
- `tabular_size = 162`
- `sequence_length = 180`
- no treino das classes de falha, manutencao apenas de observacoes cujo `class` local indica falha ou transiente
- `split` preservado com `2228` series: `1559` treino, `334` validacao e `335` teste
- classes preservadas no problema: `0..9`
- os artefatos antigos de `segmentos_negativos` seguem apenas como referencia historica, nao como resultado principal

### Leitura comparativa final da classificacao

Pontos mais importantes:

- o `RandomForest` permaneceu como melhor modelo global em todas as versoes de classificacao;
- o `XGBoost` foi a segunda baseline mais forte e, em geral, ficou muito proximo do `RandomForest`;
- a `versao6` foi a primeira LSTM realmente competitiva;
- a `versao7` mostrou que mais camadas, sozinhas, nao resolviam o problema;
- a `versao8` teve a mesma performance da `versao7`, mas tornou o experimento mais auditavel;
- a `versao9` melhorou bastante a rede recorrente com uma leitura hierarquica por janelas;
- a `versao10` foi a melhor rede recorrente do projeto em `macro-F1` e `balanced accuracy`.
- a `versao11` confirmou que o problema e muito sensivel ao pre-processamento: a leitura por `class` observacional e conceitualmente melhor, mas nao trouxe ganho final sobre a `versao10`.

Gaps centrais da `versao10`:

- sobre a `versao9`, houve ganho de `0.0149` em `accuracy`, `0.0140` em `macro-F1` e `0.0157` em `balanced accuracy`;
- sobre a melhor `LSTM` anterior da `versao6`, houve perda de `0.0060` em `accuracy`, mas ganho de `0.0223` em `macro-F1` e `0.0205` em `balanced accuracy`;
- o `RandomForest` ainda abre `0.0478` em `accuracy`, `0.0403` em `macro-F1` e `0.0172` em `balanced accuracy` sobre a `versao10`;
- o `XGBoost` ainda abre `0.0418` em `accuracy`, `0.0367` em `macro-F1` e `0.0142` em `balanced accuracy` sobre a `versao10`.

Leitura da `versao11`:

- a execucao mais recente ficou tecnicamente correta e todos os notebooks rodaram sem erro;
- a documentacao agora deve tomar como referencia `classificacao_v11_foco_por_class_observacional`, nao `segmentos_negativos`;
- a `LSTM` da `versao11` ficou `0.0179` abaixo da `versao10` em `accuracy`, `0.0291` abaixo em `macro-F1` e `0.0334` abaixo em `balanced accuracy`;
- as baselines continuaram fortes: na propria `versao11`, o `RandomForest` abriu `0.0657` de `accuracy` sobre a `LSTM`, e o `XGBoost` abriu `0.0731` de `macro-F1`;
- por isso, a `versao11` deve ser lida como ablacao metodologica util, mas nao como nova melhor versao do projeto.

Classes em que a `versao10` ainda mais sofre frente aos baselines:

- contra o `RandomForest`: `7`, `4`, `0` e `1`
- contra o `XGBoost`: `7`, `4`, `1` e `0`

## Ranking final

Se o criterio for o melhor modelo global de classificacao:

1. `RandomForest`
2. `XGBoost`
3. `LSTM multitarefa sensivel a fonte (versao10)`
4. `LSTM hibrida hierarquica (versao9)`
5. `LSTM multiclasse original (versao6)`
6. `LSTM profunda explicita/profunda (versoes7 e 8)`

Se o criterio for a melhor rede recorrente:

1. `versao10`
2. `versao9`
3. `versao6`
4. `versoes7 e 8`

Observacao sobre ranking:

- a `versao11` entra como ablacao comparavel, mas nao altera o ranking final porque ficou abaixo da `versao10` e das baselines.

## Conclusao geral

O projeto mostrou duas licoes principais:

- em previsao temporal, a `persistencia` e um baseline extremamente forte e precisa ser tratada como referencia seria;
- em classificacao multiclasse, as baselines tabulares com estatisticas agregadas permaneceram muito competitivas.

A principal contribuicao cientifica das ultimas iteracoes foi mostrar que:

- o ganho da `versao10` nao veio de apenas aumentar profundidade;
- o ganho veio de usar melhor a estrutura do dataset, incluindo `27` variaveis, mascaras operacionais, contexto de fonte e supervisao temporal auxiliar.
- a `versao11` reforcou, por contraste, que mudar o pre-processamento pode alterar bastante o desempenho final e precisa ser tratado com muito cuidado.

Isso nao foi suficiente para derrotar o `RandomForest`, mas foi suficiente para produzir a melhor `LSTM` do projeto.
