# Versao 11

A `versao11` preserva a arquitetura multitarefa da `versao10`, mas muda deliberadamente o pre-processamento. O objetivo desta versao e testar se vale a pena treinar as classes de falha apenas com trechos em que o `state` ja esta em fase transiente ou de falha, alem de remover features totalmente vazias.

## O que a versao11 propoe

- remocao das features totalmente vazias: `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`;
- reducao das entradas de `27` para `18` variaveis e de `243` para `162` atributos tabulares;
- recorte por `state` no treino das classes globais `1..9`, mantendo apenas `1 = transiente` e `2 = falha`;
- manutencao da serie completa para a classe `0`;
- mesma arquitetura multitarefa da `versao10`.

## Execucao atual

Os quatro notebooks da `versao11` executaram sem erro.

Durante a auditoria dos resultados, foi corrigida uma inconsistencia na exportacao de avaliacao: antes, os arquivos `metrics.json` eram calculados sobre as classes presentes, mas o `classification_report.csv` ainda listava tambem classes ausentes. Agora os dois artefatos usam o mesmo conjunto efetivo de classes observado na avaliacao.

## O que aconteceu com a base

Com a regra atual de filtro por `state`, a `versao11` ficou muito mais restritiva do que as versoes anteriores:

- series originais: `2228`
- series mantidas: `605`
- series descartadas por `no_negative_state_segment`: `1623`
- `split`: `train = 423`, `validation = 90`, `test = 92`
- classes sobreviventes: apenas `0` e `8`

Isso muda bastante a interpretacao do experimento. Na pratica, a execucao atual da `versao11` deixou de representar o problema multiclasse amplo das versoes `6` a `10` e passou a ser uma tarefa reduzida, dominada pelas classes `0` e `8`.

## Resultados atuais

`LSTM multitarefa v11`

- validacao: `accuracy = 1.0000`, `macro-F1 = 1.0000`, `balanced accuracy = 1.0000`
- teste: `accuracy = 1.0000`, `macro-F1 = 1.0000`, `balanced accuracy = 1.0000`

Baselines em teste:

- `RandomForest`: `accuracy = 1.0000`, `macro-F1 = 1.0000`, `balanced accuracy = 1.0000`
- `LGBM`: `accuracy = 1.0000`, `macro-F1 = 1.0000`, `balanced accuracy = 1.0000`
- `XGBoost`: `accuracy = 0.9891`, `macro-F1 = 0.8972`, `balanced accuracy = 0.9944`

## Leitura metodologica

Esses numeros perfeitos nao significam que a `versao11` superou a `versao10` no problema original. Eles significam que, com o filtro por `state` implementado do jeito atual, a tarefa ficou muito mais facil e muito menos abrangente.

Portanto:

- a execucao esta tecnicamente correta;
- os artefatos agora estao consistentes entre `metrics.json` e `classification_report.csv`;
- mas a `versao11`, no estado atual, nao deve ser tratada como nova melhor rede recorrente do projeto;
- a melhor comparacao multiclasse ainda continua sendo a `versao10`.

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`
