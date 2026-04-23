# Versao 11

A `versao11` preserva a arquitetura multitarefa da `versao10`, mas muda o recorte observacional do treino. A execucao mais recente dos notebooks corresponde ao experimento `classificacao_v11_foco_por_class_observacional`, que usa `class` observacional para focar trechos de falha e transiente sem reduzir o problema a apenas duas classes.

## Ideia central

No `3W`, cada observacao pode carregar dois rotulos diferentes:

- `class`: identifica o evento observado naquele instante;
- `state`: identifica o estado operacional do poco naquele instante.

Na `versao11`, o foco do treino passa a respeitar essa diferenca:

- `class = 0` representa operacao normal;
- `class = 1..9` representa falhas;
- `class = 101..109` representa estados transitorios associados a eventos;
- `state` continua util como metadado auxiliar, mas nao como criterio principal de corte.

## O que mudou em relacao a `versao10`

- remove `9` features totalmente vazias: `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`;
- reduz a entrada sequencial de `27` para `18` variaveis;
- reduz a representacao tabular de `243` para `162` atributos;
- usa `sequence_length = 180`;
- mantem a mesma arquitetura multitarefa da `versao10`;
- preserva todas as `2228` series no `split`;
- para as classes globais `1..9`, usa no treino apenas as observacoes cujo `class` local indica falha ou transiente;
- preserva a serie completa da classe `0`, para manter o contraste com operacao normal.

## Resumo do pre-processamento

Nos artefatos atuais de `classificacao_v11_foco_por_class_observacional`:

- series totais: `2228`
- `split`: `train = 1559`, `validation = 334`, `test = 335`
- classes preservadas: `0..9`
- distribuicao por classe: `0: 594`, `1: 128`, `2: 38`, `3: 106`, `4: 343`, `5: 450`, `6: 221`, `7: 46`, `8: 95`, `9: 207`
- series com pelo menos uma observacao em foco: `1591`

O arquivo `series_quality_report.csv` deve ser lido assim:

- `n_observation_class_zero_rows`: observacoes normais da serie;
- `n_observation_class_fault_rows`: observacoes em `class = 1..9`;
- `n_observation_class_transient_rows`: observacoes em classes transitorias;
- `n_observation_class_focus_rows`: observacoes que entram no foco de treino da `v11`.

## Resultados mais recentes

### Teste - `classificacao_v11_foco_por_class_observacional`

| Modelo | Accuracy | Macro-F1 | Balanced Accuracy | Leitura |
| --- | ---: | ---: | ---: | --- |
| `LSTM multitarefa` | `0.9194` | `0.9118` | `0.9238` | ablacao valida, mas abaixo da `versao10` |
| `RandomForest` | `0.9851` | `0.9812` | `0.9752` | melhor `accuracy` da `versao11` |
| `LGBM` | `0.9791` | `0.9798` | `0.9714` | baseline forte e estavel |
| `XGBoost` | `0.9821` | `0.9850` | `0.9814` | melhor baseline em `macro-F1` e `balanced accuracy` |

Leitura curta:

- a `versao11` nao superou a `versao10`;
- a melhor rede recorrente do projeto continua sendo a `versao10`;
- as baselines tabulares continuam dominando tambem nesta ablacao.

## Classes mais dificeis para a LSTM

No teste da `versao11`, os pontos mais sensiveis da `LSTM` foram:

- classe `3`: `precision = 0.5517`, apesar de `recall = 1.0000`;
- classe `5`: `recall = 0.8382`;
- classe `2`: `recall = 0.8000`;
- classe `4`: ainda competitiva, mas com erros relevantes contra a classe `0`.

## Sobre o experimento antigo

O diretorio `artifacts/reports_v11/classificacao_v11_segmentos_negativos/` continua existindo como registro de uma rodada anterior. Nela, o filtro por `state` reduziu a avaliacao pratica as classes `0` e `8`, gerando metricas quase perfeitas. Esses numeros nao devem ser usados como resultado consolidado da `versao11`.

## Conclusao

A `versao11` foi importante para testar a hipotese de que um recorte observacional mais coerente com o `3W` poderia ajudar a `LSTM`. A execucao completa dos notebooks mostrou que a ideia e metodologicamente melhor que o filtro por `state`, mas nao trouxe ganho final sobre a `versao10`.

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`
