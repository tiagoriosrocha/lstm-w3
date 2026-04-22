# README - Versao 8

## Escopo

A `versao8` preserva o mesmo problema da `versao7`: **classificacao multiclasse de series temporais** no dataset `3W`, tratando cada arquivo `parquet` como uma amostra rotulada entre as classes `0` e `9`.

O diferencial desta versao nao e criar uma nova familia de modelos, e sim tornar o experimento muito mais auditavel e didatico. Os blocos centrais que antes estavam concentrados no `pipeline_v7.py` passam a aparecer explicitamente nos notebooks:

- a classe `LSTMSeriesClassifier`;
- a rotina de treinamento da `LSTM`;
- a rotina de inferencia da `LSTM`;
- a construcao das baselines `RandomForest` e `XGBoost`;
- partes importantes do pre-processamento, como a reamostragem temporal e a criacao do vetor tabular de estatisticas.

## Objetivo pedagogico

A `versao8` foi pensada para um contexto de apresentacao, correcao ou defesa academica. A ideia e permitir que o professor acompanhe a sequencia completa da construcao dos modelos sem depender apenas do pipeline auxiliar.

Em termos práticos, esta versao facilita responder perguntas como:

- onde exatamente a `LSTM` e declarada?
- como o `forward` combina os resumos temporais?
- qual classe do `scikit-learn` implementa o baseline?
- como o `RandomForest` e o `XGBoost` sao instanciados?
- como o melhor checkpoint e escolhido?
- como as metricas finais sao calculadas e comparadas?

## O que a versao8 manteve da versao7

Do ponto de vista experimental, a `versao8` mantem a mesma proposta arquitetural da `versao7` para a rede recorrente:

- `hidden_size = 192`
- `num_layers = 4`
- `dropout = 0.30`
- `bidirectional = True`
- combinacao de tres resumos temporais:
  - ultimo estado oculto
  - media temporal
  - pooling por atencao

Em termos conceituais:

```text
serie temporal -> LSTM empilhada -> resumos temporais -> MLP classificadora -> softmax -> classe 0..9
```

As baselines tabulares tambem seguem a mesma ideia da `versao7`:

- `RandomForestClassifier` sobre `X_tab`
- `XGBClassifier` sobre `X_tab`, quando a biblioteca `xgboost` estiver instalada

## Resultados da execucao realizada

Na execucao registrada em `artifacts/reports_v8/classificacao_v8_explicita/`, os resultados principais foram:

- `LSTM explicita` em validacao:
  - `accuracy = 0.9251`
  - `macro-F1 = 0.8997`
  - `balanced accuracy = 0.9101`
- `LSTM explicita` em teste:
  - `accuracy = 0.9104`
  - `macro-F1 = 0.8964`
  - `balanced accuracy = 0.9191`
- `RandomForest` em teste:
  - `accuracy = 0.9851`
  - `macro-F1 = 0.9811`
  - `balanced accuracy = 0.9744`
- `XGBoost` em teste:
  - `accuracy = 0.9821`
  - `macro-F1 = 0.9791`
  - `balanced accuracy = 0.9733`

Leitura quantitativa:

- a `versao8` reproduziu exatamente os numeros da `versao7` para a `LSTM`;
- portanto, a abertura do codigo no notebook nao mudou o comportamento do experimento;
- o `RandomForest` permaneceu liderando com vantagem de `0.0746` em `accuracy`, `0.0847` em `macro-F1` e `0.0553` em `balanced accuracy` sobre a `LSTM`;
- o `XGBoost` tambem ficou acima da `LSTM`, com vantagem de `0.0716` em `accuracy`, `0.0827` em `macro-F1` e `0.0542` em `balanced accuracy`.

Leitura metodologica:

- a principal contribuicao da `versao8` nao foi superar a baseline, mas tornar o experimento mais transparente;
- essa transparencia e especialmente util para artigo, defesa oral e avaliacao docente;
- a equivalencia numerica entre `versao7` e `versao8` reforca a confiabilidade da reproducao.

## Estrutura

1. `1-visao-geral-dos-dados.ipynb`
   Reapresenta o dataset e reforca a ideia de que cada arquivo representa uma amostra de classificacao.
2. `2-pre-processamento.ipynb`
   Mostra o fluxo de preparacao dos dados e expõe, no proprio notebook, a reamostragem temporal e a geracao de descritores estatisticos.
3. `3-classificacao-multiclasse-lstm-profunda.ipynb`
   Treina e avalia a `LSTM`, com a classe do modelo e a rotina de treino escritas explicitamente no notebook.
4. `4-comparacao-lstm-profunda-vs-baseline.ipynb`
   Treina `RandomForest` e `XGBoost` com as classes explicitamente visiveis e compara os resultados com a `LSTM`.
5. `pipeline_v8.py`
   Continua existindo como suporte reutilizavel para leitura de artefatos, metricas e orquestracao do experimento.

## Diferenca metodologica em relacao as versoes anteriores

As `versoes 6` e `7` ja eram reprodutiveis, mas parte relevante da logica estava encapsulada no pipeline. Na `versao8`, o foco muda:

- menos "caixa-preta" nos notebooks;
- mais visibilidade de classes, funcoes e hiperparametros;
- mais comentarios no codigo;
- mais facilidade para inspecao detalhada por um professor ou aluno iniciante.

Por isso, a `versao8` e especialmente adequada para:

- apresentacoes em sala;
- relatorios de disciplina;
- estudo guiado;
- revisao para prova;
- defesa do raciocinio experimental do projeto.

## Leitura recomendada

Se a ideia for entender o projeto como um todo, a sequencia sugerida e:

1. abrir `1-visao-geral-dos-dados.ipynb`;
2. executar `2-pre-processamento.ipynb`;
3. ler com calma a classe `LSTMSeriesClassifier` no notebook 3;
4. comparar a `LSTM` com `RandomForest` e `XGBoost` no notebook 4;
5. usar o `faq.md` da raiz para revisar os conceitos teoricos e as metricas.

## Conclusao

A `versao8` nao existe para trocar o melhor modelo do projeto, e sim para tornar a experiencia mais transparente. Depois da execucao, ela se consolidou como a melhor versao para escrita do artigo, porque combina:

- resultados reais ja medidos;
- reproducao fiel da `versao7`;
- exposicao explicita das classes e rotinas principais;
- facilidade de auditoria do experimento.

Ela transforma os notebooks em um material de estudo e avaliacao mais completo, no qual o leitor consegue ver diretamente:

- como os dados sao transformados;
- como cada modelo e declarado;
- como o treinamento acontece;
- como a avaliacao e feita;
- e como a comparacao final e sustentada pelos numeros.
