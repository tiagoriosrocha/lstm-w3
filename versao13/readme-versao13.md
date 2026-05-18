# Versao 13

A `versao13` adapta ao projeto `3W` uma arquitetura inspirada no artigo de `BiGRU + Multi-Head Attention`, mas sem abrir mao das licoes praticas que vieram das versoes anteriores do repositorio.

## Ideia central

O objetivo desta versao e testar uma rede mais simples que a `versao12`, porem ainda forte em modelagem sequencial:

- `BiGRU` bidirecional com `48` unidades por direcao;
- `dropout + layer normalization` apos a saida recorrente;
- `multi-head self-attention` com conexao residual;
- `global average pooling`;
- bloco final totalmente conectado com `ReLU`, `dropout` e `softmax`.

Em vez de manter a arquitetura multitarefa, hierarquica e multi-ramo da `versao12`, a `versao13` volta a um fluxo mais direto: **uma unica sequencia entra, uma classe global sai**.

## Escolha de pre-processamento

Para a `versao13`, a melhor combinacao com essa arquitetura foi:

- manter a limpeza da `versao11`, removendo as `9` features totalmente vazias;
- manter o protocolo robusto da `versao10` e `versao12` para `split`, reamostragem temporal para comprimento fixo e padronizacao por feature;
- manter `sequence_length = 180`, que ja esta consolidado nas versoes mais recentes do projeto;
- preservar `X_tab`, `X_missing`, `X_frozen`, `y_step_class`, `y_step_state` e `source_id` nos artefatos para comparabilidade e baselines;
- gerar adicionalmente `X_seq_bigru`, uma versao **univariada por timestep** da sequencia.

### Por que nao usar o recorte observacional da `versao11`

A `versao11` foi importante como ablacao, mas seu foco observacional reduziu desempenho final frente a `versao10`. Como a `versao13` quer medir o efeito da arquitetura `BiGRU + MHA`, ela herda da `versao11` apenas a limpeza de features, e nao o recorte de treino.

### Por que a sequencia vira univariada

O artigo de referencia descreve a entrada como `X in R^(T x 1)` apos a adaptacao da sequencia. No `3W`, depois da limpeza da `versao11`, ainda temos `18` features por timestep. Se simplesmente achatarmos `180 x 18` para uma sequencia longa, a etapa de `self-attention` fica quadraticamente mais cara e perde viabilidade pratica.

Por isso, a `versao13` usa uma projecao univariada por timestep, ajustada no `train` a partir da primeira componente principal das `18` features padronizadas. Assim:

- preservamos a estrutura temporal em `180` passos;
- satisfazemos a exigencia de entrada univariada da arquitetura;
- evitamos inflar artificialmente o comprimento efetivo da sequencia;
- continuamos apoiados no pre-processamento mais estavel ja validado no projeto.

## Artefatos gerados

Nos artefatos de `classificacao_v13_bigru_mha`, cada split passa a conter:

- `X_seq`: sequencia multivariada padronizada com `18` features;
- `X_seq_bigru`: sequencia univariada pronta para o modelo `BiGRU + MHA`;
- `X_tab`: atributos agregados para baselines tabulares;
- `X_missing` e `X_frozen`: mantidos para comparabilidade metodologica;
- `y`: classe global da serie;
- `y_step_class` e `y_step_state`: supervisionamentos locais preservados como registro;
- `source_id`: metadado de origem da amostra.

O diretorio tambem salva `univariate_projection_v13.json`, que documenta a projecao usada para transformar `X_seq` em `X_seq_bigru`.

## Status atual

Nesta etapa, a `versao13` ja possui:

- `pipeline_v13.py`;
- notebooks de visao geral, pre-processamento, treino e comparacao;
- integracao com o mesmo protocolo de avaliacao e baselines das versoes recentes.

As metricas finais da `versao13` ainda dependem da execucao completa dos notebooks.

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-bigru-mha.ipynb`
- `4-comparacao-bigru-mha-vs-baselines.ipynb`
