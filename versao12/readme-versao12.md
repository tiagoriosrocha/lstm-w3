# Versao 12

A `versao12` foi criada como uma sintese arquitetural das melhores ideias exploradas nas versoes anteriores.

Em vez de escolher entre profundidade, hierarquia temporal, informacao tabular, mascaras operacionais ou supervisao auxiliar, a proposta agora e juntar tudo isso em uma unica arquitetura coerente.

## O que a versao12 propoe

A arquitetura central da `versao12` combina:

- profundidade alta nos codificadores recorrentes locais e no codificador de contexto;
- pooling avancado com `last hidden + mean pooling + attention pooling`;
- hierarquia temporal por janelas, inspirada na `versao9`;
- separacao explicita entre variaveis continuas e variaveis de estado;
- uso de `X_tab` como ramo complementar de atributos agregados;
- mascaras de `missing values` e `frozen variables`, herdadas da `versao10`;
- embedding de contexto por tipo de fonte da serie (`well`, `simulated`, `drawn`);
- multitarefa com cabeca principal para a classe global e cabecas auxiliares para `class` e `state` por observacao.

## Decisoes de pre-processamento

Na implementacao atual, a `versao12` reaproveita duas decisoes que se mostraram uteis nas versoes anteriores:

- mantem a estrutura multitarefa e os artefatos de `X_seq`, `X_tab`, `X_missing`, `X_frozen`, `y_step_class`, `y_step_state` e `source_id`;
- adota a limpeza de features totalmente vazias introduzida na `versao11`, reduzindo a entrada sequencial para `18` variaveis realmente informativas.

Ao mesmo tempo, ela **nao herda o recorte observacional da `versao11` para treino das classes `1..9`**. A `versao12` volta a usar a serie completa, para preservar a comparabilidade direta com a `versao10` e nao carregar uma ablacao que reduziu desempenho.

## Arquitetura em alto nivel

O fluxo da rede fica assim:

`sequencia completa -> separacao em ramos continuo/estado -> janelas temporais -> LSTMs locais profundas -> pooling avancado por janela -> codificador de contexto entre janelas -> cabecas auxiliares temporais -> fusao com X_tab e embedding de source -> classificacao global`

Isso faz a `versao12` virar, na pratica, uma fusao entre:

- a profundidade e o pooling rico da `versao7`;
- a hierarquia temporal e o uso de `X_tab` da `versao9`;
- as mascaras, o embedding de contexto e a multitarefa da `versao10`;
- a selecao de features nao vazias da `versao11`.

## Status atual

Nesta etapa, a `versao12` foi criada no arquivo `pipeline_v12.py` e esta pronta para:

- gerar artefatos preprocessados em `artifacts/reports_v12/`;
- treinar o modelo profundo hierarquico multitarefa;
- exportar checkpoint, historico e metricas no mesmo padrao das versoes recentes.

As metricas da `versao12` ainda dependem da execucao dos experimentos.

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`
