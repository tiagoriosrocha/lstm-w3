# Versao 11

A `versao11` preserva a arquitetura multitarefa da `versao10`, mas muda deliberadamente o que entra no treino. O foco agora e medir se a rede melhora quando deixa de aprender as classes de falha a partir de observacoes cujo `state` ainda e ausente ou normal.

## O que a versao11 propoe

- remocao das features totalmente vazias: `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`;
- recorte por `state` no treino das classes globais `1..9`, mantendo apenas `1 = transiente` e `2 = falha`;
- manutencao da serie completa para a classe `0`;
- avaliacao final na mesma arquitetura multitarefa da `versao10`.

## Estado atual

Resultados registrados para a `versao11`:

- validacao: `accuracy = 0.8851`, `macro-F1 = 0.8724`, `balanced accuracy = 0.8921`
- teste: `accuracy = 0.9213`, `macro-F1 = 0.9155`, `balanced accuracy = 0.9225`
- comparacao com a `versao10`: `accuracy = -0.0160`, `macro-F1 = -0.0253`, `balanced accuracy = -0.0347`
- `RandomForest` em teste: `accuracy = 0.9888`, `macro-F1 = 0.9833`, `balanced accuracy = 0.9750`

## Sequencia recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`
