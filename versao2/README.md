# Projeto 3W - Versao 2

## Visao geral

Esta pasta documenta a segunda versao do experimento com o dataset 3W. A meta da `versao2` foi manter um cenario controlado de **arquivo unico**, mas substituir a LSTM simples da `versao1` por um pipeline mais forte:

- entrada enriquecida com features derivadas
- modelo hibrido com convolucao, `GRU` bidirecional e atencao
- previsao residual sobre a persistencia
- selecao de checkpoint baseada em comparacao direta com o baseline

O arquivo usado foi:

`3W/dataset/0/WELL-00001_20170201010207.parquet`

## Estrutura

- `2-pre-processamento.ipynb`
  - limpa o arquivo unico
  - divide em treino, validacao e teste
  - cria `24` features de entrada
- `3-treino-validacao.ipynb`
  - monta janelas de `60` passos
  - treina o modelo hibrido residual
  - salva o melhor checkpoint
- `4-teste.ipynb`
  - avalia o checkpoint no trecho de teste
  - compara com o baseline de persistencia
  - analisa metricas globais e por feature

## Pre-processamento

O pre-processamento da `versao2` ficou parecido com o da `versao1` na limpeza base, mas diferente na representacao da entrada.

As 6 variaveis alvo continuam:

- `P-ANULAR`
- `P-JUS-CKGL`
- `P-MON-CKP`
- `P-TPT`
- `T-JUS-CKP`
- `T-TPT`

Nenhuma coluna auxiliar foi aproveitada nesta execucao, porque todas as candidatas ficaram constantes no trecho de treino. Mesmo assim, a entrada final cresceu de `6` para `24` features, porque cada alvo gerou:

- `raw__`: valor padronizado
- `diff1__`: variacao instante a instante
- `dev_roll5__`: desvio em relacao a media curta
- `std_roll5__`: volatilidade local

Divisao temporal usada:

- treino: `15.031` linhas
- validacao: `3.221` linhas
- teste: `3.222` linhas

## Modelo

A arquitetura da `versao2` foi:

- `input_size = 24`
- `sequence_length = 60`
- projecao linear inicial para `128`
- `3` blocos convolucionais residuais
- `GRU` bidirecional com `hidden_size = 128` e `2` camadas
- pooling por atencao
- cabeca densa final para prever `6` variaveis

Total de parametros treinaveis:

- `891.815`

A previsao e residual: o modelo aprende uma correcao sobre o ultimo valor observado da janela, em vez de prever o proximo passo do zero.

## Treino e validacao

Resultado observado no notebook `3-treino-validacao.ipynb`:

- dispositivo: `cpu`
- janelas de treino: `14.971`
- janelas de validacao: `3.161`
- melhor epoch: `10`
- melhor `val_mae`: `0.007879`
- `val_mae` da persistencia: `0.020887`
- ganho sobre a persistencia na validacao: `62,28%`
- early stopping acionado na epoca `18`

Leitura curta:

- as curvas ficaram estaveis
- o modelo superou a persistencia cedo e manteve essa vantagem
- nao houve sinal forte de colapso ou overfitting severo

## Teste

No notebook `4-teste.ipynb`, o modelo foi avaliado em `3.162` janelas de teste.

### Metricas globais na escala padronizada

| Modelo | RMSE | MAE |
|---|---:|---:|
| `Modelo_v2` | `0.022600` | `0.007636` |
| `Persistencia` | `0.051535` | `0.020833` |

Ganhos globais da `versao2`:

- `55,28%` melhor em `RMSE`
- `65,01%` melhor em `MAE`

### Metricas globais na escala original

| Modelo | RMSE | MAE |
|---|---:|---:|
| `Modelo_v2` | `3761,92` | `914,72` |
| `Persistencia` | `8411,83` | `2614,48` |

## Leitura por feature

Melhores ganhos da `versao2`:

- `P-MON-CKP`
  - `MAE`: `5405,69` contra `15655,21` da persistencia
  - ganho de `65,47%`
- `T-JUS-CKP`
  - ganho de `73,06%` em `MAE`
- `T-TPT`
  - ganho de `18,77%` em `MAE`

Pontos ainda fracos:

- `P-ANULAR` ficou pior que a persistencia
- `P-TPT` melhorou pouco em `RMSE` e perdeu em `MAE`
- `P-JUS-CKGL` ficou praticamente constante no teste, então a persistencia teve erro zero e virou um baseline muito dificil de bater

## Interpretacao correta dos resultados

A `versao2` foi um avanço real em relacao a `versao1`. Desta vez, o modelo venceu a persistencia no teste quando olhamos os erros globais mais importantes.

Existe um detalhe importante: o `R2 medio` da persistencia ficou maior que o do modelo. Isso aconteceu porque:

- `MAE` e `RMSE` globais pesam mais as variaveis de grande escala
- `R2_medio` da o mesmo peso a cada feature

Na pratica, o modelo ganhou muito nas variaveis mais dinamicas e com maior impacto absoluto no erro total, mas ainda nao ficou melhor em todas as variaveis individualmente.

## Conclusao

Como experimento de arquivo unico, a `versao2` foi bem-sucedida. Ela mostrou que:

- enriquecer a entrada com dinamica local ajuda
- prever o delta sobre a persistencia funciona melhor do que a abordagem simples da `versao1`
- a arquitetura hibrida residual tem potencial real

O proximo passo faz sentido ser incremental:

1. ajustar melhor `P-ANULAR` e `P-TPT`
2. investigar tratamento especifico para variaveis quase constantes
3. depois disso, expandir a mesma arquitetura para muitos arquivos do 3W
