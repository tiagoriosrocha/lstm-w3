# Projeto LSTM com Dataset 3W

## Visao geral

Este projeto mostra um pipeline completo de series temporais com o dataset 3W da Petrobras:

1. entender os dados
2. limpar e preparar as variaveis
3. treinar uma rede LSTM
4. avaliar o desempenho no conjunto de teste

O foco aqui nao foi apenas "rodar um modelo", mas construir uma esteira didatica e reprodutivel, separando cada etapa em um notebook diferente.

## Estrutura dos notebooks

- `1-visao-geral-dos-dados.ipynb`
  - faz a leitura da instancia escolhida do dataset
  - mostra `head`, `info`, estatisticas, nulos e visualizacoes
- `2-pre-processamento.ipynb`
  - remove colunas inviaveis
  - separa sinais analogicos e sinais de estado
  - trata faltantes
  - divide em treino, validacao e teste
  - gera dados escalados e salva artefatos
- `3-treino-validacao.ipynb`
  - monta sequencias temporais
  - cria a LSTM
  - treina com early stopping
  - salva o melhor checkpoint
- `4-teste.ipynb`
  - carrega o melhor modelo salvo
  - avalia no conjunto de teste
  - calcula metricas globais e por feature

## Dataset utilizado

Foi usada uma instancia local do clone do repositorio oficial `3W`, no arquivo:

`3W/dataset/0/WELL-00001_20170201010207.parquet`

Resumo da instancia analisada:

- 21.474 observacoes
- 29 colunas na leitura bruta
- periodo: `2017-02-01 01:02:07` ate `2017-02-01 07:00:00`

## Como os dados chegam

No dado bruto, existem sensores analogicos, variaveis de estado e labels. Um exemplo real da primeira linha lida foi:

```text
timestamp    = 2017-02-01 01:02:07
ESTADO-DHSV  = 1.0
P-ANULAR     = 12767730.0
P-JUS-CKGL   = 1563422.0
P-MON-CKP    = 1627884.0
P-PDG        = 0.0
P-TPT        = 10074540.0
QGL          = 0.0
T-JUS-CKP    = 84.64463
T-PDG        = 0.0
T-TPT        = 119.0781
class        = NaN
state        = NaN
```

Esse exemplo ja mostra algo importante: o dataset nao chega "pronto para modelagem". Ha colunas vazias, colunas constantes e labels ausentes em parte da serie.

## O que a exploracao mostrou

Durante a EDA, apareceram alguns pontos centrais:

- 9 colunas estavam 100% nulas:
  - `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`
- `class` e `state` tinham 3.600 valores ausentes cada, cerca de `16,76%`
- varias colunas eram constantes ou quase triviais nessa instancia
- algumas variaveis como `P-PDG`, `QGL` e `T-PDG` nao traziam variacao util para esse experimento

Em outras palavras: o primeiro ganho do projeto veio muito mais da limpeza correta dos dados do que de qualquer truque na rede.

## Como foi feito o pre-processamento

O notebook `2-pre-processamento.ipynb` aplicou a seguinte logica:

1. remover colunas 100% nulas
2. remover labels e colunas auxiliares das features
3. separar sinais de estado/binarios de sinais analogicos
4. interpolar apenas os sinais analogicos
5. usar `ffill` e `bfill` para completar faltantes residuais
6. remover colunas constantes apos a limpeza
7. dividir a serie temporal sem embaralhar
8. gerar tres versoes escaladas:
   - `MinMaxScaler`
   - `StandardScaler`
   - `RobustScaler`

### Features finais

Depois da limpeza, restaram apenas 6 features realmente usadas no modelo:

- `P-ANULAR`
- `P-JUS-CKGL`
- `P-MON-CKP`
- `P-TPT`
- `T-JUS-CKP`
- `T-TPT`

Resultado da limpeza:

- shape final: `(21474, 6)`
- valores nulos restantes: `0`

### Exemplo didatico de transformacao

A mesma primeira linha, depois da limpeza, ficou assim:

```text
P-ANULAR    = 12767730.0
P-JUS-CKGL  = 1563422.0
P-MON-CKP   = 1627884.0
P-TPT       = 10074540.0
T-JUS-CKP   = 84.64463
T-TPT       = 119.0781
```

Depois do `StandardScaler`, essa observacao passou a ser representada por valores normalizados:

```text
P-ANULAR    =  1.997102
P-JUS-CKGL  = -1.731634
P-MON-CKP   =  0.273858
P-TPT       =  0.495554
T-JUS-CKP   =  0.771591
T-TPT       =  0.714380
```

Isso e importante porque a LSTM treina melhor quando as variaveis estao em uma escala mais comparavel.

## Divisao temporal

A divisao respeitou a ordem cronologica:

- treino: `15.031` linhas
- validacao: `3.221` linhas
- teste: `3.222` linhas

Sem embaralhamento. Em serie temporal isso e obrigatorio para evitar vazamento de informacao do futuro para o passado.

## Como a rede foi montada

A rede usada foi uma `LSTMForecaster` com:

- entrada com `6` features por passo temporal
- `sequence_length = 30`
- `hidden_size = 64`
- `num_layers = 2`
- `dropout = 0.2`
- camada final `Linear(64 -> 6)`

Quantidade total de parametros:

- `52.102` parametros treinaveis

### Formato dos tensores

Depois da montagem das janelas temporais:

- `X_train = (15001, 30, 6)`
- `y_train = (15001, 6)`
- `X_val   = (3191, 30, 6)`
- `y_val   = (3191, 6)`
- `X_test  = (3192, 30, 6)`
- `y_test  = (3192, 6)`

## Como a informacao passa pela rede

Uma maneira didatica de pensar a rede e esta:

```text
Janela de 30 passos x 6 features
        |
        v
LSTM camada 1
        |
        v
LSTM camada 2
        |
        v
Ultimo estado oculto da janela (vetor de tamanho 64)
        |
        v
Camada linear 64 -> 6
        |
        v
Predicao da proxima observacao multivariada
```

Ou seja:

- a entrada e uma sequencia dos ultimos 30 instantes
- a LSTM resume essa janela em um estado oculto
- a camada linear transforma esse resumo em uma previsao para o proximo instante

## Exemplo didatico de valor esperado vs valor predito

No conjunto de teste, para a primeira janela avaliada, o proximo timestamp previsto foi:

`2017-02-01 06:06:49`

Os valores abaixo estao na escala padronizada (`StandardScaler`), nao nas unidades fisicas originais.

| Feature | Esperado | Predito | Erro absoluto |
|---|---:|---:|---:|
| `P-ANULAR` | -1.144632 | -1.778522 | 0.633890 |
| `P-JUS-CKGL` | 2.481718 | 2.107269 | 0.374449 |
| `P-MON-CKP` | 0.215236 | 0.258580 | 0.043344 |
| `P-TPT` | 0.422314 | 0.174464 | 0.247850 |
| `T-JUS-CKP` | -0.484902 | -0.616885 | 0.131983 |
| `T-TPT` | -0.416548 | 0.018804 | 0.435353 |

Interpretacao:

- em algumas variaveis o modelo chegou relativamente perto, como `P-MON-CKP`
- em outras, o erro foi maior, como `P-ANULAR` e `T-TPT`
- isso mostra que a rede aprendeu parte da dinamica, mas ainda nao modela todas as variacoes com a mesma qualidade

## Treino e validacao

O treino foi configurado com:

- `MAX_EPOCHS = 50`
- `PATIENCE = 7`
- `early stopping`

Resultado observado:

- melhor epoch de validacao: `23`
- melhor `val_loss`: `0.066476`
- epocas executadas: `30`

Isso significa que o treinamento nao ficou preso a um numero fixo de epocas. O modelo foi salvo no melhor ponto de validacao e o treino foi interrompido quando a validacao deixou de melhorar por varias rodadas.

## Resultados no teste

Metricas globais no conjunto de teste:

| Metrica | Valor |
|---|---:|
| `MSE`  | 0.428819 |
| `RMSE` | 0.654842 |
| `MAE`  | 0.475725 |

### Erro por feature

As features mais dificeis e mais faceis para o modelo ficaram assim:

| Feature | MAE | RMSE |
|---|---:|---:|
| `P-JUS-CKGL` | 1.215497 | 1.313882 |
| `P-ANULAR`   | 0.577404 | 0.594144 |
| `P-TPT`      | 0.458250 | 0.531194 |
| `T-TPT`      | 0.331179 | 0.393002 |
| `T-JUS-CKP`  | 0.148393 | 0.179450 |
| `P-MON-CKP`  | 0.123623 | 0.157481 |

## Analise dos resultados

### O que foi bom

- o pipeline ficou limpo e bem separado em etapas
- o pre-processamento eliminou ruido estrutural importante
- a validacao melhorou ao longo do treino
- o early stopping capturou um checkpoint melhor do que simplesmente usar o ultimo epoch
- a rede conseguiu aprender alguma estrutura temporal real do problema

### O que ainda esta fraco

- o erro no teste ainda e relativamente alto para algumas variaveis
- `P-JUS-CKGL` foi claramente a feature mais dificil de prever
- o `MAE` global de `0.4757` em escala padronizada mostra que o baseline ainda esta longe de uma previsao "muito precisa"
- o gap entre treino e validacao foi classificado no notebook como moderado, ou seja: o modelo melhorou, mas ainda ha espaco para refinamento

### Leitura pedagogica

Este projeto mostra um ponto essencial de data science:

> um modelo razoavel em dados bem preparados costuma ensinar mais do que um modelo sofisticado em dados mal tratados

Aqui, a maior melhora veio de:

- remover colunas totalmente vazias
- remover colunas constantes
- separar variaveis analogicas de sinais de estado
- escalar corretamente os dados
- avaliar com divisao temporal correta

So depois disso a LSTM passou a fazer sentido.

## Proximos passos recomendados

Se a ideia for evoluir este baseline, os proximos testes mais promissores sao:

1. inverter o scaler para analisar erros nas unidades fisicas reais
2. testar outros comprimentos de janela, como `15`, `60` ou `120`
3. experimentar outro alvo:
   - prever apenas uma variavel especifica
   - prever varios passos a frente
4. comparar com baselines simples:
   - persistencia
   - regressao linear
   - MLP
5. ajustar melhor a arquitetura:
   - mais ou menos camadas
   - hidden size diferente
   - dropout diferente
6. trabalhar com mais instancias do dataset, e nao apenas um unico arquivo

## Conclusao

Os 4 notebooks constroem uma historia completa:

- primeiro entendemos o dado
- depois limpamos e organizamos o problema
- em seguida treinamos uma LSTM de forma correta
- por fim avaliamos o que ela realmente conseguiu aprender

Como baseline didatico, o projeto foi bem-sucedido: ele mostra claramente a passagem do dado bruto para a previsao temporal. Como solucao final de negocio, ainda ha espaco para melhorar bastante, principalmente na precisao de algumas features e na interpretacao dos erros em unidades reais.
