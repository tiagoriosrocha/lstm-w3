# README - Versao 3

## O que foi feito nesta versao

A versao 3 organiza o projeto em quatro notebooks encadeados:

1. `1-visao-geral-dos-dados.ipynb`
   Analisa a estrutura do dataset 3W, a distribuicao das classes, o tipo de origem das series e o tamanho temporal de cada arquivo.
2. `2-pre-processamento.ipynb`
   Divide a base em treino, validacao e teste, seleciona variaveis auxiliares, ajusta normalizacao e gera features temporais derivadas.
3. `3-treino-validacao.ipynb`
   Monta janelas temporais, treina o modelo da versao 3 e escolhe o melhor checkpoint com base na comparacao contra a baseline de persistencia.
4. `4-teste.ipynb`
   Executa a inferencia no conjunto de teste em modo streaming, calcula metricas globais e detalhadas e gera visualizacoes.

Tambem foi feita uma revisao didatica dos notebooks para transformá-los em material de estudo:

- comentarios no codigo no padrao `#comentario`
- explicacoes entre blocos para conectar uma etapa a outra
- interpretacao dos principais resultados salvos nas execucoes
- contextualizacao acessivel para iniciantes em series temporais e modelos recorrentes

## Como era o modelo da versao 3

Apesar de o projeto mencionar LSTM como contexto de estudo, a arquitetura usada na versao 3 nao e uma LSTM pura. O modelo salvo como `hybrid_residual_forecaster_v3` combina:

- projecao inicial das features de entrada
- blocos convolucionais residuais para capturar padroes locais
- GRU bidirecional para modelar dependencia temporal
- mecanismo de atencao para pesar melhor os passos mais relevantes da sequencia
- embedding do poço para fornecer contexto da origem da serie
- cabeca residual que aprende um delta em relacao ao ultimo valor observado

Em termos de entrada e saida:

- entrada: janelas com 60 passos de tempo e 36 features por passo
- saida: previsao do proximo passo para 6 variaveis-alvo
- quantidade de parametros treinaveis: 894.031

## Principais resultados observados

### 1.Analise exploratoria dos dados

A leitura inicial da base mostrou:

- 2228 arquivos ao todo
- 10 classes distintas
- 42 wells distintos
- 3 tipos de origem: dados de poço real, dados simulados e dados desenhados

Tambem ficou claro que a base e desbalanceada. Exemplos:

- classe `0`: 594 arquivos
- classe `5`: 450 arquivos
- classe `2`: apenas 38 arquivos

Esse desbalanceamento ja sugere que algumas classes serao naturalmente mais faceis de aprender do que outras.

### 2.Pre-processamento

A divisao gerada foi:

- treino: 1554 series
- validacao: 314 series
- teste: 360 series

O bundle final do pre-processamento ficou com:

- 36 features de entrada
- 6 variaveis-alvo
- 12 variaveis auxiliares selecionadas
- 42 identificadores de poço

Durante o pre-processamento apareceram warnings de `overflow encountered in cast`. Eles indicam que algumas colunas brutas apresentam escala numerica muito alta. Como o pipeline aplica clipping e normalizacao, o fluxo continuou funcional, mas esse sinal merece investigacao futura.

### 3.Treino e validacao

A etapa de janelamento mostrou um ponto importante de escala:

- treino: 54.427.623 janelas possiveis
- validacao: 9.642.628 janelas possiveis

Para tornar o treino viavel por epoca, a versao 3 amostrou:

- 120.000 janelas de treino
- 40.000 janelas de validacao

No melhor checkpoint da validacao, os resultados foram:

- `best_epoch`: 15
- `best_val_mae`: 0.00011215
- `best_val_persistence_mae`: 0.00012156
- `best_ratio_to_persistence`: 0.9226

Interpretacao:

- a persistencia foi uma baseline forte durante quase todo o treino
- o modelo conseguiu supera-la no melhor momento
- a melhora foi modesta, de aproximadamente 7,74%
- houve oscilacao entre epocas, sugerindo um problema dificil e sensivel a amostragem

### 4.Teste final

No conjunto de teste, a execucao completa processou:

- 360 series
- 12.383.387 janelas
- 24.187 batches
- tempo total de cerca de 339,62 segundos
- uso de memoria do processo em torno de 4,69 GB nos logs da inferencia streaming

Isso mostra que a abordagem streaming foi fundamental para tornar o teste viavel.

Metricas globais na escala original:

- modelo: `MAE = 703,94`, `RMSE = 4595,72`
- persistencia: `MAE = 1037,84`, `RMSE = 7137,56`
- melhora do modelo em MAE: aproximadamente 32,17%

Leituras importantes do teste:

- o ganho global foi bom
- o R² ficou muito alto para modelo e baseline, o que era esperado em series suaves de curto prazo
- nem todas as variaveis melhoraram igualmente
- `P-MON-CKP` teve melhora forte
- `P-ANULAR`, `P-TPT` e `P-JUS-CKGL` continuaram dificeis em comparacao com a persistencia em algumas metricas
- classes `6`, `7` e `9` apresentaram erros absolutos mais altos, sugerindo dinamica mais dificil ou maior amplitude dos sinais

## Leitura geral da qualidade do modelo

A versao 3 nao e um fracasso, mas tambem nao fecha o problema por completo. Ela parece estar em uma zona intermediaria interessante:

- aprende padroes reais da base
- supera a persistencia no agregado
- ainda enfrenta dificuldade em variaveis muito estaveis ou em classes mais desafiadoras
- se beneficia muito de uma implementacao de teste eficiente em memoria

Em outras palavras, a versao 3 e uma boa base experimental e pedagogica, mas ainda nao parece ser a forma final do projeto.

## Sugestoes de melhoria para a versao 4

### 1.Comparar com uma LSTM pura

Como o projeto tem foco educacional em LSTM, faz sentido incluir na versao 4 uma baseline com LSTM pura, treinada nas mesmas janelas e com as mesmas features. Isso ajudaria a responder se a arquitetura hibrida realmente entrega vantagem sobre uma solucao mais classica.

### 2.Trabalhar melhor as variaveis com escala extrema

Os warnings de overflow sugerem revisar colunas com amplitude muito alta, como algumas variaveis auxiliares analogicas. Boas direcoes:

- investigar outliers por coluna
- experimentar transformacoes logaritmicas quando fizer sentido fisico
- testar escalonadores mais robustos para certas features

### 3.Treinar com objetivo mais alinhado ao teste

Hoje o modelo ganha no agregado, mas perde para a persistencia em algumas variaveis. A versao 4 pode explorar:

- pesos diferentes por target na loss
- losses robustas por variavel
- monitoramento de metricas por target durante o treino

### 4.Analisar melhor classes mais dificeis

As classes `6`, `7` e `9` merecem atencao especial. Ideias:

- analise de erro por classe e por poço
- graficos de exemplos tipicos de falha
- amostragem mais balanceada de janelas por classe

### 5.Incluir previsao multi-step

A versao 3 faz previsao do proximo passo. A versao 4 pode testar horizontes maiores, por exemplo prever 5 ou 10 passos a frente. Isso torna o problema mais realista para apoio operacional e tambem mais desafiador do ponto de vista de series temporais.

### 6.Refinar a avaliacao

Seria util ampliar a parte de relatorio com:

- tabelas automáticas por classe e por variavel
- exportacao de erros por serie
- comparacao entre treino, validacao e teste em um unico resumo
- destaque automatico das features em que a persistencia ainda vence

### 7.Explorar eficiencia adicional

A inferencia streaming ja resolveu o gargalo principal de memoria. Ainda assim, a versao 4 pode testar:

- `num_workers` maior no `DataLoader`
- `pin_memory` e perfis de transferencia CPU-GPU
- exportacao opcional de previsoes por classe ou por serie
- mixed precision no teste e no treino, se a estabilidade permitir

## Resumo final

A versao 3 e uma etapa madura do projeto porque entrega ao mesmo tempo:

- pipeline completo de dados ate teste final
- modelo recorrente/temporal funcional
- comparacao seria contra baseline de persistencia
- inferencia escalavel para mais de 12 milhoes de janelas
- notebooks agora comentados como material de estudo

A versao 4 deve focar menos em “fazer funcionar” e mais em “fazer entender melhor e melhorar onde ainda doi”: targets em que a persistencia vence, classes mais duras, tratamento de escalas extremas e comparacao direta com uma LSTM classica.
