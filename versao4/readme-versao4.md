# README - Versao 4

## O que foi implementado

A versao 4 foi criada para responder lacunas importantes que ficaram abertas na versao 3. Em vez de apenas repetir o pipeline anterior, ela introduziu mudancas estruturais:

- comparacao direta entre uma `LSTM` mais classica e o modelo hibrido residual
- previsao multi-step com horizonte de 5 passos
- escalonamento robusto no pre-processamento
- perfil numerico explicito para investigar colunas com amplitude extrema
- criterio conservador para decidir se alguma coluna deveria usar transformacao logaritmica
- amostragem de janelas balanceada por classe no treino
- loss robusta com pesos por target
- avaliacao streaming detalhada por feature, classe, poco, serie e horizonte

Os notebooks tambem foram revisados para estudo: agora eles trazem mais comentarios no codigo e mais interpretacoes dos resultados observados na execucao real.

## Estrutura da versao 4

1. `1-visao-geral-dos-dados.ipynb`
   Rele a base com foco nos problemas que motivaram a versao 4: desbalanceamento, classes dificeis e escalas numericas extremas.
2. `2-pre-processamento.ipynb`
   Gera o manifesto da versao 4, seleciona variaveis auxiliares, ajusta o bundle robusto e salva as series processadas.
3. `3-treino-validacao.ipynb`
   Treina duas arquiteturas, compara a validacao contra a persistencia e salva o modelo campeao interno.
4. `4-teste.ipynb`
   Executa o teste final em modo streaming e gera relatorios detalhados.
5. `pipeline_v4.py`
   Centraliza a logica nova da versao 4, incluindo dataset multi-step, modelos, loss robusta e avaliacao detalhada.

## O que os notebooks mostraram nesta execucao

### 1. Visao geral dos dados

A leitura inicial confirmou que a base continua bastante desbalanceada:

- `2228` arquivos
- `10` classes
- `42` wells
- `3` tipos de origem

Distribuicao por classe:

- classe `0`: `594` arquivos
- classe `5`: `450`
- classe `4`: `343`
- classe `7`: `46`
- classe `2`: `38`

Distribuicao por origem:

- `well`: `1119`
- `simulated`: `1089`
- `drawn`: `20`

Leitura importante:

- a classe `7` e a classe `2` continuam raras em numero de arquivos
- a origem `drawn` e muito pequena, entao qualquer conclusao sobre ela precisa ser tratada com cautela
- o desbalanceamento por arquivo nao conta toda a historia, porque algumas classes raras têm series muito longas

### 2. Pre-processamento

A divisao final ficou em:

- treino: `1554` series
- validacao: `314`
- teste: `360`

O bundle da versao 4 ficou com:

- `36` features de entrada
- `6` targets
- `12` variaveis auxiliares
- `42` ids de poco
- horizonte recomendado de `5` passos
- `scaler_strategy = robust`

O perfil numerico confirmou a presenca de colunas com escala extrema, especialmente `P-PDG` e `T-PDG`. Mesmo assim, o criterio automatico nao recomendou transformacao logaritmica para nenhuma coluna. O motivo foi simples: varias dessas variaveis possuem muitos zeros ou sinais mistos, entao a heuristica preferiu nao aplicar log para nao distorcer a interpretacao.

No total, o pre-processamento gerou aproximadamente:

- treino: `54.520.863` linhas processadas
- validacao: `9.661.468`
- teste: `12.404.987`
- total geral: `76.587.318`

Leitura importante:

- a classe `7` continua rara em numero de arquivos, mas muito pesada em numero de linhas
- isso ajuda a explicar por que “poucos arquivos” nao significa necessariamente “poucas janelas”

### 3. Treino e validacao

A etapa de janelamento mostrou novamente o tamanho do problema:

- treino total possivel: `54.421.407` janelas
- validacao total possivel: `9.641.372` janelas

Por epoca, a versao 4 usou:

- treino ativo: `140.000` janelas
- validacao ativa: `50.000` janelas

Comparacao entre arquiteturas:

- `pure_lstm_forecaster_v4`
  - melhor epoch: `18`
  - `best_val_mae = 0.000239`
  - `best_val_persistence_mae = 0.000208`
  - `best_ratio_to_persistence = 1.149055`
  - `features_where_persistence_wins = 5`
- `hybrid_residual_forecaster_v4`
  - melhor epoch: `3`
  - `best_val_mae = 0.000298`
  - `best_val_persistence_mae = 0.000208`
  - `best_ratio_to_persistence = 1.431792`
  - `features_where_persistence_wins = 6`

Leitura principal da validacao:

- a `LSTM pura` foi melhor do que o modelo hibrido
- mas nenhum dos dois superou a persistencia na validacao amostrada
- a persistencia continuou fortissima, especialmente porque a tarefa ainda e de curto horizonte
- a melhoria da LSTM depois da reducao da taxa de aprendizado sugere que ela precisava de atualizacoes mais suaves para estabilizar

No melhor checkpoint da LSTM:

- `P-TPT` foi a unica feature que venceu a persistencia em MAE
- `P-MON-CKP` perdeu em MAE, mas ganhou em RMSE

Isso e uma boa licao pedagogica: `MAE` e `RMSE` nao contam exatamente a mesma historia.

### 4. Teste final

O modelo escolhido automaticamente foi:

- `pure_lstm_forecaster_v4`

O teste completo processou:

- `360` series
- `12.381.947` janelas
- `32.245` batches
- horizonte de `5` passos
- cerca de `61.909.735` previsoes no nivel linha a linha
- tempo total de aproximadamente `241,85` segundos
- memoria do processo em torno de `4,47 GB` nos logs

Esse ponto foi uma vitoria tecnica clara: o modo streaming continuou funcionando mesmo com previsao multi-step.

#### Resultado global do teste

Escala original:

- modelo: `MAE = 1938,90`
- persistencia: `MAE = 2236,45`
- melhora em MAE: `13,30%`
- melhora em RMSE: `19,01%`

Escala padronizada:

- modelo: `MAE = 0.000419`
- persistencia: `MAE = 0.000464`
- melhora em MAE: `9,71%`
- melhora em RMSE: `22,80%`

Essa foi a maior surpresa da versao 4:

- a validacao sugeria que a LSTM ainda perdia para a persistencia
- o teste completo mostrou ganho global real

## Como interpretar a divergencia entre validacao e teste

Essa divergencia nao significa necessariamente que a validacao “falhou”, mas indica que ela nao representou perfeitamente o criterio final de uso. As explicacoes mais provaveis sao:

- a validacao foi feita por amostragem, nao em streaming completo
- o criterio principal de selecao estava na escala padronizada
- o teste completo agregou a base inteira e nas unidades originais
- targets de grande escala, como `P-MON-CKP`, influenciam muito o MAE global

Em termos didaticos, esse e um excelente exemplo de um ponto central em Machine Learning:

- o melhor modelo segundo uma validacao interna pode nao ser o melhor modelo segundo a metrica operacional final

Isso nao invalida o experimento. Pelo contrario: mostra por que a forma de avaliacao importa tanto quanto a arquitetura.

## Onde o modelo realmente melhorou e onde ainda falhou

### Por feature

No teste em escala original:

- melhorou em `2` das `6` features no MAE
- piorou em `4` das `6` features no MAE

As maiores leituras foram:

- `P-MON-CKP`: melhora forte em MAE e RMSE
- `P-TPT`: melhora relevante
- `P-ANULAR`: piora forte em MAE
- `T-JUS-CKP` e `T-TPT`: piora em MAE, embora com RMSE ainda competitivo

Interpretacao:

- o ganho global veio principalmente de poucas features muito importantes e de grande escala
- portanto, a versao 4 nao resolveu o problema inteiro; ela resolveu melhor algumas partes do problema

### Por classe

No teste:

- `4` das `10` classes melhoraram no MAE
- `6` das `10` classes ainda ficaram piores do que a persistencia

Classes com ganho claro:

- classe `7`: melhora de `32,21%`
- classe `3`: melhora de `35,87%`
- classe `4`: melhora de `11,69%`
- classe `9`: melhora pequena, mas positiva

Classes que continuaram problematicas:

- classe `5`: `-138,92%`
- classe `1`: `-57,90%`
- classe `2`: `-44,17%`
- classe `0`: `-29,63%`

Leitura importante:

- uma classe pode continuar com erro absoluto alto e ainda assim mostrar ganho relativo sobre a persistencia
- por isso, “classe dificil” nao significa automaticamente “classe em que o modelo fracassou”

### Por poco e por serie

No teste:

- `14` dos `27` pocos melhoraram em MAE
- `78` das `360` series melhoraram em MAE

Isso mostra que a melhora global ficou concentrada. O modelo venceu de forma forte em alguns subsets, mas nao generalizou esse ganho para toda a base.

Exemplos:

- `WELL-00024` teve melhora forte
- varios trechos de `WELL-00001` continuaram piores do que a persistencia
- a serie `9__WELL-00014_20170214190000` apareceu como a mais dificil em MAE

Tambem surgiram casos de `R²` muito estranho, inclusive extremamente negativo, em alguns pocos como `WELL-00035`. Isso deve ser interpretado com cuidado:

- em subsets com variancia muito baixa ou comportamento quase constante, `R²` pode ficar instavel
- nesses casos, `MAE` e `RMSE` sao referencias mais confiaveis

### Por horizonte

Todos os 5 passos do horizonte venceram a persistencia no teste:

- passo 1: melhora de `10,03%` em MAE
- passo 2: `16,07%`
- passo 3: `15,56%`
- passo 4: `13,44%`
- passo 5: `11,39%`

Interpretacao:

- o erro cresce com o horizonte, como esperado
- mas a vantagem sobre a persistencia foi mantida em todos os passos

Esse resultado e importante porque mostra que a versao 4 nao ganhou apenas no primeiro passo. O ganho persistiu ate o fim do horizonte de previsao considerado.

## Leitura geral da versao 4

A versao 4 foi um experimento mais rico e mais honesto do que a versao 3. Ela nao entrega uma narrativa linear do tipo “mudamos tudo e melhorou em tudo”. O que ela mostra e algo mais realista:

- a `LSTM pura` foi a melhor arquitetura testada nesta configuracao
- a persistencia ainda continuou muito forte na validacao
- o teste completo revelou um ganho global real
- esse ganho nao se distribuiu uniformemente entre features, classes, pocos e series

Portanto, a versao 4 nao fecha o problema, mas avanca em duas frentes importantes ao mesmo tempo:

1. melhora a qualidade de avaliacao do projeto
2. mostra que ha sinal preditivo util alem da persistencia, especialmente em algumas variaveis e subsets

## Sugestoes para a proxima versao

### 1. Fazer validacao streaming completa

A maior licao da versao 4 foi a divergencia entre validacao amostrada e teste completo. A versao 5 deveria incluir validacao streaming completa para reduzir esse desencontro.

### 2. Usar criterio de selecao mais proximo do objetivo final

Hoje o modelo e escolhido pela razao entre MAE do modelo e da persistencia na validacao amostrada. Para a proxima versao, faria sentido testar:

- criterio combinado entre escala padronizada e escala original
- criterio por horizonte
- criterio que penalize features onde a persistencia continua vencendo

### 3. Refinar o tratamento de colunas extremas

O perfil numerico mostrou que o problema de escala continua real. Como o log automatico nao foi ativado, a proxima versao pode investigar:

- transformacoes assinadas mais elaboradas
- escalonadores diferentes por grupo de variaveis
- tratamento especifico para variaveis muito esparsas ou com muitos zeros

### 4. Atacar explicitamente as fatias em que o modelo ainda perde

Como o relatorio agora mostra por classe, poco e serie onde o modelo falha, a versao 5 pode usar isso de forma ativa:

- amostragem orientada por classes dificeis
- analise focada em pocos problematicos
- modelos ou losses adaptados para subsets especificos

### 5. Tornar a comparacao ainda mais didatica

Ja que a LSTM pura venceu o modelo hibrido nesta configuracao, uma proxima etapa interessante seria comparar:

- LSTM direta vs residual
- LSTM unidirecional vs bidirecional
- previsao de 1 passo vs 5 passos vs 10 passos

## Resumo final

A versao 4 foi bem-sucedida em tres sentidos:

- evoluiu tecnicamente o pipeline
- manteve o teste viavel em memoria mesmo com horizonte multi-step
- transformou os notebooks em material melhor de estudo, porque agora eles mostram nao so o que foi feito, mas tambem o que os resultados significam

O principal aprendizado da execucao foi este:

- a validacao sugeriu cautela
- o teste mostrou ganho global real
- o ganho existe, mas e localizado e desigual

Isso faz da versao 4 uma etapa muito boa para estudo serio do problema e uma base forte para uma futura versao 5.
