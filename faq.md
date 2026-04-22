# FAQ - Machine Learning, Deep Learning e Entendimento do Projeto

Este arquivo foi escrito como material de revisao sobre os conceitos de `machine learning`, `deep learning`, series temporais, metricas de avaliacao e a logica do projeto baseado no dataset `3W` e será utilizado para estudar os conceitos da disciplina.

## 1. Fundamentos de Machine Learning

**1. O que e machine learning?**  
Machine learning e uma area da inteligencia artificial em que um modelo aprende padroes a partir de dados, em vez de receber todas as regras manualmente.

**2. O que e deep learning?**  
Deep learning e uma subarea de machine learning baseada em redes neurais com varias camadas, capazes de aprender representacoes mais complexas dos dados.

**3. Qual a diferenca entre machine learning e deep learning?**  
Todo deep learning e machine learning, mas nem todo machine learning e deep learning. Modelos como `RandomForest` e `XGBoost` sao de machine learning classico; `LSTM` e deep learning.

**4. O que e aprendizado supervisionado?**  
E o tipo de aprendizado em que o modelo recebe entradas e tambem os rotulos corretos durante o treino.

**5. O que e rotulo?**  
Rotulo e a resposta correta que o modelo deve aprender a prever. No projeto, o rotulo e a classe `0` a `9`.

**6. O que e uma amostra?**  
E uma unidade de dado usada pelo modelo. Nas versoes `6` e `7`, cada arquivo `parquet` e tratado como uma amostra.

**7. O que e feature ou atributo?**  
E uma variavel de entrada usada pelo modelo para aprender padroes.

**8. O que e target?**  
E a variavel que o modelo tenta prever. Em classificacao, o target e a classe; em regressao, e um valor numerico.

**9. O que e regressao?**  
E um problema em que o modelo precisa prever um valor continuo, como temperatura, pressao ou proximo valor de uma serie.

**10. O que e classificacao?**  
E um problema em que o modelo precisa escolher uma classe entre categorias predefinidas.

**11. O que e classificacao multiclasse?**  
E a classificacao em que existem mais de duas classes possiveis. Neste projeto, ha `10` classes.

**12. Qual a diferenca entre regressao e classificacao no contexto deste projeto?**  
Nas versoes anteriores, o foco principal era prever o proximo valor de variaveis do processo. Nas versoes `6` em diante, o foco passou a ser classificar cada serie completa em uma classe `0` a `9`.

**13. O que e baseline?**  
Baseline e um modelo de referencia usado para comparar desempenho. Ele ajuda a responder se uma solucao mais complexa realmente trouxe ganho.

**14. O que e overfitting?**  
E quando o modelo aprende muito bem os dados de treino, mas perde capacidade de generalizar para novos dados.

**15. O que e generalizacao?**  
E a capacidade do modelo de ter bom desempenho em dados que nao foram usados para treina-lo.

**16. O que e underfitting?**  
E quando o modelo nao aprende bem nem os dados de treino, geralmente por ser simples demais ou mal ajustado.

**17. O que e inferencia?**  
E a fase em que um modelo ja treinado recebe novos dados e produz previsoes.

**18. O que e pipeline em machine learning?**  
E a sequencia organizada de etapas, como leitura dos dados, limpeza, pre-processamento, treino, avaliacao e exportacao de resultados.

## 2. Series Temporais

**19. O que e uma serie temporal?**  
E uma sequencia de observacoes ordenadas no tempo.

**20. Por que a ordem temporal importa?**  
Porque o significado de um valor pode depender do que aconteceu antes e depois dele.

**21. O que e uma janela temporal?**  
E um trecho da serie usado como unidade de entrada para o modelo.

**22. O que significa sequence-to-one?**  
Significa receber uma sequencia completa como entrada e devolver uma unica saida. Nas versoes `6` e `7`, a saida e a classe da serie.

**23. O que significa sequence-to-sequence?**  
Significa receber uma sequencia e prever outra sequencia, ou varios passos temporais de saida.

**24. O que e reamostragem temporal?**  
E o processo de converter series com comprimentos diferentes para um mesmo comprimento fixo.

**25. Por que foi necessario reamostrar as series?**  
Porque a `LSTM` precisa receber tensores com formato padronizado, por exemplo `(N, T, F)`, onde `T` precisa ser o mesmo para todas as amostras.

**26. Qual foi o comprimento fixo usado nas versoes 6 e 7?**  
Foi usado `sequence_length = 120`.

**27. O que e interpolacao no contexto de series temporais?**  
E uma forma de estimar valores intermediarios ao ajustar a serie para um novo conjunto de posicoes temporais.

**28. O que e um sinal continuo?**  
E uma variavel numerica que pode assumir varios valores, como pressao, temperatura ou vazao.

**29. O que e um sinal discreto de estado?**  
E um sinal que representa estados operacionais, como aberto/fechado ou ligado/desligado.

**30. O que e tendencia em uma serie?**  
E a direcao geral de crescimento, queda ou estabilidade ao longo do tempo.

**31. O que e variabilidade em uma serie?**  
E o quanto os valores oscilam ao longo do tempo.

## 3. Dataset 3W e Estrutura do Projeto

**32. O que e o dataset 3W?**  
E um conjunto de dados industriais voltado para diagnostico de anomalias em sistemas de elevacao artificial de petroleo.

**33. O que significa tratar cada arquivo `parquet` como uma amostra?**  
Significa que a unidade de classificacao nao e cada linha do arquivo, mas a serie completa contida no arquivo.

**34. Quais sao as classes do problema nas versoes 6 e 7?**  
As classes sao `0` a `9`.

**35. O que significa a classe 0?**  
A classe `0` representa `Normal Operation`.

**36. O que significam as demais classes?**  
`1`: `Abrupt Increase of BSW`  
`2`: `Spurious Closure of DHSV`  
`3`: `Severe Slugging`  
`4`: `Flow Instability`  
`5`: `Rapid Productivity Loss`  
`6`: `Quick Restriction in PCK`  
`7`: `Scaling in PCK`  
`8`: `Hydrate in Production Line`  
`9`: `Hydrate in Service Line`

**37. Por que o projeto foi versionado em varias pastas?**  
Para preservar a evolucao do raciocinio experimental sem sobrescrever abordagens anteriores.

**38. O que as versões 1 a 5 estudaram?**  
Uma comparacao entre modelos de previsao e um modelo de persistencia, com foco mais academico.

**39. O que a versao 6 estudou?**  
A `versao6` reformulou o problema para classificacao multiclasse de series temporais e comparou `LSTM`, `RandomForest` e `XGBoost`.

**40. O que a versao 7 estudou?**  
A `versao7` testou uma `LSTM` mais profunda para verificar se maior capacidade recorrente reduziria o gap para as baselines tabulares.

**41. Qual foi a principal conclusao atual do projeto?**  
O melhor modelo do projeto continua sendo a baseline tabular com `RandomForest`. Entre as redes recorrentes, a melhor continua sendo a `versao10`. A `versao11` funcionou como uma ablacao de pre-processamento e, na rodada exploratoria documentada, nao superou a `versao10`.

## 4. Pre-processamento e Representacao dos Dados

**42. O que e pre-processamento?**  
E o conjunto de etapas que transforma os dados brutos em entradas adequadas para os modelos.

**43. Quais foram as principais etapas de pre-processamento nas versoes 6 e 7?**  
Descoberta dos arquivos, limpeza basica, `split` estratificado, selecao de colunas, reamostragem para comprimento fixo, padronizacao e geracao de `X_seq` e `X_tab`.

**44. O que e split de treino, validacao e teste?**  
E a separacao dos dados em tres grupos: um para aprender, um para ajustar o experimento e um para medir desempenho final.

**45. Para que serve o conjunto de treino?**  
Serve para ajustar os parametros do modelo.

**46. Para que serve o conjunto de validacao?**  
Serve para comparar epocas, escolher hiperparametros e monitorar generalizacao durante o desenvolvimento.

**47. Para que serve o conjunto de teste?**  
Serve para a avaliacao final, simulando dados nao vistos.

**48. O que e split estratificado?**  
E uma divisao em que se tenta manter a proporcao das classes nos conjuntos de treino, validacao e teste.

**49. Por que o split estratificado e importante aqui?**  
Porque o problema e desbalanceado, e um split aleatorio simples poderia piorar a representatividade de classes raras.

**50. O que e desbalanceamento de classes?**  
E quando algumas classes tem muito mais exemplos do que outras.

**51. Por que o desbalanceamento pode ser um problema?**  
Porque o modelo pode aprender a favorecer classes mais frequentes e aparentar bom desempenho sem realmente aprender as classes raras.

**52. O que e padronizacao?**  
E a transformacao dos dados para uma escala comum, normalmente com media proxima de zero e desvio-padrao proximo de um.

**53. Qual transformacao foi usada para padronizar os sinais continuos?**  
Foi usado `StandardScaler`.

**54. O que e `X_seq`?**  
E a representacao sequencial usada pela `LSTM`, com formato aproximado `(numero_de_amostras, 120, numero_de_features)`.

**55. O que e `X_tab`?**  
E a representacao tabular usada pelas baselines classicas, baseada em estatisticas agregadas da serie.

**56. Quantas colunas foram selecionadas para modelagem nas versoes 6 e 7?**  
Foram usadas `18` colunas.

**57. Como essas 18 colunas se dividem?**  
Em `9` colunas de estado discreto e `9` colunas continuas.

**58. Que tipos de sinais entram no modelo?**  
Entram principalmente sinais de pressao, temperatura, vazao e estados discretos de componentes e valvulas.

**59. Quais exemplos de colunas usadas nas versoes 6 e 7?**  
Exemplos de sinais continuos: `P-ANULAR`, `P-TPT`, `QGL`, `T-TPT`, `T-PDG`. Exemplos de estados discretos: `ESTADO-DHSV`, `ESTADO-M1`, `ESTADO-W1`, `ESTADO-W2`, `ESTADO-XO`.

**60. Por que combinar sinais continuos e estados discretos?**  
Porque o comportamento do sistema depende tanto da evolucao fisica das grandezas analogicas quanto da configuracao operacional dos componentes.

**61. Quais estatisticas foram usadas para construir `X_tab`?**  
`mean`, `std`, `min`, `max`, `median`, `first`, `last`, `slope` e `mean_abs_diff`.

### 🔹 Nível (valores da série)
- **mean**: média dos valores (valor típico)
- **median**: valor central (robusto a outliers)
- **min**: menor valor da janela
- **max**: maior valor da janela
- **first**: valor inicial da sequência
- **last**: valor mais recente (muito importante para previsão)

### 🔹 Variabilidade
- **std**: desvio padrão (mede variação dos dados)
- **mean_abs_diff**: média de |xₜ - xₜ₋₁| (mede “agitação” da série)

### 🔹 Tendência
- **slope**: inclinação da série ao longo do tempo  
  - positivo → crescente 
  - negativo → decrescente 
  - próximo de zero → estável

**62. Quantos atributos tabulares finais foram gerados?**  
Foram `162` atributos, porque ha `18` colunas e `9` estatisticas por coluna.

**63. Por que `X_tab` pode ser tao forte?**  
Porque ele condensa em um vetor informacoes de nivel, variabilidade, tendencia, inicio e fim da serie, o que pode ser muito discriminativo para classificacao.

**64. O que e engenharia de atributos?**  
E o processo de criar representacoes informativas a partir dos dados originais para facilitar o aprendizado do modelo.

**65. O que e vazamento de dados ou data leakage?**  
E quando informacoes do conjunto de validacao ou teste entram indevidamente no treino.

**66. Como evitar leakage?**  
Fazendo o `split` antes de ajustar transformacoes dependentes dos dados e usando treino, validacao e teste de forma separada.

## 5. Metricas de Avaliacao

**67. O que e accuracy?**  
E a proporcao total de previsoes corretas.

**68. Por que accuracy sozinha pode enganar?**  
Porque em dados desbalanceados ela pode parecer alta mesmo quando o modelo vai mal em classes raras.

**69. O que e precision?**  
E a fração das previsoes positivas de uma classe que realmente pertencem a essa classe.

**70. O que e recall?**  
E a fração dos exemplos reais de uma classe que o modelo conseguiu recuperar.

**71. O que e F1-score?**  
E a media harmonica entre precision e recall.

**72. O que e macro-F1?**  
E a media do F1 calculada igualmente entre as classes, dando o mesmo peso para classes grandes e pequenas.

**73. O que e balanced accuracy?**  
E a media dos recalls por classe. Ela mostra se o modelo esta equilibrado entre as classes.

**74. O que e support?**  
E o numero de exemplos reais de cada classe no conjunto avaliado.

**75. O que e matriz de confusao?**  
E uma tabela que mostra, para cada classe real, em que classes o modelo previu.

**76. Como interpretar uma matriz de confusao?**  
A diagonal principal mostra os acertos. Os valores fora da diagonal mostram as confusoes.

**77. O que significa uma matriz de confusao normalizada?**  
Significa que os valores foram convertidos em proporcoes, o que facilita comparar classes com tamanhos diferentes.

**78. Por que o projeto destaca `macro-F1` e `balanced accuracy`?**  
Porque o dataset e desbalanceado e essas metricas ajudam a avaliar justica entre classes.

**79. O que e MAE?**  
`Mean Absolute Error` e a media dos erros absolutos. Foi importante nas versoes de regressao do projeto.

**80. O que e RMSE?**  
`Root Mean Squared Error` e a raiz da media dos erros ao quadrado. Ele penaliza mais fortemente erros grandes.

**81. Qual a diferenca entre MAE e RMSE?**  
O `MAE` e mais robusto a erros extremos; o `RMSE` da mais peso a erros grandes.

**82. O que significa dizer que um modelo teve melhor validacao do que teste?**  
Pode indicar leve queda de generalizacao, ou que o conjunto de teste e mais dificil do que o de validacao.

## 6. Baselines Tabulares

**83. O que e `RandomForest`?**  
E um conjunto de muitas arvores de decisao treinadas em subconjuntos dos dados e combinadas por voto.

**84. Por que `RandomForest` costuma ser uma baseline forte?**  
Porque lida bem com nao linearidades, variaveis heterogeneas e interacoes complexas sem exigir muito pre-processamento manual.

**85. O que e `XGBoost`?**  
E um metodo de boosting de arvores, no qual novas arvores sao treinadas para corrigir erros das anteriores.

**86. Qual a intuicao por tras do boosting?**  
Construir um modelo forte somando varios modelos fracos de forma sequencial.

**87. Qual a diferenca geral entre `RandomForest` e `XGBoost`?**  
`RandomForest` combina arvores mais independentes; `XGBoost` cria arvores sequencialmente para corrigir erros.

**88. Qual baseline teve melhor desempenho no projeto?**  
Na execucao atual, o `RandomForest` foi o melhor no teste, embora o `XGBoost` tenha ficado muito proximo e ate superado em alguns indicadores de validacao.

**89. Quais hiperparametros principais do `RandomForest` foram usados?**  
`n_estimators = 400`, `max_depth = None`, `class_weight = balanced_subsample` e `random_state = 42`.

**90. Quais hiperparametros principais do `XGBoost` foram usados?**  
`n_estimators = 400`, `max_depth = 6`, `learning_rate = 0.05`, `subsample = 0.9`, `colsample_bytree = 0.9` e `objective = multi:softmax`.

**91. Por que uma baseline pode vencer uma rede neural?**  
Porque a representacao de entrada pode estar mais favoravel para ela, ou porque o problema pode ser resolvido muito bem por estatisticas agregadas.

## 7. Redes Neurais e Conceitos de Deep Learning

**92. O que e uma rede neural?**  
E um modelo composto por camadas de transformacoes numericas que aprendem pesos para mapear entradas em saidas.

**93. O que e uma camada?**  
E uma etapa de transformacao entre a entrada e a saida de um modelo.

**94. O que e neuronio em uma rede neural?**  
E uma unidade computacional que combina entradas, pesos e uma funcao de ativacao.

**95. O que e funcao de ativacao?**  
E a funcao nao linear que permite a rede aprender relacoes complexas.

**96. O que e `GELU`?**  
E uma funcao de ativacao usada em modelos modernos, suave e nao linear.

**97. O que e `softmax`?**  
E a funcao que transforma os logits finais em valores comparaveis a probabilidades para classes mutuamente exclusivas.

**98. O que sao logits?**  
Sao os valores brutos produzidos pela ultima camada antes do `softmax`.

**99. O que e `CrossEntropyLoss`?**  
E a funcao de perda padrao para classificacao multiclasse com logits.

**100. O que e funcao de perda?**  
E a medida que informa ao modelo o quanto suas previsoes estao distantes das respostas corretas.

**101. O que e otimizador?**  
E o algoritmo que atualiza os pesos do modelo para reduzir a perda.

**102. O que e `AdamW`?**  
E um otimizador bastante usado em deep learning, com correcoes adaptativas e regularizacao por `weight decay`.

**103. O que e learning rate?**  
E o tamanho do passo dado na atualizacao dos pesos.

**104. O que e weight decay?**  
E uma forma de regularizacao que penaliza pesos excessivamente grandes.

**105. O que e batch size?**  
E o numero de amostras processadas antes de cada atualizacao dos pesos.

**106. O que e epoca?**  
E uma passada completa pelos dados de treino.

**107. O que e early stopping por paciencia?**  
E interromper o treino quando a validacao para de melhorar por varias epocas consecutivas.

**108. O que e `ReduceLROnPlateau`?**  
E um agendador que reduz o learning rate quando a metrica de validacao estagna.

**109. O que e gradient clipping?**  
E limitar o tamanho do gradiente para evitar instabilidade no treino.

**110. O que e dropout?**  
E uma tecnica de regularizacao que desativa aleatoriamente parte das unidades durante o treino.

**111. O que e regularizacao?**  
E qualquer estrategia usada para reduzir overfitting e melhorar generalizacao.

**112. O que e `LayerNorm`?**  
E uma normalizacao aplicada dentro da rede para estabilizar o processamento.

**113. O que e checkpoint?**  
E o arquivo salvo com os pesos do melhor modelo encontrado durante o treino.

## 8. LSTM e Arquitetura do Projeto

**114. O que e LSTM?**  
`Long Short-Term Memory` e um tipo de rede recorrente criada para lidar melhor com dependencias temporais e reduzir o problema de desaparecimento do gradiente.

**115. Por que usar LSTM em series temporais?**  
Porque ela consegue processar a sequencia na ordem temporal e manter um estado interno ao longo do tempo.

**116. O que e rede recorrente?**  
E uma rede que processa entradas sequenciais mantendo memoria do que ja foi observado.

**117. O que significa `hidden_size` em uma LSTM?**  
E a dimensionalidade do estado oculto da rede. Quanto maior, maior a capacidade representacional.

**118. O que significa `num_layers` em uma LSTM?**  
E o numero de camadas recorrentes empilhadas.

**119. O que significa `bidirectional = True`?**  
Significa que a rede processa a sequencia em dois sentidos, para frente e para tras, e combina as duas representacoes.

**120. Qual era a configuracao da LSTM da versao 6?**  
`hidden_size = 128`, `num_layers = 2`, `dropout = 0.20`, `bidirectional = True`.

**121. Qual era a configuracao da LSTM da versao 7?**  
`hidden_size = 192`, `num_layers = 4`, `dropout = 0.30`, `bidirectional = True`.

**122. O que muda entre a LSTM da versao 6 e a da versao 7?**  
A `versao7` aumentou profundidade, largura, `dropout` e tambem enriqueceu a cabeca de classificacao.

**123. O que e cabeca de classificacao?**  
E a parte final da rede que pega a representacao aprendida e a transforma em logits de classe.

**124. Como era a cabeca da versao 6?**  
Era mais simples, baseada principalmente no ultimo estado oculto combinado com uma pequena rede densa.

**125. Como era a cabeca da versao 7?**  
Ela combinava tres resumos: ultimo estado oculto, media temporal e `attention pooling`, seguidos por uma MLP mais profunda.

**126. O que e `attention pooling`?**  
E um mecanismo que aprende a dar pesos diferentes aos instantes da sequencia para formar um resumo mais informativo.

**127. O que e `mean pooling`?**  
E a media das representacoes ao longo do tempo.

**128. Por que combinar ultimo estado, media e atencao?**  
Porque cada resumo captura um aspecto diferente da sequencia: estado final, comportamento global e instantes mais relevantes.

**129. O que significa dizer que a LSTM faz classificacao sequence-to-one?**  
Significa que ela le toda a sequencia e devolve um unico rotulo para a amostra inteira.

**130. Por que foi usada `CrossEntropyLoss` com pesos de classe?**  
Para tentar compensar o desbalanceamento e nao deixar classes raras completamente dominadas pelas mais frequentes.

**131. O que sao pesos de classe?**  
Sao pesos maiores dados a classes menos frequentes dentro da funcao de perda.

**132. Como os pesos de classe foram calculados?**  
Com `compute_class_weight`.

**133. Qual foi o learning rate da versao 6?**  
`1e-3`.

**134. Qual foi o learning rate da versao 7?**  
`7e-4`.

**135. Qual foi o batch size da versao 6?**  
`64`.

**136. Qual foi o batch size da versao 7?**  
`48`.

**137. Qual foi o numero maximo de epocas da versao 6?**  
`30`.

**138. Qual foi o numero maximo de epocas da versao 7?**  
`40`.

**139. Qual foi a paciencia da versao 6?**  
`6`.

**140. Qual foi a paciencia da versao 7?**  
`8`.

## 9. Interpretacao dos Resultados do Projeto

**141. Quais foram os resultados da LSTM da versao 6 no teste?**  
`accuracy = 0.9433`, `macro-F1 = 0.9185` e `balanced accuracy = 0.9367`.

**142. Quais foram os resultados do RandomForest no teste da versao 6?**  
`accuracy = 0.9851`, `macro-F1 = 0.9811` e `balanced accuracy = 0.9744`.

**143. Quais foram os resultados do XGBoost no teste da versao 6?**  
`accuracy = 0.9821`, `macro-F1 = 0.9791` e `balanced accuracy = 0.9733`.

**144. O que esses resultados mostram sobre a versao 6?**  
Mostram que a `LSTM` foi forte, mas as baselines tabulares foram melhores.

**145. Quais foram os resultados da LSTM profunda da versao 7 no teste?**  
`accuracy = 0.9104`, `macro-F1 = 0.8964` e `balanced accuracy = 0.9191`.

**146. O que aconteceu com a versao 7 em relacao a versao 6?**  
A rede mais profunda nao melhorou a rede anterior; ela piorou nos tres indicadores principais.

**147. Qual foi a principal leitura metodologica da versao 7?**  
Que aumentar profundidade recorrente, sozinho, nao garantiu melhoria.

**148. Por que o professor pode considerar a versao 7 importante mesmo com piora?**  
Porque um experimento negativo ainda testa uma hipotese e gera conhecimento sobre o problema.

**149. O que significa dizer que um resultado negativo tambem e cientificamente util?**  
Significa que ele ajuda a descartar hipoteses fracas e orienta experimentos futuros de forma mais informada.

**150. Por que o RandomForest continuou tao forte?**  
Porque a representacao tabular agregada parece capturar muito bem o padrao discriminativo das series neste problema.

**151. Por que o XGBoost tambem foi forte?**  
Porque boosting de arvores consegue explorar bem interacoes entre atributos agregados e correcoes finas de erro.

**152. O que se aprende ao comparar modelos complexos com baselines simples?**  
Aprende-se que complexidade maior nao significa automaticamente melhor desempenho.

**153. O que um professor pode querer ouvir como conclusao final do aluno?**  
Que o projeto mostrou, com base experimental, que a formulacao do problema, a representacao dos dados e a escolha das metricas sao tao importantes quanto a complexidade do modelo.

## 10. Perguntas Integradoras de Prova

**154. Por que o problema das versoes 6 e 7 e um problema de classificacao e nao de regressao?**  
Porque a saida desejada e um rotulo discreto entre `0` e `9`, e nao um valor continuo.

**155. Por que nao basta olhar somente accuracy neste projeto?**  
Porque o dataset e desbalanceado, e accuracy pode esconder falhas importantes em classes raras.

**156. Se o RandomForest venceu a LSTM, isso quer dizer que LSTM nao serve para series temporais?**  
Nao. Quer dizer apenas que, nesta configuracao e com esta representacao, a baseline tabular foi mais eficiente.

**157. Por que o uso de `X_tab` ajuda tanto os modelos classicos?**  
Porque ele resume a sequencia em atributos estatisticos diretamente comparaveis e muito informativos.

**158. O que significa dizer que a representacao pode ser mais importante que o modelo?**  
Significa que um modelo simples pode vencer se receber uma representacao muito boa dos dados.

**159. Por que a versao 7 nao invalida a versao 6?**  
Porque ela nao foi criada para substituir automaticamente a anterior, e sim para testar uma hipotese arquitetural.

**160. O que seria uma resposta madura para justificar a manutencao do RandomForest como melhor modelo atual?**  
Que a escolha do melhor modelo deve ser baseada em desempenho observado, nao em preferencia por modelos mais sofisticados.

**161. Qual seria uma boa pergunta de professor sobre metodologia cientifica neste projeto?**  
Por que foi importante comparar `LSTM`, `RandomForest` e `XGBoost` sob o mesmo `split` e o mesmo pre-processamento?

**162. Qual e a resposta para essa pergunta?**  
Porque isso torna a comparacao justa e reduz o risco de atribuir diferencas de desempenho a mudancas no dado, e nao ao modelo.

**163. O que o aluno deve demonstrar ao explicar o projeto como um todo?**  
Que entendeu a diferenca entre previsao e classificacao, a estrutura do `3W`, o papel do pre-processamento, o motivo das metricas escolhidas, a arquitetura das LSTMs, o valor das baselines e a interpretacao honesta dos resultados.

## 11. Resumo Final Para Revisao Rapida

**164. Qual e a ideia central do projeto?**  
E estudar diferentes abordagens para modelar dados do `3W`, preservando cada iteracao experimental em uma versao separada.

**165. Qual e a ideia central das versoes 6 e 7?**  
Classificar series temporais completas nas classes `0` a `9`.

**166. Qual foi a melhor rede recorrente?**  
A `LSTM` multitarefa da `versao10`.

**167. Qual foi o melhor modelo geral?**  
O `RandomForest`.

**168. O que o projeto ensina de mais importante em termos de ciencia de dados?**  
Que boas comparacoes experimentais exigem baselines fortes, metricas adequadas, interpretacao cuidadosa e disposicao para aceitar quando o modelo mais sofisticado nao vence.

## 12. Perguntas Sobre A Versao 8

**169. O que a versao 8 acrescentou ao projeto?**  
Ela tornou a construcao da `LSTM`, do `RandomForest` e do `XGBoost` explicitamente visivel nos notebooks.

**170. A versao 8 mudou os resultados da versao 7?**  
Nao. A `LSTM` da `versao8` reproduziu exatamente os numeros da `versao7`.

**171. O que significa obter os mesmos resultados na versao 7 e na versao 8?**  
Significa que a abertura do codigo no notebook aumentou a transparencia sem alterar o experimento.

**172. Por que a versao 8 e especialmente util para um artigo cientifico?**  
Porque ela permite mostrar com clareza a metodologia, a arquitetura e a comparacao entre modelos sem esconder etapas centrais no pipeline auxiliar.

**173. Qual foi a principal conclusao experimental reforcada pela versao 8?**  
Que o `RandomForest` continua sendo o melhor modelo global, enquanto a melhor `LSTM` continua sendo a da `versao6`.

**174. O que a versao 8 ensina em termos de reprodutibilidade?**  
Ensina que um experimento bem reproduzido deve manter os mesmos resultados mesmo quando a implementacao fica mais explicita e auditavel.

## 13. Perguntas Sobre A Versao 9 E O Artigo Do Dataset

**175. Quais foram os resultados da versao 9 no teste?**  
`accuracy = 0.9224`, `macro-F1 = 0.9268` e `balanced accuracy = 0.9415`.

**176. A versao 9 foi melhor que a versao 8?**  
Sim. Ela melhorou em `accuracy`, `macro-F1` e `balanced accuracy`.

**177. A versao 9 foi melhor que a melhor LSTM anterior da versao 6?**  
Ela nao foi melhor em `accuracy`, mas superou a `versao6` em `macro-F1` e `balanced accuracy`.

**178. A versao 9 venceu o RandomForest?**  
Nao. O `RandomForest` continuou melhor no conjunto de teste.

**179. O que a versao 9 mostrou de mais importante?**  
Mostrou que combinar leitura temporal com `X_tab` ajuda bastante, mas ainda nao e suficiente para superar as baselines tabulares.

**180. Segundo o artigo do dataset, quais tipos de rotulo existem no 3W 2.0.0?**  
Existem o `class label`, ligado a normalidade ou evento indesejado, e o `state label`, ligado ao estado operacional do poco.

**181. O que sao as classes transitorias 101 a 109?**  
Sao codigos que representam periodos de transicao entre operacao normal e o estado estacionario de certos eventos indesejados.

**182. Por que isso e importante para o projeto?**  
Porque mostra que o dataset nao foi pensado apenas para classificar a instancia inteira, mas tambem para modelar a evolucao temporal do evento.

**183. O projeto atual usa totalmente esse potencial do dataset?**  
Nao. Ate a `versao9`, o foco principal ainda esta na classificacao da instancia como um todo.

**184. O que o artigo diz sobre missing values, frozen variables e outliers nas instancias reais?**  
Que essas caracteristicas sao mantidas propositalmente, para incentivar metodologias capazes de lidar com desafios reais.

**185. O que isso sugere para uma proxima versao do projeto?**  
Sugere usar mascaras de missing e frozen values como informacao explicita do modelo, em vez de apenas tratar esses casos de forma indireta.

**186. O artigo reforca que todas as 27 variaveis devem ser consideradas?**  
Ele mostra que o dataset 2.0.0 foi concebido com 27 variaveis presentes nas instancias, mesmo quando algumas estao ausentes em certos casos.

**187. O que isso implica para a versao 10?**  
Implica que a `versao10` deve buscar usar o conjunto completo de variaveis, junto com os rotulos por observacao e o contexto operacional.

## 14. Perguntas Sobre A Versao 10

**188. Qual e a principal diferenca conceitual da versao 10?**  
A `versao10` deixa de tratar a serie apenas como uma instancia global e passa a modelar tambem informacao temporal interna da amostra.

**189. Quais informacoes novas entram explicitamente na versao 10?**  
Entram as `27` variaveis do dataset, mascaras de `missing values`, mascaras de `frozen values`, rotulos por observacao de `class` e `state`, e a origem da amostra.

**190. O que significa dizer que a versao 10 e multitarefa?**  
Significa que o modelo aprende mais de uma tarefa ao mesmo tempo: a classe global da instancia e tarefas auxiliares ligadas aos rotulos sequenciais.

**191. Por que usar o state label como tarefa auxiliar pode ajudar?**  
Porque o `state` pode ensinar ao modelo regularidades operacionais intermediarias que ajudam a formar uma representacao temporal mais informativa.

**192. Por que a origem da amostra pode ser relevante?**  
Porque amostras reais, simuladas e desenhadas podem ter caracteristicas estatisticas e dinamicas diferentes, e o modelo pode se beneficiar desse contexto.

**193. Qual e a hipotese cientifica da versao 10?**  
A hipotese e que uma arquitetura mais fiel ao desenho do dataset tem mais chance de competir com as baselines do que uma rede apenas mais profunda.

**194. Quais foram os resultados da versao 10 no teste?**  
`accuracy = 0.9373`, `macro-F1 = 0.9409` e `balanced accuracy = 0.9572`.

**195. A versao 10 foi melhor que a versao 9?**  
Sim. Ela melhorou nas tres metricas principais e reduziu o gap para as baselines.

**196. A versao 10 foi melhor que a melhor LSTM anterior da versao 6?**  
Ela perdeu pouco em `accuracy`, mas superou a `versao6` em `macro-F1` e `balanced accuracy`.

**197. A versao 10 venceu o RandomForest ou o XGBoost?**  
Nao. Ela ficou mais proxima, mas o `RandomForest` e o `XGBoost` ainda terminaram acima no teste.

**198. O que a versao 10 mostrou de mais importante?**  
Mostrou que aproveitar melhor a estrutura do dataset gera ganho real, e que o caminho promissor nao era apenas aumentar a profundidade da LSTM.

## 15. Perguntas Sobre A Versao 11

**199. Qual foi a ideia central da versao 11?**  
Manter a arquitetura multitarefa da `versao10`, mas mudar o pre-processamento para testar uma hipotese mais especifica: remover features totalmente vazias e treinar as classes de falha usando apenas trechos com `state` transiente ou de falha.

**200. Quais features foram removidas na versao 11 por estarem totalmente vazias?**  
`ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS` e `T-MON-CKP`.

**201. Quantas entradas ficaram na versao 11 depois dessa remocao?**  
Ficaram `18` colunas para `X_seq` e `162` atributos tabulares em `X_tab`, porque sao `18` colunas multiplicadas por `9` estatisticas por coluna.

**202. O que o pre-processamento atual da versao 11 fez com a base?**  
Nos artefatos atuais, de `2228` series originais, `605` foram mantidas. O `split` ficou em `423` series de treino, `90` de validacao e `92` de teste. Com a regra atual, sobreviveram apenas as classes `0` e `8`.

**203. Por que isso exige cuidado ao interpretar a versao 11?**  
Porque, nesse estado atual, a `versao11` deixou de ser diretamente comparavel ao problema multiclasse mais amplo das versoes anteriores. Na pratica, os artefatos atuais ficaram muito mais proximos de um problema reduzido do que da tarefa original de `10` classes.

**204. Quais sao os resultados atuais da versao 11?**  
Na execucao atual, a `LSTM` da `versao11` ficou com `accuracy = 1.0000`, `macro-F1 = 1.0000` e `balanced accuracy = 1.0000` tanto em validacao quanto em teste. No mesmo teste, `RandomForest` e `LGBM` tambem ficaram com `1.0000`, enquanto o `XGBoost` ficou com `accuracy = 0.9891`, `macro-F1 = 0.8972` e `balanced accuracy = 0.9944`.

**205. Isso quer dizer que a versao 11 virou a melhor versao do projeto?**  
Nao. Esses resultados perfeitos aconteceram porque o pre-processamento atual preservou apenas as classes `0` e `8`. Entao a tarefa ficou muito mais simples e deixou de ser diretamente comparavel ao problema multiclasse mais amplo das versoes anteriores.

**206. Qual e a leitura metodologica mais honesta sobre a versao 11?**  
Que ela foi util como experimento de ablacao e mostrou a alta sensibilidade do projeto ao pre-processamento. A execucao atual esta tecnicamente correta, mas o filtro por `state` ficou tao severo que a `versao11` deve ser interpretada como uma tarefa reduzida, e nao como substituta direta da `versao10`.
