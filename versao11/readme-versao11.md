# Versao 11

A `versao11` preserva a arquitetura multitarefa da `versao10`, mas corrige a hipotese de pre-processamento. O ponto central desta revisao e simples: no `3W`, `class` e `state` nao significam a mesma coisa, entao o recorte do treino nao deve usar `state` como se ele fosse o rotulo do evento.

## Correcao conceitual

No `3W`, cada observacao pode carregar dois rotulos diferentes:

- `class`: identifica o evento observado naquele instante;
- `state`: identifica o estado operacional do poco naquele instante.

Em termos práticos:

- `class = 0` significa operacao normal;
- `class = 1..9` significa algum evento indesejado;
- `class = 101..109` representa fase transitoria de eventos;
- `state` nao diz qual evento ocorreu; ele descreve o modo operacional do poco.

Esse foi o mal-entendido corrigido nesta versao. O recorte antigo por `state` acabava removendo series inteiras por interpretar `state` como se fosse o proprio rotulo de falha. A leitura correta e: quem distingue normal, falha e transiente em nivel observacional e o `class`.

## O Que a Versao 11 Passa a Fazer

- remove as features totalmente vazias: `ABER-CKGL`, `ABER-CKP`, `P-JUS-BS`, `P-JUS-CKP`, `P-MON-CKGL`, `P-MON-SDV-P`, `PT-P`, `QBS`, `T-MON-CKP`;
- reduz as entradas de `27` para `18` variaveis e de `243` para `162` atributos tabulares;
- mantem todas as series no `split`, sem descartar series inteiras por `state`;
- no treino das classes globais `1..9`, tenta manter apenas observacoes cujo `class` observacional indica evento ou transiente, isto e, `1..9` ou `101..109`;
- preserva a serie completa da classe global `0`, para que o modelo continue vendo exemplos normais no treino;
- se uma serie de falha ficaria vazia apos remover observacoes com `class = 0`, o pipeline faz `fallback` para a serie completa, evitando perder cobertura daquela classe;
- mantem a mesma arquitetura multitarefa da `versao10`.

## O Que Mudou no Pre-processamento

Com a regra corrigida:

- series originais: `2228`
- series mantidas para `split`: `2228`
- series descartadas por filtro de `state`: `0`
- `split`: `train = 1559`, `validation = 334`, `test = 335`
- classes globais preservadas no problema: `0..9`

No treino:

- `1143` series de falha pedem recorte observacional por `class`;
- `1112` conseguem usar apenas observacoes com `class != 0`;
- `31` precisam de `fallback` para a serie completa porque ficariam vazias apos o recorte;
- todas essas series com `fallback` pertencem a classe global `9`.

## Como Ler o Novo Relatorio

O arquivo `series_quality_report.csv` agora deve ser lido assim:

- `n_observation_class_zero_rows`: quantas observacoes da serie estao com `class = 0`;
- `n_observation_class_fault_rows`: quantas observacoes estao em `class = 1..9`;
- `n_observation_class_transient_rows`: quantas observacoes estao em classes transitorias;
- `n_observation_class_focus_rows`: quantas observacoes podem ser mantidas no foco de treino da `v11`.

As colunas derivadas de `state` continuam aparecendo no relatorio como metadado descritivo, mas deixaram de ser o criterio de corte desta versao.

## Leitura Metodologica

Depois desta correcao, a `versao11` deixa de ser a ablacao "segmentos negativos por `state`" e passa a ser uma ablacao "foco por `class` observacional". Isso e muito mais coerente com o significado real dos rotulos do `3W`.

Em outras palavras:

- `class` responde "qual evento esta acontecendo?";
- `state` responde "em que modo operacional o poco esta?".

Logo, se a intencao e remover trechos normais e manter trechos ligados a falhas ou transientes, o recorte correto deve usar `class`, nao `state`.

## Status dos Resultados

Os resultados perfeitos documentados anteriormente para a `versao11` pertenciam a configuracao antiga, baseada em filtro por `state`. Eles nao representam mais a configuracao atual.

Portanto:

- o pre-processamento da `versao11` foi corrigido;
- o codigo, o `split` e o relatorio de preparo agora refletem a interpretacao correta de `class` e `state`;
- as metricas finais da `LSTM` e das baselines precisam ser recalculadas com os notebooks `3` e `4`.

## Sequencia Recomendada

- `1-visao-geral-dos-dados.ipynb`
- `2-pre-processamento.ipynb`
- `3-classificacao-multiclasse-lstm-multitarefa.ipynb`
- `4-comparacao-lstm-multitarefa-vs-baselines.ipynb`
