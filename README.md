# Projeto 3W com Versionamento de Experimentos

## Visao geral

Este repositorio esta organizado para preservar o historico de desenvolvimento em duas etapas separadas:

- `versao1/`
- `versao2/`

A ideia e simples:

- `versao1` guarda o primeiro pipeline, mais basico e didatico
- `versao2` guarda a iteracao seguinte, com melhorias de pre-processamento, arquitetura e avaliacao

Assim, voce consegue evoluir o projeto sem sobrescrever aquilo que aprendeu nas versoes anteriores.

## Estrutura da raiz

- `3W/`
  - clone local do dataset 3W
  - fica apenas na pasta raiz
- `versao1/`
  - notebooks e `README` da primeira versao
  - possui seu proprio diretorio `artifacts/`
- `versao2/`
  - notebooks, `README` e codigo auxiliar da segunda versao
  - possui seu proprio diretorio `artifacts/`

## Como usar

Fluxo recomendado:

1. manter o dataset apenas em `3W/`
2. executar os notebooks sempre de dentro da versao desejada
3. deixar os resultados gerados em `versao1/artifacts/` ou `versao2/artifacts/`

Isso evita mistura entre artefatos de experimentos diferentes.

## Sobre o `.gitignore`

O repositório ignora:

- `3W/`
- `versao1/artifacts/`
- `versao2/artifacts/`

Com isso, o Git guarda o codigo, os notebooks e a documentacao, mas nao versiona:

- o clone bruto do dataset
- arquivos pesados de pre-processamento
- checkpoints de modelos
- saidas intermediarias

## Documentacao das versoes

Para detalhes de cada experimento, veja:

- [versao1/README.md](/Users/tiagoriosdarocha/Desktop/lstm-w3/versao1/README.md)
- [versao2/README.md](/Users/tiagoriosdarocha/Desktop/lstm-w3/versao2/README.md)
