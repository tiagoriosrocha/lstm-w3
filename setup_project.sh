#!/usr/bin/env bash
set -euo pipefail

#instala dependencias Python do projeto
python -m pip install -r requirements.txt

#clona o dataset 3W apenas se ele ainda nao existir na raiz
if [ ! -d "3W/.git" ]; then
  git clone https://github.com/petrobras/3W.git
else
  echo "Repositorio 3W ja existe em ./3W"
fi
