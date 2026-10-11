#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
FLOWRF_PYTHON="${PYTHON_BIN:-python3}"
"$FLOWRF_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install '.[plot]'
.venv/bin/flowrf install-grtm
.venv/bin/flowrf doctor
printf '\nActivate with: source .venv/bin/activate\n'
