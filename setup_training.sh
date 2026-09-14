#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " Setup training environment for Pippo"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERRORE] $PYTHON_BIN non trovato."
  echo "Installa Python 3.12, ad esempio con:"
  echo "  brew install python@3.12"
  echo "Poi rilancia questo script."
  exit 1
fi

echo "[1/4] Versione Python: $("$PYTHON_BIN" --version)"

if [ ! -d ".venv312" ]; then
  echo "[2/4] Creo virtualenv .venv312 con $PYTHON_BIN ..."
  "$PYTHON_BIN" -m venv .venv312
else
  echo "[2/4] Virtualenv .venv312 già esistente, lo riuso."
fi

# shellcheck disable=SC1091
source .venv312/bin/activate

echo "[3/4] Aggiorno pip e installo dipendenze..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Avvio training: pippo su dataset di sopravvivenza ..."
python training/scripts/train_sft.py
