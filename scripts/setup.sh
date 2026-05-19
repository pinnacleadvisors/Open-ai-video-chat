#!/usr/bin/env bash
# Install Python deps, frontend deps, third-party model code, and download
# the open-source model checkpoints needed for the default pipeline.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

echo "==> python venv at $VENV"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "==> installing server requirements"
pip install -r server/requirements.txt

echo "==> installing frontend deps"
( cd web && npm install --no-audit --no-fund )

mkdir -p models/checkpoints third_party media/personas

# ---- MuseTalk ----------------------------------------------------------------
if [ ! -d third_party/MuseTalk ]; then
  echo "==> cloning MuseTalk (Tencent / MIT)"
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk third_party/MuseTalk
fi

# ---- Model weights ----------------------------------------------------------
"$VENV/bin/python" scripts/download_models.py

# ---- Default persona --------------------------------------------------------
if [ ! -f media/personas/default.png ]; then
  echo "==> no default persona installed (drop one at media/personas/default.png"
  echo "    or upload via the web UI)"
fi

cat <<EOF

setup complete.

next steps:
  1. Start an LLM server:    ollama serve & ollama pull llama3.1:8b
  2. Copy .env.example:      cp .env.example .env
  3. Start the stack:        ./scripts/start.sh

EOF
