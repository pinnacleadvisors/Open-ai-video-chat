#!/usr/bin/env bash
# Start the backend (FastAPI/Uvicorn) and the frontend (Next.js dev server).
# Usage: ./scripts/start.sh [--virtual-camera]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example"
fi

if [[ "${1:-}" == "--virtual-camera" ]]; then
  export ENABLE_VIRTUAL_CAMERA=true
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> starting backend on :8000"
( python -m server.main ) &

echo "==> starting frontend on :3000"
( cd web && npm run dev ) &

wait
