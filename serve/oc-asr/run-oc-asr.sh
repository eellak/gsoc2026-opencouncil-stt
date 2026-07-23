#!/usr/bin/env bash
# Start the self-hosted OpenCouncil ASR endpoint (faster-whisper + fine-tuned LoRA).
# Binds 127.0.0.1 only; put a Cloudflare Tunnel in front for public access.
#
#   OC_ASR_API_KEY=your-secret ./run-oc-asr.sh [port]
#
# Overridable: OC_ASR_VENV (python venv), OC_ASR_MODEL_DIR (CT2 model dir).
set -euo pipefail

PORT="${1:-8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${OC_ASR_VENV:-/home/harold/opencouncil-fine-tuning/.venv-eval}"

: "${OC_ASR_API_KEY:?set OC_ASR_API_KEY to a secret string first}"
export OC_ASR_API_KEY
export OC_ASR_MODEL_DIR="${OC_ASR_MODEL_DIR:-$HERE/ct2}"

exec "$VENV/bin/python" -m uvicorn oc_asr_server:app \
  --app-dir "$HERE" --host 127.0.0.1 --port "$PORT"
