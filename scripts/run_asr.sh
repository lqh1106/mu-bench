#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
export FW_MODEL_PATH="${FW_MODEL_PATH:-$MUBENCH_REPO/external/weights/asr}"
export FW_DEVICE="${FW_DEVICE:-cuda}"
export FW_COMPUTE_TYPE="${FW_COMPUTE_TYPE:-float16}"
exec "${ASR_PYTHON:-$MUBENCH_REPO/.venv-asr/bin/python}" services/asr/server.py --host "${ASR_BIND_HOST:-127.0.0.1}" --port "${ASR_PORT:-18991}" --preload
