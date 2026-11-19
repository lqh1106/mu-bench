#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
: "${FREEZE_MODEL_PATH:?Set the absolute Freeze-Omni checkpoint path}"
: "${FREEZE_LLM_PATH:?Set the absolute Qwen2-7B-Instruct checkpoint path}"
export PYTHONPATH="$MUBENCH_REPO/third_party/models/Freeze-Omni/source${PYTHONPATH:+:$PYTHONPATH}"
exec "$MUBENCH_REPO/.venv-freeze_omni/bin/python" third_party/models/Freeze-Omni/source/bin/server.py --model_path "$FREEZE_MODEL_PATH" --llm_path "$FREEZE_LLM_PATH" --ip 127.0.0.1 --port "${FREEZE_PORT:-8226}" --max_users "${FREEZE_MAX_USERS:-2}" --llm_exec_nums "${FREEZE_MAX_USERS:-2}" --timeout 6000
