#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
MODEL="${1:?Pass minicpm, personaplex, raon, lychee, moshi, freeze_omni, flm_audio or voila}"
PYTHON_BIN="${MODEL_PYTHON:-python3.10}"
"$PYTHON_BIN" -m venv ".venv-$MODEL"
PY="$MUBENCH_REPO/.venv-$MODEL/bin/python"
"$PY" -m pip install --upgrade pip
# Each backend owns its torch/transformers stack. Never install into the runner venv.
"$PY" -m pip install -r "environments/$MODEL/requirements.lock"
"$PY" -m pip install pydantic soundfile scipy loguru psutil requests
"$PY" -m pip install -c "environments/$MODEL/requirements.lock" -r "environments/$MODEL/requirements.extra.txt"
case "$MODEL" in
  freeze_omni) "$PY" -m pip install -c "environments/$MODEL/requirements.lock" -r third_party/models/Freeze-Omni/requirements.txt ;;
  flm_audio) "$PY" -m pip install -c "environments/$MODEL/requirements.lock" -r third_party/models/flm-audio/requirements-server.txt ;;
  lychee) "$PY" -m pip install -c "environments/$MODEL/requirements.lock" -r third_party/models/Lychee-FD/requirements.txt ;;
esac
printf 'Created isolated model environment: %s\n' "$MUBENCH_REPO/.venv-$MODEL"
