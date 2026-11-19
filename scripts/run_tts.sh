#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
: "${COSYVOICE_REPO:?Set the absolute CosyVoice source directory}"
: "${COSYVOICE_MODEL_DIR:?Set the absolute Fun-CosyVoice3-0.5B-2512 checkpoint directory}"
: "${TTS_TOKEN_FILE:?Set an absolute token-file path}"
TTS_PYTHON="${TTS_PYTHON:-$MUBENCH_REPO/.venv-tts/bin/python}"
export MUBENCH_RUNTIME_ROOT="$MUBENCH_REPO/runtime/core"
export PYTHONPATH="$MUBENCH_RUNTIME_ROOT/src:$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS"
exec "$TTS_PYTHON" services/tts/server.py --model-dir "$COSYVOICE_MODEL_DIR" --cosyvoice-repo "$COSYVOICE_REPO" --voice-catalog "$MUBENCH_REPO/assets/voices/catalog/voice_catalog.json" --voice-binding-manifest "$MUBENCH_REPO/assets/voices/bindings.json" --cache-dir "$MUBENCH_REPO/outputs/services/tts_cache" --device cuda:0 --host "${TTS_BIND_HOST:-127.0.0.1}" --port "${TTS_PORT:-22986}" --advertise-host "${TTS_ADVERTISE_HOST:-127.0.0.1}" --api-key-file "$TTS_TOKEN_FILE" --endpoint-file "$MUBENCH_REPO/outputs/services/tts.json" --replica-id speech-1
