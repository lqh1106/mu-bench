#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
PY="${MUBENCH_PYTHON:-$MUBENCH_REPO/.venv/bin/python}"
"$PY" -m mubench validate
"$PY" scripts/check_entrypoints.py
"$PY" -m unittest discover -s tests -v
mkdir -p outputs/release_tests
export MUBENCH_TEST_OUTPUT="$MUBENCH_REPO/outputs/release_tests"
"$PY" tests/tool_augmented/test_request_revision.py
"$PY" tests/tool_augmented/test_caller_protocol.py
"$PY" tests/tool_augmented/test_text_contracts.py
"$PY" tests/tool_augmented/reference_check.py
