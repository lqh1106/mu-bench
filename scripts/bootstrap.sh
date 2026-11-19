#!/usr/bin/env bash
set -euo pipefail
MUBENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MUBENCH_REPO"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m mubench validate
.venv/bin/python scripts/check_entrypoints.py
.venv/bin/python -m unittest discover -s tests -v
