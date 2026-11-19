"""Package entrypoint for the standalone OpenAI Realtime runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "output" / "run_tau_mu_voice_realtime.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "tau_mu_voice_realtime_impl", _RUNNER_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Realtime runner: {_RUNNER_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

from experiments.tau_multiuser_voice.tts_backends import CosyVoiceTTS


if __name__ == "__main__":
    _MODULE.main()
