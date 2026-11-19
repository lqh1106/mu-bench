"""Preferred μ-Bench OpenAI Realtime entrypoint."""

from experiments.tau_multiuser_voice.run_realtime import *  # noqa: F403
from experiments.tau_multiuser_voice.run_realtime import _MODULE


if __name__ == "__main__":
    _MODULE.main()
