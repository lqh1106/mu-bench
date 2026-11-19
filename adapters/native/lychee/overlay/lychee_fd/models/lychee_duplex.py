"""Compatibility aliases for the Lychee-FD official training package.

The released realtime runtime imports ``lychee_fd.models.lychee_duplex``,
while the checkpoint repository keeps the actual implementation under
``Training_package/Code/Models/LycheeFD.py``.  Keep the adapter thin and
delegate all model behavior to the official class.
"""

from Training_package.Code.Models.LycheeFD import LycheeFD, LycheeFDConfig

LycheeDuplexConfig = LycheeFDConfig
LycheeDuplexForCausalLM = LycheeFD

__all__ = ["LycheeDuplexConfig", "LycheeDuplexForCausalLM"]
