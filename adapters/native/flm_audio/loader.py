"""FLM-Audio backend loader — smoke rebuild 2026-10-17 / hubfix 2026-10-18.

Contract: FLMAudioNative (patch_natives / flm smoke native) expects
FLMAudioBackend with .mimi .lm_gen .torch .device .frame_size .tokenizer
and mimi.sample_rate; quantum 80 ms.
PYTHONPATH must include $OUT/repos/flm-audio (see 24_start_scripts.txt).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class FLMAudioBackend:
    def __init__(self, args: Any):
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_error = None
        model_path = Path(args.model_path)
        mimi_path = Path(getattr(args, "mimi_path", model_path))

        self.mimi = None
        self.lm_gen = None
        self.tokenizer = None
        self.frame_size = 1920  # 80ms @ 24kHz placeholder until mimi loads

        allow_stub = str(
            getattr(args, "allow_mimi_stub", os.environ.get("allow_mimi_stub", "0"))
        ).strip().lower() in ("1", "true", "yes")

        try:
            from flmaudio.models.loaders import get_mimi
            from flmaudio.models import LMGen, FLMAudioForCausalLM

            self.mimi = get_mimi(str(mimi_path), device=self.device)
            sr = int(getattr(self.mimi, "sample_rate", 24000))
            self.frame_size = sr * 80 // 1000

            lm = FLMAudioForCausalLM.from_pretrained(
                str(model_path),
                device_map=str(self.device),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.lm_gen = LMGen(lm)

            # Ops: FLMAudioNative calls o.mimi/lm_gen.reset_streaming directly — wrap so forever precedes reset.
            def _wrap_reset(mod):
                if mod is None or not hasattr(mod, "reset_streaming"):
                    return
                orig = mod.reset_streaming
                def _reset(*a, __orig=orig, __mod=mod, **k):
                    if getattr(__mod, "_streaming_state", None) is None and hasattr(__mod, "streaming_forever"):
                        __mod.streaming_forever(1)
                    return __orig(*a, **k)
                mod.reset_streaming = _reset
            _wrap_reset(self.mimi)
            _wrap_reset(self.lm_gen)

            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path), trust_remote_code=True
                )
            except Exception:
                class _Tok:
                    def decode(self, ids):
                        return ""

                self.tokenizer = _Tok()
        except Exception as e:
            self._init_error = e
            if allow_stub:
                class _MimiStub:
                    sample_rate = 24000

                    def reset_streaming(self):
                        pass

                    def encode(self, x):
                        raise RuntimeError(f"FLM mimi stub — init failed: {e}")

                    def decode(self, t):
                        raise RuntimeError(f"FLM mimi stub — init failed: {e}")

                class _LmStub:
                    def reset_streaming(self):
                        pass

                    def step(self, codes):
                        return None

                    class lm_model:
                        class config:
                            class mm_token_info:
                                text_wait_token_id = None

                self.mimi = _MimiStub()
                self.lm_gen = _LmStub()
                self.frame_size = 1920
                if self.tokenizer is None:
                    class _Tok:
                        def decode(self, ids):
                            return ""

                    self.tokenizer = _Tok()
            else:
                # fail-closed: do not silently stub; surface on start/sessions
                raise RuntimeError(f"FLMAudioBackend init failed: {e}") from e

    def ensure_streaming(self, batch_size: int = 1) -> None:
        if self._init_error is not None:
            raise RuntimeError(f"FLMAudioBackend init failed: {self._init_error}") from self._init_error
        for mod in (self.mimi, self.lm_gen):
            if mod is None:
                continue
            if getattr(mod, "_streaming_state", None) is None and hasattr(mod, "streaming_forever"):
                mod.streaming_forever(batch_size)

    def reset_streaming(self) -> None:
        self.ensure_streaming(1)
        if self.mimi is not None and hasattr(self.mimi, "reset_streaming"):
            self.mimi.reset_streaming()
        if self.lm_gen is not None and hasattr(self.lm_gen, "reset_streaming"):
            self.lm_gen.reset_streaming()
