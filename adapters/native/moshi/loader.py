"""Moshi backend loader — smoke rebuild 2026-10-17 / hubfix 2026-10-18.

Contract sources (not recovered bytes):
- Ops DRAFT native_config backend_args (weights under benchmark1-share)
- OPS_MOSHI_FIX_CONTINUE: moshi.models LMGen / loaders
- $V native.py _module(loader_module, loader_sha256) pin
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class MoshiBackend:
    """In-process Moshi (moshika) streaming backend for mu-bench openweight native."""

    def __init__(self, args: Any):
        import torch
        from moshi.models import loaders

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO(smoke): confirm historical dtype/device map if formal parity needed
        model_path = Path(getattr(args, "model_path", args.moshi_weight)).parent if hasattr(args, "moshi_weight") else Path(args.model_path)
        moshi_weight = Path(args.moshi_weight) if hasattr(args, "moshi_weight") else model_path / "model.safetensors"
        mimi_weight = Path(args.mimi_weight) if hasattr(args, "mimi_weight") else None
        tokenizer = Path(args.tokenizer) if hasattr(args, "tokenizer") else None

        # Prefer official loaders helpers when present
        self.mimi = None
        self.lm = None
        self.lm_gen = None
        self.sample_rate = 24000
        self.frame_size = self.sample_rate * 80 // 1000  # 80 ms quantum default
        try:
            if hasattr(loaders, "get_mimi") and mimi_weight is not None:
                self.mimi = loaders.get_mimi(str(mimi_weight), device=self.device)
                if hasattr(self.mimi, "sample_rate"):
                    self.sample_rate = int(self.mimi.sample_rate)
                    self.frame_size = self.sample_rate * 80 // 1000
            if hasattr(loaders, "get_moshi_lm"):
                self.lm = loaders.get_moshi_lm(str(moshi_weight), device=self.device)
            from moshi.models import LMGen
            if self.lm is not None:
                self.lm_gen = LMGen(self.lm)
        except Exception as e:
            # Keep importable; start()/step will surface clear errors on card
            self._init_error = e
        else:
            self._init_error = None
        self.tokenizer_path = tokenizer
        self.text_tokenizer = None
        if tokenizer is not None and tokenizer.exists():
            try:
                from sentencepiece import SentencePieceProcessor
                self.text_tokenizer = SentencePieceProcessor(model_file=str(tokenizer))
            except Exception:
                pass

    def ensure_streaming(self, batch_size: int = 1) -> None:
        if self._init_error is not None:
            raise RuntimeError(f"MoshiBackend init failed: {self._init_error}") from self._init_error
        # Mirror Kyutai moshi/server.py ServerState.__init__: streaming_forever BEFORE any reset.
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
