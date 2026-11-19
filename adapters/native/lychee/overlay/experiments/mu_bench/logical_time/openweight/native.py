"""Source-aligned native lifecycles; real-GPU equivalence is not yet validated.

Loading uses the existing pinned legacy *constructors*, never run_session.
Model objects then stay alive in one dedicated actor process. No custom prompt
is accepted. No whole-track input, reset-on-pause, or transcript-based replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
from pathlib import Path
import random
import sys
from typing import Any

from .session import NativeOutput, SessionProtocolError


def _module(path: str, expected_sha256: str):
    source = Path(path).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise SessionProtocolError("native loader source differs from pinned SHA256")
    manifest = source.parent / "SOURCE_MANIFEST.json"
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())["files_sha256"].items():
            if Path(name).name != name or hashlib.sha256((source.parent/name).read_bytes()).hexdigest() != digest:
                raise SessionProtocolError(f"native loader dependency differs from pinned SHA256: {name}")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("mu_v14_native_loader", source)
    if spec is None or spec.loader is None:
        raise SessionProtocolError("cannot load native source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pcm(waveform) -> bytes:
    import numpy as np
    return (np.clip(np.asarray(waveform).reshape(-1), -1, 1) * 32767).astype("<i2").tobytes()


def _float(pcm: bytes):
    import numpy as np
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0


def _seed(torch, seed):
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _arguments(config):
    args = dict(config["backend_args"])
    for key in ("repo_path", "model_path", "ref_audio", "mimi_weight", "tokenizer", "moshi_weight"):
        if key in args:
            args[key] = Path(args[key])
    return argparse.Namespace(**args)


class MiniCPMNative:
    input_sample_rate = 16000
    output_sample_rate = 24000
    native_quantum_ms = 1000

    def __init__(self, module, config):
        self.owner = module.MiniCPMBackend(_arguments(config))

    def start(self, *, seed, max_media_ms):
        _seed(self.owner.torch, seed)
        with self.owner.torch.inference_mode():
            self.owner.duplex.prepare(prefix_system_prompt=None, ref_audio=self.owner.ref_audio,
                                      prompt_wav_path=self.owner.prompt_wav_path)

    def step(self, pcm16):
        with self.owner.torch.inference_mode():
            result = self.owner.duplex.streaming_prefill(audio_waveform=_float(pcm16))
            if not result.get("success", False):
                raise RuntimeError(f"MiniCPM streaming_prefill failed: {result}")
            generated = self.owner.duplex.streaming_generate(
                prompt_wav_path=self.owner.prompt_wav_path,
                max_new_speak_tokens_per_chunk=20, decode_mode="sampling")
            waveform = generated.get("audio_waveform")
            listen = bool(generated.get("is_listen", True))
            return NativeOutput(
                pcm16=_pcm(waveform) if waveform is not None and not listen else b"",
                diagnostic_text=str(generated.get("text", "")),
                events=[{"type": "native_state", "is_listen": listen,
                         "end_of_turn": bool(generated.get("end_of_turn", False))}])

    def finish(self):
        self.owner.duplex.set_session_stop()


class PersonaPlexNative:
    native_quantum_ms = 80

    def __init__(self, module, config):
        self.owner = module.PersonaPlexBackend(_arguments(config))
        self.default_prompt = module.PERSONAPLEX_OFFICIAL_DEFAULT_SYSTEM_PROMPT
        self.input_sample_rate = self.output_sample_rate = int(self.owner.mimi.sample_rate)
        if self.owner.frame_size * 1000 != self.input_sample_rate * self.native_quantum_ms:
            raise SessionProtocolError("unexpected PersonaPlex frame size")

    def start(self, *, seed, max_media_ms):
        from moshi.offline import seed_all, wrap_with_system_tags
        seed_all(seed)
        o = self.owner
        with o.torch.inference_mode():
            o.lm_gen.text_prompt_tokens = o.tokenizer.encode(wrap_with_system_tags(self.default_prompt))
            o.mimi.reset_streaming(); o.other_mimi.reset_streaming(); o.lm_gen.reset_streaming()
            o.lm_gen.step_system_prompts(o.mimi)
            o.mimi.reset_streaming()

    def step(self, pcm16):
        from moshi.offline import decode_tokens_to_pcm
        import numpy as np
        o = self.owner
        frames, pieces = [], []
        with o.torch.inference_mode():
            x = o.torch.from_numpy(_float(pcm16)).reshape(1, 1, -1).to(o.device)
            codes = o.mimi.encode(x)
            for col in range(codes.shape[-1]):
                tokens = o.lm_gen.step(codes[:, :, col:col+1])
                if tokens is None:
                    continue
                frames.append(decode_tokens_to_pcm(o.mimi, o.other_mimi, o.lm_gen, tokens))
                token = int(tokens[0, 0, 0].item())
                if token not in (0, 3):
                    pieces.append(o.tokenizer.id_to_piece(token).replace("▁", " "))
        return NativeOutput(_pcm(np.concatenate(frames)) if frames else b"", "".join(pieces))

    def finish(self):
        # The process owns these streaming objects and is released after finish.
        # In particular, no system prompt is re-fed at a planning pause.
        pass


class RaonNative:
    native_quantum_ms = 80

    def __init__(self, module, config):
        self.owner = module.RaonBackend(_arguments(config))
        self.plan_cache = module.plan_raon_cache_sequence_length
        self.pipeline = self.owner.pipeline
        self.input_sample_rate = self.output_sample_rate = int(self.pipeline.processor.sampling_rate)
        self.state = None
        self.plan = None
        if round(1000 / self.pipeline.processor.frame_rate) != self.native_quantum_ms:
            raise SessionProtocolError("unexpected Raon frame rate")

    def start(self, *, seed, max_media_ms):
        p = self.pipeline
        torch = self.owner.torch
        _seed(torch, seed)
        cfg = p.duplex_params
        default = cfg.get("system_prompt", "You are engaging in real-time conversation.")
        with torch.inference_mode():
            inputs = p.processor([{"role": "system", "content": default}],
                                 add_generation_prompt=False, device=p.device, dtype=p.dtype)
            tokens = inputs["input_ids"]
            self.plan = self.plan_cache(
                system_token_count=int(tokens.shape[-1]),
                audio_sample_count=(max_media_ms * self.input_sample_rate + 999)//1000,
                sample_rate=self.input_sample_rate, frame_rate=p.processor.frame_rate,
                minimum=self.owner.minimum_sequence_length,
                maximum=self.owner.maximum_sequence_length)
            self.owner.active_sequence_length = self.plan["planned_sequence_length"]
            self.state = p.model.init_duplex_decoding_state(
                sequences=tokens, attention_mask=torch.ones_like(tokens), do_sample=True,
                temperature=cfg.get("temperature", 0.9), top_p=cfg.get("top_p", 0.95),
                top_k=cfg.get("top_k", 66), sil_penalty=cfg.get("sil_penalty", 0.0),
                bc_penalty=cfg.get("bc_penalty", 0.0), eos_penalty=cfg.get("eos_penalty", 0.0),
                speak_first=False, max_sequence_length=self.plan["planned_sequence_length"])

    def step(self, pcm16):
        p = self.pipeline
        with self.owner.torch.inference_mode():
            # Cache cap checked before any native CUDA operation on this frame.
            if int(self.state.sequences.shape[-1]) + 3 >= self.plan["planned_sequence_length"]:
                raise SessionProtocolError("Raon KV budget exhausted before CUDA step")
            x = self.owner.torch.from_numpy(_float(pcm16)).reshape(1, -1).to(device=p.device, dtype=p.dtype)
            self.state, y = p.model.duplex_decoding_step(state=self.state, audio_input=x)
            # Native text/audio is delayed; do not claim generated text as spoken.
            phase = getattr(getattr(self.state, "machine_state", None), "phase", None)
            return NativeOutput(_pcm(y.detach().float().cpu().numpy()), events=[
                {"type": "native_state", "phase": getattr(phase, "name", None)}])

    def finish(self):
        if self.state is not None:
            self.pipeline.model.free_duplex_decoding_state(self.state)
            self.state = None


class LycheeNative:
    """Guarded direct incremental runtime, never the old asynchronous HTTP app.

    A runtime factory must retain HF stream session AND audio renderer state and
    expose advance_pcm/barrier/close. The stock app's /chunk acknowledgement is
    only an enqueue acknowledgement and does NOT satisfy this contract.
    """
    native_quantum_ms = 400
    input_sample_rate = 16000
    output_sample_rate = 24000

    def __init__(self, module, config):
        if config.get("owned_hf_runtime", False):
            from .lychee import OwnedLycheeRuntime
            self.runtime = OwnedLycheeRuntime(config["backend_args"])
        else:
            factory = getattr(module, config.get("factory_name", "build_logical_lychee_runtime"), None)
            if not callable(factory):
                raise SessionProtocolError(
                    "Lychee needs a synchronous owned runtime factory retaining HF session and token2wav "
                    "state; legacy /chunk + SSE is not a pause barrier. Real Lychee binding is not validated.")
            self.runtime = factory(config["backend_args"])
        required = {"persistent_kv", "persistent_codec", "synchronous_audio_barrier", "no_background_generation"}
        if not required.issubset(set(self.runtime.capabilities)):
            raise SessionProtocolError("Lychee runtime lacks complete context/barrier capabilities")

    def start(self, *, seed, max_media_ms):
        self.runtime.start(seed=seed, max_media_ms=max_media_ms, system_prompt=None)

    def step(self, pcm16):
        value = self.runtime.advance_pcm(pcm16)
        self.runtime.barrier()
        if not isinstance(value, NativeOutput):
            raise SessionProtocolError("Lychee factory must return NativeOutput")
        return value

    def finish(self):
        self.runtime.barrier()
        self.runtime.close()


def build_native_backend(config: dict[str, Any]):
    if config.get("system_prompt") is not None:
        raise SessionProtocolError("only original official model defaults are allowed")
    module = None if config.get("model") == "lychee" and config.get("owned_hf_runtime") else _module(config["loader_module"], config["loader_sha256"])
    classes = {"minicpm": MiniCPMNative, "personaplex": PersonaPlexNative,
               "raon": RaonNative, "lychee": LycheeNative}
    if config["model"] not in classes:
        raise SessionProtocolError("unknown native model")
    return classes[config["model"]](module, config)
