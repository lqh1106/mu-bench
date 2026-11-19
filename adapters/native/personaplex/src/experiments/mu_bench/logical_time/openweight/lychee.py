"""Owned HF V9 + synchronous Token2Wav lifecycle, no existing HTTP server.

The public app's raw event -> StreamingDecoder -> vocoder hop/lookahead logic
is preserved, but its asynchronous TTS queue is replaced by direct calls. Thus
every advance returns with all eligible synthesis complete; short unrenderable
lookahead tails stay in the same object without being force-flushed on pause.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from .session import NativeOutput, SessionProtocolError


class SynchronousToken2Wav:
    """Same buffering thresholds as official RealtimeTTSPool._worker_loop."""
    def __init__(self, model, *, prompt_wav: str, hop: int, lookahead: int):
        self.model = model
        self.prompt_wav = prompt_wav
        self.hop = hop
        self.lookahead = lookahead
        self.tokens = []
        self.initialized = False
        self.event_active = False
        self.calls = 0
        if hop <= 0 or lookahead < 0:
            raise SessionProtocolError("invalid token2wav hop/lookahead")

    def event_start(self):
        self.tokens.clear()
        self.initialized = False
        self.event_active = True

    def _synthesize(self, tokens, last):
        self.calls += 1
        return bytes(self.model.stream(tokens, prompt_wav=self.prompt_wav, last_chunk=last))

    def audio_chunk(self, tokens):
        self.tokens.extend(int(t) for t in tokens if 0 <= int(t) < 6561)
        if not self.initialized:
            if not self.tokens:
                return b""
            self.model.set_stream_cache(self.prompt_wav)
            self.initialized = True
        output = bytearray()
        window = self.hop + self.lookahead
        # Strict > follows the official pool, not a hand-tuned latency change.
        while len(self.tokens) > window:
            current = self.tokens[:window]
            self.tokens = self.tokens[self.hop:]
            output.extend(self._synthesize(current, False))
        return bytes(output)

    def event_end(self, *, interrupt=False):
        output = b""
        if not interrupt and self.tokens:
            output = self._synthesize(list(self.tokens), True)
        self.tokens.clear()
        self.initialized = False
        self.event_active = False
        return output

    def barrier(self):
        # No task queue or background vocoder thread exists. All stream() calls
        # above are synchronous. This method MUST NOT flush lookahead tails.
        return {"pending_background_tasks": 0, "retained_lookahead_tokens": len(self.tokens)}


class LycheeEventEngine:
    """Dependency-injected core; exercised with fake native generator/codec."""
    capabilities = {"persistent_kv", "persistent_codec", "synchronous_audio_barrier", "no_background_generation"}

    def __init__(self, stream_session, decoder, vocoder, *, synchronize=lambda: None):
        self.stream = stream_session
        self.decoder = decoder
        self.vocoder = vocoder
        self.synchronize = synchronize
        self.advancing = False
        self.closed = False

    def advance_float(self, audio):
        if self.closed or self.advancing:
            raise SessionProtocolError("Lychee session is closed or already advancing")
        self.advancing = True
        output = bytearray()
        diagnostic = []
        controls = []
        try:
            raw = self.stream.advance_stream(audio, emit_generation_complete=False,
                                             audio_is_incremental=True, flush_audio_tail=False)
            for event in raw:
                for decoded in self.decoder.feed(event):
                    kind = decoded.get("type")
                    if kind == "event_start":
                        self.vocoder.event_start()
                    elif kind == "audio_chunk":
                        output.extend(self.vocoder.audio_chunk(decoded["stoken_ids"]))
                    elif kind == "event_end":
                        interrupted = bool(decoded.get("interrupt", False))
                        tail = self.vocoder.event_end(interrupt=interrupted)
                        if interrupted:
                            # The native control decision cancels generated but
                            # not yet committed audio, never prior played PCM.
                            output.clear()
                            controls.append({"type": "audio_interrupt",
                                             "reason": decoded.get("interrupt_reason") or "model_control"})
                        else:
                            output.extend(tail)
                    elif kind == "text_delta":
                        diagnostic.append(str(decoded.get("delta", decoded.get("text", ""))))
                    elif kind == "state_change":
                        controls.append({"type": "native_state", "from": decoded.get("from"),
                                         "to": decoded.get("to"), "interrupt": bool(decoded.get("interrupt", False))})
            self.vocoder.barrier()
            self.synchronize()
            return NativeOutput(bytes(output), "".join(diagnostic), controls)
        finally:
            self.advancing = False

    def barrier(self):
        if self.advancing:
            raise SessionProtocolError("cannot acknowledge pause during native advance")
        self.synchronize()
        return self.vocoder.barrier()

    def close(self):
        self.barrier()
        self.stream.close()
        self.closed = True


class OwnedLycheeRuntime:
    capabilities = LycheeEventEngine.capabilities

    def __init__(self, config):
        self.config = dict(config)
        self.engine = None
        repo = Path(config["repo_path"]).resolve()
        # Pin the actual deployed sources; version labels alone are insufficient.
        required = ("lychee_fd/runtime/hf_v9_realtime.py", "lychee_fd/runtime/vllm_generation.py")
        hashes = config.get("source_sha256", {})
        for relative in required:
            observed = hashlib.sha256((repo/relative).read_bytes()).hexdigest()
            if hashes.get(relative) != observed:
                raise SessionProtocolError(f"Lychee native source not pinned or changed: {relative}")
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(config.get("stepaudio_source", repo/"third_party/Step-Audio2")))

    def start(self, *, seed, max_media_ms, system_prompt=None):
        if system_prompt is not None:
            raise SessionProtocolError("Lychee official default only")
        import torch
        # The public Lychee checkpoint is served through its official vLLM
        # integration. The HF V9 path expects a separate lychee_fd.models
        # package that is not present in this checkpoint/runtime bundle.
        from lychee_fd.runtime.vllm_generation import VLLMGenerationFramework
        from lychee_fd.runtime.vllm_generation import StreamingDecoder
        from token2wav import Token2wav
        from .native import _seed
        _seed(torch, seed)
        self.torch = torch
        config = self.config
        # Defaults mirror official HF V9 realtime app. No identity/permissions
        # prefix is injected; self-introductions arrive as ordinary user audio.
        self.framework = VLLMGenerationFramework(
            model_type="FD", model_path=str(config["model_path"]), device="cuda",
            attn_implementation=config.get("attn_implementation", "eager"),
            torch_dtype=torch.bfloat16, align_audio_input=True, allowing_backchannel=False,
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.60)),
            max_model_len=int(config.get("max_model_len", 4096)),
            enforce_eager=True, enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_num_batched_tokens=int(config.get("max_num_batched_tokens", 1024)))
        self.token2wav = Token2wav(str(config["token2wav_path"]))
        stream = self.framework.create_incremental_stream_session(
            prefix=None, initial_listening_state="l", start_speak_token_factor=1.2,
            start_listen_token_factor=1.2, end_speak_token_factor=1.0,
            audio_incremental_mode=True)
        decoder = StreamingDecoder(self.framework.tokenizer, self.framework,
                                   tts_chunk_size=int(config.get("decoder_tts_chunk_size", 10)),
                                   end_event_on_generation_complete=False)
        vocoder = SynchronousToken2Wav(self.token2wav, prompt_wav=str(config["ref_audio"]),
                                      hop=int(config.get("vocoder_hop_size", 25)),
                                      lookahead=int(getattr(self.token2wav.flow, "pre_lookahead_len", 0)))
        self.engine = LycheeEventEngine(stream, decoder, vocoder, synchronize=torch.cuda.synchronize)

    def advance_pcm(self, pcm16):
        from .native import _float
        with self.torch.inference_mode():
            return self.engine.advance_float(_float(pcm16))

    def barrier(self):
        return self.engine.barrier()

    def close(self):
        # No synthetic EOS or audio tail is forced at finish. Any desired tail
        # must be reached by explicitly advancing observed logical silence.
        self.engine.close()
