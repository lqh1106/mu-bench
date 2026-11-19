"""CPU-only deterministic stateful backend. Not a model or a score source."""
import random
import struct
from .session import NativeOutput


class DeterministicNative:
    input_sample_rate = output_sample_rate = 8000

    def __init__(self, quantum_ms=80):
        self.native_quantum_ms = quantum_ms
        self.steps = 0
        self.starts = 0
        self.closed = False
        self.kv = []
        self.codec_history = bytearray()

    def start(self, *, seed, max_media_ms):
        self.starts += 1
        self.rng = random.Random(seed)

    def step(self, pcm16):
        assert not self.closed
        self.steps += 1
        self.kv.append(hash(pcm16))
        self.codec_history.extend(pcm16[-4:])
        value = self.rng.randint(-30000, 30000)
        return NativeOutput(struct.pack("<h", value) * (len(pcm16)//2),
                            diagnostic_text=f"FUTURE_DIAGNOSTIC_{value}")

    def finish(self):
        self.closed = True
