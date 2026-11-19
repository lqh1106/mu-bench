"""Freeze-Omni SocketIO bridge backend for mu-Bench logical-time wiring (2026-10-15).

Talks to the already-running Freeze-Omni Flask-SocketIO server on the same worker.
Does not reload weights; does not modify shared openweight loaders.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from typing import Any


class FreezeOmniSocketBackend:
    """Client-side bridge: PCM16 frames <-> local Freeze-Omni SocketIO server."""

    INPUT_SR = 16000
    OUTPUT_SR = 24000
    QUANTUM_MS = 160
    CHUNK_SAMPLES = 2560

    def __init__(self, args: argparse.Namespace) -> None:
        import numpy as np

        self.np = np
        self.base_url = str(getattr(args, "socketio_url", "https://127.0.0.1:8222"))
        self.verify_ssl = bool(getattr(args, "verify_ssl", False))
        self.sio = None
        self.out_q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._connected = False
        self.pending_out = bytearray()
        self.frame_size = self.CHUNK_SAMPLES

    def connect(self) -> None:
        import socketio

        if self._connected and self.sio is not None and getattr(self.sio, "connected", False):
            return
        sio = socketio.Client(reconnection=True, reconnection_attempts=5, ssl_verify=self.verify_ssl)

        @sio.on("audio")
        def _on_audio(data):  # noqa: ANN001
            if data is None:
                return
            raw = bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)
            if raw:
                self.out_q.put(raw)

        @sio.on("stop_tts")
        def _on_stop(_data=None):  # noqa: ANN001
            while not self.out_q.empty():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    break
            self.pending_out.clear()

        @sio.on("disconnect")
        def _on_disconnect():
            self._connected = False

        sio.connect(
            self.base_url,
            transports=["websocket"],
            wait_timeout=60,
            socketio_path="socket.io",
            namespaces=["/"],
        )
        self.sio = sio
        self._connected = True
        try:
            sio.emit("recording-started", namespace="/")
        except Exception:
            pass

    def _ensure_connected(self) -> None:
        if self._connected and self.sio is not None and getattr(self.sio, "connected", False):
            return
        try:
            if self.sio is not None:
                try:
                    self.sio.disconnect()
                except Exception:
                    pass
        finally:
            self.sio = None
            self._connected = False
        self.connect()

    def reset_stream(self) -> None:
        while not self.out_q.empty():
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                break
        self.pending_out.clear()
        self._ensure_connected()
        if self.sio is not None:
            try:
                self.sio.emit("recording-started", namespace="/")
            except Exception:
                self._connected = False
                self._ensure_connected()

    def step_pcm(self, pcm16: bytes) -> bytes:
        """Send one 160ms @16k user frame; return up to 160ms @24k assistant PCM."""
        import time as _time

        self._ensure_connected()
        assert self.sio is not None
        need = self.CHUNK_SAMPLES * 2
        if len(pcm16) < need:
            pcm16 = pcm16 + bytes(need - len(pcm16))
        elif len(pcm16) > need:
            pcm16 = pcm16[:need]
        payload = json.dumps({"audio": list(pcm16), "sample_rate": self.INPUT_SR})
        for attempt in range(2):
            try:
                with self._lock:
                    self.sio.emit("audio", payload, namespace="/")
                break
            except Exception:
                if attempt == 0:
                    self._connected = False
                    self._ensure_connected()
                    continue
                raise
        _time.sleep(0.01)
        while not self.out_q.empty():
            try:
                self.pending_out.extend(self.out_q.get_nowait())
            except queue.Empty:
                break
        out_need = int(self.OUTPUT_SR * self.QUANTUM_MS / 1000) * 2
        if len(self.pending_out) >= out_need:
            out = bytes(self.pending_out[:out_need])
            del self.pending_out[:out_need]
            return out
        have = bytes(self.pending_out)
        self.pending_out.clear()
        if len(have) < out_need:
            have = have + bytes(out_need - len(have))
        return have[:out_need]

    def close(self) -> None:
        if self.sio is not None:
            try:
                self.sio.emit("recording-stopped", namespace="/")
            except Exception:
                pass
            try:
                self.sio.disconnect()
            except Exception:
                pass
        self.sio = None
        self._connected = False
