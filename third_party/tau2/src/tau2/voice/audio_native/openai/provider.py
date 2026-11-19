"""
OpenAI Realtime API provider for end-to-end voice/text processing.
"""

import asyncio
import base64
import json
import os
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import websockets
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from tau2.config import (
    DEFAULT_OPENAI_NOISE_REDUCTION,
    DEFAULT_OPENAI_REALTIME_BASE_URL,
    DEFAULT_OPENAI_REALTIME_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_VAD_THRESHOLD,
    DEFAULT_OPENAI_VOICE,
)
from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT, AudioFormat
from tau2.environment.tool import Tool
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    TimeoutEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.utils.openai_utils import audio_format_to_openai

load_dotenv()

PINE_MODEL_PREFIX = "pine-"
PINE_API_KEY_ENV = "PINE_API_KEY"
PINE_REALTIME_BASE_URL_ENV = "PINE_REALTIME_BASE_URL"


class OpenAIVADMode(str, Enum):
    """Voice Activity Detection modes supported by OpenAI's Realtime API.

    Attributes:
        SERVER_VAD: Server-side VAD using audio level thresholds and silence detection.
        SEMANTIC_VAD: Semantic-aware VAD that understands speech patterns and pauses.
        MANUAL: Manual turn detection where the client explicitly commits audio turns.
    """

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"
    MANUAL = "manual"


## TODO: We should have enum to specify output modality (text, audio, text_and_audio).
## TODO: Not sure where speech_in_speech_out and speech_in_text_out should go.


class OpenAIVADConfig(BaseModel):
    """Configuration for OpenAI's Voice Activity Detection.

    Configures how the API detects when the user has finished speaking.
    Different parameters apply depending on the selected mode.

    Attributes:
        mode: The VAD mode to use. Defaults to SERVER_VAD.
        threshold: Audio level threshold for SERVER_VAD (0.0-1.0).
            Higher values require louder speech to trigger. Default: 0.5.
        prefix_padding_ms: Milliseconds of audio to include before detected
            speech start (SERVER_VAD only). Default: 300.
        silence_duration_ms: Milliseconds of silence required to end a turn
            (SERVER_VAD only). Default: 500.
        eagerness: How eagerly to end turns for SEMANTIC_VAD mode.
            One of "low", "medium", "high". Default: "medium".
    """

    mode: OpenAIVADMode = OpenAIVADMode.SERVER_VAD
    threshold: float = DEFAULT_OPENAI_VAD_THRESHOLD
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    eagerness: str = "medium"  # For semantic_vad mode


class OpenAIRealtimeProvider:
    """OpenAI Realtime API provider with WebSocket-based communication.

    This provider manages a persistent WebSocket connection to OpenAI's Realtime API,
    enabling real-time bidirectional communication for voice and text processing.
    It supports configurable Voice Activity Detection (VAD), tool/function calling,
    and both audio and text modalities.

    Attributes:
        base_url: The WebSocket endpoint selected for the requested model.
        DEFAULT_MODEL: The default model to use for realtime sessions.
        api_key: The OpenAI API key for authentication.
        model: The model identifier to use for the session.
        ws: The active WebSocket connection, or None if disconnected.

    Example:
        ```python
        provider = OpenAIRealtimeProvider()
        await provider.connect()
        await provider.configure_session(
            system_prompt="You are a helpful assistant.",
            tools=[],
            vad_config=OpenAIVADConfig(),
            modality="text"
        )
        async for event in provider.receive_events():
            print(event)
        await provider.disconnect()
        ```
    """

    BASE_URL = DEFAULT_OPENAI_REALTIME_BASE_URL
    DEFAULT_MODEL = DEFAULT_OPENAI_REALTIME_MODEL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize the OpenAI Realtime provider.

        Args:
            api_key: Explicit API key. If not provided, Pine models read from
                PINE_API_KEY and OpenAI models read from OPENAI_API_KEY.
            model: Model identifier to use. Defaults to DEFAULT_MODEL.
            reasoning_effort: Reasoning effort for thinking models ("minimal",
                "low", "medium", "high"). If None, not sent to the API.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.model = model or self.DEFAULT_MODEL
        if self.model.startswith(PINE_MODEL_PREFIX):
            self.base_url = os.environ.get(PINE_REALTIME_BASE_URL_ENV)
            self.api_key = api_key or os.environ.get(PINE_API_KEY_ENV)
            if not self.base_url:
                raise ValueError(
                    "Pine Realtime base URL not provided. Set "
                    f"{PINE_REALTIME_BASE_URL_ENV}."
                )
            if not self.api_key:
                raise ValueError(f"Pine API key not provided. Set {PINE_API_KEY_ENV}.")
        else:
            self.base_url = self.BASE_URL
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY env var."
                )

        self.reasoning_effort = reasoning_effort
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_vad_config: Optional[OpenAIVADConfig] = None
        self._audio_format: AudioFormat = TELEPHONY_AUDIO_FORMAT
        self.session_id: Optional[str] = None
        # Drain downlink frames continuously.  If the 200 ms media tick itself
        # owns ws.recv(), a fast audio response can fill the WebSocket/proxy
        # receive buffers and backpressure the simultaneous user-audio send.
        self._event_queue: Optional[asyncio.Queue] = None
        self._receiver_task: Optional[asyncio.Task] = None
        # Serialize all post-configuration client messages through a permanent
        # sender task.  Media ticks must never wait for the WebSocket transport:
        # a slow proxy or a full TCP buffer would otherwise stop the local media
        # clock and delay every following 200 ms input chunk.
        self._outbound_queue: Optional[asyncio.Queue[str]] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._sender_error: Optional[BaseException] = None
        self._max_outbound_queue_depth: int = 0

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active.

        Returns:
            True if connected and the WebSocket is in OPEN state, False otherwise.
        """
        if self.ws is None:
            return False
        from websockets.protocol import State

        return self.ws.state == State.OPEN

    @property
    def audio_format(self) -> AudioFormat:
        """Get the configured audio format.

        Returns:
            The AudioFormat configured for this session.
        """
        return self._audio_format

    @websocket_retry
    async def connect(self) -> None:
        """Establish a WebSocket connection to the OpenAI Realtime API.

        Opens a new WebSocket connection and waits for the session.created event
        to confirm successful connection. If already connected, this is a no-op.

        Raises:
            RuntimeError: If the initial handshake fails or receives unexpected response.
        """
        if self.is_connected:
            return

        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        self.ws = await websockets.connect(url, additional_headers=headers)

        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {data.get('type')}")

        # Store and log session ID for debugging with OpenAI
        session_data = data.get("session", {})
        self.session_id = session_data.get("id")
        logger.info(
            f"OpenAI Realtime API: session created (session_id={self.session_id})"
        )

    async def disconnect(self) -> None:
        """Close the WebSocket connection.

        Gracefully closes the WebSocket connection if one exists.
        Safe to call even if not connected.
        """
        for task_name in ("_sender_task", "_receiver_task"):
            task = getattr(self, task_name)
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            setattr(self, task_name, None)

        self._outbound_queue = None
        self._event_queue = None
        self._sender_error = None
        self._max_outbound_queue_depth = 0
        if self.ws:
            logger.info("OpenAI Realtime API: disconnecting WebSocket connection")
            await self.ws.close()
            self.ws = None
            logger.info("OpenAI Realtime API: WebSocket connection closed")

    async def _receiver_loop(self) -> None:
        """Continuously drain WebSocket frames into the media-tick event queue."""
        assert self.ws is not None
        assert self._event_queue is not None
        while self.is_connected:
            try:
                raw_message = await self.ws.recv()
                await self._event_queue.put(parse_realtime_event(json.loads(raw_message)))
            except asyncio.CancelledError:
                raise
            except websockets.ConnectionClosed as exc:
                await self._event_queue.put(
                    UnknownEvent(
                        type="error",
                        raw={
                            "error": (
                                "WebSocket closed while receiving events: "
                                f"code={exc.code}, reason={exc.reason or 'none'}"
                            )
                        },
                    )
                )
                break
            except Exception as exc:
                await self._event_queue.put(
                    UnknownEvent(type="error", raw={"error": str(exc)})
                )

    async def _sender_loop(self) -> None:
        """Continuously send queued client messages in their original order."""
        assert self.ws is not None
        assert self._outbound_queue is not None
        while self.is_connected:
            payload = await self._outbound_queue.get()
            try:
                await self.ws.send(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._sender_error = exc
                if self._event_queue is not None:
                    await self._event_queue.put(
                        UnknownEvent(
                            type="error",
                            raw={
                                "error": (
                                    "WebSocket failed while sending queued realtime "
                                    f"message: {type(exc).__name__}: {exc}"
                                )
                            },
                        )
                    )
                break
            finally:
                self._outbound_queue.task_done()

    def _ensure_receiver(self) -> None:
        """Start the downlink pump after session configuration has completed."""
        if not self.is_connected or self.ws is None:
            raise RuntimeError("Not connected to API")
        if self._event_queue is None:
            self._event_queue = asyncio.Queue()
        if self._receiver_task is None or self._receiver_task.done():
            self._receiver_task = asyncio.create_task(self._receiver_loop())

    def _ensure_sender(self) -> None:
        """Start the ordered uplink pump after session configuration."""
        if not self.is_connected or self.ws is None:
            raise RuntimeError("Not connected to API")
        if self._sender_error is not None:
            raise RuntimeError(
                "Realtime outbound sender previously failed"
            ) from self._sender_error
        if self._outbound_queue is None:
            self._outbound_queue = asyncio.Queue()
        if self._sender_task is None or self._sender_task.done():
            self._sender_task = asyncio.create_task(self._sender_loop())

    def _enqueue_json(self, message: Dict) -> None:
        """Queue a post-configuration message without blocking the media tick."""
        self._ensure_sender()
        assert self._outbound_queue is not None
        self._outbound_queue.put_nowait(json.dumps(message))
        self._max_outbound_queue_depth = max(
            self._max_outbound_queue_depth, self._outbound_queue.qsize()
        )

    @property
    def outbound_queue_depth(self) -> int:
        """Number of messages waiting for the WebSocket sender."""
        return self._outbound_queue.qsize() if self._outbound_queue is not None else 0

    @property
    def max_outbound_queue_depth(self) -> int:
        """Largest observed uplink backlog for the current session."""
        return self._max_outbound_queue_depth

    def _build_turn_detection_config(
        self, vad_config: OpenAIVADConfig
    ) -> Optional[Dict]:
        """Build the turn detection configuration for the API.

        Converts the internal VAD configuration to the format expected by
        OpenAI's Realtime API.

        Args:
            vad_config: The VAD configuration to convert.

        Returns:
            A dictionary with turn detection settings, or None for manual mode.

        Raises:
            ValueError: If the VAD mode is unknown.
        """
        if vad_config.mode == OpenAIVADMode.MANUAL:
            return None
        elif vad_config.mode == OpenAIVADMode.SERVER_VAD:
            return {
                "type": "server_vad",
                "threshold": vad_config.threshold,
                "prefix_padding_ms": vad_config.prefix_padding_ms,
                "silence_duration_ms": vad_config.silence_duration_ms,
            }
        elif vad_config.mode == OpenAIVADMode.SEMANTIC_VAD:
            return {
                "type": "semantic_vad",
                "eagerness": vad_config.eagerness,
            }
        else:
            raise ValueError(f"Unknown VAD mode: {vad_config.mode}")

    def _format_tools_for_api(self, tools: List[Tool]) -> List[Dict]:
        """Format tools for the OpenAI Realtime API.

        Converts internal Tool objects to the format expected by the API,
        extracting the function name, description, and parameters from
        each tool's OpenAI schema.

        Args:
            tools: List of Tool objects to format.

        Returns:
            List of dictionaries in OpenAI's tool format.
        """
        formatted_tools = []
        for tool in tools:
            schema = tool.openai_schema
            formatted_tools.append(
                {
                    "type": "function",
                    "name": schema["function"]["name"],
                    "description": schema["function"]["description"],
                    "parameters": schema["function"]["parameters"],
                }
            )
        return formatted_tools

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: OpenAIVADConfig,
        modality: str = "text",
        audio_format: Optional[AudioFormat] = None,
    ) -> None:
        """Configure the realtime session with instructions, tools, and settings.

        Sets up the session with the provided system prompt, available tools,
        VAD configuration, and modality settings. Waits for confirmation from
        the API before returning.

        Args:
            system_prompt: The system instructions for the assistant.
            tools: List of tools available for the assistant to use.
            vad_config: Voice Activity Detection configuration.
            modality: The input/output modality. One of:
                - "text": Text-only input and output.
                - "audio": Audio input and audio output (with text transcription).
                - "audio_in_text_out": Audio input with text-only output.
            audio_format: Audio format for input/output. Defaults to telephony
                (8kHz μ-law). Must be compatible with OpenAI Realtime API:
                g711_ulaw (8kHz), g711_alaw (8kHz), or pcm16 (24kHz).

        Raises:
            RuntimeError: If not connected or if session configuration fails.
            ValueError: If an unknown modality is specified or audio format unsupported.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")

        if modality == "text":
            modalities = ["text"]
        elif modality == "audio":
            modalities = ["audio"]
        elif modality == "audio_in_text_out":
            modalities = ["text"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Default to telephony format if not specified
        if audio_format is None:
            audio_format = TELEPHONY_AUDIO_FORMAT

        # Store audio format for reference
        self._audio_format = audio_format

        audio_fmt = audio_format_to_openai(audio_format)

        session = {
            "type": "realtime",
            "instructions": system_prompt,
            "output_modalities": modalities,
            "tools": self._format_tools_for_api(tools),
            "tool_choice": "auto",
        }

        if self.reasoning_effort is not None:
            session["reasoning"] = {"effort": self.reasoning_effort}

        if modality in ("audio", "audio_in_text_out"):
            session["audio"] = {
                "input": {
                    "format": audio_fmt,
                    "transcription": {
                        "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
                        "language": "en",
                    },
                    "noise_reduction": {"type": DEFAULT_OPENAI_NOISE_REDUCTION},
                    "turn_detection": self._build_turn_detection_config(vad_config),
                },
            }

        if modality == "audio":
            session.setdefault("audio", {})["output"] = {
                "format": audio_fmt,
                "voice": DEFAULT_OPENAI_VOICE,
            }

        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            event_type = data.get("type", "")

            if event_type == "session.updated":
                self._current_vad_config = vad_config
                break
            elif event_type == "error":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Session configuration failed: {error_msg}")

        # Configuration is the final operation that reads directly from the
        # socket.  From here on, permanent sender/receiver tasks exclusively own
        # WebSocket I/O and ticks interact only with queues.
        self._ensure_receiver()
        self._ensure_sender()

    async def send_audio(self, audio_data: bytes) -> None:
        """Append audio data to the input audio buffer.

        Sends raw audio bytes to the API's input buffer. The audio is
        base64-encoded before transmission. With server VAD, the API commits
        turns automatically from buffered audio.

        Args:
            audio_data: Raw audio bytes in the configured input format (g711_ulaw).

        Raises:
            RuntimeError: If not connected to the API.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        self._enqueue_json(message)

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        """Send a tool result and optionally request a continuation response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        self._enqueue_json(item_create)

        if request_response:
            self._enqueue_json({"type": "response.create"})

    async def truncate_item(
        self,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Tell the server how much audio was played before interruption."""
        if not self.is_connected:
            raise RuntimeError("Not connected to API")

        truncate_event = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": audio_end_ms,
        }
        self._enqueue_json(truncate_event)
        logger.debug(
            f"Truncate sent: item_id={item_id}, content_index={content_index}, "
            f"audio_end_ms={audio_end_ms}"
        )

    async def receive_events(self) -> AsyncGenerator[BaseRealtimeEvent, None]:
        """Async generator yielding parsed events from the WebSocket.

        Yields TimeoutEvent when no message arrives within 10ms.
        Raises RuntimeError if the connection closes unexpectedly.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        self._ensure_receiver()
        assert self._event_queue is not None

        while self.is_connected:
            try:
                yield await asyncio.wait_for(self._event_queue.get(), timeout=0.01)
            except asyncio.TimeoutError:
                yield TimeoutEvent(type="timeout")
            except Exception as e:
                logger.error(
                    f"OpenAI Realtime API: Error receiving event: {type(e).__name__}: {e}"
                )
                yield UnknownEvent(type="error", raw={"error": str(e)})

    async def receive_events_for_duration(
        self, duration_seconds: float
    ) -> List[BaseRealtimeEvent]:
        """Receive events for a specified duration.

        Collects all non-timeout events within the time window.

        Args:
            duration_seconds: How long to collect events.

        Returns:
            List of events received during the duration.
        """
        events = []
        end_time = asyncio.get_event_loop().time() + duration_seconds

        async for event in self.receive_events():
            if not isinstance(event, TimeoutEvent):
                events.append(event)

            if asyncio.get_event_loop().time() >= end_time:
                break

        return events

    async def receive_event_until(
        self, deadline: float
    ) -> Optional[BaseRealtimeEvent]:
        """Return the next queued event, waiting no later than ``deadline``.

        The permanent receiver owns ``ws.recv``.  Keeping the deadline here
        means an idle downlink consumes the rest of the media tick, while an
        event flood is handled incrementally by the adapter and can be left in
        the queue for the next tick.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        self._ensure_receiver()
        assert self._event_queue is not None

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        try:
            return await asyncio.wait_for(self._event_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            return None
