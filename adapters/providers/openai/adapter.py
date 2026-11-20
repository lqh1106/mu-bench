"""A corrected logical-time wrapper around tau2's OpenAI Realtime adapter.

The upstream tau2 adapter deliberately exposes a synchronous, tick-shaped API.
This subclass keeps that API and the same persistent provider session, while
fixing interruption bookkeeping that must be expressed in *conversation item*
time rather than in one-tick time.

Logical pause is intentionally local: no audio is sent and no provider events
are consumed while paused.  It does not claim to freeze server-side inference.
With server VAD the provider owns response cancellation; this adapter never
sends an unscoped ``response.cancel`` that could race with a newer response.
"""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
from copy import deepcopy
import json
import threading
import time
from typing import Any, Callable, Iterable, Optional

from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioTranscriptDeltaEvent,
    ConversationItemTruncatedEvent,
    ErrorEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseCancelledEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
)
from tau2.voice.audio_native.tick_result import TickResult


StrictTranscriptFn = Callable[[bytes, Any, int], str]


class LogicalTimeOpenAIAdapter(DiscreteTimeOpenAIAdapter):
    """OpenAI discrete-time adapter with item-relative interruption state.

    The public streaming surface intentionally matches the official adapter:
    ``connect``, ``disconnect``, ``send_tool_result`` and synchronous
    ``run_tick``.  Additional pause and audit methods are consumed by the
    single-session μ-Bench orchestrator.
    """

    _BENIGN_TERMINAL_ERROR_FRAGMENTS = (
        "no active response",
        "not an active response",
        "already completed",
        "already complete",
        "already cancelled",
        "already canceled",
    )

    def __init__(
        self,
        *args: Any,
        strict_transcript_fn: StrictTranscriptFn | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._strict_transcript_fn = strict_transcript_fn
        self._played_bytes_by_item: dict[str, int] = defaultdict(int)
        self._last_played_item_id: str | None = None
        self._item_response_id: dict[str, str] = {}
        self._active_response_ids: set[str] = set()
        self._terminal_response_ids: set[str] = set()
        self._suppressed_response_ids: set[str] = set()
        self._suppressed_item_ids: set[str] = set()
        self._delete_sent_item_ids: set[str] = set()
        self._anonymous_active_responses = 0
        self._created_response_ids: set[str] = set()
        self._created_event_ids: set[str] = set()
        self._pending_truncate_acks: dict[str, list[int]] = defaultdict(list)
        self._pre_ack_race_keys: set[tuple[int, str, str]] = set()
        self._deferred_provider_events: list[Any] = []
        self._tool_races: list[dict[str, Any]] = []
        self._interruption_epoch = 0
        self._interruption_open = False
        self._paused = False
        self._pause_started_wall: float | None = None
        self._pause_logical_time_ms: int | None = None
        self._audit_records: list[dict[str, Any]] = []
        self._audit_sequence = 0
        self._audit_lock = threading.RLock()
        self._current_logical_time_ms = 0
        self._tool_output_epoch = 0
        self._tool_output_unacked: dict[str, int] = {}
        self._tool_output_committed_epoch = 0
        self._continuation_wanted_epoch = 0
        self._continuation_covered_epoch = 0
        self._continuation_inflight = False
        self._continuation_request_ids: list[str] = []
        self._continuation_serial = 0
        self._continuation_not_before = 0.0

    # ------------------------------------------------------------------
    # Public compatibility and logical-pause contract
    # ------------------------------------------------------------------

    @property
    def audit_records(self) -> list[dict[str, Any]]:
        """A defensive snapshot of pause/interruption/race evidence."""
        with self._audit_lock:
            return deepcopy(self._audit_records)

    @property
    def tool_races(self) -> list[dict[str, Any]]:
        return deepcopy(self._tool_races)

    @property
    def pending_truncate_acks(self) -> dict[str, list[int]]:
        return {key: list(values) for key, values in self._pending_truncate_acks.items()}

    @property
    def logical_clock_paused(self) -> bool:
        return self._paused

    @property
    def transcript_alignment(self) -> str:
        return (
            "strict_played_audio_callback"
            if self._strict_transcript_fn is not None
            else "upstream_proportional_not_word_aligned"
        )

    def pause_logical_clock(
        self,
        reason: str,
        logical_time_ms: int | None = None,
    ) -> None:
        """Freeze benchmark media progress without touching provider state."""
        if self._paused:
            raise RuntimeError("logical clock is already paused")
        self._paused = True
        self._pause_started_wall = time.monotonic()
        self._pause_logical_time_ms = (
            self._current_logical_time_ms
            if logical_time_ms is None
            else int(logical_time_ms)
        )
        self._audit(
            "logical_pause_started",
            reason=reason,
            logical_time_ms=self._pause_logical_time_ms,
            provider_action="none",
        )

    def resume_logical_clock(
        self,
        reason: str = "planner_complete",
        logical_time_ms: int | None = None,
    ) -> None:
        """Resume media progress at the unchanged logical frontier."""
        if not self._paused:
            raise RuntimeError("logical clock is not paused")
        now = time.monotonic()
        started = self._pause_started_wall or now
        paused_at = self._pause_logical_time_ms
        resumed_at = (
            self._pause_logical_time_ms
            if logical_time_ms is None
            else int(logical_time_ms)
        )
        if paused_at is None or resumed_at != paused_at:
            self._audit(
                "logical_pause_resume_rejected",
                reason=reason,
                paused_logical_time_ms=paused_at,
                requested_logical_time_ms=resumed_at,
                provider_action="none",
            )
            raise ValueError(
                "resume must use the unchanged logical-time frontier "
                f"({paused_at} ms), got {resumed_at} ms"
            )
        self._audit(
            "logical_pause_resumed",
            reason=reason,
            logical_time_ms=resumed_at,
            wall_pause_ms=round((now - started) * 1000, 3),
            provider_action="none",
        )
        self._paused = False
        self._pause_started_wall = None
        self._pause_logical_time_ms = None

    def run_tick(
        self,
        user_audio: bytes,
        tick_number: Optional[int] = None,
    ) -> TickResult:
        """Run one official tick and commit only actually retained item audio."""
        if self._paused:
            raise RuntimeError(
                "run_tick is forbidden while the logical clock is paused; "
                "resume without advancing logical time first"
            )
        effective_tick = self._tick_count if tick_number is None else tick_number
        self._current_logical_time_ms = int(effective_tick * self.tick_duration_ms)
        result = super().run_tick(user_audio=user_audio, tick_number=tick_number)

        # agent_audio_chunks excludes silence padding.  It is therefore the only
        # safe source for item-relative committed playback accounting.
        for chunk, item_id in result.agent_audio_chunks:
            if not item_id or not chunk:
                continue
            self._played_bytes_by_item[item_id] += len(chunk)
            self._last_played_item_id = item_id

        if not result.agent_audio_chunks and result.proportional_transcript:
            # A transcript without retained audio is never model-visible.
            self._audit(
                "unplayed_transcript_suppressed",
                tick_number=result.tick_number,
                characters=len(result.proportional_transcript),
            )
            result.proportional_transcript = ""

        if self._strict_transcript_fn is not None:
            played = b"".join(chunk for chunk, _ in result.agent_audio_chunks)
            result.proportional_transcript = (
                self._strict_transcript_fn(played, self.audio_format, result.tick_number)
                if played
                else ""
            )

        self._audit(
            "tick_completed",
            tick_number=result.tick_number,
            logical_time_ms=self._current_logical_time_ms,
            retained_audio_bytes=sum(len(chunk) for chunk, _ in result.agent_audio_chunks),
            transcript_alignment=self.transcript_alignment,
        )
        return result

    def configure_auto_response(
        self,
        enabled: bool,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Order a server-VAD policy update and wait for its acknowledgement.

        The transport-decoupled provider owns the WebSocket after initial
        configuration.  Its ordered sender must therefore carry this update;
        sending directly on ``ws`` could overtake queued audio or truncation.
        Non-configuration events observed while waiting are retained in exact
        receive order for the next logical tick.
        """
        if not self.is_connected:
            raise RuntimeError("cannot configure auto response before connect")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._bg_loop.run_coroutine(
            self._async_configure_auto_response(bool(enabled), float(timeout_seconds)),
            timeout=float(timeout_seconds) + 0.5,
        )

    async def _async_configure_auto_response(
        self,
        enabled: bool,
        timeout_seconds: float,
    ) -> None:
        websocket = self.provider.ws
        if websocket is None:
            raise RuntimeError("Realtime WebSocket is not connected")
        payload = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": enabled,
                            "interrupt_response": enabled,
                        }
                    }
                },
            },
        }
        enqueue = getattr(self.provider, "_enqueue_json", None)
        if callable(enqueue):
            enqueue(payload)
            transport = "provider_ordered_sender"
        else:
            await websocket.send(json.dumps(payload))
            transport = "direct_websocket_official_provider"
        self._audit(
            "auto_response_update_sent",
            enabled=enabled,
            transport=transport,
        )

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        deferred_count = 0
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self._audit(
                    "auto_response_update_timeout",
                    enabled=enabled,
                    deferred_event_count=deferred_count,
                    fail_closed=True,
                )
                raise TimeoutError(
                    "timed out waiting for Realtime session.updated acknowledgement"
                )

            events = await self._receive_configuration_events(deadline, remaining)
            if not events:
                continue
            acknowledged = False
            for event in events:
                event_type = getattr(event, "type", "unknown")
                if event_type == "session.updated" and not acknowledged:
                    acknowledged = True
                    continue
                self._raise_if_fatal_provider_event(event)
                self._deferred_provider_events.append(event)
                deferred_count += 1
                self._audit(
                    "configuration_interleaved_event_deferred",
                    event_type=event_type,
                    enabled=enabled,
                )
            if acknowledged:
                self._audit(
                    "auto_response_update_ack",
                    enabled=enabled,
                    deferred_event_count=deferred_count,
                )
                return

    async def _receive_configuration_events(
        self,
        deadline: float,
        remaining: float,
    ) -> list[Any]:
        receive_one = getattr(self.provider, "receive_event_until", None)
        if callable(receive_one):
            event = await receive_one(deadline)
            return [] if event is None else [event]
        return list(
            await self.provider.receive_events_for_duration(min(remaining, 0.1))
        )

    def is_output_drained(self) -> bool:
        """Conservative completion check for a Probe/tail boundary."""
        return not any(
            (
                self._buffered_agent_audio,
                self._active_response_ids,
                self._anonymous_active_responses,
                self._pending_truncate_acks,
                self._pending_tool_results,
                self._tool_output_unacked,
                self._continuation_inflight,
                self._continuation_wanted_epoch > self._continuation_covered_epoch,
                self._deferred_provider_events,
            )
        )

    def disconnect(self) -> None:
        if self._paused:
            self._audit(
                "disconnect_while_logically_paused",
                logical_time_ms=self._pause_logical_time_ms,
            )
            self._paused = False
            self._pause_started_wall = None
            self._pause_logical_time_ms = None
        super().disconnect()

    # ------------------------------------------------------------------
    # Corrected OpenAI event processing
    # ------------------------------------------------------------------

    async def _flush_pending_tool_results(self) -> None:
        # Every result is sent once. Only continuation requests are coalesced.
        for call_id, result_str, request_response, _is_error in self._pending_tool_results:
            self._tool_output_epoch += 1
            epoch = self._tool_output_epoch
            self._tool_output_unacked[call_id] = epoch
            await self.provider.send_tool_result(call_id, result_str, False)
            if request_response:
                self._continuation_wanted_epoch = epoch
                self._continuation_not_before = time.monotonic() + 0.2
            self._audit('tool_output_queued', call_id=call_id, epoch=epoch,
                        continuation_requested=request_response)
        self._pending_tool_results.clear()

    async def _maybe_request_continuation(self) -> None:
        if (self._continuation_wanted_epoch <= self._continuation_covered_epoch
                or self._continuation_inflight or self._tool_output_unacked
                or self._active_response_ids or self._anonymous_active_responses
                or self._paused or self._interruption_open or self._pending_truncate_acks
                or time.monotonic() < self._continuation_not_before):
            return
        self._continuation_serial += 1
        event_id = 'mu_continuation_' + str(self._continuation_serial)
        await self.provider.request_response(event_id)
        self._continuation_inflight = True
        self._continuation_request_ids = (self._continuation_request_ids + [event_id])[-16:]
        self._audit('continuation_requested', event_id=event_id,
                    committed_epoch=self._tool_output_committed_epoch)

    async def _execute_tick(
        self,
        user_audio: bytes,
        tick_number: int,
        result: TickResult,
        tick_start: float,
    ) -> None:
        """Use the fixed upstream tick contract even beside a patched tau2 tree.

        The benchmark's older Realtime runtime may have locally changed this
        method to enqueue one complete tick at once.  v1.4 is defined against
        the audited tau2 revision: provider-native 20 ms sends are paced while
        receive remains open for the rest of the logical tick.
        """

        async def receive_events() -> list[Any]:
            elapsed = asyncio.get_running_loop().time() - tick_start
            remaining = max(0.01, (self.tick_duration_ms / 1000) - elapsed)
            return await self.provider.receive_events_for_duration(remaining)

        deferred = self._deferred_provider_events
        self._deferred_provider_events = []
        _, received_events = await asyncio.gather(
            self._send_audio_chunked(
                user_audio,
                self.provider.send_audio,
                self._chunk_size,
            ),
            receive_events(),
        )
        for event in [*deferred, *received_events]:
            await self._process_event(result, event)
        await self._maybe_request_continuation()

    async def _process_event(self, result: TickResult, event: Any) -> None:
        event_type = getattr(event, "type", "unknown")

        if event_type in {'conversation.item.created', 'conversation.item.done'}:
            raw = getattr(event, 'raw', None) or {}
            item = getattr(event, 'item', None) or raw.get('item', {})
            call_id = item.get('call_id')
            if item.get('type') == 'function_call_output' and call_id in self._tool_output_unacked:
                epoch = self._tool_output_unacked.pop(call_id)
                # WebSocket event order, not wall time, proves server receipt.
                if not self._tool_output_unacked:
                    self._tool_output_committed_epoch = self._tool_output_epoch
                self._audit('tool_output_committed', call_id=call_id, epoch=epoch)


        if self._pending_truncate_acks and (
            event_type == "response.created" or event_type.startswith("response.output")
        ):
            self._record_pre_ack_provider_event(event)

        if isinstance(event, AudioDeltaEvent):
            response_id = getattr(event, "response_id", None)
            item_id = getattr(event, "item_id", None)
            if response_id and response_id not in self._terminal_response_ids:
                self._active_response_ids.add(response_id)
                if item_id:
                    self._item_response_id[item_id] = response_id
            if self._must_suppress(item_id=item_id, response_id=response_id):
                result.events.append(event)
                discarded = len(base64.b64decode(event.delta))
                result.truncated_audio_bytes += discarded
                if item_id and item_id not in self._suppressed_item_ids:
                    self._suppressed_item_ids.add(item_id)
                if item_id and item_id != self._last_played_item_id:
                    await self._delete_unplayed_item(item_id)
                self._audit(
                    "post_interrupt_audio_suppressed",
                    item_id=item_id,
                    response_id=response_id,
                    bytes=discarded,
                )
                return

        if isinstance(event, AudioTranscriptDeltaEvent):
            response_id = getattr(event, "response_id", None)
            item_id = getattr(event, "item_id", None)
            if response_id and response_id not in self._terminal_response_ids:
                self._active_response_ids.add(response_id)
                if item_id:
                    self._item_response_id[item_id] = response_id
            if self._must_suppress(item_id=item_id, response_id=response_id):
                result.events.append(event)
                self._audit(
                    "post_interrupt_transcript_suppressed",
                    item_id=item_id,
                    response_id=response_id,
                    characters=len(getattr(event, "delta", "") or ""),
                )
                return

        if isinstance(event, SpeechStartedEvent):
            await self._handle_speech_started(result, event)
            return

        if isinstance(event, SpeechStoppedEvent):
            await super()._process_event(result, event)
            self._interruption_open = False
            self._audit(
                "user_speech_stopped",
                audio_end_ms=getattr(event, "audio_end_ms", None),
            )
            return

        if event_type == "response.created":
            raw = getattr(event, "raw", None) or {}
            response_id = (getattr(event, "response_id", None)
                           or (raw.get("response") or {}).get("id"))
            event_id = getattr(event, "event_id", None) or raw.get("event_id")
            duplicate = bool((response_id and response_id in self._created_response_ids)
                             or (event_id and event_id in self._created_event_ids))
            if event_id:
                self._created_event_ids.add(event_id)
            if response_id:
                self._created_response_ids.add(response_id)
                if response_id not in self._terminal_response_ids:
                    self._active_response_ids.add(response_id)
            elif not duplicate:
                self._anonymous_active_responses += 1
            if not duplicate:
                self._continuation_inflight = False
                # A response created before output acknowledgement cannot consume it.
                self._continuation_covered_epoch = max(self._continuation_covered_epoch,
                                                        self._tool_output_committed_epoch)
            await super()._process_event(result, event)
            self._audit("response_created", response_id=response_id, event_id=event_id,
                        duplicate=duplicate, active_response_ids=sorted(self._active_response_ids),
                        anonymous_active_responses=self._anonymous_active_responses)
            return

        if isinstance(event, ResponseCancelledEvent):
            await super()._process_event(result, event)
            self._mark_response_terminal(getattr(event, "response_id", None), "cancelled")
            self._audit(
                "server_vad_cancelled_response",
                response_id=getattr(event, "response_id", None),
            )
            return

        if isinstance(event, ResponseDoneEvent):
            response_id = getattr(event, "response_id", None)
            status = getattr(event, "status", None) or "unknown"
            await super()._process_event(result, event)
            was_suppressed = bool(response_id and response_id in self._suppressed_response_ids)
            self._mark_response_terminal(response_id, status)
            if was_suppressed and status == "completed":
                self._audit(
                    "cancel_completed_race",
                    response_id=response_id,
                    disposition="benign_history_reconciled_by_truncate_delete",
                )
            return

        if isinstance(event, ConversationItemTruncatedEvent):
            await super()._process_event(result, event)
            self._record_truncate_ack(
                item_id=getattr(event, "item_id", None),
                audio_end_ms=getattr(event, "audio_end_ms", None),
            )
            return

        if isinstance(event, FunctionCallArgumentsDoneEvent):
            await super()._process_event(result, event)
            if self._interruption_open:
                race = {
                    "call_id": getattr(event, "call_id", None),
                    "name": getattr(event, "name", None),
                    "interruption_epoch": self._interruption_epoch,
                    "policy": "surfaced_unchanged_but_requires_runner_race_check",
                }
                self._tool_races.append(race)
                self._audit("tool_call_interruption_race", **race)
            return

        if (isinstance(event, ErrorEvent)
                and event.code == 'conversation_already_has_active_response'
                and getattr(event, 'related_event_id', None) in self._continuation_request_ids):
            # Reconcile only an identified request made by this scheduler.
            import re
            match = re.search(r'in progress: ([A-Za-z0-9_-]+)', event.message or '')
            if match and match[1] not in self._terminal_response_ids:
                self._active_response_ids.add(match[1])
            self._continuation_inflight = False
            self._audit('continuation_conflict_deferred', event_id=event.related_event_id,
                        active_response_id=match[1] if match else None,
                        wanted_epoch=self._continuation_wanted_epoch,
                        covered_epoch=self._continuation_covered_epoch,
                        tool_outputs_replayed=False)
            return

        if isinstance(event, ErrorEvent):
            await super()._process_event(result, event)
            self._raise_if_fatal_provider_event(event)
            return

        # The benchmark's transport-decoupled provider reports sender/receiver
        # failures as UnknownEvent(type="error", raw={...}).  They are not API
        # ErrorEvent instances, but allowing them through would turn a closed
        # socket into hundreds of empty logical ticks.
        if event_type == "error":
            self._raise_if_fatal_provider_event(event)
            return

        if event_type == "conversation.item.deleted":
            await super()._process_event(result, event)
            raw = getattr(event, "raw", None) or {}
            self._audit("unplayed_item_delete_ack", item_id=raw.get("item_id"))
            return

        await super()._process_event(result, event)

    async def _handle_speech_started(
        self,
        result: TickResult,
        event: SpeechStartedEvent,
    ) -> None:
        """Reconcile provider output with the audio actually played at barge-in."""
        result.events.append(event)
        result.vad_events.append("speech_started")
        self._interruption_epoch += 1
        self._interruption_open = True

        audio_start_ms = int(getattr(event, "audio_start_ms", None) or 0)
        within_tick_ms = audio_start_ms - result.cumulative_user_audio_at_tick_start_ms
        tick_ms = int(result.bytes_per_tick / result.bytes_per_second * 1000)
        within_tick_ms = max(0, min(within_tick_ms, tick_ms))
        byte_limit = int(within_tick_ms * result.bytes_per_second / 1000)

        original = list(result.agent_audio_chunks)
        buffered = list(self._buffered_agent_audio)
        kept, discarded = self._split_chunks_at(original, byte_limit)
        current_tick_played: dict[str, int] = defaultdict(int)
        current_item_id: str | None = None
        for chunk, item_id in kept:
            if item_id and chunk:
                current_tick_played[item_id] += len(chunk)
                current_item_id = item_id
        if current_item_id is None:
            current_item_id = self._last_played_item_id

        response_ids = {
            self._item_response_id[item_id]
            for _, item_id in [*original, *buffered]
            if item_id and item_id in self._item_response_id
        }
        self._suppressed_response_ids.update(response_ids)

        result.was_truncated = bool(original or self._buffered_agent_audio or current_item_id)
        result.interruption_audio_start_ms = audio_start_ms
        result.agent_audio_chunks = kept
        result.truncated_audio_bytes += sum(len(chunk) for chunk, _ in discarded)
        if buffered:
            result.truncated_audio_bytes += sum(
                len(chunk) for chunk, _ in buffered
            )
            self._buffered_agent_audio.clear()

        unplayed_item_ids: list[str] = []
        for _, item_id in [*discarded, *buffered]:
            if not item_id or item_id == current_item_id:
                continue
            if self._played_bytes_by_item.get(item_id, 0) == 0 and current_tick_played.get(item_id, 0) == 0:
                if item_id not in unplayed_item_ids:
                    unplayed_item_ids.append(item_id)

        truncate_end_ms: int | None = None
        if current_item_id is not None:
            result.skip_item_id = current_item_id
            self._suppressed_item_ids.add(current_item_id)
            played_bytes = self._played_bytes_by_item.get(current_item_id, 0)
            played_bytes += current_tick_played.get(current_item_id, 0)
            truncate_end_ms = int(played_bytes / result.bytes_per_second * 1000)
            await self.provider.truncate_item(
                item_id=current_item_id,
                content_index=0,
                audio_end_ms=truncate_end_ms,
            )
            self._pending_truncate_acks[current_item_id].append(truncate_end_ms)
            self._audit(
                "truncate_sent",
                item_id=current_item_id,
                content_index=0,
                audio_end_ms=truncate_end_ms,
                prior_item_played_bytes=self._played_bytes_by_item.get(current_item_id, 0),
                current_tick_item_played_bytes=current_tick_played.get(current_item_id, 0),
            )

        for item_id in unplayed_item_ids:
            self._suppressed_item_ids.add(item_id)
            self._utterance_transcripts.pop(item_id, None)
            await self._delete_unplayed_item(item_id)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                race = {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "interruption_epoch": self._interruption_epoch,
                    "policy": "same_tick_order_requires_runner_race_check",
                }
                self._tool_races.append(race)
                self._audit("tool_call_interruption_race", **race)

        self._audit(
            "speech_started_reconciled",
            audio_start_ms=audio_start_ms,
            within_tick_ms=within_tick_ms,
            current_item_id=current_item_id,
            truncate_end_ms=truncate_end_ms,
            deleted_unplayed_item_ids=unplayed_item_ids,
            cancel_strategy="server_vad_auto_no_unscoped_client_cancel",
        )

    @staticmethod
    def _split_chunks_at(
        chunks: Iterable[tuple[bytes, str | None]],
        byte_limit: int,
    ) -> tuple[list[tuple[bytes, str | None]], list[tuple[bytes, str | None]]]:
        kept: list[tuple[bytes, str | None]] = []
        discarded: list[tuple[bytes, str | None]] = []
        remaining = max(0, byte_limit)
        for chunk, item_id in chunks:
            take = min(len(chunk), remaining)
            if take:
                kept.append((chunk[:take], item_id))
            if take < len(chunk):
                discarded.append((chunk[take:], item_id))
            remaining -= take
        return kept, discarded

    def _must_suppress(self, *, item_id: str | None, response_id: str | None) -> bool:
        return bool(
            (item_id and item_id in self._suppressed_item_ids)
            or (response_id and response_id in self._suppressed_response_ids)
        )

    async def _delete_unplayed_item(self, item_id: str) -> None:
        if item_id in self._delete_sent_item_ids:
            return
        if not self.provider.is_connected or self.provider.ws is None:
            self._audit("unplayed_item_delete_not_sent", item_id=item_id, reason="not_connected")
            return
        await self.provider.ws.send(
            json.dumps({"type": "conversation.item.delete", "item_id": item_id})
        )
        self._delete_sent_item_ids.add(item_id)
        self._audit("unplayed_item_delete_sent", item_id=item_id)

    def _record_truncate_ack(
        self,
        *,
        item_id: str | None,
        audio_end_ms: int | None,
    ) -> None:
        expected = self._pending_truncate_acks.get(item_id or "", [])
        matched = audio_end_ms in expected if audio_end_ms is not None else False
        if matched and item_id is not None:
            expected.remove(audio_end_ms)
            if not expected:
                self._pending_truncate_acks.pop(item_id, None)
        self._audit(
            "truncate_ack",
            item_id=item_id,
            audio_end_ms=audio_end_ms,
            matched=matched,
            expected_audio_end_ms=list(expected),
        )

    def _record_pre_ack_provider_event(self, event: Any) -> None:
        response_id = getattr(event, "response_id", None) or "anonymous"
        event_type = getattr(event, "type", "unknown")
        family = "created" if event_type == "response.created" else "output"
        key = (self._interruption_epoch, response_id, family)
        if key in self._pre_ack_race_keys:
            return
        self._pre_ack_race_keys.add(key)
        self._audit(
            "provider_event_before_truncate_ack",
            event_type=event_type,
            response_id=None if response_id == "anonymous" else response_id,
            interruption_epoch=self._interruption_epoch,
            pending_truncate_acks=self.pending_truncate_acks,
            strict_history_ordering=False,
            policy="event_preserved_race_requires_runner_audit",
        )

    def _mark_response_terminal(self, response_id: str | None, status: str) -> None:
        if response_id and response_id not in self._terminal_response_ids:
            self._terminal_response_ids.add(response_id)
            self._active_response_ids.discard(response_id)
            if response_id not in self._created_response_ids:
                self._anonymous_active_responses = max(0, self._anonymous_active_responses - 1)
        elif response_id is None:
            self._anonymous_active_responses = max(0, self._anonymous_active_responses - 1)
        self._audit("response_terminal", response_id=response_id, status=status,
                    active_response_ids=sorted(self._active_response_ids),
                    anonymous_active_responses=self._anonymous_active_responses)

    def _raise_if_fatal_provider_event(self, event: Any) -> None:
        if isinstance(event, ErrorEvent):
            detail = getattr(event, "message", None) or "no provider message"
            message = detail.lower()
            benign = any(
                fragment in message
                for fragment in self._BENIGN_TERMINAL_ERROR_FRAGMENTS
            )
            self._audit(
                "provider_error",
                code=getattr(event, "code", None),
                message=detail,
                benign_terminal_race=benign,
            )
            if benign:
                return
            code = getattr(event, "code", None) or "unknown"
            raise RuntimeError(
                f"OpenAI Realtime provider error (code={code}): {detail}"
            )

        if getattr(event, "type", None) == "error":
            raw = getattr(event, "raw", None) or {}
            detail = raw.get("error") or repr(event)
            self._audit(
                "provider_transport_error",
                message=str(detail),
                fail_closed=True,
            )
            raise RuntimeError(f"OpenAI Realtime transport error: {detail}")

    def _audit(self, kind: str, **fields: Any) -> None:
        with self._audit_lock:
            self._audit_sequence += 1
            self._audit_records.append(
                {
                    "sequence": self._audit_sequence,
                    "kind": kind,
                    "wall_monotonic": time.monotonic(),
                    **fields,
                }
            )


def make_logical_adapter(
    agent_module: Any,
    tick_duration_ms: int,
    audio_format: Any,
) -> LogicalTimeOpenAIAdapter:
    """Factory used by the runner's scoped ``create_adapter`` installation.

    The runner first installs its endpoint-aware factory on ``agent_module``.
    We ask that factory for an ordinary OpenAI adapter, transfer the resulting
    provider (including gateway URL, credential and model selection), and then
    replace only the discrete-time bookkeeping layer.
    """
    if not hasattr(agent_module, "create_adapter"):
        raise TypeError("agent_module must expose the official create_adapter factory")

    base_adapter, resolved_model = agent_module.create_adapter(
        provider="openai",
        tick_duration_ms=tick_duration_ms,
        send_audio_instant=False,
        model=None,
        reasoning_effort=None,
        audio_format=audio_format,
    )
    if isinstance(base_adapter, LogicalTimeOpenAIAdapter):
        return base_adapter
    if not isinstance(base_adapter, DiscreteTimeOpenAIAdapter):
        raise TypeError(
            "installed OpenAI factory returned an incompatible adapter: "
            f"{type(base_adapter).__name__}"
        )

    provider = base_adapter.provider
    from experiments.mu_bench.logical_time.provider_event_identity import install_response_identity_capture
    install_response_identity_capture(provider)
    owns_provider = bool(getattr(base_adapter, "_owns_provider", False))
    base_adapter._owns_provider = False
    adapter = LogicalTimeOpenAIAdapter(
        tick_duration_ms=tick_duration_ms,
        send_audio_instant=bool(base_adapter.send_audio_instant),
        reasoning_effort=getattr(base_adapter, "reasoning_effort", None),
        provider=provider,
        audio_format=base_adapter.audio_format,
    )
    adapter.model = getattr(base_adapter, "model", None) or resolved_model
    adapter._owns_provider = owns_provider
    adapter._audit(
        "endpoint_aware_factory_wrapped",
        provider_class=type(provider).__name__,
        resolved_model=resolved_model,
        owns_provider=owns_provider,
    )
    return adapter
