"""Backend-neutral Probe executor used by IterativeReplayCompiler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from experiments.mu_bench.runtime.contracts import SessionResult, SessionSpec, SessionState
from experiments.mu_bench.runtime.orchestrator import UnifiedOrchestrator
from experiments.tau_multiuser_voice.iterative_replay import (
    FrozenUserEvent,
    ProbeAssistantFragment,
    ProbeAssistantInterval,
    ProbeRunResult,
    ProbeToolObservation,
)
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase


@dataclass(frozen=True)
class PreparedProbeSession:
    """Artifacts rendered on the development machine before submission."""

    spec: SessionSpec
    spec_path: Path
    task_start_ms: int
    artifact_dir: Path


class ProbeSessionPreparer(Protocol):
    """Render enrollment, frozen events, TTS, prompt, and SessionSpec."""

    def prepare_probe(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe_index: int,
    ) -> PreparedProbeSession:
        """Return one immutable fresh-probe SessionSpec."""


class UnifiedProbeExecutor:
    """Run Probes through either a local API backend or an HTTP GPU Worker."""

    def __init__(
        self,
        *,
        orchestrator: UnifiedOrchestrator,
        preparer: ProbeSessionPreparer,
        timeout_seconds: float,
        quiet_guard_ms: int,
    ) -> None:
        self.orchestrator = orchestrator
        self.preparer = preparer
        self.timeout_seconds = timeout_seconds
        self.quiet_guard_ms = quiet_guard_ms

    @staticmethod
    def _task_relative(result: SessionResult, task_start_ms: int) -> tuple[list, list, list]:
        fragments = [
            ProbeAssistantFragment(
                timestamp_ms=max(0, item.timestamp_ms - task_start_ms),
                text=item.text,
            )
            for item in result.assistant_fragments
            if item.timestamp_ms >= task_start_ms
        ]
        intervals = [
            ProbeAssistantInterval(
                start_ms=max(0, item.start_ms - task_start_ms),
                end_ms=max(0, item.end_ms - task_start_ms),
            )
            for item in result.assistant_intervals
            if item.end_ms > task_start_ms
        ]
        tools = [
            ProbeToolObservation(
                timestamp_ms=max(0, item.timestamp_ms - task_start_ms),
                kind=item.kind,
                tool_name=item.tool_name,
                result=(
                    str(item.payload.get("result"))
                    if item.kind == "tool_result"
                    else None
                ),
            )
            for item in result.tool_observations
            if item.timestamp_ms >= task_start_ms
        ]
        return fragments, intervals, tools

    def run_probe(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe_index: int,
    ) -> ProbeRunResult:
        prepared = self.preparer.prepare_probe(
            case=case,
            frozen_events=frozen_events,
            probe_index=probe_index,
        )
        result = self.orchestrator.run(
            prepared.spec,
            prepared.spec_path,
            timeout_seconds=self.timeout_seconds,
        )
        if result.state != SessionState.SUCCEEDED:
            detail = result.error.message if result.error else result.state
            raise RuntimeError(f"probe session failed: {detail}")
        fragments, intervals, tools = self._task_relative(
            result, prepared.task_start_ms
        )
        return ProbeRunResult(
            probe_index=probe_index,
            frozen_event_ids=[event.event_id for event in frozen_events],
            session_id=result.session_id,
            assistant_fragments=fragments,
            assistant_intervals=intervals,
            tool_observations=tools,
            task_media_end_ms=max(
                0, result.media_duration_ms - prepared.task_start_ms
            ),
            response_done_seen=True,
            playback_drained=True,
            quiet_guard_ms=self.quiet_guard_ms,
            completion_reason="session_result_succeeded_and_playback_drained",
            artifact_dir=str(prepared.artifact_dir),
        )
