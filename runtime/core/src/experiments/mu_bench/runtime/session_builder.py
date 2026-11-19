"""Build versioned SessionSpec objects from Probe/Freeze/Final artifacts."""

from __future__ import annotations

from pathlib import Path

from experiments.mu_bench.runtime.contracts import (
    ArtifactManifest,
    ExecutionMode,
    MediaClockSpec,
    SessionPhase,
    SessionSpec,
)
from experiments.tau_multiuser_voice.iterative_replay import FrozenEventPlan
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
from experiments.tau_multiuser_voice.version import (
    BENCHMARK_VERSION,
    CASE_CATALOG_VERSION,
    EXECUTION_PROTOCOL_VERSION,
    CLOSED_SOURCE_SYSTEM_PROMPT_POLICY,
    OPEN_WEIGHT_SYSTEM_PROMPT_POLICY,
    SCORING_CONTRACT_VERSION,
)


class SessionSpecBuilder:
    """Create backend-neutral session specifications on the development machine."""

    def __init__(
        self,
        *,
        model_name: str,
        backend_name: str,
        tick_ms: int = 200,
        input_sample_rate: int = 16_000,
        pacing: str = "realtime",
        execution_mode: ExecutionMode = ExecutionMode.CAUSAL_STREAMING,
    ) -> None:
        self.model_name = model_name
        self.backend_name = backend_name
        self.media_clock = MediaClockSpec(
            tick_ms=tick_ms,
            input_sample_rate=input_sample_rate,
            pacing=pacing,
        )
        self.execution_mode = execution_mode

    def build(
        self,
        *,
        case: MultiUserVoiceCase,
        plan: FrozenEventPlan,
        phase: SessionPhase,
        attempt_id: int,
        input_wav: Path,
        output_dir: Path,
        system_prompt_path: Path | None,
        plan_path: Path,
        system_prompt_policy: str | None = None,
        probe_index: int | None = None,
        voice_manifest_path: Path | None = None,
        required_capabilities: dict[str, bool] | None = None,
    ) -> SessionSpec:
        if plan.case_id != case.case_id:
            raise ValueError("frozen plan belongs to another case")
        if phase == SessionPhase.FINAL and not plan.complete:
            raise ValueError("Final requires a complete frozen event plan")
        prompt_policy = system_prompt_policy or (
            CLOSED_SOURCE_SYSTEM_PROMPT_POLICY
            if system_prompt_path is not None
            else OPEN_WEIGHT_SYSTEM_PROMPT_POLICY
        )
        if prompt_policy not in {
            CLOSED_SOURCE_SYSTEM_PROMPT_POLICY,
            OPEN_WEIGHT_SYSTEM_PROMPT_POLICY,
        }:
            raise ValueError(f"unknown system prompt policy: {prompt_policy}")
        session_suffix = (
            f"probe-{probe_index:02d}" if phase == SessionPhase.PROBE else phase.value
        )
        session_id = f"{case.case_id}--{session_suffix}--attempt-{attempt_id}"
        result_path = output_dir / "session_result.json"
        return SessionSpec(
            session_id=session_id,
            attempt_id=attempt_id,
            case_id=case.case_id,
            phase=phase,
            probe_index=probe_index,
            model_name=self.model_name,
            backend_name=self.backend_name,
            execution_mode=self.execution_mode,
            seed=case.seed + (probe_index or 0),
            media_clock=self.media_clock,
            artifacts=ArtifactManifest(
                input_wav=str(input_wav.resolve()),
                output_dir=str(output_dir.resolve()),
                result_json=str(result_path.resolve()),
                system_prompt_path=(
                    str(system_prompt_path.resolve())
                    if system_prompt_path is not None
                    else None
                ),
                frozen_event_plan_path=str(plan_path.resolve()),
                voice_manifest_path=(
                    str(voice_manifest_path.resolve())
                    if voice_manifest_path is not None
                    else None
                ),
                trace_json=str((output_dir / "trace.json").resolve()),
            ),
            frozen_event_ids=[event.event_id for event in plan.events],
            frozen_plan_sha256=plan.plan_sha256,
            required_capabilities=required_capabilities or {},
            control_context={
                "visibility": "evaluator_orchestrator_only",
                "assistant_name": case.assistant_activation.assistant_name,
                "enrollment_transport": case.voice_enrollment.transport,
                "oracle_participants": [
                    {
                        "participant_id": case.public_participant_id(user.user_id),
                        "display_name": user.display_name,
                        "role": user.role,
                    }
                    for user in case.users
                ],
            },
            metadata={
                "benchmark_version": BENCHMARK_VERSION,
                "case_catalog_version": CASE_CATALOG_VERSION,
                "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
                "scoring_contract_version": SCORING_CONTRACT_VERSION,
                "score_final_only": case.iterative_replay.score_final_only,
                "system_prompt_policy": prompt_policy,
                "custom_system_prompt_sent": (
                    system_prompt_path is not None
                    and prompt_policy == CLOSED_SOURCE_SYSTEM_PROMPT_POLICY
                ),
                "identity_delivery": "spoken_tts_self_introduction",
                "assistant_name": case.assistant_activation.assistant_name,
                "registration_silence_required": case.voice_enrollment.silence_required,
                "formal_start_after_activation": True,
            },
        )
