"""Run deterministic μ-Bench smoke episodes and render oracle audio stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.tau_multiuser_voice.audio import DeterministicToneTTS
from experiments.tau_multiuser_voice.cases import CaseCatalog
from experiments.tau_multiuser_voice.enrollment import InSessionVoiceEnrollment
from experiments.tau_multiuser_voice.models import ObservationKind, WorldObservation
from experiments.tau_multiuser_voice.sessions import ScriptedUserSession
from experiments.tau_multiuser_voice.simulator import MultiUserVoiceSimulator
from experiments.tau_multiuser_voice.version import (
    BENCHMARK_DISPLAY_NAME,
    BENCHMARK_SLUG,
    BENCHMARK_VERSION,
    CASE_CATALOG_VERSION,
    EXECUTION_PROTOCOL_VERSION,
)


def run_case(
    case_id: str,
    output_root: Path,
    duration_ms: int,
    enrollment_transport: str = "audio_self_identification",
) -> dict:
    """Run one deterministic event-coverage smoke episode."""
    case = CaseCatalog.get(case_id).model_copy(deep=True)
    if enrollment_transport != "audio_self_identification":
        raise ValueError(f"unsupported enrollment transport: {enrollment_transport}")
    case.voice_enrollment = case.voice_enrollment.model_copy(
        update={"transport": enrollment_transport}
    )
    sessions = {profile.user_id: ScriptedUserSession(profile) for profile in case.users}
    simulator = MultiUserVoiceSimulator(
        case=case,
        sessions=sessions,
        tts=DeterministicToneTTS(),
    )
    enrollment = InSessionVoiceEnrollment(
        case=case,
        tts=simulator.tts,
        tick_ms=simulator.tick_ms,
    )
    simulator.trace.voice_enrollment_mode = (
        "in_session" if case.voice_enrollment.mode == "in_session" else "not_run"
    )
    simulator.trace.task_start_ms = enrollment.task_start_ms
    simulator.trace.voice_enrollments = list(enrollment.traces)
    simulator.trace.assistant_activation = enrollment.activation_trace
    decisions = []
    for sim_time_ms in range(0, duration_ms, simulator.tick_ms):
        kind = (
            ObservationKind.CONVERSATION_START
            if sim_time_ms == 0
            else ObservationKind.CLOCK
        )
        result = simulator.update(WorldObservation(sim_time_ms=sim_time_ms, kind=kind))
        if result.schedule.selected or result.schedule.rejected:
            decisions.append(result.schedule.model_dump(mode="json"))

    case_dir = output_root / case.case_id
    task_audio_paths = simulator.write_audio(case_dir / "audio" / "task", duration_ms)
    session_audio_paths = enrollment.write_session_audio(
        simulator.queues,
        case_dir / "audio" / "session",
        duration_ms,
    )
    trace_path = case_dir / "trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(simulator.trace.model_dump_json(indent=2), encoding="utf-8")
    decisions_path = case_dir / "scheduler_decisions.json"
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    control_events_path = case_dir / "session_control_events.json"
    control_events_path.write_text(
        json.dumps(enrollment.control_events(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enrollment_path = case_dir / "voice_enrollment.json"
    enrollment_path.write_text(
        json.dumps(
            {
                "mode": simulator.trace.voice_enrollment_mode,
                "transport": case.voice_enrollment.transport,
                "assistant_name": case.assistant_activation.assistant_name,
                "session_preamble": case.voice_enrollment.session_preamble_template,
                "formal_start_instruction": case.voice_enrollment.self_identification_activation_template,
                "uses_dynamic_control_items": (
                    case.voice_enrollment.uses_dynamic_control_items
                ),
                "task_start_ms": enrollment.task_start_ms,
                "enrollment_complete_ms": enrollment.enrollment_complete_ms,
                "assistant_activation": (
                    enrollment.activation_trace.model_dump(mode="json")
                    if enrollment.activation_trace
                    else None
                ),
                "oracle_audio_bindings": enrollment.prompt_bindings(),
                "control_events_file": control_events_path.name,
                "model_visible_identity_source": "spoken_tts_self_introduction_only",
                "system_prompt_policy": "model_native_default_or_none",
                "note": "Offline artifact only; the control-event file is intentionally empty.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "benchmark_name": BENCHMARK_SLUG,
        "benchmark_version": BENCHMARK_VERSION,
        "case_catalog_version": CASE_CATALOG_VERSION,
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "case_id": case.case_id,
        "enrollment_transport": case.voice_enrollment.transport,
        "system_prompt_policy": "model_native_default_or_none",
        "identity_delivery": "spoken_tts_self_introduction",
        "required_events": sum(event.required for event in case.events),
        "scheduled_events": len(simulator.trace.spoken_events),
        "task_start_ms": enrollment.task_start_ms,
        "assistant_activation": (
            enrollment.activation_trace.model_dump(mode="json")
            if enrollment.activation_trace
            else None
        ),
        "task_audio": {name: str(path) for name, path in task_audio_paths.items()},
        "session_audio": {
            name: str(path) for name, path in session_audio_paths.items()
        },
        "trace": str(trace_path),
        "scheduler_decisions": str(decisions_path),
        "session_control_events": str(control_events_path),
        "voice_enrollment": str(enrollment_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp/mu_bench_smoke"),
    )
    parser.add_argument(
        "--enrollment-transport",
        choices=("audio_self_identification",),
        default="audio_self_identification",
    )
    parser.add_argument("--duration-ms", type=int, default=20000)
    args = parser.parse_args()

    case_ids = (
        [case.case_id for case in CaseCatalog.load_default()]
        if args.case_id == "all"
        else [args.case_id]
    )
    summaries = [
        run_case(
            case_id,
            args.output_dir,
            args.duration_ms,
            args.enrollment_transport,
        )
        for case_id in case_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": BENCHMARK_DISPLAY_NAME,
                "version": BENCHMARK_VERSION,
                "summary": str(summary_path),
                "cases": len(summaries),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
