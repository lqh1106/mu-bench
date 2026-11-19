"""Literal copy of the preceding openweight eligibility wrapper.

Only the imports and this provenance header are new.  The function body and
signature are byte-identical to the frozen source, including historical track
labels.  A caller may annotate its distinct execution protocol separately;
wall-clock comparability must never be inferred from ``realtime_eligible``.
"""
from __future__ import annotations

from typing import Any

from experiments.mu_bench.runtime.contracts import CausalInputIsolation, ExecutionMode
from experiments.tau_multiuser_voice.version import (
    BENCHMARK_VERSION,
    EXECUTION_PROTOCOL_VERSION,
    SCORING_CONTRACT_VERSION,
)

SOURCE_COMMIT = "ac77eb96b351b220d6ee0b26b4a8c668f66bb592"
SOURCE_PATH = "scripts/run_openweight_benchmark.py"
SOURCE_FILE_SHA256 = "a12083e99739801379b79088c3ea66cd657386e36ff572d99d3708066fb36a3f"
FUNCTION_SHA256 = "735f3e7ab4c80c7df58e14e91ad80ed62b419ff188542f98d8810bd320309c2c"
FUNCTION_AST_SHA256 = "f01bf42e3ecd7891bdfb704f468f4d6963e7b8e9102a802029cdb6b83b3038f0"


def compatibility_score(
    report: Any,
    *,
    case_id: str,
    model: str,
    tool_required: bool,
    leaderboard_track: str,
    execution_mode: ExecutionMode = ExecutionMode.WHOLE_TRACK_COMPAT,
    causal_input_isolation: CausalInputIsolation = CausalInputIsolation.NOT_ENFORCED,
    plan_source: str = "model_conditioned_on_policy",
) -> dict[str, Any]:
    realtime_eligible = (
        execution_mode == ExecutionMode.CAUSAL_STREAMING
        and causal_input_isolation == CausalInputIsolation.ENFORCED
    )
    interaction_score = report.interaction_score if realtime_eligible else None
    interaction_success = report.interaction_success if realtime_eligible else None
    task_supported = not tool_required
    task_success = report.task_success if task_supported else None
    strict_success = (
        task_success and interaction_success
        if task_supported and realtime_eligible
        else None
    )
    measurement_track = "realtime" if realtime_eligible else "batch_semantic"
    return {
        "case_id": case_id,
        "model": model,
        "benchmark_version": BENCHMARK_VERSION,
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "leaderboard_track": leaderboard_track,
        "plan_source": plan_source,
        "probe_execution": (
            "skipped_target_probes_shared_plan"
            if plan_source == "shared_frozen_plan"
            else "target_model_on_policy_probes"
        ),
        "measurement_track": measurement_track,
        "execution_mode": execution_mode.value,
        "causal_input_isolation": causal_input_isolation.value,
        "realtime_eligible": realtime_eligible,
        "wall_clock_eligible": False,
        "tool_required": tool_required,
        "task_contract_supported": task_supported,
        "task_success": task_success,
        "attribution_supported": False,
        "attribution_success": None,
        "interaction_score": interaction_score,
        "interaction_success": interaction_success,
        "interaction_score_supported_contract": realtime_eligible,
        "interaction_success_supported_contract": realtime_eligible,
        "batch_semantic_score": report.interaction_score if not realtime_eligible else None,
        "strict_joint_success": strict_success,
        "strict_joint_success_supported_contract": realtime_eligible and task_supported,
        "included_interaction_components": [
            "floor_control",
            "timing",
            "interruption_recovery",
            "semantic_routing_from_spoken_behavior",
        ],
        "excluded_as_not_supported": [
            "explicit_speaker_attribution",
            "explicit_addressee_attribution",
            "native_function_calling",
            "permission_decision_accuracy_from_tool_actions",
            "causal_realtime_interaction_comparability",
            "wall_clock_latency_eligibility_for_offline_native_replay",
        ],
        "unsupported_reason": (
            "native_tool_transport_unavailable"
            if tool_required
            else (
                "whole_track_input_prevents_causal_realtime_comparison"
                if not realtime_eligible
                else None
            )
        ),
        "diagnostics": {
            "required_event_coverage": report.required_event_coverage,
            "goal_disposition_accuracy": report.goal_disposition_accuracy,
            "timing_pass_rate": report.timing_pass_rate,
            "floor_control_score": report.floor_control_score,
            "timing_score": report.timing_score,
            "interruption_recovery_score": report.interruption_recovery_score,
            "semantic_routing_score": report.semantic_routing_score,
            "semantic_assertion_accuracy": report.semantic_assertion_accuracy,
            "privacy_leak_count": report.privacy_leak_count,
            "interaction_mechanism_realized": report.interaction_mechanism_realized,
            "run_valid": report.run_valid,
        },
    }
