"""Static admission checks for τ-MU Voice case releases."""

from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.tau_multiuser_voice.models import (
    BenchmarkTaskType,
    CapabilityCategory,
    ComprehensionLevel,
    GoalDisposition,
    GoalRelationType,
    InteractionMechanism,
    MultiUserVoiceCase,
    ParticipantScale,
    SpeechCondition,
    TriggerKind,
)

IDENTITY_CATEGORIES = {
    CapabilityCategory.PARTICIPANT_SCENE,
    CapabilityCategory.ADDRESSEE,
    CapabilityCategory.DISCOURSE_PROVENANCE,
    CapabilityCategory.MULTI_THREAD,
    CapabilityCategory.TARGETED_REPAIR,
    CapabilityCategory.GROUP_COLLABORATION,
    CapabilityCategory.FAIRNESS,
    CapabilityCategory.USER_MEMORY,
}


def audit_scaled_case_pack(cases: list[MultiUserVoiceCase]) -> dict[str, Any]:
    """Reject shallow, duplicated, or half-duplex-solvable v0.6 expansions."""
    failures: list[str] = []
    families = [case.coverage.scenario_family for case in cases]
    if len(families) != len(set(families)):
        duplicates = sorted(
            family for family, count in Counter(families).items() if count > 1
        )
        failures.append(f"duplicate scenario families: {duplicates}")

    for case in cases:
        prefix = case.case_id
        goal_dispositions = {
            goal.goal_id: goal.expected_disposition
            for user in case.users
            for goal in user.goals
        }
        for relation in case.goal_relations:
            left = goal_dispositions[relation.left_goal_id]
            right = goal_dispositions[relation.right_goal_id]
            if (
                relation.relation == GoalRelationType.CONFLICTING
                and left == GoalDisposition.FULFILL
                and right == GoalDisposition.FULFILL
            ):
                failures.append(
                    f"{prefix}: conflicting goals cannot both require fulfillment"
                )
            if (
                relation.relation == GoalRelationType.UNAUTHORIZED
                and right == GoalDisposition.FULFILL
            ):
                failures.append(
                    f"{prefix}: unauthorized live goal cannot require fulfillment"
                )
        event_actors = {event.actor_id for event in case.events}
        has_live_overlap = any(
            event.overlap_allowed
            and event.trigger.kind
            in {
                TriggerKind.ASSISTANT_SPEECH_START,
                TriggerKind.ASSISTANT_PREFIX,
                TriggerKind.TOOL_RESULT,
            }
            for event in case.events
        )
        has_realtime_timing = any(
            requirement.metric
            in {
                "yield_latency_ms",
                "resume_latency_ms",
                "thread_resume_latency_ms",
                "clarification_latency_ms",
                "false_interruptions",
            }
            for requirement in case.timing_requirements
        )
        has_identity_assertion = any(
            assertion.category in IDENTITY_CATEGORIES
            for assertion in case.capability_assertions
        )
        has_spoken_result = any(
            assertion.category == CapabilityCategory.RESPONSE_RENDERING
            for assertion in case.capability_assertions
        )
        has_start_deadline = any(
            expected.before_event_id is not None
            and expected.before_event_anchor == "start"
            for expected in case.expected_tool_actions
        )

        checks = {
            "needs two speaking users": len(event_actors) >= 2,
            "needs assistant-relative overlap": has_live_overlap,
            "needs realtime timing assertion": has_realtime_timing,
            "needs identity-sensitive assertion": has_identity_assertion,
            "needs spoken result assertion": has_spoken_result,
            "needs source adaptation": bool(case.coverage.source_adaptations),
            "needs interaction mechanism": bool(
                case.coverage.interaction_mechanisms
            ),
            "needs identity decision factor": bool(
                case.coverage.identity_decision_factors
            ),
            "needs explicit anti-shortcut proof": len(
                case.coverage.anti_shortcut_proof
            )
            >= 2,
            "needs case anti-wait constraint": bool(case.anti_wait_constraints),
            "needs a causal deadline or realtime failure mode": (
                has_start_deadline or has_realtime_timing
            ),
        }
        if case.coverage.task_type not in {
            BenchmarkTaskType.KNOWLEDGE_QA,
            BenchmarkTaskType.COLLABORATIVE_REASONING,
        }:
            checks["tool task needs an expected tool action"] = bool(
                case.expected_tool_actions
            )
        for description, passed in checks.items():
            if not passed:
                failures.append(f"{prefix}: {description}")

    domain_counts = Counter(case.environment.domain for case in cases)
    task_counts = Counter(case.coverage.task_type.value for case in cases)
    mechanism_counts = Counter(
        mechanism.value
        for case in cases
        for mechanism in case.coverage.interaction_mechanisms
    )
    identity_counts = Counter(
        factor.value
        for case in cases
        for factor in case.coverage.identity_decision_factors
    )
    source_counts = Counter(
        source
        for case in cases
        for source in case.coverage.source_adaptations
    )
    if len(domain_counts) < 8:
        failures.append("scale pack must span at least eight domains")
    if len(task_counts) < 4:
        failures.append("scale pack must span at least four task types")
    if len(mechanism_counts) < 12:
        failures.append("scale pack must cover at least twelve interaction mechanisms")
    if len(identity_counts) < 8:
        failures.append("scale pack must cover at least eight identity decision factors")
    if len(source_counts) < 6:
        failures.append("scale pack must adapt at least six source benchmarks")
    if failures:
        raise ValueError("scaled case audit failed:\n" + "\n".join(failures))
    return {
        "case_count": len(cases),
        "domain_count": len(domain_counts),
        "task_type_count": len(task_counts),
        "mechanism_count": len(mechanism_counts),
        "identity_factor_count": len(identity_counts),
        "source_count": len(source_counts),
        "cases": len(cases),
        "domains": dict(sorted(domain_counts.items())),
        "task_types": dict(sorted(task_counts.items())),
        "interaction_mechanisms": dict(sorted(mechanism_counts.items())),
        "identity_decision_factors": dict(sorted(identity_counts.items())),
        "source_adaptations": dict(sorted(source_counts.items())),
    }


def audit_factorial_case_pack(cases: list[MultiUserVoiceCase]) -> dict[str, Any]:
    """Prove the v0.7 release is a real 100 x 2 x 2 x 2 case grid."""
    from experiments.tau_multiuser_voice.business_tooling_v07 import (
        contract_for_domain,
        domain_tool_names,
        operation_for_scenario,
    )

    failures: list[str] = []
    if len(cases) != 800:
        failures.append(f"expected 800 runnable cases, found {len(cases)}")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        failures.append("case ids are not unique")
    if any(case.variant is None for case in cases):
        failures.append("every v0.7 case must declare variant coordinates")

    by_scenario: dict[str, list[MultiUserVoiceCase]] = {}
    for case in cases:
        if case.variant is not None:
            by_scenario.setdefault(case.variant.scenario_id, []).append(case)
    if len(by_scenario) != 100:
        failures.append(f"expected 100 base scenarios, found {len(by_scenario)}")
    expected_grid = {
        (scale, speech, comprehension)
        for scale in ParticipantScale
        for speech in SpeechCondition
        for comprehension in ComprehensionLevel
    }
    for scenario_id, variants in by_scenario.items():
        actual_grid = {
            (
                case.variant.participant_scale,
                case.variant.speech_condition,
                case.variant.comprehension_level,
            )
            for case in variants
            if case.variant is not None
        }
        if len(variants) != 8 or actual_grid != expected_grid:
            failures.append(f"{scenario_id}: incomplete 2 x 2 x 2 grid")

    for case in cases:
        if case.variant is None:
            continue
        prefix = case.case_id
        scale = case.variant.participant_scale
        speech = case.variant.speech_condition
        comprehension = case.variant.comprehension_level
        speaker_ids = {event.actor_id for event in case.events}
        user_ids = {user.user_id for user in case.users}
        if speaker_ids != user_ids:
            failures.append(f"{prefix}: every participant must have a task event")
        assertion_ids = {
            assertion.assertion_id for assertion in case.capability_assertions
        }
        mechanism = (
            case.coverage.interaction_mechanisms[0]
            if case.coverage.interaction_mechanisms
            else None
        )
        if mechanism == InteractionMechanism.BACKCHANNEL_SELECTIVITY:
            required_floor_ids = {
                "a_backchannel_non_takeover",
                "a_backchannel_original_continuation",
            }
        elif mechanism == InteractionMechanism.FLOOR_FAIRNESS:
            required_floor_ids = {
                "a_targeted_floor_invitation",
                "a_no_premature_closure",
                "a_quiet_input_integration",
            }
        else:
            required_floor_ids = set()
        if not required_floor_ids.issubset(assertion_ids):
            failures.append(f"{prefix}: floor split assertion contract is incomplete")
        if mechanism not in {
            InteractionMechanism.BACKCHANNEL_SELECTIVITY,
            InteractionMechanism.FLOOR_FAIRNESS,
        } and {
            "a_backchannel_non_takeover",
            "a_backchannel_original_continuation",
            "a_targeted_floor_invitation",
            "a_no_premature_closure",
            "a_quiet_input_integration",
        }.intersection(assertion_ids):
            failures.append(f"{prefix}: floor split assertions appear on a non-floor case")

        expected_recognition_ids = {
            "a_primary_return_spoken_anchor",
            "a_primary_return_behavior_continuity",
            "a_live_return_spoken_anchor",
            "a_live_return_behavior_continuity",
        }
        if scale == ParticipantScale.LARGE:
            expected_recognition_ids.update(
                {
                    "a_extra1_return_spoken_anchor",
                    "a_extra1_return_behavior_continuity",
                }
            )
        if not expected_recognition_ids.issubset(assertion_ids):
            failures.append(f"{prefix}: recognition split contract is incomplete")
        if not all(
            assertion.diagnostic_only
            and assertion.critical is False
            for assertion in case.capability_assertions
            if assertion.assertion_id.endswith("_return_spoken_anchor")
            or assertion.assertion_id.endswith("_return_behavior_continuity")
        ):
            failures.append(f"{prefix}: recognition split must remain diagnostic-only")
        if scale == ParticipantScale.SMALL and not 2 <= len(case.users) <= 3:
            failures.append(f"{prefix}: small group is not two to three people")
        if scale == ParticipantScale.LARGE and not 4 <= len(case.users) <= 7:
            failures.append(f"{prefix}: large group is not four to seven people")
        accents = {user.voice.accent_label for user in case.users}
        if speech == SpeechCondition.CLEAN:
            if accents != {"General American"} or case.difficulty.audio_conditions:
                failures.append(f"{prefix}: clean condition contains accent or noise")
        elif len(accents) < 2 or not case.difficulty.audio_conditions:
            failures.append(f"{prefix}: realistic condition lacks acoustic diversity")
        if mechanism == InteractionMechanism.FLOOR_FAIRNESS and not {
            ("floor_invitation_latency_ms", "e_live"),
            ("post_invitation_false_interruptions", "e_live"),
        }.issubset(
            {
                (requirement.metric, requirement.event_id)
                for requirement in case.timing_requirements
            }
        ):
            failures.append(
                f"{prefix}: fairness invitation timing contract is incomplete"
            )
        repair_events = {
            event.event_id for event in case.events if event.event_id == "e_comprehension_repair"
        }
        if comprehension == ComprehensionLevel.STRONG:
            if repair_events or case.difficulty.comprehension_challenges:
                failures.append(f"{prefix}: strong comprehension includes repair")
        else:
            assertion_ids = {
                assertion.assertion_id for assertion in case.capability_assertions
            }
            repair_event = next(
                (
                    event
                    for event in case.events
                    if event.event_id == "e_comprehension_repair"
                ),
                None,
            )
            repair_requirement = next(
                (
                    requirement
                    for requirement in case.timing_requirements
                    if requirement.metric == "clarification_latency_ms"
                    and requirement.event_id == "e_comprehension_repair"
                ),
                None,
            )
            timing_keys = {
                (requirement.metric, requirement.event_id)
                for requirement in case.timing_requirements
            }
            if (
                repair_events != {"e_comprehension_repair"}
                or not case.difficulty.comprehension_challenges
                or "a_comprehension_repair" not in assertion_ids
                or ("clarification_latency_ms", "e_comprehension_repair")
                not in timing_keys
            ):
                failures.append(f"{prefix}: weak comprehension loop is incomplete")
            if repair_event is None or repair_requirement is None:
                failures.append(f"{prefix}: weak repair timing contract is missing")
            else:
                if repair_event.max_delay_ms > repair_requirement.threshold:
                    failures.append(
                        f"{prefix}: weak repair planner window exceeds clarification gate"
                    )
                if repair_event.trigger.fallback_at_ms is None:
                    failures.append(
                        f"{prefix}: weak repair has no liveness fallback"
                    )
                if repair_event.trigger.fallback_after_event_id != "e_primary_commit":
                    failures.append(
                        f"{prefix}: weak repair must remain after e_primary_commit"
                    )
        live_events = [event for event in case.events if event.event_id == "e_live"]
        if len(live_events) != 1 or not live_events[0].overlap_allowed:
            failures.append(f"{prefix}: missing required live overlap event")
        elif live_events[0].trigger.kind not in {
            TriggerKind.ASSISTANT_SPEECH_START,
            TriggerKind.TOOL_RESULT,
        }:
            failures.append(f"{prefix}: e_live is not model-conditioned")
        if not case.coverage.source_adaptations:
            failures.append(f"{prefix}: missing benchmark provenance")
        if len(case.coverage.anti_shortcut_proof) < 3:
            failures.append(f"{prefix}: anti-shortcut proof is incomplete")
        if not any(
            assertion.category == CapabilityCategory.RESPONSE_RENDERING
            for assertion in case.capability_assertions
        ):
            failures.append(f"{prefix}: missing spoken-result requirement")
        if case.environment.domain != f"v07_{case.capability_tags[0]}":
            failures.append(f"{prefix}: wrong domain-specific sandbox")
        if case.assistant_activation.allow_brief_acknowledgement:
            failures.append(f"{prefix}: activation may leak into formal task time")
        if live_events and live_events[0].trigger.fallback_after_event_id != "e_open":
            failures.append(f"{prefix}: e_live fallback may overlap unfinished e_open")
        contract = contract_for_domain(case.environment.domain)
        if contract is None:
            if case.environment.tool_names or case.expected_tool_actions:
                failures.append(f"{prefix}: tool-free domain declares a domain tool")
        else:
            allowed_names = set(domain_tool_names(case.environment.domain))
            from experiments.tau_multiuser_voice.completion_tools import definitions as completion_definitions, VIEW, DOC
            completion = completion_definitions(case.environment)
            if completion:
                obsolete = 'record_viewing_availability' if case.variant.scenario_id == VIEW else 'stage_separate_document_updates'
                allowed_names.discard(obsolete)
                allowed_names.update(completion)
            operation = operation_for_scenario(case.variant.scenario_id)
            if (
                case.environment.tool_names
                and set(case.environment.tool_names) != allowed_names
            ):
                failures.append(f"{prefix}: contains a cross-domain or generic tool")
            for action in case.expected_tool_actions:
                if action.tool_name not in allowed_names:
                    failures.append(f"{prefix}: action uses a non-domain tool")
                resource_key = completion[action.tool_name].resource_key if action.tool_name in completion else contract.resource_key
                if resource_key not in action.argument_subset:
                    failures.append(f"{prefix}: action omits its domain resource")
                if "actor_id" in action.argument_subset:
                    failures.append(f"{prefix}: business args leak requester identity")
                if (
                    operation is not None
                    and action.allowed
                    and action.tool_name == operation.tool_name
                    and not set(operation.expected_arguments).issubset(
                        action.argument_subset
                    )
                ):
                    failures.append(
                        f"{prefix}: business operation omits concrete parameters"
                    )

    domain_counts = Counter(case.capability_tags[0] for case in cases)
    scenario_domain_counts = Counter(
        cases_for_scenario[0].capability_tags[0]
        for cases_for_scenario in by_scenario.values()
        if cases_for_scenario
    )
    origin_counts = Counter(case.capability_tags[1] for case in cases)
    scenario_origin_counts = Counter(
        cases_for_scenario[0].capability_tags[1]
        for cases_for_scenario in by_scenario.values()
        if cases_for_scenario
    )
    if len(domain_counts) != 20 or set(domain_counts.values()) != {40}:
        failures.append("expected twenty domains with forty variants each")
    if set(scenario_domain_counts.values()) != {5}:
        failures.append("expected five base scenarios per domain")
    if origin_counts != {"direct": 320, "interaction_extension": 480}:
        failures.append("expected 320 direct and 480 interaction-extension cases")
    if scenario_origin_counts != {"direct": 40, "interaction_extension": 60}:
        failures.append("expected 40 direct and 60 interaction-extension scenarios")
    mechanism_counts = Counter(
        case.coverage.interaction_mechanisms[0].value for case in cases
    )
    expected_mechanisms = {mechanism.value for mechanism in InteractionMechanism}
    if set(mechanism_counts) != expected_mechanisms:
        failures.append("not all interaction mechanisms are represented")
    task_counts = Counter(case.coverage.task_type.value for case in cases)
    if set(task_counts) != {task.value for task in BenchmarkTaskType}:
        failures.append("not all task types are represented")
    source_counts = Counter(
        source
        for variants in by_scenario.values()
        for source in variants[0].coverage.source_adaptations
    )
    participant_counts = Counter(len(case.users) for case in cases)
    if set(participant_counts) != {2, 3, 4, 5, 6, 7}:
        failures.append("participant-count gradient does not cover two through seven")
    tool_cases = [case for case in cases if case.environment.tool_names]
    declared_tool_names = {
        name for case in tool_cases for name in case.environment.tool_names
    }
    generic_names = {
        "inspect_case_record",
        "commit_case_resolution",
        "execute_case_action",
        "record_case_decision",
    }
    if len(tool_cases) != 408 or len(cases) - len(tool_cases) != 392:
        failures.append("tool/tool-free factorial coverage changed unexpectedly")
    expected_domain_tools = {
        tool_name
        for domain in domain_counts
        for tool_name in domain_tool_names(domain)
    }
    from experiments.tau_multiuser_voice.completion_tools import definitions as completion_definitions
    expected_domain_tools.update(name for case in tool_cases for name in completion_definitions(case.environment))
    if declared_tool_names != expected_domain_tools or declared_tool_names & generic_names:
        failures.append("v0.7 does not expose the complete stable domain APIs")
    if failures:
        raise ValueError("factorial case audit failed:\n" + "\n".join(failures))
    return {
        "runnable_case_count": len(cases),
        "base_scenario_count": len(by_scenario),
        "variants_per_scenario": 8,
        "domain_count": len(domain_counts),
        "domains": dict(sorted(domain_counts.items())),
        "origins": dict(sorted(origin_counts.items())),
        "participant_counts": dict(sorted(participant_counts.items())),
        "task_types": dict(sorted(task_counts.items())),
        "interaction_mechanisms": dict(sorted(mechanism_counts.items())),
        "source_benchmarks": dict(sorted(source_counts.items())),
        "tool_case_count": len(tool_cases),
        "tool_free_case_count": len(cases) - len(tool_cases),
        "unique_domain_tool_count": len(declared_tool_names),
    }
