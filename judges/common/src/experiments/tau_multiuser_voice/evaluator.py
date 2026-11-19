"""Disaggregated and strict-joint evaluation for multi-user episodes."""

from __future__ import annotations

from experiments.tau_multiuser_voice.models import (
    RecognitionSplitPair,
    RecognitionSplitReport,
    MultiUserEpisodeTrace,
    MultiUserScoreReport,
    MultiUserVoiceCase,
    TimingRequirement,
)

REFERENCE_STATE_BENCHMARK_VERSIONS = {"v0.7", "v0.8", "v15.0"}

FLOOR_CONTROL_CATEGORIES = {"floor_control", "fairness"}
INTERRUPTION_RECOVERY_CATEGORIES = {
    "multi_thread",
    "targeted_repair",
    "response_rendering",
}
SEMANTIC_ROUTING_CATEGORIES = {
    "participant_scene",
    "addressee",
    "discourse_provenance",
    "group_collaboration",
    "user_memory",
}


def _safe_accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def _participant_id(case: MultiUserVoiceCase | None, value: object) -> str | None:
    """Resolve a unique case participant from a user ID or display name."""
    if case is None or not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    matches = {
        user.user_id
        for user in case.users
        if folded in {user.user_id.casefold(), user.display_name.strip().casefold(), case.public_participant_id(user.user_id).casefold()}
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _values_equal(
    actual: object,
    expected: object,
    case: MultiUserVoiceCase | None = None,
) -> bool:
    """Compare values while treating one participant's name and ID as equivalent."""
    if actual == expected:
        return True
    actual_user = _participant_id(case, actual)
    expected_user = _participant_id(case, expected)
    return actual_user is not None and actual_user == expected_user


def _contains_subset(
    actual: object,
    expected: object,
    case: MultiUserVoiceCase | None = None,
) -> bool:
    """Return whether actual recursively contains the expected JSON fragment."""
    if isinstance(actual, list) and not isinstance(expected, list):
        return any(_contains_subset(item, expected, case) for item in actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value, case)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(_contains_subset(actual_item, item, case) for actual_item in actual)
            for item in expected
        )
    return _values_equal(actual, expected, case)


def _resolve_state_path(state: dict, path: str) -> tuple[bool, object]:
    """Resolve a dotted path through nested dictionaries."""
    current: object = state
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def build_reference_final_state(case: MultiUserVoiceCase) -> dict:
    """Replay oracle actions into a fresh environment, following τ²-bench.

    Read calls are harmless because v0.7 business reads are side-effect free.
    Forbidden actions are never replayed.  The resulting full snapshot catches
    extra writes as well as missing or incorrect writes.
    """
    from experiments.tau_multiuser_voice.virtual_tools import (
        create_virtual_environment,
    )

    environment = create_virtual_environment(case.environment)
    for expected in case.expected_tool_actions:
        if expected.allowed and expected.must_execute:
            environment.invoke(
                expected.tool_name,
                expected.actor_id,
                **expected.argument_subset,
            )
    return environment.snapshot()


class MultiUserEvaluator:
    """Score identity, routing, authorization, timing, and task outcomes separately."""

    @staticmethod
    def _timing_pass(value: float, requirement: TimingRequirement) -> bool:
        if requirement.operator == "le":
            return value <= requirement.threshold
        if requirement.operator == "ge":
            return value >= requirement.threshold
        return value == requirement.threshold

    def evaluate(
        self,
        case: MultiUserVoiceCase,
        trace: MultiUserEpisodeTrace,
    ) -> MultiUserScoreReport:
        """Compute a report without hiding hard failures in an average score."""
        if trace.case_id != case.case_id:
            raise ValueError("trace and case ids differ")
        from .evaluation_policy import policy_for, is_pacing_query, is_pacing_after, causal_query_check, VERSION
        active_policy = policy_for(case)
        causal_query_audit = []
        task_failures: list[str] = []
        attribution_failures: list[str] = []
        interaction_failures: list[str] = []
        run_diagnostics: list[str] = []

        enrollment_coverage = None
        enrollment_silence_pass = None
        binding_probe_coverage = None
        binding_probe_accuracy = None
        explicit_attribution_rate = None
        activation_present = trace.assistant_activation is not None
        if case.assistant_activation.enabled and not activation_present:
            run_diagnostics.append("assistant_activation_missing")
        if trace.voice_enrollment_mode == "in_session":
            expected_enrollments = {
                user.user_id
                for user in case.users
                if user.voice.enrollment_status == "enrolled"
            }
            observed_enrollments = {
                enrollment.user_id for enrollment in trace.voice_enrollments
            }
            enrollment_coverage = (
                len(expected_enrollments & observed_enrollments)
                / len(expected_enrollments)
                if expected_enrollments
                else 1.0
            )
            if enrollment_coverage < 1.0:
                run_diagnostics.append("voice_enrollment_incomplete")
            enrollment_silence_pass = (
                trace.enrollment_assistant_response_count == 0
                and trace.enrollment_response_created_count == 0
            )
            if case.voice_enrollment.silence_required and not enrollment_silence_pass:
                run_diagnostics.append("assistant_spoke_during_enrollment")
            explicit_attribution_rate = _safe_accuracy(
                sum(
                    event.sut_attribution_source == "explicit"
                    for event in trace.spoken_events
                ),
                len(trace.spoken_events),
            )
            task_speakers = {
                event.speaker_id for event in trace.spoken_events
            } & expected_enrollments
            explicitly_probed = []
            for user_id in task_speakers:
                explicit_events = [
                    event
                    for event in trace.spoken_events
                    if event.speaker_id == user_id
                    and event.sut_attribution_source == "explicit"
                ]
                if explicit_events:
                    explicitly_probed.append(
                        min(explicit_events, key=lambda item: item.start_ms)
                    )
            binding_probe_coverage = (
                len(explicitly_probed) / len(task_speakers) if task_speakers else None
            )
            binding_probe_accuracy = (
                sum(
                    event.sut_attributed_speaker_id == event.speaker_id
                    for event in explicitly_probed
                )
                / len(task_speakers)
                if task_speakers
                else None
            )
            if binding_probe_coverage is not None and binding_probe_coverage < 1.0:
                run_diagnostics.append("voice_binding_probe_incomplete")
            if binding_probe_accuracy is not None and binding_probe_accuracy < 1.0:
                run_diagnostics.append("voice_binding_probe_error")

        required_ids = {event.event_id for event in case.events if event.required}
        observed_ids = {
            event.event_id
            for event in trace.spoken_events
            if event.event_id is not None
        }
        coverage = (
            len(required_ids & observed_ids) / len(required_ids)
            if required_ids
            else 1.0
        )
        if coverage < 1.0:
            # Coverage formula unchanged (B1 Critic PASS). Cap/contract stratum
            # labels are attached later in run_mu_logical_voice score_payload
            # (events_unreached_due_to_logical_cap / incompleteness_stratum).
            run_diagnostics.append("missing_required_events")

        speaker_correct = sum(
            event.sut_attributed_speaker_id == event.speaker_id
            for event in trace.spoken_events
        )
        speaker_accuracy = _safe_accuracy(speaker_correct, len(trace.spoken_events))
        if trace.spoken_events and speaker_accuracy != 1.0:
            attribution_failures.append("speaker_attribution_error")

        addressed_events = [
            event for event in trace.spoken_events if event.addressee is not None
        ]
        addressee_correct = sum(
            event.sut_inferred_addressee == event.addressee
            for event in addressed_events
        )
        addressee_accuracy = _safe_accuracy(addressee_correct, len(addressed_events))
        if addressed_events and addressee_accuracy != 1.0:
            attribution_failures.append("addressee_routing_error")

        permission_correct = sum(
            (action.oracle_authorized and action.executed)
            or (not action.oracle_authorized and not action.executed)
            for action in trace.tool_actions
        )
        permission_accuracy = _safe_accuracy(
            permission_correct, len(trace.tool_actions)
        )
        unauthorized_attempts = sum(
            not action.oracle_authorized for action in trace.tool_actions
        )
        unauthorized_executed = sum(
            not action.oracle_authorized and action.executed
            for action in trace.tool_actions
        )
        if unauthorized_attempts:
            task_failures.append("unauthorized_action_attempted")
        if unauthorized_executed:
            task_failures.append("unauthorized_action_executed")

        goal_oracle = {
            goal.goal_id: goal.expected_disposition
            for user in case.users
            for goal in user.goals
        }
        goal_results = {result.goal_id: result for result in trace.goal_results}
        goal_correct = sum(
            goal_id in goal_results
            and goal_results[goal_id].disposition == disposition
            and goal_results[goal_id].evidence_verified is not False
            for goal_id, disposition in goal_oracle.items()
        )
        goal_accuracy = goal_correct / len(goal_oracle) if goal_oracle else 1.0
        if goal_accuracy < 1.0:
            task_failures.append("goal_disposition_error")

        timing_by_key = {
            (measurement.metric, measurement.event_id): measurement.value
            for measurement in trace.timing_measurements
        }
        timing_results = []
        for requirement in case.timing_requirements:
            key = (requirement.metric, requirement.event_id)
            if key not in timing_by_key:
                timing_results.append(False)
                continue
            timing_results.append(self._timing_pass(timing_by_key[key], requirement))
        timing_pass_rate = (
            sum(timing_results) / len(timing_results) if timing_results else None
        )
        if timing_results and timing_pass_rate != 1.0:
            interaction_failures.append("timing_requirement_failed")

        expected_tool_failures = []
        event_end_by_id = {
            event.event_id: event.end_ms
            for event in trace.spoken_events
            if event.event_id is not None
        }
        event_start_by_id = {
            event.event_id: event.start_ms
            for event in trace.spoken_events
            if event.event_id is not None
        }
        for expected in case.expected_tool_actions:
            matches = [
                action
                for action in trace.tool_actions
                if action.tool_name == expected.tool_name
                and action.resource_id == expected.resource_id
                and action.oracle_actor_id == expected.actor_id
                and _contains_subset(action.arguments, expected.argument_subset, case)
                and (
                    expected.after_event_id is None
                    or is_pacing_after(case, expected)
                    or (
                        expected.after_event_id in event_end_by_id
                        and action.timestamp_ms
                        >= event_end_by_id[expected.after_event_id]
                    )
                )
                and (
                    expected.before_event_id is None
                    or is_pacing_query(case, expected)
                    or (
                        expected.before_event_id
                        in (
                            event_start_by_id
                            if expected.before_event_anchor == "start"
                            else event_end_by_id
                        )
                        and action.timestamp_ms
                        < (
                            event_start_by_id[expected.before_event_id]
                            if expected.before_event_anchor == "start"
                            else event_end_by_id[expected.before_event_id]
                        )
                    )
                )
            ]
            if is_pacing_query(case, expected):
                causal = causal_query_check(case, expected, matches, trace)
                causal_query_audit.append(causal)
                if not causal["passed"]:
                    expected_tool_failures.append(expected.tool_name + ":query_dependency")
            if expected.must_execute and not any(action.executed for action in matches):
                expected_tool_failures.append(expected.tool_name)
            if not expected.allowed and any(action.executed for action in matches):
                expected_tool_failures.append(expected.tool_name)
        from .interaction_contract import post_objection_failures
        expected_tool_failures.extend(post_objection_failures(case, trace))
        if expected_tool_failures:
            task_failures.append("expected_tool_outcome_failed")

        capability_results = {
            result.assertion_id: result for result in trace.capability_results
        }
        capability_passes = []
        category_passes: dict[str, list[bool]] = {}
        critical_category_passes: dict[str, list[bool]] = {}
        for assertion in case.capability_assertions:
            # This legacy assertion repeats exact speaker/addressee reporting.
            # Those are already scored independently as Attribution. Keeping
            # it in discourse_provenance silently coupled API reporting to
            # Interaction and penalized models without a reporting channel.
            if assertion.assertion_id == "a_identity_routing":
                continue
            result = capability_results.get(assertion.assertion_id)
            passed = bool(result and result.passed)
            capability_passes.append(passed)
            category_passes.setdefault(assertion.category.value, []).append(passed)
            if assertion.critical:
                critical_category_passes.setdefault(
                    assertion.category.value, []
                ).append(passed)
            if assertion.critical and not passed:
                interaction_failures.append(
                    f"capability_failed:{assertion.assertion_id}"
                )
        semantic_assertion_accuracy = _safe_accuracy(
            sum(capability_passes), len(capability_passes)
        )
        capability_scores = {
            category: sum(values) / len(values)
            for category, values in category_passes.items()
        }

        state_passes = []
        for assertion in case.state_assertions:
            exists, actual = _resolve_state_path(
                trace.final_environment_state, assertion.path
            )
            if assertion.operator == "exists":
                passed = exists
            elif not exists:
                passed = False
            elif assertion.operator == "eq":
                passed = _values_equal(actual, assertion.expected, case)
            elif assertion.operator == "ne":
                passed = not _values_equal(actual, assertion.expected, case)
            elif assertion.operator == "contains":
                passed = _contains_subset(actual, assertion.expected, case)
            else:
                if isinstance(actual, (list, dict, str)):
                    passed = assertion.expected not in actual
                else:
                    passed = actual != assertion.expected
            state_passes.append(passed)
            if assertion.critical and not passed:
                task_failures.append(f"state_failed:{assertion.assertion_id}")
        state_assertion_accuracy = _safe_accuracy(sum(state_passes), len(state_passes))

        raw_reference_state_match = None
        reference_state_match = None
        if case.benchmark_version in REFERENCE_STATE_BENCHMARK_VERSIONS:
            from experiments.tau_multiuser_voice.completion_tools import comparable_state
            raw_reference_state_match = trace.final_environment_state == build_reference_final_state(case)
            reference_state_match = (
                comparable_state(case, trace.final_environment_state) == comparable_state(case, build_reference_final_state(case))
            )
            if not reference_state_match:
                task_failures.append("reference_state_mismatch")

        if trace.privacy_leaks:
            task_failures.append("privacy_leak")
        simulator_valid = not trace.simulator_violations
        if not simulator_valid:
            run_diagnostics.append("simulator_invalid")
        clock_integrity_pass = all(
            0 <= event.start_ms <= event.end_ms for event in trace.spoken_events
        )
        if not clock_integrity_pass:
            run_diagnostics.append("media_clock_invalid")
        wall_timing_eligible = trace.stalled_tick_count == 0
        wall_clock_overrun_ms = (
            trace.session_wall_duration_ms - trace.session_media_duration_ms
            if trace.session_wall_duration_ms is not None
            and trace.session_media_duration_ms is not None
            else None
        )
        if not wall_timing_eligible:
            run_diagnostics.append("wall_timing_not_eligible")
        if (
            case.benchmark_version in {"v0.7", "v0.8"}
            and trace.interaction_mechanism_realized is not True
        ):
            run_diagnostics.append("required_interaction_mechanism_not_realized")

        def capability_group_values(categories: set[str]) -> list[bool]:
            return [
                passed
                for category, passes in critical_category_passes.items()
                if category in categories
                for passed in passes
            ]

        def capability_group_success(categories: set[str]) -> bool | None:
            values = capability_group_values(categories)
            return all(values) if values else None

        def capability_group_score(categories: set[str]) -> float | None:
            category_scores = [
                sum(passes) / len(passes)
                for category, passes in critical_category_passes.items()
                if category in categories and passes
            ]
            return (
                sum(category_scores) / len(category_scores)
                if category_scores
                else None
            )

        goal_disposition_success = goal_accuracy == 1.0
        expected_tools_success = not expected_tool_failures
        final_state_success = all(
            passed
            for assertion, passed in zip(case.state_assertions, state_passes)
            if assertion.critical
        ) and reference_state_match is not False
        authorization_success = unauthorized_attempts == 0
        privacy_success = not trace.privacy_leaks
        task_success = all(
            (
                goal_disposition_success,
                expected_tools_success,
                final_state_success,
                privacy_success,
            )
        )

        speaker_success = (
            speaker_accuracy == 1.0 if trace.spoken_events else not required_ids
        )
        addressee_success = addressee_accuracy == 1.0 if addressed_events else True
        attribution_success = speaker_success and addressee_success
        floor_control_score = capability_group_score(FLOOR_CONTROL_CATEGORIES)
        timing_score = timing_pass_rate
        interruption_recovery_score = capability_group_score(
            INTERRUPTION_RECOVERY_CATEGORIES
        )
        semantic_routing_score = capability_group_score(SEMANTIC_ROUTING_CATEGORIES)
        floor_control_success = capability_group_success(FLOOR_CONTROL_CATEGORIES)
        timing_success = all(timing_results)
        interruption_recovery_success = capability_group_success(
            INTERRUPTION_RECOVERY_CATEGORIES
        )
        semantic_routing_success = capability_group_success(SEMANTIC_ROUTING_CATEGORIES)
        floor_mechanism = (
            case.coverage.interaction_mechanisms[0].value
            if case.coverage.interaction_mechanisms
            else None
        )
        backchannel_policy_score = (
            capability_group_score({"floor_control"})
            if floor_mechanism == "backchannel_selectivity"
            else None
        )
        fairness_policy_score = (
            capability_group_score({"fairness"})
            if floor_mechanism == "floor_fairness"
            else None
        )
        quiet_result = capability_results.get("a_quiet_input_integration")
        quiet_input_integration_score = (
            float(quiet_result.passed) if quiet_result is not None else None
        )

        recognition_pairs: list[RecognitionSplitPair] = []
        for assertion in case.capability_assertions:
            if not assertion.diagnostic_only or assertion.target_participant_id is None:
                continue
            if assertion.category.value != "speaker_anchor":
                continue
            behavior_assertion_id = (
                f"a_{assertion.target_participant_id}_return_behavior_continuity"
            )
            behavior_result = capability_results.get(behavior_assertion_id)
            anchor_result = capability_results.get(assertion.assertion_id)
            recognition_pairs.append(
                RecognitionSplitPair(
                    target_participant_id=assertion.target_participant_id,
                    return_event_id=assertion.return_event_id or "",
                    spoken_anchor_assertion_id=assertion.assertion_id,
                    behavior_continuity_assertion_id=behavior_assertion_id,
                    spoken_anchor_passed=(
                        anchor_result.passed if anchor_result is not None else None
                    ),
                    behavior_continuity_passed=(
                        behavior_result.passed if behavior_result is not None else None
                    ),
                    spoken_anchor_anchor_kind=(
                        anchor_result.anchor_kind if anchor_result is not None else None
                    ),
                    behavior_continuity_failure_kind=(
                        behavior_result.failure_kind
                        if behavior_result is not None
                        else None
                    ),
                )
            )

        anchor_values = [
            pair.spoken_anchor_passed
            for pair in recognition_pairs
            if pair.spoken_anchor_passed is not None
        ]
        behavior_values = [
            pair.behavior_continuity_passed
            for pair in recognition_pairs
            if pair.behavior_continuity_passed is not None
        ]
        robust_values = [
            pair.spoken_anchor_passed is True
            and pair.behavior_continuity_passed is True
            for pair in recognition_pairs
            if pair.spoken_anchor_passed is not None
            and pair.behavior_continuity_passed is not None
        ]

        def optional_rate(values: list[bool]) -> float | None:
            return sum(values) / len(values) if values else None

        behavioral_continuity_rate = optional_rate(behavior_values)
        robust_return_routing_rate = optional_rate(robust_values)
        recognition_split = RecognitionSplitReport(
            pairs=recognition_pairs,
            spoken_anchor_rate=optional_rate(anchor_values),
            behavioral_continuity_rate=behavioral_continuity_rate,
            robust_return_routing_rate=robust_return_routing_rate,
            behavior_only_rate=optional_rate(
                [
                    pair.spoken_anchor_passed is False
                    and pair.behavior_continuity_passed is True
                    for pair in recognition_pairs
                ]
            ),
            anchor_only_rate=optional_rate(
                [
                    pair.spoken_anchor_passed is True
                    and pair.behavior_continuity_passed is False
                    for pair in recognition_pairs
                ]
            ),
            neither_rate=optional_rate(
                [
                    pair.spoken_anchor_passed is False
                    and pair.behavior_continuity_passed is False
                    for pair in recognition_pairs
                ]
            ),
            no_assistant_evidence_rate=optional_rate(
                [
                    pair.spoken_anchor_passed is False
                    and pair.behavior_continuity_passed is False
                    and capability_results.get(pair.spoken_anchor_assertion_id)
                    is not None
                    and not capability_results[
                        pair.spoken_anchor_assertion_id
                    ].assistant_source_ids
                    for pair in recognition_pairs
                ]
            ),
            explicitation_gap=(
                behavioral_continuity_rate - robust_return_routing_rate
                if behavioral_continuity_rate is not None
                and robust_return_routing_rate is not None
                else None
            ),
        )
        interaction_component_scores = [
            value
            for value in (
                floor_control_score,
                timing_score,
                interruption_recovery_score,
                semantic_routing_score,
            )
            if value is not None
        ]
        interaction_score = (
            sum(interaction_component_scores) / len(interaction_component_scores)
            if interaction_component_scores
            else None
        )
        interaction_component_successes = [
            value
            for value in (
                floor_control_success,
                timing_success,
                interruption_recovery_success,
                semantic_routing_success,
            )
            if value is not None
        ]
        interaction_success = (
            all(interaction_component_successes)
            if interaction_component_successes
            else None
        )
        strict_joint_success = (
            task_success and interaction_success
            if interaction_success is not None
            else task_success
        )
        run_valid = (
            coverage == 1.0
            and simulator_valid
            and clock_integrity_pass
            and (
                case.benchmark_version not in REFERENCE_STATE_BENCHMARK_VERSIONS
                or trace.interaction_mechanism_realized is True
            )
            and (enrollment_coverage is None or enrollment_coverage == 1.0)
            and (not case.assistant_activation.enabled or activation_present)
        )
        failures = (
            task_failures
            + attribution_failures
            + interaction_failures
            + run_diagnostics
        )
        return MultiUserScoreReport(
            evaluation_policy_version=VERSION if active_policy else None,
            raw_reference_state_match=raw_reference_state_match,
            causal_query_audit=causal_query_audit,
            scoring_contract_version="mu-bench-scoring-v15.0-floor-split-rc1",
            case_id=case.case_id,
            voice_enrollment_coverage=enrollment_coverage,
            enrollment_silence_pass=enrollment_silence_pass,
            activation_present=activation_present,
            activation_assistant_response_count=(
                trace.activation_assistant_response_count
            ),
            voice_binding_probe_coverage=binding_probe_coverage,
            voice_binding_probe_accuracy=binding_probe_accuracy,
            explicit_attribution_rate=explicit_attribution_rate,
            session_wall_duration_ms=trace.session_wall_duration_ms,
            session_media_duration_ms=trace.session_media_duration_ms,
            max_tick_processing_ms=trace.max_tick_processing_ms,
            stalled_tick_count=trace.stalled_tick_count,
            clock_integrity_pass=clock_integrity_pass,
            wall_timing_eligible=wall_timing_eligible,
            wall_clock_overrun_ms=wall_clock_overrun_ms,
            interaction_mechanism_realized=trace.interaction_mechanism_realized,
            required_event_coverage=coverage,
            speaker_attribution_accuracy=speaker_accuracy,
            addressee_accuracy=addressee_accuracy,
            permission_decision_accuracy=permission_accuracy,
            unauthorized_action_attempts=unauthorized_attempts,
            unauthorized_actions_executed=unauthorized_executed,
            goal_disposition_accuracy=goal_accuracy,
            timing_pass_rate=timing_pass_rate,
            semantic_assertion_accuracy=semantic_assertion_accuracy,
            capability_scores=capability_scores,
            state_assertion_accuracy=state_assertion_accuracy,
            reference_state_match=reference_state_match,
            privacy_leak_count=len(trace.privacy_leaks),
            simulator_valid=simulator_valid,
            run_valid=run_valid,
            goal_disposition_success=goal_disposition_success,
            expected_tools_success=expected_tools_success,
            final_state_success=final_state_success,
            authorization_success=authorization_success,
            privacy_success=privacy_success,
            task_success=task_success,
            speaker_success=speaker_success,
            addressee_success=addressee_success,
            attribution_success=attribution_success,
            floor_control_score=floor_control_score,
            timing_score=timing_score,
            interruption_recovery_score=interruption_recovery_score,
            semantic_routing_score=semantic_routing_score,
            floor_mechanism=floor_mechanism,
            backchannel_policy_score=backchannel_policy_score,
            fairness_policy_score=fairness_policy_score,
            quiet_input_integration_score=quiet_input_integration_score,
            interaction_score=interaction_score,
            recognition_split=recognition_split,
            floor_control_success=floor_control_success,
            timing_success=timing_success,
            interruption_recovery_success=interruption_recovery_success,
            semantic_routing_success=semantic_routing_success,
            interaction_success=interaction_success,
            strict_joint_success=strict_joint_success,
            task_failures=task_failures,
            attribution_failures=attribution_failures,
            interaction_failures=interaction_failures,
            run_diagnostics=run_diagnostics,
            failures=failures,
        )
