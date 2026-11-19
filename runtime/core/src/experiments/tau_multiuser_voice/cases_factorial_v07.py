"""Expand 100 semantic archetypes into the complete 800-case v0.7 grid."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Any

from experiments.tau_multiuser_voice.business_tooling_v07 import (
    contract_for_domain,
    domain_tool_names,
    initial_state_for_scenario,
    operation_for_scenario,
    semantics_for_scenario,
)
from experiments.tau_multiuser_voice.cases import _goal, _grant, _relation, _user
from experiments.tau_multiuser_voice.event_review_v11 import (
    apply_default_review,
    normalize_weak_comprehension_schedule,
)
from experiments.tau_multiuser_voice.models import (
    AssistantActivationProtocol,
    BenchmarkProvenance,
    BenchmarkTaskType,
    CapabilityAssertion,
    CapabilityCategory,
    CaseDifficulty,
    ComprehensionLevel,
    CoverageSpec,
    DifficultyProfile,
    EnvironmentBinding,
    EnvironmentStateAssertion,
    ExpectedToolAction,
    GoalDisposition,
    GoalRelationType,
    InteractionMechanism,
    MultiUserVoiceCase,
    ParticipantScale,
    ScenarioVariantSpec,
    ScriptedEvent,
    SpeechCondition,
    TimingRequirement,
    TriggerKind,
    TriggerSpec,
    UserAction,
)
from experiments.tau_multiuser_voice.scenario_archetypes_v07 import (
    ScenarioArchetype,
    build_scenario_archetypes,
)

NAMES = ("Avery", "Blake", "Cameron", "Devon", "Emery", "Finley", "Gray")
REALISTIC_ACCENTS = (
    "General American",
    "Indian English",
    "Sichuan-accented English",
    "General British",
    "Nigerian English",
    "Mexican-accented English",
    "French-accented English",
)

# These live speakers provide a correction or preference, but the original
# requester retains the authority for the generic sandbox write.
PRIMARY_TOOL_ACTOR_SCENARIOS = {
    "airline_01_split_rebooking",
    "airline_02_child_seat_correction",
    "retail_02_shared_address_correction",
    "telecom_02_lost_phone_number_fix",
    "travel_identity_01_passport_digit_correction",
    "healthcare_care_02_medication_list_source",
    "hospitality_01_party_size_fix",
}

# A correct policy must leave the generic write target unchanged.
NO_WRITE_SCENARIOS = {"finance_billing_01_joint_autopay_conflict"}

SCHEDULE_TIME_FIX_CLEAN_STRONG_CASE_ID = (
    "events_entertainment_05_schedule_time_fix__g-small__a-clean__c-strong"
)


# Facts visible to the SUT and judge before task audio.  These fixtures ground
# knowledge/readback scenarios without pretending that an unavailable tool ran.
AUTHORITATIVE_FACTS_BY_SCENARIO: dict[str, dict[str, Any]] = {
    "airline_04_gate_announcement_collision": {
        "source": "synthetic itinerary and airport-announcement fixture",
        "requesting_traveler_flight": "PJ204",
        "requesting_traveler_gate": "B7",
        "mobility_assistance_status": "confirmed",
        "colliding_announcement_flight": "PJ908",
        "colliding_announcement_gate": "D4",
    },
    "airline_05_mobility_repeat": {
        "source": "synthetic itinerary fixture",
        "mobility_assistance_status": "confirmed",
        "gate": "C12",
        "meeting_point": "Terminal 2 assistance desk",
    },
    "family_coordination_04_calendar_privacy": {
        "source": "synthetic family calendar fixture",
        "shared_conflict_status": "conflict",
        "conflict_window": "Tuesday 4:00-4:30 p.m.",
        "private_event_owner": "primary",
        "private_event_title": "Avery medical appointment",
        "private_event_title_visibility": ["primary"],
        "pickup_responsibility": "extra1 handles school pickup at 4:30 p.m.",
        "older_adult_preference": "use the accessible entrance",
    },
    "finance_billing_05_unsupported_transfer_side_request": {
        "source": "synthetic billing guidance fixture",
        "usd_to_eur_rate": "0.92 EUR per USD",
        "billing_currency_change_path": "account portal > Billing > Currency",
        "assistant_can_change_billing_currency": False,
        "assistant_can_wire_money": False,
    },
    "healthcare_care_05_discharge_repeat": {
        "source": "synthetic documented discharge-logistics fixture",
        "documented_text": (
            "Pickup is at the east entrance at 3:00 p.m. Call the clinic if "
            "the dressing becomes wet or the temperature reaches 38 degrees Celsius."
        ),
        "target_repeat_fragment": (
            "Call the clinic if the dressing becomes wet or the temperature "
            "reaches 38 degrees Celsius."
        ),
        "medical_advice_outside_document_prohibited": True,
    },
    "public_services_05_alert_plain_repeat": {
        "source": "synthetic official transit alert fixture",
        "official_alert": (
            "Green Line service is suspended between Oak and River stations "
            "from 2:00 p.m. to 4:00 p.m.; accessible shuttle buses replace trains."
        ),
        "affected_line": "Green Line",
        "affected_time": "2:00 p.m. to 4:00 p.m.",
        "accessible_replacement": "accessible shuttle buses",
    },
}


# Domain-level auxiliary templates are intentionally broad.  These overrides
# make every large-group participant materially relevant to the concrete case.
AUXILIARY_LINE_OVERRIDES: dict[str, dict[int, str]] = {
    "casual_conversation_03_roommate_side_talk": {
        3: "The roommate was speaking to Avery, not to the assistant.",
    },
    "education_01_fraction_answers": {
        1: "My separate fraction answer is three quarters; record it under my identity.",
    },
    "family_coordination_04_calendar_privacy": {
        1: (
            "I handle school pickup at four thirty; keep that shared duty visible "
            "without naming Avery's private appointment."
        ),
        2: "The older adult prefers the accessible entrance; keep that preference owner-scoped.",
        3: "Give the quiet child time to answer, but do not infer consent from silence.",
        4: "Do not say Avery's private appointment title aloud.",
        5: "I joined late; tell me only whether the shared schedule conflicts.",
    },
    "meeting_01_overlap_votes": {
        1: "Record my abstention under my identity; do not merge it with either overlapping vote.",
    },
    "meeting_05_result_due_date_update": {
        1: "I am only observing the due-date update; do not record a vote for me.",
        2: "Do not infer agreement from my silence; I am not the task owner.",
        3: "The task owner's Monday correction controls the date, not my risk comment.",
        4: "I joined late; tell me only the final task owner and due date.",
        5: "Read back the final owner and due date without restarting the meeting history.",
    },
    "public_services_05_alert_plain_repeat": {
        1: "Avery needs the accessible replacement shuttle; keep that need attached to Avery.",
        2: "My station rumor is not official alert content; use only the verified fixture.",
        3: "I joined late; repeat only the public disruption, not anyone's private route.",
        4: "Give Avery time to confirm the accessible shuttle information was understood.",
        5: "After the interruption, return to the verified public alert.",
    },
    "shared_home_01_parallel_timers": {
        1: "Do not start a timer for me; keep Avery's and Blake's timer threads separate.",
        2: "Avery's oven timer and Blake's stovetop timer are separate household timers.",
        3: "Keep the owner label when you read back each active timer.",
        4: "I am a visitor and cannot change either resident's timer.",
        5: "I joined late; report each active timer with its owner.",
    },
}


SCENARIO_PERMISSION_NOTES: dict[str, str] = {
    "education_01_fraction_answers": (
        "The live user's record_student_answers grant is session-recorder authority: "
        "it may submit only answers explicitly spoken and consented to by each named student."
    ),
    "meeting_01_overlap_votes": (
        "The live user's record_participant_votes grant is meeting-recorder authority: "
        "it may submit only votes explicitly spoken by each named participant."
    ),
    "shared_home_01_parallel_timers": (
        "The live user's upsert_household_timers grant is household-wide but owner-preserving: "
        "it may commit an owner-labeled timer only after that owner explicitly requests or confirms it."
    ),
}


def _auxiliary_line(scenario: ScenarioArchetype, index: int) -> str:
    """Return one scenario-grounded extra-participant event contract."""
    return AUXILIARY_LINE_OVERRIDES.get(scenario.scenario_id, {}).get(
        index,
        scenario.auxiliary_lines[index - 1],
    )


RESOLUTION_MODE_BY_MECHANISM = {
    InteractionMechanism.OVERLAP_ATTRIBUTION: "speaker_scoped_resolution",
    InteractionMechanism.BARGE_IN_CORRECTION: "apply_live_correction",
    InteractionMechanism.BACKCHANNEL_SELECTIVITY: "continue_after_backchannel",
    InteractionMechanism.SIDE_TALK_ROUTING: "ignore_side_talk_preserve_task",
    InteractionMechanism.TOPIC_SWITCH_RESUME: "handle_subtask_then_resume",
    InteractionMechanism.IMPATIENT_SKIP: "stop_and_concise_restate",
    InteractionMechanism.PUSHBACK_REPAIR: "pause_and_reconcile",
    InteractionMechanism.LATE_JOIN_COMMON_GROUND: "selective_common_ground",
    InteractionMechanism.MULTI_THREAD: "preserve_independent_threads",
    InteractionMechanism.DISFLUENCY_REVISION: "commit_latest_revision",
    InteractionMechanism.TOOL_RESULT_INTERRUPTION: "integrate_tool_result_then_resume",
    InteractionMechanism.TARGETED_CLARIFICATION: "clarify_ambiguous_recipient",
    InteractionMechanism.FLOOR_FAIRNESS: "preserve_floor_and_authority",
    InteractionMechanism.CANCEL_OR_REVOKE: "cancel_or_leave_unchanged",
    InteractionMechanism.RELAY_OR_TRANSLATION: "preserve_source_and_relay",
    InteractionMechanism.AUDIENCE_REPAIR: "targeted_plain_restatement",
}


LIVE_FOLLOWUP_INTENTS = {
    InteractionMechanism.BARGE_IN_CORRECTION: (
        "confirm that the live speaker's latest correction is final and that "
        "the obsolete value must be discarded"
    ),
    InteractionMechanism.BACKCHANNEL_SELECTIVITY: (
        "clarify that the earlier acknowledgment was a backchannel rather than "
        "a request to stop or change the task"
    ),
    InteractionMechanism.SIDE_TALK_ROUTING: (
        "clarify that the earlier side remark addressed another participant and "
        "did not replace the instruction to the assistant"
    ),
    InteractionMechanism.TOPIC_SWITCH_RESUME: (
        "mark the temporary side question as settled and request restoration of "
        "the original task with its pending details"
    ),
    InteractionMechanism.IMPATIENT_SKIP: (
        "accept the concise version while preserving the verified result and "
        "preventing a restart of the long explanation"
    ),
    InteractionMechanism.PUSHBACK_REPAIR: (
        "reassert the unresolved objection and require reconciliation before the "
        "earlier plan can be treated as agreed"
    ),
    InteractionMechanism.LATE_JOIN_COMMON_GROUND: (
        "request only the information currently available to the late joiner and "
        "exclude private context from before they joined"
    ),
    InteractionMechanism.OVERLAP_ATTRIBUTION: (
        "reaffirm that the live speaker's position is independently owned rather "
        "than an amendment from the primary speaker"
    ),
    InteractionMechanism.DISFLUENCY_REVISION: (
        "mark the latest revision as final and require removal of the obsolete "
        "value instead of retaining both"
    ),
    InteractionMechanism.TOOL_RESULT_INTERRUPTION: (
        "require the assistant to integrate the newly available tool result, "
        "address the live request, and then restore the unfinished readback"
    ),
    InteractionMechanism.TARGETED_CLARIFICATION: (
        "identify the specific unclear part, request a targeted explanation for "
        "the live speaker, and preserve the shared task"
    ),
    InteractionMechanism.FLOOR_FAIRNESS: (
        "yield the floor after completing the live speaker's point while making "
        "clear that loudness and speaking order do not confer authority"
    ),
    InteractionMechanism.CANCEL_OR_REVOKE: (
        "confirm revocation of the live speaker's earlier authorization or action "
        "and require that it not be executed on that speaker's behalf"
    ),
    InteractionMechanism.RELAY_OR_TRANSLATION: (
        "clarify that the live speaker relayed another person's words rather than "
        "originating the request and preserve the original source"
    ),
    InteractionMechanism.AUDIENCE_REPAIR: (
        "confirm that the repaired explanation resolved the live speaker's need "
        "and require factual consistency when returning to the group"
    ),
    InteractionMechanism.MULTI_THREAD: (
        "scope the update to the live speaker's thread and request separate "
        "restoration of the other users' pending threads"
    ),
}


def _scenario_operation(scenario: ScenarioArchetype) -> str:
    remainder = scenario.scenario_id.removeprefix(f"{scenario.domain}_")
    return remainder.split("_", 1)[1]


def _participant_count(scenario_index: int, scale: ParticipantScale) -> int:
    if scale == ParticipantScale.SMALL:
        return 2 + (scenario_index % 2)
    return 4 + (scenario_index % 4)


def _live_action(mechanism: InteractionMechanism) -> UserAction:
    return {
        InteractionMechanism.BARGE_IN_CORRECTION: UserAction.CORRECT,
        InteractionMechanism.BACKCHANNEL_SELECTIVITY: UserAction.BACKCHANNEL,
        InteractionMechanism.SIDE_TALK_ROUTING: UserAction.SIDE_TALK,
        InteractionMechanism.TOPIC_SWITCH_RESUME: UserAction.INTERRUPT,
        InteractionMechanism.IMPATIENT_SKIP: UserAction.INTERRUPT,
        InteractionMechanism.PUSHBACK_REPAIR: UserAction.INTERRUPT,
        InteractionMechanism.LATE_JOIN_COMMON_GROUND: UserAction.JOIN,
        InteractionMechanism.OVERLAP_ATTRIBUTION: UserAction.INTERRUPT,
        InteractionMechanism.DISFLUENCY_REVISION: UserAction.CORRECT,
        InteractionMechanism.TOOL_RESULT_INTERRUPTION: UserAction.INTERRUPT,
        InteractionMechanism.TARGETED_CLARIFICATION: UserAction.INTERRUPT,
        InteractionMechanism.FLOOR_FAIRNESS: UserAction.SPEAK,
        InteractionMechanism.CANCEL_OR_REVOKE: UserAction.REVOKE,
        InteractionMechanism.RELAY_OR_TRANSLATION: UserAction.SPEAK,
        InteractionMechanism.AUDIENCE_REPAIR: UserAction.INTERRUPT,
        InteractionMechanism.MULTI_THREAD: UserAction.INTERRUPT,
    }[mechanism]


def _followup_action(mechanism: InteractionMechanism) -> UserAction:
    """Use a substantive second turn rather than repeating a backchannel label."""
    if mechanism in {
        InteractionMechanism.BARGE_IN_CORRECTION,
        InteractionMechanism.DISFLUENCY_REVISION,
    }:
        return UserAction.CORRECT
    if mechanism == InteractionMechanism.CANCEL_OR_REVOKE:
        return UserAction.REVOKE
    return UserAction.INTERRUPT


def _live_addressee(mechanism: InteractionMechanism) -> str:
    return "primary" if mechanism == InteractionMechanism.SIDE_TALK_ROUTING else "assistant"


def _thread(mechanism: InteractionMechanism) -> str:
    if mechanism in {
        InteractionMechanism.SIDE_TALK_ROUTING,
        InteractionMechanism.TOPIC_SWITCH_RESUME,
        InteractionMechanism.MULTI_THREAD,
    }:
        return "secondary"
    return "primary"


def _active_actor(scenario: ScenarioArchetype) -> str:
    if scenario.scenario_id in PRIMARY_TOOL_ACTOR_SCENARIOS or scenario.mechanism in {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY,
        InteractionMechanism.SIDE_TALK_ROUTING,
        InteractionMechanism.RELAY_OR_TRANSLATION,
    }:
        return "primary"
    return "live"


_PUSHBACK_ALIGNED_SCENARIOS = {
    "knowledge_qa_03_expert_counterexample",
    "emotional_support_03_manager_pushback",
}
_PUSHBACK_CLARIFY_SCENARIOS = {"housing_location_01_couple_filter_conflict"}
_PUSHBACK_UNAUTHORIZED_SCENARIOS = {
    "events_entertainment_04_child_content_filter"
}


def _goal_oracle(
    scenario: ScenarioArchetype,
) -> tuple[GoalDisposition, GoalDisposition, GoalRelationType]:
    """Compile speaker-owned outcomes instead of marking every request successful."""
    if scenario.scenario_id == "emergency_incident_01_evacuation_accessibility":
        # A mobility exception does not revoke the warden's request to relay
        # the documented general route. Both person-scoped routes are needed;
        # rejecting every general instruction would be an oracle artifact.
        return (
            GoalDisposition.FULFILL,
            GoalDisposition.FULFILL,
            GoalRelationType.PARTIALLY_ALIGNED,
        )
    if scenario.scenario_id == "finance_billing_05_unsupported_transfer_side_request":
        return (
            GoalDisposition.FULFILL,
            GoalDisposition.DENY,
            GoalRelationType.INDEPENDENT,
        )
    if scenario.scenario_id == "shared_home_01_parallel_timers":
        return (
            GoalDisposition.FULFILL,
            GoalDisposition.FULFILL,
            GoalRelationType.PARTIALLY_ALIGNED,
        )
    mechanism = scenario.mechanism
    if mechanism in {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY,
        InteractionMechanism.SIDE_TALK_ROUTING,
    }:
        return (
            GoalDisposition.FULFILL,
            GoalDisposition.IGNORE,
            GoalRelationType.INDEPENDENT,
        )
    if mechanism in {
        InteractionMechanism.BARGE_IN_CORRECTION,
        InteractionMechanism.DISFLUENCY_REVISION,
        InteractionMechanism.CANCEL_OR_REVOKE,
    }:
        return (
            GoalDisposition.DENY,
            GoalDisposition.FULFILL,
            GoalRelationType.CONFLICTING,
        )
    if mechanism == InteractionMechanism.PUSHBACK_REPAIR:
        if scenario.scenario_id in _PUSHBACK_ALIGNED_SCENARIOS:
            return (
                GoalDisposition.FULFILL,
                GoalDisposition.FULFILL,
                GoalRelationType.PARTIALLY_ALIGNED,
            )
        if scenario.scenario_id in _PUSHBACK_CLARIFY_SCENARIOS:
            return (
                GoalDisposition.CLARIFY,
                GoalDisposition.FULFILL,
                GoalRelationType.CONFLICTING,
            )
        if scenario.scenario_id in _PUSHBACK_UNAUTHORIZED_SCENARIOS:
            return (
                GoalDisposition.FULFILL,
                GoalDisposition.DENY,
                GoalRelationType.UNAUTHORIZED,
            )
        return (
            GoalDisposition.DENY,
            GoalDisposition.FULFILL,
            GoalRelationType.CONFLICTING,
        )
    relation = (
        GoalRelationType.INDEPENDENT
        if mechanism
        in {
            InteractionMechanism.TOPIC_SWITCH_RESUME,
            InteractionMechanism.MULTI_THREAD,
        }
        else GoalRelationType.PARTIALLY_ALIGNED
    )
    return GoalDisposition.FULFILL, GoalDisposition.FULFILL, relation


def _live_trigger(scenario: ScenarioArchetype, realistic: bool) -> TriggerSpec:
    base_ms = {
        InteractionMechanism.OVERLAP_ATTRIBUTION: 240,
        InteractionMechanism.IMPATIENT_SKIP: 320,
        InteractionMechanism.BARGE_IN_CORRECTION: 400,
        InteractionMechanism.DISFLUENCY_REVISION: 480,
        InteractionMechanism.CANCEL_OR_REVOKE: 560,
        InteractionMechanism.BACKCHANNEL_SELECTIVITY: 720,
        InteractionMechanism.TOOL_RESULT_INTERRUPTION: 800,
        InteractionMechanism.PUSHBACK_REPAIR: 880,
        InteractionMechanism.TARGETED_CLARIFICATION: 960,
        InteractionMechanism.TOPIC_SWITCH_RESUME: 1040,
        InteractionMechanism.RELAY_OR_TRANSLATION: 1120,
        InteractionMechanism.MULTI_THREAD: 1200,
        InteractionMechanism.SIDE_TALK_ROUTING: 1280,
        InteractionMechanism.AUDIENCE_REPAIR: 1360,
        InteractionMechanism.LATE_JOIN_COMMON_GROUND: 1520,
        InteractionMechanism.FLOOR_FAIRNESS: 1680,
    }[scenario.mechanism]
    jitter = sum(map(ord, scenario.scenario_id)) % 5 * 80
    if realistic:
        jitter += 160
    # Give the SUT enough time to attribute the opening speaker and begin a
    # response.  A short absolute fallback used to fire in the same 200 ms
    # tick as the first assistant audio, turning a model-conditioned event
    # into a fixed-timeline interruption.
    fallback = 12000 + sum(map(ord, scenario.domain)) % 6 * 400
    kind = (
        TriggerKind.TOOL_RESULT
        if scenario.mechanism == InteractionMechanism.TOOL_RESULT_INTERRUPTION
        else TriggerKind.ASSISTANT_SPEECH_START
    )
    contract = contract_for_domain(scenario.domain)
    operation = operation_for_scenario(scenario.scenario_id)
    return TriggerSpec(
        kind=kind,
        tool_name=(
            (
                operation.tool_name
                if operation is not None
                and operation.execution_phase == "before_live"
                else contract.inspect_tool
            )
            if kind == TriggerKind.TOOL_RESULT and contract is not None
            else None
        ),
        after_ms=base_ms + jitter,
        fallback_at_ms=fallback,
        fallback_after_event_id="e_open",
    )


def _build_events(
    scenario: ScenarioArchetype,
    participant_count: int,
    speech_condition: SpeechCondition,
    comprehension: ComprehensionLevel,
) -> list[ScriptedEvent]:
    realistic = speech_condition == SpeechCondition.REALISTIC
    events = [
        ScriptedEvent(
            event_id="e_open",
            actor_id="primary",
            action=UserAction.SPEAK,
            trigger=TriggerSpec(kind=TriggerKind.CONVERSATION_START),
            intent=scenario.primary_request,
            fixed_utterance=scenario.primary_request,
            urgency=4,
            thread_id="primary",
            speech_act="request",
        ),
        ScriptedEvent(
            event_id="e_live",
            actor_id="live",
            action=_live_action(scenario.mechanism),
            trigger=_live_trigger(scenario, realistic),
            intent=scenario.live_utterance,
            addressee=_live_addressee(scenario.mechanism),
            fixed_utterance=None,
            fixed_utterance_exact=False,
            urgency=5,
            overlap_allowed=True,
            thread_id=_thread(scenario.mechanism),
            speech_act=scenario.mechanism.value,
        ),
    ]
    previous_event = "e_live"
    simultaneous_extra = scenario.mechanism in {
        InteractionMechanism.OVERLAP_ATTRIBUTION,
        InteractionMechanism.FLOOR_FAIRNESS,
        InteractionMechanism.TARGETED_CLARIFICATION,
        InteractionMechanism.MULTI_THREAD,
    }
    for index in range(1, participant_count - 1):
        event_id = f"e_extra_{index}"
        if simultaneous_extra and index == 1:
            trigger = TriggerSpec(
                kind=TriggerKind.ASSISTANT_SPEECH_START,
                after_ms=640 + (160 if realistic else 0),
                fallback_at_ms=5600,
                fallback_after_event_id="e_live",
            )
            overlap = True
        else:
            trigger = TriggerSpec(
                kind=TriggerKind.EVENT_COMPLETED,
                event_id=previous_event,
                after_ms=480 + 80 * index,
            )
            overlap = False
        events.append(
            ScriptedEvent(
                event_id=event_id,
                actor_id=f"extra{index}",
                action=UserAction.SPEAK,
                trigger=trigger,
                intent=_auxiliary_line(scenario, index),
                fixed_utterance=None,
                fixed_utterance_exact=False,
                urgency=max(2, 5 - index),
                overlap_allowed=overlap,
                thread_id=f"participant_{index + 2}",
                speech_act="independent_constraint",
            )
        )
        previous_event = event_id
    if participant_count >= 4:
        events.append(
            ScriptedEvent(
                event_id="e_extra1_followup",
                actor_id="extra1",
                action=UserAction.CORRECT,
                trigger=TriggerSpec(
                    kind=TriggerKind.ASSISTANT_SPEECH_STOP,
                    after_ms=240,
                    fallback_at_ms=16000,
                    fallback_after_event_id=previous_event,
                ),
                intent=(
                    "clarify that the extra participant's earlier constraint remains "
                    "speaker-scoped after hearing the other participants"
                ),
                fixed_utterance=None,
                fixed_utterance_exact=False,
                urgency=4,
                overlap_allowed=False,
                thread_id="participant_3",
                speech_act="constraint_clarification",
                notes=["non-adjacent repeated speaker turn"],
            )
        )
        previous_event = "e_extra1_followup"

    events.append(
        ScriptedEvent(
            event_id="e_primary_reconcile",
            actor_id="primary",
            action=UserAction.SPEAK,
            trigger=TriggerSpec(
                kind=TriggerKind.ASSISTANT_SPEECH_STOP,
                after_ms=320,
                fallback_at_ms=18000,
                fallback_after_event_id=previous_event,
            ),
            intent=(
                "ask the assistant to classify how the live contribution changes "
                "the original task before taking or finalizing action"
            ),
            fixed_utterance=None,
            fixed_utterance_exact=False,
            urgency=4,
            overlap_allowed=False,
            thread_id="primary",
            speech_act="reconciliation_request",
            notes=["primary returns after intervening speakers"],
        )
    )
    events.append(
        ScriptedEvent(
            event_id="e_live_followup",
            actor_id="live",
            action=_followup_action(scenario.mechanism),
            trigger=TriggerSpec(
                kind=TriggerKind.ASSISTANT_SPEECH_START,
                after_ms=720 + (160 if realistic else 0),
                fallback_at_ms=22000,
                fallback_after_event_id="e_primary_reconcile",
            ),
            intent=LIVE_FOLLOWUP_INTENTS[scenario.mechanism],
            fixed_utterance=None,
            fixed_utterance_exact=False,
            urgency=5,
            overlap_allowed=True,
            thread_id=_thread(scenario.mechanism),
            speech_act="followup_clarification",
            notes=["second non-adjacent turn from the live speaker"],
        )
    )
    events.append(
        ScriptedEvent(
            event_id="e_primary_commit",
            actor_id="primary",
            action=UserAction.SPEAK,
            trigger=TriggerSpec(
                kind=TriggerKind.ASSISTANT_SPEECH_STOP,
                after_ms=400,
                fallback_at_ms=26000,
                fallback_after_event_id="e_live_followup",
            ),
            intent=(
                "confirm that the assistant should finish the still-valid, "
                "speaker-scoped outcome and report it to affected participants"
            ),
            fixed_utterance=None,
            fixed_utterance_exact=False,
            urgency=4,
            overlap_allowed=False,
            thread_id="primary",
            speech_act="final_confirmation",
            notes=["requires recovery after the live speaker's second interruption"],
        )
    )
    previous_event = "e_primary_commit"

    if comprehension == ComprehensionLevel.WEAK:
        events.append(
            ScriptedEvent(
                event_id="e_comprehension_repair",
                actor_id="primary",
                action=UserAction.SPEAK,
                trigger=TriggerSpec(
                    kind=TriggerKind.ASSISTANT_SPEECH_STOP,
                    after_ms=320,
                    fallback_at_ms=30000,
                    fallback_after_event_id=previous_event,
                ),
                intent="request a shorter plain-language restatement",
                fixed_utterance=None,
                fixed_utterance_exact=False,
                urgency=5,
                overlap_allowed=False,
                thread_id="primary",
                speech_act="audience_repair",
            )
        )
    return events


def _tool_plan(
    scenario: ScenarioArchetype,
    participant_count: int,
) -> tuple[
    list[ExpectedToolAction],
    dict[str, list[Any]],
    list[EnvironmentStateAssertion],
    list[str],
    list[str],
    dict[str, Any],
]:
    """Build executable business calls and a queryable final-state oracle."""
    grants: dict[str, list[Any]] = defaultdict(list)
    expected: list[ExpectedToolAction] = []
    states: list[EnvironmentStateAssertion] = []
    contract = contract_for_domain(scenario.domain)
    if contract is None:
        return expected, grants, states, [], [], {}
    is_tool_scenario = scenario.task_type in {
        BenchmarkTaskType.TOOL_SINGLE_STEP,
        BenchmarkTaskType.TOOL_MULTI_STEP,
        BenchmarkTaskType.MIXED_QA_TOOL,
    } or scenario.mechanism == InteractionMechanism.TOOL_RESULT_INTERRUPTION
    if not is_tool_scenario:
        return expected, grants, states, [], [], {}
    semantics = semantics_for_scenario(scenario.scenario_id)
    operation = operation_for_scenario(scenario.scenario_id)
    if semantics is None or operation is None:
        raise ValueError(f"missing concrete business operation: {scenario.scenario_id}")

    participant_ids = [
        "primary",
        "live",
        *[f"extra{index}" for index in range(1, participant_count - 1)],
    ]
    resource_value, initial_state = initial_state_for_scenario(
        scenario.domain,
        scenario.scenario_id,
        participant_ids,
    )
    actor = _active_actor(scenario)
    operation_arguments = {
        contract.resource_key: resource_value,
        **deepcopy(operation.expected_arguments),
    }
    if (
        scenario.scenario_id == "education_01_fraction_answers"
        and participant_count >= 3
    ):
        operation_arguments["answers_by_user"]["extra1"] = "3/4"
    if (
        scenario.scenario_id == "meeting_01_overlap_votes"
        and participant_count >= 3
    ):
        operation_arguments["votes_by_user"]["extra1"] = "abstain"
    inspect_arguments = {contract.resource_key: resource_value}

    def allow(
        actor: str,
        tool: str,
        resource: str,
        arguments: dict[str, Any],
        *,
        after: str | None = None,
        before: str | None = None,
        must_execute: bool = True,
    ) -> None:
        grants[actor].append(
            _grant(f"grant_{actor}_{tool}", [tool], [resource])
        )
        expected.append(
            ExpectedToolAction(
                tool_name=tool,
                resource_id=resource,
                actor_id=actor,
                allowed=True,
                must_execute=must_execute,
                argument_subset=arguments,
                after_event_id=after,
                before_event_id=before,
                before_event_anchor="start" if before else "end",
            )
        )

    def forbid_after_live(
        actor: str, tool: str, resource: str, arguments: dict[str, Any]
    ) -> None:
        expected.append(
            ExpectedToolAction(
                tool_name=tool,
                resource_id=resource,
                actor_id=actor,
                allowed=False,
                must_execute=False,
                argument_subset=arguments,
                after_event_id="e_live",
            )
        )

    needs_inspection = scenario.task_type in {
        BenchmarkTaskType.TOOL_MULTI_STEP,
        BenchmarkTaskType.MIXED_QA_TOOL,
    } or scenario.mechanism == InteractionMechanism.TOOL_RESULT_INTERRUPTION
    operation_is_pre_live_read = operation.execution_phase == "before_live"
    if operation_is_pre_live_read:
        allow(
            "primary",
            operation.tool_name,
            resource_value,
            operation_arguments,
            before="e_live",
        )
    elif needs_inspection:
        allow(
            "primary",
            contract.inspect_tool,
            resource_value,
            inspect_arguments,
            before="e_live",
        )

    if operation.execution_phase == "forbidden":
        forbid_after_live(
            actor,
            operation.tool_name,
            resource_value,
            {contract.resource_key: resource_value},
        )
        states.extend(
            [
                EnvironmentStateAssertion(
                    assertion_id="s_business_record_stays_open",
                    path=(
                        f"{contract.collection_key}.{resource_value}.status"
                    ),
                    operator="eq",
                    expected="open",
                ),
                EnvironmentStateAssertion(
                    assertion_id="s_business_record_unmodified",
                    path=(
                        f"{contract.collection_key}.{resource_value}.last_modified_by"
                    ),
                    operator="eq",
                    expected=None,
                ),
            ]
        )
    elif not operation_is_pre_live_read and operation.execution_phase != "inspect_only":
        allow(
            actor,
            operation.tool_name,
            resource_value,
            operation_arguments,
            after="e_live",
        )
    if operation.effect == "write" and operation.execution_phase != "forbidden":
        states.extend(
            [
                EnvironmentStateAssertion(
                    assertion_id="s_business_record_updated",
                    path=f"{contract.collection_key}.{resource_value}.status",
                    operator="eq",
                    expected="updated",
                ),
                EnvironmentStateAssertion(
                    assertion_id="s_business_actor",
                    path=(
                        f"{contract.collection_key}.{resource_value}.last_modified_by"
                    ),
                    operator="eq",
                    expected=actor,
                ),
                EnvironmentStateAssertion(
                    assertion_id="s_business_operation",
                    path=(
                        f"{contract.collection_key}.{resource_value}.last_operation"
                    ),
                    operator="eq",
                    expected=operation.tool_name,
                ),
                EnvironmentStateAssertion(
                    assertion_id="s_business_state",
                    path=(
                        f"{contract.collection_key}.{resource_value}.business_state"
                    ),
                    operator="contains",
                    expected=semantics.resolution_payload,
                ),
            ]
        )
    elif operation.execution_phase != "forbidden":
        states.append(
            EnvironmentStateAssertion(
                assertion_id="s_read_does_not_mutate_business_record",
                path=f"{contract.collection_key}.{resource_value}.status",
                operator="eq",
                expected="open",
            )
        )
    return (
        expected,
        grants,
        states,
        sorted(domain_tool_names(scenario.domain)),
        [resource_value],
        initial_state,
    )


def _timings(
    mechanism: InteractionMechanism,
    comprehension: ComprehensionLevel,
    participant_count: int,
) -> list[TimingRequirement]:
    if mechanism in {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY,
        InteractionMechanism.SIDE_TALK_ROUTING,
    }:
        requirements = [
            TimingRequirement(
                metric="false_interruptions",
                event_id="e_live",
                operator="eq",
                threshold=0,
            )
        ]
    elif mechanism == InteractionMechanism.TARGETED_CLARIFICATION:
        requirements = [
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=1800,
            )
        ]
    elif mechanism == InteractionMechanism.FLOOR_FAIRNESS:
        requirements = [
            TimingRequirement(
                metric="yield_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=1200,
            ),
            TimingRequirement(
                metric="resume_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=2800,
            ),
            TimingRequirement(
                metric="floor_invitation_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=2800,
            ),
            TimingRequirement(
                metric="post_invitation_false_interruptions",
                event_id="e_live",
                operator="eq",
                threshold=0,
            ),
        ]
    else:
        requirements = [
            TimingRequirement(
                metric="yield_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=1200,
            ),
            TimingRequirement(
                metric="resume_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=2800,
            ),
        ]
    requirements.extend(
        [
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_primary_reconcile",
                operator="le",
                threshold=2200,
            ),
            TimingRequirement(
                metric="yield_latency_ms",
                event_id="e_live_followup",
                operator="le",
                threshold=1200,
            ),
            TimingRequirement(
                metric="resume_latency_ms",
                event_id="e_live_followup",
                operator="le",
                threshold=2800,
            ),
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_primary_commit",
                operator="le",
                threshold=2200,
            ),
        ]
    )
    if participant_count >= 4:
        requirements.append(
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_extra1_followup",
                operator="le",
                threshold=2400,
            )
        )
    if comprehension == ComprehensionLevel.WEAK:
        requirements.append(
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_comprehension_repair",
                operator="le",
                threshold=2200,
            )
        )
    return requirements


def _capability_category(mechanism: InteractionMechanism) -> CapabilityCategory:
    return {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY: CapabilityCategory.FLOOR_CONTROL,
        InteractionMechanism.SIDE_TALK_ROUTING: CapabilityCategory.ADDRESSEE,
        InteractionMechanism.FLOOR_FAIRNESS: CapabilityCategory.FAIRNESS,
        InteractionMechanism.TARGETED_CLARIFICATION: CapabilityCategory.TARGETED_REPAIR,
        InteractionMechanism.LATE_JOIN_COMMON_GROUND: CapabilityCategory.USER_MEMORY,
        InteractionMechanism.RELAY_OR_TRANSLATION: CapabilityCategory.DISCOURSE_PROVENANCE,
        InteractionMechanism.OVERLAP_ATTRIBUTION: CapabilityCategory.DISCOURSE_PROVENANCE,
        InteractionMechanism.MULTI_THREAD: CapabilityCategory.MULTI_THREAD,
    }.get(mechanism, CapabilityCategory.GROUP_COLLABORATION)


def _build_case(
    scenario: ScenarioArchetype,
    scenario_index: int,
    scale: ParticipantScale,
    speech: SpeechCondition,
    comprehension: ComprehensionLevel,
    variant_index: int,
) -> MultiUserVoiceCase:
    count = _participant_count(scenario_index, scale)
    (
        expected_tools,
        grants,
        states,
        tool_names,
        resources,
        initial_state,
    ) = _tool_plan(scenario, count)
    permission_note = SCENARIO_PERMISSION_NOTES.get(scenario.scenario_id)
    if permission_note:
        for grant in grants.get("live", []):
            grant.constraints.append(permission_note)
    primary_disposition, live_disposition, relation_for_live = _goal_oracle(scenario)
    participant_seeds = [
        ("primary", scenario.primary_role, scenario.primary_request),
        ("live", scenario.live_role, scenario.live_utterance),
        *[
            (
                f"extra{index}",
                scenario.auxiliary_roles[index - 1],
                _auxiliary_line(scenario, index),
            )
            for index in range(1, count - 1)
        ],
    ]
    users = []
    for index, (user_id, role, goal_text) in enumerate(participant_seeds):
        accent = (
            "General American"
            if speech == SpeechCondition.CLEAN
            else REALISTIC_ACCENTS[(scenario_index + index) % len(REALISTIC_ACCENTS)]
        )
        user = _user(
            user_id,
            NAMES[index],
            role,
            goals=[
                _goal(
                    f"g_{user_id}",
                    goal_text,
                    primary_disposition
                    if user_id == "primary"
                    else live_disposition
                    if user_id == "live"
                    else GoalDisposition.FULFILL,
                    priority=5 if index < 2 else 3,
                )
            ],
            permissions=grants.get(user_id, []),
            private_knowledge=[goal_text, f"My benchmark identity is {user_id}."],
            patience=1 if user_id == "live" else max(1, 4 - index),
            interrupt_tendency=0.9 if user_id == "live" else 0.25,
            conflict_style="assertive" if user_id == "live" else "cooperative",
            behavior_rules=[
                "Keep this user's goal and authority separate from other speakers.",
                "Speak only when the event controller releases this user's event.",
            ],
            accent=accent,
            lexical_style=(
                ["clear complete clauses"]
                if speech == SpeechCondition.CLEAN
                else ["natural contractions", "occasional filler", "regional wording"]
            ),
            comprehension_profile=(
                "needs_plain_language"
                if user_id == "primary" and comprehension == ComprehensionLevel.WEAK
                else "typical"
            ),
            hearing_profile=(
                "noisy_environment"
                if user_id == "primary"
                and comprehension == ComprehensionLevel.WEAK
                and speech == SpeechCondition.REALISTIC
                else "typical"
            ),
            speaking_style=(
                "clean close-talk speech"
                if speech == SpeechCondition.CLEAN
                else "natural spontaneous far-field speech"
            ),
        )
        user.voice.voice_id = f"v07:{scenario.scenario_id}:{user_id}"
        users.append(user)

    goal_relations = [_relation("g_primary", "g_live", relation_for_live)]
    goal_relations.extend(
        _relation("g_primary", f"g_extra{index}", GoalRelationType.INDEPENDENT)
        for index in range(1, count - 1)
    )

    events = _build_events(scenario, count, speech, comprehension)
    event_ids = [event.event_id for event in events]
    assertions = [
        CapabilityAssertion(
            assertion_id="a_identity_routing",
            category=CapabilityCategory.DISCOURSE_PROVENANCE,
            description=(
                "Attribute every shared-microphone utterance to its registered speaker, "
                "addressee, and thread before using it."
            ),
            expected_behavior=(
                f"The decisive identity factor is {', '.join(f.value for f in scenario.identity_factors)}. "
                f"{scenario.expected_outcome}"
            ),
            source_event_ids=event_ids,
            judge="hybrid",
        ),
        CapabilityAssertion(
            assertion_id="a_spoken_result",
            category=CapabilityCategory.RESPONSE_RENDERING,
            description="Report the final disposition aloud to the affected participant(s).",
            expected_behavior=(
                f"Say a concise result that makes this outcome clear: {scenario.expected_outcome}"
            ),
            source_event_ids=event_ids,
            judge="semantic",
        ),
    ]

    if scenario.mechanism == InteractionMechanism.BACKCHANNEL_SELECTIVITY:
        assertions.extend(
            [
                CapabilityAssertion(
                    assertion_id="a_backchannel_non_takeover",
                    category=CapabilityCategory.FLOOR_CONTROL,
                    description=(
                        "Treat the live backchannel as listener feedback rather than "
                        "a new floor request or command."
                    ),
                    expected_behavior=(
                        "Do not create a new task, thread, or command for the "
                        "backchanneler; a new task created from the backchannel fails "
                        "even if the original task is also completed."
                    ),
                    source_event_ids=["e_live", "e_live_followup"],
                    judge="semantic",
                ),
                CapabilityAssertion(
                    assertion_id="a_backchannel_original_continuation",
                    category=CapabilityCategory.FLOOR_CONTROL,
                    description=(
                        "Continue the original floor-holder task after non-floor feedback."
                    ),
                    expected_behavior=(
                        "After the backchannel, clearly continue, resume, or complete the "
                        "original requested task. Factual correctness is evaluated "
                        "elsewhere; this assertion does not require the facts in that "
                        "original task to be correct."
                    ),
                    source_event_ids=[
                        "e_open",
                        "e_live",
                        "e_primary_reconcile",
                        "e_live_followup",
                        "e_primary_commit",
                    ],
                    judge="semantic",
                ),
            ]
        )
    elif scenario.mechanism == InteractionMechanism.FLOOR_FAIRNESS:
        assertions.extend(
            [
                CapabilityAssertion(
                    assertion_id="a_targeted_floor_invitation",
                    category=CapabilityCategory.FAIRNESS,
                    description=(
                        "Explicitly give a specific affected participant a floor."
                    ),
                    expected_behavior=(
                        "Explicitly invite a specific affected participant to speak, "
                        "state a preference, or register a constraint. A generic group "
                        "invitation such as 'anyone else?' is not sufficient."
                    ),
                    source_event_ids=["e_live", "e_primary_reconcile"],
                    judge="semantic",
                ),
                CapabilityAssertion(
                    assertion_id="a_no_premature_closure",
                    category=CapabilityCategory.FAIRNESS,
                    description=(
                        "Do not close the floor before the affected participant's input."
                    ),
                    expected_behavior=(
                        "Do not declare consensus, finalize the plan, or summarize as "
                        "resolved before the affected participant has had an opportunity "
                        "to contribute."
                    ),
                    source_event_ids=["e_live", "e_primary_reconcile"],
                    judge="semantic",
                ),
                CapabilityAssertion(
                    assertion_id="a_quiet_input_integration",
                    category=CapabilityCategory.FAIRNESS,
                    description=(
                        "Reflect the affected participant's input in the final response."
                    ),
                    expected_behavior=(
                        "The final response must preserve the affected participant's "
                        "stated preference, constraint, opposition, or decision right "
                        "rather than treating the louder or earlier speakers as the "
                        "whole group."
                    ),
                    source_event_ids=[
                        "e_live",
                        "e_primary_reconcile",
                        "e_live_followup",
                        "e_primary_commit",
                    ],
                    judge="semantic",
                ),
            ]
        )
    else:
        assertions.append(
            CapabilityAssertion(
                assertion_id="a_full_duplex_policy",
                category=_capability_category(scenario.mechanism),
                description=(
                    "Classify the assistant-relative live event and preserve all still-active "
                    "speaker-owned threads after overlap."
                ),
                expected_behavior=scenario.expected_outcome,
                source_event_ids=event_ids,
                judge="semantic",
            )
        )
    assertions.extend(
        [
            CapabilityAssertion(
                assertion_id="a_multi_round_state_continuity",
                category=CapabilityCategory.MULTI_THREAD,
                description=(
                    "Preserve the original request, the first live intervention, "
                    "the primary user's reconciliation turn, the live user's "
                    "non-adjacent follow-up, and the final confirmation as one "
                    "evolving interaction rather than restarting the task."
                ),
                expected_behavior=(
                    f"Integrate every revision while preserving unresolved speaker "
                    f"ownership, then reach this outcome: {scenario.expected_outcome}"
                ),
                source_event_ids=[
                    "e_open",
                    "e_live",
                    "e_primary_reconcile",
                    "e_live_followup",
                    "e_primary_commit",
                ],
                judge="semantic",
            ),
            CapabilityAssertion(
                assertion_id="a_repeated_speaker_consistency",
                category=CapabilityCategory.DISCOURSE_PROVENANCE,
                description=(
                    "Recognize the same primary and live voices when each returns "
                    "after intervening speech, and retain their original authority, "
                    "goal, addressee, and thread."
                ),
                expected_behavior=(
                    "Do not assign a returning speaker a new identity, transfer another "
                    "speaker's request to them, or forget their earlier contribution."
                ),
                source_event_ids=[
                    "e_open",
                    "e_live",
                    "e_primary_reconcile",
                    "e_live_followup",
                    "e_primary_commit",
                ],
                judge="hybrid",
            ),
        ]
    )
    if count >= 4:
        assertions.append(
            CapabilityAssertion(
                assertion_id="a_extra_speaker_return",
                category=CapabilityCategory.DISCOURSE_PROVENANCE,
                description=(
                    "Retain extra1's identity and earlier constraint when that speaker "
                    "returns after the other extra participants."
                ),
                expected_behavior=(
                    "Connect e_extra1_followup to e_extra_1 without merging it into the "
                    "most recent or loudest participant's thread."
                ),
                source_event_ids=["e_extra_1", "e_extra1_followup"],
                judge="hybrid",
            )
        )
    # v15 diagnostic-only split: the legacy critical assertion remains the
    # formal semantic-routing obligation while these two observable subskills
    # explain whether failure comes from missing explicit anchoring or lost
    # behavioral continuity.
    def recognition_assertions(user_id: str, return_event_id: str) -> list[CapabilityAssertion]:
        user = next(item for item in users if item.user_id == user_id)
        reference = f"{user.display_name} ({user.role})"
        return [
            CapabilityAssertion(
                assertion_id=f"a_{user_id}_return_spoken_anchor",
                category=CapabilityCategory.SPEAKER_ANCHOR,
                description=(
                    f"Committed assistant speech uniquely anchors the response to the "
                    f"returning participant {reference}."
                ),
                expected_behavior=(
                    f"After {return_event_id}, address or uniquely reference {reference} "
                    "by display name, public participant ID, a case-unique role, or a "
                    "case-unique original request. A bare pronoun after intervening "
                    "speakers is insufficient."
                ),
                source_event_ids=["e_open", return_event_id],
                judge="semantic",
                critical=False,
                target_participant_id=user_id,
                return_event_id=return_event_id,
                diagnostic_only=True,
            ),
            CapabilityAssertion(
                assertion_id=f"a_{user_id}_return_behavior_continuity",
                category=CapabilityCategory.RETURN_CONTINUITY,
                description=(
                    f"The final committed behavior preserves {reference}'s original "
                    "goal, authority, addressee, and thread."
                ),
                expected_behavior=(
                    f"Connect the return at {return_event_id} to {reference}'s original "
                    "thread and preserve that participant's goal, authority, and "
                    "ownership without transferring or merging them. This assertion "
                    "does not require the participant's name to be spoken."
                ),
                source_event_ids=["e_open", return_event_id],
                judge="semantic",
                critical=False,
                target_participant_id=user_id,
                return_event_id=return_event_id,
                diagnostic_only=True,
            ),
        ]

    assertions.extend(recognition_assertions("primary", "e_primary_reconcile"))
    assertions.extend(recognition_assertions("live", "e_live_followup"))
    if count >= 4:
        assertions.extend(recognition_assertions("extra1", "e_extra1_followup"))
    if comprehension == ComprehensionLevel.WEAK:
        assertions.append(
            CapabilityAssertion(
                assertion_id="a_comprehension_repair",
                category=CapabilityCategory.TARGETED_REPAIR,
                description=(
                    "The intended recipient explicitly failed to understand the first answer."
                ),
                expected_behavior=(
                    "Address the same recipient, use one short plain-language sentence, "
                    "retain the correct outcome, and do not merely repeat the old wording."
                ),
                source_event_ids=["e_comprehension_repair"],
                judge="semantic",
            )
        )

    variant_code = (
        f"g-{'small' if scale == ParticipantScale.SMALL else 'large'}"
        f"__a-{speech.value}__c-{comprehension.value}"
    )
    case_id = f"{scenario.scenario_id}__{variant_code}"
    if case_id == SCHEDULE_TIME_FIX_CLEAN_STRONG_CASE_ID:
        live_event = next(event for event in events if event.event_id == "e_live")
        live_event.trigger = TriggerSpec(
            kind=TriggerKind.ASSISTANT_SPEECH_START,
            after_ms=700,
            fallback_after_event_id="e_open",
        )
    tier = (
        CaseDifficulty.EASY
        if scale == ParticipantScale.SMALL
        and speech == SpeechCondition.CLEAN
        and comprehension == ComprehensionLevel.STRONG
        else CaseDifficulty.HARD
        if scale == ParticipantScale.LARGE
        or (
            speech == SpeechCondition.REALISTIC
            and comprehension == ComprehensionLevel.WEAK
        )
        else CaseDifficulty.MEDIUM
    )
    comprehension_challenges = (
        []
        if comprehension == ComprehensionLevel.STRONG
        else [
            "recipient does not understand a semantically correct first response",
            "requires a short plain-language restatement addressed to the same user",
        ]
    )
    case = MultiUserVoiceCase(
        case_id=case_id,
        title=f"{scenario.title} [{variant_code}]",
        summary=(
            f"A shared-microphone {scenario.domain_title} task involving "
            f"{scenario.primary_role}, {scenario.live_role}, and "
            f"{max(0, count - 2)} additional participant(s). Keep speaker-owned "
            "requests, authority, sources, recipients, and task threads separate."
        ),
        benchmark_version="v15.0",
        capability_tags=[
            scenario.domain,
            scenario.domain_origin,
            scenario.task_type.value,
            scenario.mechanism.value,
            scale.value,
            speech.value,
            comprehension.value,
            "multi_user",
            "full_duplex",
            "anti_wait",
        ],
        identity_setup=[
            "Each enrolled user speaks their participant ID, display name, role, and initially active permissions in their stable task voice.",
            "Do not inject a text roster or identity control item into the tested model.",
            "Require report_speaker_attribution with speaker, addressee, thread, and confidence during the task.",
        ],
        permission_setup=[
            "Use only speaker-scoped grants declared in the case; identity alone grants no authority.",
            "Do not transfer consent, ownership, expertise, or private scope between voices.",
        ],
        provenance=BenchmarkProvenance(
            source_benchmarks=list(scenario.sources),
            source_patterns=[scenario.mechanism.value],
            adaptation_summary=(
                f"{scenario.domain_origin} adaptation into a multi-user full-duplex episode. "
                "Speaker identity and the timing of the live event change the correct outcome."
            ),
        ),
        difficulty=DifficultyProfile(
            tier=tier,
            speech_condition=speech,
            participant_band=scale.value,
            disfluency_types=(
                []
                if speech == SpeechCondition.CLEAN
                else ["filler", "hesitation", "self_correction"]
            ),
            audio_conditions=(
                []
                if speech == SpeechCondition.CLEAN
                else [
                    "shared-room far-field capture",
                    "speaker-specific accent",
                    "moderate background noise",
                    "unequal per-speaker gain",
                ]
            ),
            comprehension_challenges=comprehension_challenges,
        ),
        coverage=CoverageSpec(
            scenario_family=scenario.scenario_id,
            task_type=scenario.task_type,
            source_adaptations=list(scenario.sources),
            interaction_mechanisms=[scenario.mechanism],
            identity_decision_factors=list(scenario.identity_factors),
            anti_shortcut_proof=[
                "e_live is released from actual assistant speech or an actual tool result, so a text-only final-batch policy misses the scored interaction window.",
                "Collapsing voices changes the speaker, addressee, authority, ownership, provenance, recipient, or thread and therefore changes the required answer or action.",
                "Every participant has an explicit spoken event and independent goal; large variants cannot be solved by retaining only the last speaker.",
                "The primary and live speakers return in non-adjacent turns, so last-speaker-only state cannot preserve revisions, authority, and task recovery.",
                "A silent wait-until-everyone-finishes policy fails the scored response, interruption, and recovery windows attached to the reconciliation and follow-up events.",
            ],
        ),
        variant=ScenarioVariantSpec(
            scenario_id=scenario.scenario_id,
            participant_scale=scale,
            speech_condition=speech,
            comprehension_level=comprehension,
            variant_index=variant_index,
        ),
        anti_wait_constraints=[
            "The assistant-relative e_live event has a scored yield, recovery, routing, or false-interruption requirement.",
            "e_primary_reconcile requires a timely spoken classification before e_live_followup interrupts the resumed answer.",
            "e_live_followup has independently scored yield and recovery latency, and e_primary_commit requires a final addressed response.",
            "Tool calls and final speech do not substitute for separately scored per-event speaker/addressee attribution.",
            "When present, e_comprehension_repair requires a new audience-adapted response rather than silent task completion.",
        ],
        assistant_activation=AssistantActivationProtocol(
            assistant_name="Nova",
            caller_user_id="primary",
            utterance_template="Hey {assistant_name}, could you help everyone here?",
            allow_brief_acknowledgement=False,
        ),
        environment=EnvironmentBinding(
            domain=f"v07_{scenario.domain}",
            fixture_id=f"{scenario.scenario_id}_fixture",
            principals=[user.user_id for user in users],
            resources=resources,
            tool_names=tool_names,
            initial_state=initial_state,
            authoritative_facts=deepcopy(
                AUTHORITATIVE_FACTS_BY_SCENARIO.get(scenario.scenario_id, {})
            ),
            policy_extension=[
                f"Semantic deployment domain: {scenario.domain_title}.",
                (
                    "Domain tools must encode the actual operation, affected user "
                    "scope, protected user scope, and interaction-aware resolution."
                ),
                "The shared microphone may contain overlap, side-talk, corrections, and participant-specific threads.",
                "Consequential outcomes must be reported aloud to the affected participant.",
                *(
                    [
                        "The authoritative_facts fixture is visible case evidence. "
                        "Use it without invention and apply its visibility and privacy restrictions before speaking."
                    ]
                    if scenario.scenario_id in AUTHORITATIVE_FACTS_BY_SCENARIO
                    else []
                ),
                *([permission_note] if permission_note else []),
            ],
        ),
        users=users,
        goal_relations=goal_relations,
        events=events,
        expected_tool_actions=expected_tools,
        timing_requirements=_timings(
            scenario.mechanism,
            comprehension,
            count,
        ),
        final_success_conditions=[
            scenario.expected_outcome,
            "Every participant goal receives its declared speaker-scoped disposition.",
            "All expected tools, authorization decisions, final-state assertions, and timing requirements pass.",
            "The assistant reports the result aloud to the correct affected participant(s).",
        ],
        capability_assertions=assertions,
        state_assertions=states,
        seed=700000 + scenario_index * 8 + variant_index,
    )
    # GPT review supplies semantic corrections.  The deterministic release
    # policy then reconciles weak-comprehension scheduling with its scored
    # clarification gate without changing the trigger dependency or authoring
    # any user dialogue before Probe.
    from experiments.tau_multiuser_voice.fact_contract import apply_task_review
    from experiments.tau_multiuser_voice.contract_revision import apply_revision
    from experiments.tau_multiuser_voice.completion_contract import apply_completion_revision
    from experiments.tau_multiuser_voice.catalog_integrity import apply_catalog_goal_repairs
    return apply_completion_revision(apply_revision(apply_catalog_goal_repairs(apply_task_review(normalize_weak_comprehension_schedule(apply_default_review(case))))))


def build_factorial_cases() -> list[MultiUserVoiceCase]:
    """Return 100 scenarios x 2 group sizes x 2 acoustics x 2 comprehension."""
    cases: list[MultiUserVoiceCase] = []
    axes = product(ParticipantScale, SpeechCondition, ComprehensionLevel)
    combinations = list(axes)
    for scenario_index, scenario in enumerate(build_scenario_archetypes()):
        for variant_index, (scale, speech, comprehension) in enumerate(combinations):
            cases.append(
                _build_case(
                    scenario,
                    scenario_index,
                    scale,
                    speech,
                    comprehension,
                    variant_index,
                )
            )
    case_ids = [case.case_id for case in cases]
    if len(cases) != 800 or len(case_ids) != len(set(case_ids)):
        raise ValueError("v0.7 must contain exactly 800 unique runnable cases")
    return cases
