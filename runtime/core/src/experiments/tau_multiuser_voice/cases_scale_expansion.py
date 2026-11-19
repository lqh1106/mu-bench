"""Sixty-four diverse v0.6 cases bringing the default catalog to 100."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.tau_multiuser_voice.case_audit import audit_scaled_case_pack
from experiments.tau_multiuser_voice.cases import _goal, _grant, _relation, _user
from experiments.tau_multiuser_voice.models import (
    AssistantActivationProtocol,
    BenchmarkProvenance,
    BenchmarkTaskType,
    CapabilityAssertion,
    CapabilityCategory,
    CaseDifficulty,
    CoverageSpec,
    DifficultyProfile,
    EnvironmentBinding,
    EnvironmentStateAssertion,
    ExpectedToolAction,
    GoalDisposition,
    GoalRelationType,
    IdentityDecisionFactor,
    InteractionMechanism,
    MultiUserVoiceCase,
    ScriptedEvent,
    SpeechCondition,
    TimingRequirement,
    TriggerKind,
    TriggerSpec,
    UserAction,
)


@dataclass(frozen=True)
class ExtraSpeaker:
    """One additional participant whose independent constraint must survive."""

    role: str
    goal: str
    utterance: str
    action: UserAction = UserAction.SPEAK
    addressee: str = "assistant"
    thread_id: str = "main"


@dataclass(frozen=True)
class ToolStep:
    """Expected virtual tool call and its causal window."""

    name: str
    resource: str
    actor: str
    arguments: dict[str, Any]
    allowed: bool = True
    must_execute: bool = True
    after: str | None = None
    before: str | None = None


@dataclass(frozen=True)
class StateCheck:
    """Expected final virtual state."""

    path: str
    expected: Any
    operator: str = "eq"


@dataclass(frozen=True)
class ScaleScenario:
    """Compact human-authored seed expanded into a fully validated case."""

    case_id: str
    title: str
    domain: str
    family: str
    task_type: BenchmarkTaskType
    source_adaptations: tuple[str, ...]
    mechanisms: tuple[InteractionMechanism, ...]
    identity_factors: tuple[IdentityDecisionFactor, ...]
    summary: str
    primary_role: str
    primary_goal: str
    primary_line: str
    live_role: str
    live_goal: str
    live_line: str
    outcome: str
    spoken_result: str
    extras: tuple[ExtraSpeaker, ...] = ()
    final_line: str | None = None
    tools: tuple[ToolStep, ...] = ()
    states: tuple[StateCheck, ...] = ()
    initial_state: dict[str, Any] = field(default_factory=dict)
    resources: tuple[str, ...] = ()
    speech_condition: SpeechCondition | None = None
    disfluencies: tuple[str, ...] = ()
    comprehension: tuple[str, ...] = ()


NAMES = ("Avery", "Blake", "Cameron", "Devon", "Emery", "Finley")
ACCENTS = (
    "General American",
    "Indian English",
    "Sichuan-accented English",
    "French-accented English",
    "Bengali-accented English",
    "General British",
)


def _live_action(mechanisms: tuple[InteractionMechanism, ...]) -> UserAction:
    primary = mechanisms[0]
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
    }[primary]


def _live_addressee(mechanisms: tuple[InteractionMechanism, ...]) -> str:
    if mechanisms[0] == InteractionMechanism.SIDE_TALK_ROUTING:
        return "primary"
    return "assistant"


def _live_offsets(scenario: ScaleScenario) -> tuple[int, int]:
    """Give interaction mechanisms distinct assistant-relative timing regimes."""
    base_after = {
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
    }[scenario.mechanisms[0]]
    jitter = sum(map(ord, scenario.case_id)) % 5 * 80
    fallback = 3600 + sum(map(ord, scenario.family)) % 6 * 400
    return base_after + jitter, fallback


def _speech_condition(scenario: ScaleScenario) -> SpeechCondition:
    """Mix clean controls with realistic acoustic and linguistic conditions."""
    if scenario.speech_condition is not None:
        return scenario.speech_condition
    clean_control = (
        len(scenario.extras) <= 1
        and not scenario.disfluencies
        and not scenario.comprehension
        and sum(map(ord, scenario.case_id)) % 4 == 0
    )
    return SpeechCondition.CLEAN if clean_control else SpeechCondition.REALISTIC


def _live_trigger(
    scenario: ScaleScenario, after_ms: int, fallback_ms: int
) -> TriggerSpec:
    """Trigger ordinary overlap from speech and result-interruptions from tools."""
    if scenario.mechanisms[0] == InteractionMechanism.TOOL_RESULT_INTERRUPTION:
        prior_tools = [tool for tool in scenario.tools if tool.before == "live"]
        if not prior_tools:
            raise ValueError(
                f"{scenario.case_id}: tool-result interruption needs a pre-live tool"
            )
        return TriggerSpec(
            kind=TriggerKind.TOOL_RESULT,
            tool_name=prior_tools[-1].name,
            after_ms=after_ms,
            fallback_at_ms=fallback_ms,
        )
    return TriggerSpec(
        kind=TriggerKind.ASSISTANT_SPEECH_START,
        after_ms=after_ms,
        fallback_at_ms=fallback_ms,
    )


def _capability_category(
    mechanisms: tuple[InteractionMechanism, ...],
) -> CapabilityCategory:
    primary = mechanisms[0]
    return {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY: CapabilityCategory.FLOOR_CONTROL,
        InteractionMechanism.SIDE_TALK_ROUTING: CapabilityCategory.ADDRESSEE,
        InteractionMechanism.FLOOR_FAIRNESS: CapabilityCategory.FAIRNESS,
        InteractionMechanism.TARGETED_CLARIFICATION: CapabilityCategory.TARGETED_REPAIR,
        InteractionMechanism.LATE_JOIN_COMMON_GROUND: CapabilityCategory.USER_MEMORY,
        InteractionMechanism.RELAY_OR_TRANSLATION: CapabilityCategory.DISCOURSE_PROVENANCE,
        InteractionMechanism.OVERLAP_ATTRIBUTION: CapabilityCategory.DISCOURSE_PROVENANCE,
    }.get(primary, CapabilityCategory.MULTI_THREAD)


def _timings(scenario: ScaleScenario) -> list[TimingRequirement]:
    primary = scenario.mechanisms[0]
    if primary in {
        InteractionMechanism.BACKCHANNEL_SELECTIVITY,
        InteractionMechanism.SIDE_TALK_ROUTING,
    }:
        return [
            TimingRequirement(
                metric="false_interruptions",
                event_id="e_live",
                operator="eq",
                threshold=0,
            )
        ]
    if primary == InteractionMechanism.TARGETED_CLARIFICATION:
        return [
            TimingRequirement(
                metric="clarification_latency_ms",
                event_id="e_live",
                operator="le",
                threshold=1800,
            )
        ]
    return [
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
            threshold=2600,
        ),
    ]


def _events(scenario: ScaleScenario) -> list[ScriptedEvent]:
    live_after_ms, live_fallback_ms = _live_offsets(scenario)
    events = [
        ScriptedEvent(
            event_id="e_open",
            actor_id="primary",
            action=UserAction.SPEAK,
            trigger=TriggerSpec(kind=TriggerKind.CONVERSATION_START),
            intent=scenario.primary_goal,
            fixed_utterance=scenario.primary_line,
            urgency=4,
            thread_id="primary",
            speech_act="request",
        ),
        ScriptedEvent(
            event_id="e_live",
            actor_id="live",
            action=_live_action(scenario.mechanisms),
            trigger=_live_trigger(
                scenario,
                after_ms=live_after_ms,
                fallback_ms=live_fallback_ms,
            ),
            intent=scenario.live_goal,
            addressee=_live_addressee(scenario.mechanisms),
            fixed_utterance=scenario.live_line,
            urgency=5,
            overlap_allowed=True,
            thread_id=(
                "secondary"
                if scenario.mechanisms[0]
                in {
                    InteractionMechanism.SIDE_TALK_ROUTING,
                    InteractionMechanism.TOPIC_SWITCH_RESUME,
                    InteractionMechanism.MULTI_THREAD,
                }
                else "primary"
            ),
            speech_act=scenario.mechanisms[0].value,
        ),
    ]
    previous = "e_live"
    simultaneous_extras = scenario.mechanisms[0] in {
        InteractionMechanism.OVERLAP_ATTRIBUTION,
        InteractionMechanism.FLOOR_FAIRNESS,
        InteractionMechanism.TARGETED_CLARIFICATION,
    }
    for index, extra in enumerate(scenario.extras, start=1):
        event_id = f"e_extra_{index}"
        trigger = (
            TriggerSpec(
                kind=TriggerKind.ASSISTANT_SPEECH_START,
                after_ms=560 + 80 * index,
                fallback_at_ms=4600 + 400 * index,
            )
            if simultaneous_extras and index == 1
            else TriggerSpec(
                kind=TriggerKind.EVENT_COMPLETED,
                event_id=previous,
                after_ms=520,
            )
        )
        events.append(
            ScriptedEvent(
                event_id=event_id,
                actor_id=f"extra{index}",
                action=extra.action,
                trigger=trigger,
                intent=extra.goal,
                addressee=extra.addressee,
                fixed_utterance=extra.utterance,
                urgency=4,
                overlap_allowed=simultaneous_extras and index == 1,
                thread_id=extra.thread_id,
                speech_act=extra.action.value,
            )
        )
        previous = event_id
    if scenario.final_line:
        events.append(
            ScriptedEvent(
                event_id="e_final",
                actor_id="primary",
                action=UserAction.CONFIRM,
                trigger=TriggerSpec(
                    kind=TriggerKind.EVENT_COMPLETED,
                    event_id=previous,
                    after_ms=650,
                ),
                intent="final speaker-scoped confirmation",
                fixed_utterance=scenario.final_line,
                urgency=5,
                thread_id="primary",
                speech_act="confirmation",
            )
        )
    return events


def _resolve_event_alias(alias: str | None) -> str | None:
    if alias is None:
        return None
    if alias in {"open", "live", "final"}:
        return f"e_{alias}"
    if alias.startswith("extra"):
        return "e_extra_" + alias.removeprefix("extra")
    raise ValueError(f"unknown event alias: {alias}")


def _build_case(scenario: ScaleScenario, case_index: int) -> MultiUserVoiceCase:
    speech_condition = _speech_condition(scenario)
    participant_seeds = [
        ("primary", NAMES[0], scenario.primary_role, scenario.primary_goal),
        ("live", NAMES[1], scenario.live_role, scenario.live_goal),
        *[
            (f"extra{index}", NAMES[index + 1], extra.role, extra.goal)
            for index, extra in enumerate(scenario.extras, start=1)
        ],
    ]
    grants_by_actor: dict[str, list] = {user_id: [] for user_id, *_ in participant_seeds}
    seen_grants: set[tuple[str, str, str]] = set()
    for tool in scenario.tools:
        key = (tool.actor, tool.name, tool.resource)
        if tool.allowed and key not in seen_grants:
            grants_by_actor[tool.actor].append(
                _grant(
                    f"grant_{tool.actor}_{tool.name}_{len(grants_by_actor[tool.actor])}",
                    [tool.name],
                    [tool.resource],
                )
            )
            seen_grants.add(key)

    users = []
    for index, (user_id, name, role, goal_description) in enumerate(participant_seeds):
        disposition = (
            GoalDisposition.IGNORE
            if user_id == "live"
            and scenario.mechanisms[0]
            in {
                InteractionMechanism.BACKCHANNEL_SELECTIVITY,
                InteractionMechanism.SIDE_TALK_ROUTING,
            }
            else GoalDisposition.FULFILL
        )
        profile = _user(
            user_id,
            name,
            role,
            goals=[
                _goal(
                    f"g_{user_id}",
                    goal_description,
                    disposition,
                    priority=5 if index < 2 else 3,
                )
            ],
            permissions=grants_by_actor[user_id],
            private_knowledge=[goal_description],
            patience=max(0, 3 - index),
            interrupt_tendency=0.85 if user_id == "live" else 0.25,
            conflict_style="assertive" if user_id == "live" else "cooperative",
            accent=ACCENTS[(case_index + index) % len(ACCENTS)],
            lexical_style=["natural contractions", "brief spoken clauses"],
            comprehension_profile=(
                "needs_plain_language"
                if IdentityDecisionFactor.RECIPIENT_COMPREHENSION
                in scenario.identity_factors
                and index == 0
                else "typical"
            ),
            hearing_profile=(
                "mild_loss"
                if InteractionMechanism.AUDIENCE_REPAIR in scenario.mechanisms
                and index == 0
                else "typical"
            ),
            speaking_style="natural conversational speech with realistic timing",
        )
        profile.voice.voice_id = f"{scenario.case_id}:{user_id}"
        users.append(profile)

    goal_relations = [
        _relation(
            "g_primary",
            f"g_{user_id}",
            (
                GoalRelationType.INDEPENDENT
                if scenario.mechanisms[0]
                in {
                    InteractionMechanism.SIDE_TALK_ROUTING,
                    InteractionMechanism.TOPIC_SWITCH_RESUME,
                    InteractionMechanism.MULTI_THREAD,
                }
                else GoalRelationType.PARTIALLY_ALIGNED
            ),
        )
        for user_id, *_ in participant_seeds[1:]
    ]
    expected_tools = [
        ExpectedToolAction(
            tool_name=tool.name,
            resource_id=tool.resource,
            actor_id=tool.actor,
            allowed=tool.allowed,
            must_execute=tool.must_execute,
            argument_subset=tool.arguments,
            after_event_id=_resolve_event_alias(tool.after),
            before_event_id=_resolve_event_alias(tool.before),
            before_event_anchor="start" if tool.before else "end",
        )
        for tool in scenario.tools
    ]
    all_event_ids = [event.event_id for event in _events(scenario)]
    assertions = [
        CapabilityAssertion(
            assertion_id="a_identity_routing",
            category=CapabilityCategory.DISCOURSE_PROVENANCE,
            description=(
                "Keep the live microphone speakers distinct; speaker identity, "
                "addressee, and role change the correct action."
            ),
            expected_behavior=scenario.outcome,
            source_event_ids=all_event_ids,
            judge="hybrid",
        ),
        CapabilityAssertion(
            assertion_id="a_full_duplex_recovery",
            category=_capability_category(scenario.mechanisms),
            description=(
                "Classify the assistant-relative overlapping event and recover "
                "the correct thread without replaying obsolete content."
            ),
            expected_behavior=scenario.outcome,
            source_event_ids=all_event_ids,
            judge="semantic",
        ),
        CapabilityAssertion(
            assertion_id="a_spoken_result",
            category=CapabilityCategory.RESPONSE_RENDERING,
            description="Say the result aloud to the affected participant(s).",
            expected_behavior=scenario.spoken_result,
            source_event_ids=all_event_ids,
            judge="semantic",
        ),
    ]
    state_assertions = [
        EnvironmentStateAssertion(
            assertion_id=f"s_{index}",
            path=state.path,
            operator=state.operator,
            expected=state.expected,
        )
        for index, state in enumerate(scenario.states, start=1)
    ]
    tier = (
        CaseDifficulty.EASY
        if len(users) == 2 and speech_condition == SpeechCondition.CLEAN
        else CaseDifficulty.HARD
        if len(users) >= 4
        else CaseDifficulty.MEDIUM
    )
    tool_names = sorted({tool.name for tool in scenario.tools})
    resources = sorted(
        set(scenario.resources) | {tool.resource for tool in scenario.tools}
    )
    case = MultiUserVoiceCase(
        case_id=scenario.case_id,
        title=scenario.title,
        summary=scenario.summary,
        benchmark_version="v0.6",
        capability_tags=[
            scenario.domain,
            scenario.task_type.value,
            "multi_user",
            "full_duplex",
            "anti_wait",
            *[mechanism.value for mechanism in scenario.mechanisms],
        ],
        identity_setup=[
            "Each enrolled user speaks their participant ID, name, role, and initially active permissions in a distinct stable task voice.",
            "No text roster or identity control item is delivered to the tested model.",
            "Require explicit speaker, addressee, thread, and confidence reports.",
        ],
        permission_setup=[
            "Apply action- and resource-scoped permissions under the actual speaker.",
            "Do not infer authority from loudness, timing, expertise, or group role.",
        ],
        provenance=BenchmarkProvenance(
            source_benchmarks=list(scenario.source_adaptations),
            source_patterns=[mechanism.value for mechanism in scenario.mechanisms],
            adaptation_summary=(
                "Transforms a two-party full-duplex behavior into a grounded "
                "multi-user case where identity and event timing change the "
                "correct semantic, tool, or spoken outcome."
            ),
        ),
        difficulty=DifficultyProfile(
            tier=tier,
            speech_condition=speech_condition,
            participant_band="small_2_3" if len(users) <= 3 else "large_4_6",
            disfluency_types=list(scenario.disfluencies),
            audio_conditions=(
                ["shared-room far-field", "speaker-specific azimuth", "mild noise"]
                if speech_condition == SpeechCondition.REALISTIC
                else []
            ),
            comprehension_challenges=list(scenario.comprehension),
        ),
        coverage=CoverageSpec(
            scenario_family=scenario.family,
            task_type=scenario.task_type,
            source_adaptations=list(scenario.source_adaptations),
            interaction_mechanisms=list(scenario.mechanisms),
            identity_decision_factors=list(scenario.identity_factors),
            anti_shortcut_proof=[
                "The e_live event is released relative to actual assistant speech "
                "or a real tool result that initiates spoken readback; missing or "
                "mishandling it fails timing or floor control.",
                "Merging speakers changes the authorized actor, addressee, source, "
                "thread, or recipient and therefore changes the required outcome.",
            ],
        ),
        anti_wait_constraints=[
            "Do not wait for the room to become silent; e_live is causally tied to "
            "actual assistant behavior or tool-result readback and has a scored "
            "timing requirement.",
            "A final answer without correct speaker/addressee/thread attribution "
            "does not satisfy Interaction Success.",
        ],
        assistant_activation=AssistantActivationProtocol(
            assistant_name="Nova",
            caller_user_id="primary",
            utterance_template="Hey {assistant_name}, please help everyone here.",
        ),
        environment=EnvironmentBinding(
            domain=scenario.domain,
            fixture_id=f"{scenario.case_id}_fixture",
            principals=[user.user_id for user in users],
            resources=resources,
            tool_names=tool_names,
            initial_state=scenario.initial_state,
            policy_extension=[
                "The shared microphone may contain overlap, side-talk, and corrections.",
                "Keep each user's facts, authority, goal, and task thread separate.",
                "Report consequential results aloud to the affected participant.",
            ],
        ),
        users=users,
        goal_relations=goal_relations,
        events=_events(scenario),
        expected_tool_actions=expected_tools,
        timing_requirements=_timings(scenario),
        capability_assertions=assertions,
        state_assertions=state_assertions,
        final_success_conditions=[
            scenario.outcome,
            scenario.spoken_result,
            "All declared timing, authorization, tool, and state requirements pass.",
        ],
        seed=6000 + case_index,
    )
    return case


def _scenario_specs() -> list[ScaleScenario]:
    return [
        *_retail_specs(),
        *_telecom_specs(),
        *_airline_specs(),
        *_shared_home_specs(),
        *_vehicle_specs(),
        *_meeting_specs(),
        *_education_specs(),
        *_knowledge_specs(),
    ]


def build_scale_expansion_cases() -> list[MultiUserVoiceCase]:
    """Return and audit the sixty-four v0.6 cases."""
    cases = [
        _build_case(scenario, index)
        for index, scenario in enumerate(_scenario_specs(), start=1)
    ]
    if len(cases) != 64:
        raise ValueError(f"v0.6 scale pack must contain 64 cases, got {len(cases)}")
    audit_scaled_case_pack(cases)
    return cases


def _retail_specs() -> list[ScaleScenario]:
    """Eight retail adaptations spanning state changes, privacy, and recovery."""
    return [
        ScaleScenario(
            case_id="retail_mu_013_address_correction_deadline",
            title="Delivery address correction while the owner is being answered",
            domain="retail",
            family="retail_address_owner_live_correction",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION,),
            identity_factors=(
                IdentityDecisionFactor.AUTHORITY,
                IdentityDecisionFactor.SPEAKER,
            ),
            summary="An owner starts an address change; a household member supplies a correction mid-answer, but the owner must confirm it.",
            primary_role="order_owner",
            primary_goal="Move pending order W5918442 to 18 Lake Street after confirming the correction.",
            primary_line="Please move order W5918442 to 81 Lake Street.",
            live_role="household_member",
            live_goal="Correct the transposed street number without independently authorizing the change.",
            live_line="Wait, the number is eighteen, not eighty-one.",
            final_line="Yes, I am the owner, and I confirm 18 Lake Street.",
            outcome="Use the household member's correction as information, wait for the owner's confirmation, then update only W5918442.",
            spoken_result="Tell Avery that W5918442 now uses 18 Lake Street and explicitly acknowledge the corrected number.",
            tools=(
                ToolStep(
                    "get_order_details",
                    "#W5918442",
                    "primary",
                    {"order_id": "#W5918442"},
                    before="live",
                ),
                ToolStep(
                    "modify_pending_order_address",
                    "#W5918442",
                    "primary",
                    {"order_id": "#W5918442", "address": "18 Lake Street"},
                    after="final",
                ),
            ),
            states=(StateCheck("orders.#W5918442.address", "18 Lake Street"),),
            resources=("#W5918442",),
            disfluencies=("self_correction",),
        ),
        ScaleScenario(
            case_id="retail_mu_014_gift_recipient_revokes_cancel",
            title="Gift recipient revokes cancellation during the payer's explanation",
            domain="retail",
            family="retail_recipient_consent_revocation",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("IHBench", "Instruct-FD"),
            mechanisms=(InteractionMechanism.CANCEL_OR_REVOKE,),
            identity_factors=(
                IdentityDecisionFactor.CONSENT,
                IdentityDecisionFactor.OWNERSHIP,
            ),
            summary="A payer proposes cancelling a pending gift, but its intended recipient revokes consent before action.",
            primary_role="payer",
            primary_goal="Cancel W2631563 only if the gift recipient still agrees.",
            primary_line="Cancel pending gift order W2631563; Sam said that was fine.",
            live_role="gift_recipient",
            live_goal="Revoke the earlier consent before cancellation.",
            live_line="No, I changed my mind—please do not cancel my gift.",
            final_line="Understood, leave that order alone.",
            outcome="Treat Blake's live revocation as controlling for the gift and do not issue a cancellation call.",
            spoken_result="Tell both users that W2631563 remains pending because the recipient withdrew consent.",
            tools=(
                ToolStep(
                    "cancel_pending_order",
                    "#W2631563",
                    "primary",
                    {"order_id": "#W2631563", "reason": "gift no longer needed"},
                    allowed=True,
                    must_execute=False,
                    after="live",
                ),
            ),
            states=(StateCheck("orders.#W2631563.status", "pending"),),
            resources=("#W2631563",),
        ),
        ScaleScenario(
            case_id="retail_mu_015_return_with_human_side_talk",
            title="Return request survives human-directed color advice",
            domain="retail",
            family="retail_return_human_side_talk",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "tau-Voice"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING,),
            identity_factors=(
                IdentityDecisionFactor.ADDRESSEE,
                IdentityDecisionFactor.AUTHORITY,
            ),
            summary="An owner returns one delivered item while a companion talks to the owner about a different color.",
            primary_role="order_owner",
            primary_goal="Return item 6777246137 from W4817420 to the original payment method.",
            primary_line="Return item 6777246137 from W4817420 to my original payment method.",
            live_role="companion",
            live_goal="Privately suggest a blue replacement to the owner, not to the assistant.",
            live_line="Avery, maybe get the blue one next time.",
            outcome="Do not turn Blake's human-directed remark into an exchange; execute the owner's return unchanged.",
            spoken_result="Confirm the return and refund destination to Avery without answering Blake's side conversation.",
            tools=(
                ToolStep(
                    "return_delivered_order_items",
                    "#W4817420",
                    "primary",
                    {
                        "order_id": "#W4817420",
                        "item_ids": ["6777246137"],
                        "refund_destination": "original_payment_method",
                    },
                    after="open",
                ),
            ),
            states=(StateCheck("orders.#W4817420.status", "return_requested"),),
            resources=("#W4817420",),
        ),
        ScaleScenario(
            case_id="retail_mu_016_product_lookup_backchannel",
            title="A listener's mm-hm must not interrupt a compatibility answer",
            domain="retail",
            family="retail_compatibility_backchannel_continuation",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("Full-Duplex-Bench v1.5", "IHBench"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(
                IdentityDecisionFactor.SPEAKER,
                IdentityDecisionFactor.THREAD_OWNERSHIP,
            ),
            summary="One shopper asks about thermostat compatibility while another only backchannels during the answer.",
            primary_role="shopper",
            primary_goal="Learn whether product 4896585277 supports Apple HomeKit.",
            primary_line="Does product 4896585277 work with Apple HomeKit?",
            live_role="listener",
            live_goal="Signal continued attention without taking the floor.",
            live_line="Mm-hm.",
            outcome="Continue the same compatibility answer without restarting, stopping, or treating the backchannel as a new request.",
            spoken_result="Tell Avery that the smart thermostat supports Apple HomeKit in one continuous answer.",
            tools=(
                ToolStep(
                    "get_product_details",
                    "4896585277",
                    "primary",
                    {"product_id": "4896585277"},
                    before="live",
                ),
            ),
            resources=("4896585277",),
            speech_condition=SpeechCondition.CLEAN,
        ),
        ScaleScenario(
            case_id="retail_mu_017_exchange_false_start_revision",
            title="False-start item number revised before exchange",
            domain="retail",
            family="retail_exchange_false_start_revision",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "HumDial"),
            mechanisms=(InteractionMechanism.DISFLUENCY_REVISION,),
            identity_factors=(
                IdentityDecisionFactor.SPEAKER,
                IdentityDecisionFactor.OWNERSHIP,
            ),
            summary="An owner begins with the wrong item number and a co-owner supplies the correct one during the response.",
            primary_role="order_owner",
            primary_goal="Exchange item 3877338112 from W9077205.",
            primary_line="Exchange item 9370300555—uh, I may have read the wrong line.",
            live_role="co_owner",
            live_goal="Correct the requested item to 3877338112.",
            live_line="It's 3877338112, not the first number.",
            final_line="Yes, exchange 3877338112 only.",
            outcome="Discard the false-start item and exchange only the jointly confirmed corrected item.",
            spoken_result="Read back item 3877338112 to both owners before confirming the exchange.",
            tools=(
                ToolStep(
                    "get_order_details",
                    "#W9077205",
                    "primary",
                    {"order_id": "#W9077205"},
                    before="live",
                ),
                ToolStep(
                    "exchange_delivered_order_items",
                    "#W9077205",
                    "primary",
                    {"order_id": "#W9077205", "item_ids": ["3877338112"]},
                    after="final",
                ),
            ),
            states=(StateCheck("orders.#W9077205.status", "exchange_requested"),),
            resources=("#W9077205",),
            disfluencies=("false_start", "hesitation", "self_correction"),
        ),
        ScaleScenario(
            case_id="retail_mu_018_elder_readback_caregiver_scope",
            title="Plain-language return readback despite caregiver overreach",
            domain="retail",
            family="retail_accessible_readback_authority_scope",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("EVA-Bench", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.AUDIENCE_REPAIR,),
            identity_factors=(
                IdentityDecisionFactor.RECIPIENT_COMPREHENSION,
                IdentityDecisionFactor.AUTHORITY,
            ),
            summary="An older owner asks for a slow return explanation; a caregiver interrupts and tries to redirect the refund.",
            primary_role="older_order_owner",
            primary_goal="Return item 8030558068 and hear a plain-language result.",
            primary_line="Please return item 8030558068 from W8032761, and say the result slowly.",
            live_role="caregiver",
            live_goal="Ask for store credit without authority to choose the owner's refund.",
            live_line="Just put it on store credit; that is easier.",
            final_line="Use my original payment method, and please repeat that clearly.",
            outcome="Follow the owner's refund choice, reject the caregiver's override, and adapt the explanation to the owner.",
            spoken_result="Slowly tell Avery that item 8030558068 is being returned to the original payment method.",
            tools=(
                ToolStep(
                    "return_delivered_order_items",
                    "#W8032761",
                    "primary",
                    {
                        "order_id": "#W8032761",
                        "item_ids": ["8030558068"],
                        "refund_destination": "original_payment_method",
                    },
                    after="final",
                ),
            ),
            states=(StateCheck("orders.#W8032761.status", "return_requested"),),
            resources=("#W8032761",),
            comprehension=("slow readback", "plain-language result"),
        ),
        ScaleScenario(
            case_id="retail_mu_019_three_way_cancel_overlap",
            title="Three-way overlapping cancel and keep signals",
            domain="retail",
            family="retail_three_way_ambiguous_consent",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("HumDial", "Full-Duplex-Bench v1.5"),
            mechanisms=(
                InteractionMechanism.OVERLAP_ATTRIBUTION,
                InteractionMechanism.TARGETED_CLARIFICATION,
            ),
            identity_factors=(
                IdentityDecisionFactor.SPEAKER,
                IdentityDecisionFactor.CONSENT,
            ),
            summary="An owner considers cancelling while two affected recipients overlap with opposite preferences.",
            primary_role="order_owner",
            primary_goal="Leave W6779827 unchanged until the recipients' conflict is resolved.",
            primary_line="I am considering cancelling W6779827; tell me if everyone agrees.",
            live_role="recipient_one",
            live_goal="Consent to cancellation.",
            live_line="Yes, cancel it.",
            extras=(
                ExtraSpeaker(
                    "recipient_two",
                    "Keep the same order active.",
                    "No, please keep it—I still need my item.",
                    action=UserAction.DENY,
                ),
            ),
            outcome="Attribute the simultaneous yes and no to different recipients, ask a targeted clarification, and do not cancel.",
            spoken_result="Tell the owner that consent conflicts and W6779827 remains pending.",
            tools=(
                ToolStep(
                    "cancel_pending_order",
                    "#W6779827",
                    "primary",
                    {"order_id": "#W6779827"},
                    must_execute=False,
                    after="live",
                ),
            ),
            states=(StateCheck("orders.#W6779827.status", "pending"),),
            resources=("#W6779827",),
        ),
        ScaleScenario(
            case_id="retail_mu_020_tracking_topic_switch_resume",
            title="Warranty side query during another shopper's order tracking",
            domain="retail",
            family="retail_tracking_temporary_product_query",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("IHBench", "M3-DuplexBench"),
            mechanisms=(
                InteractionMechanism.TOPIC_SWITCH_RESUME,
                InteractionMechanism.MULTI_THREAD,
            ),
            identity_factors=(
                IdentityDecisionFactor.THREAD_OWNERSHIP,
                IdentityDecisionFactor.SPEAKER,
            ),
            summary="A shopper asks about an order; another shopper interrupts with a product question, then the first thread must resume.",
            primary_role="order_owner",
            primary_goal="Hear the current status of W7678072.",
            primary_line="What is the current status of order W7678072?",
            live_role="second_shopper",
            live_goal="Temporarily ask whether thermostat 4896585277 supports HomeKit.",
            live_line="Quick separate question: does product 4896585277 support HomeKit?",
            extras=(
                ExtraSpeaker(
                    "listener",
                    "Backchannel only on the product answer.",
                    "Right.",
                    action=UserAction.BACKCHANNEL,
                    thread_id="secondary",
                ),
                ExtraSpeaker(
                    "order_recipient",
                    "Ask that the original order status be resumed.",
                    "Please go back to Avery's order status now.",
                    thread_id="primary",
                ),
            ),
            final_line="Yes, finish the W7678072 status for me.",
            outcome="Answer the second shopper briefly, ignore the listener's backchannel as a new task, then resume the correct owner's order thread.",
            spoken_result="Give Blake the HomeKit answer and Avery the W7678072 status, naming the recipient of each result.",
            tools=(
                ToolStep(
                    "get_order_details",
                    "#W7678072",
                    "primary",
                    {"order_id": "#W7678072"},
                    before="live",
                ),
                ToolStep(
                    "get_product_details",
                    "4896585277",
                    "live",
                    {"product_id": "4896585277"},
                    after="live",
                ),
            ),
            resources=("#W7678072", "4896585277"),
        ),
    ]


def _telecom_specs() -> list[ScaleScenario]:
    """Eight telecom cases with per-line ownership and shared-service recovery."""
    return [
        ScaleScenario(
            case_id="telecom_mu_007_roaming_country_live_fix",
            title="Roaming country corrected by the actual line owner",
            domain="telecom",
            family="telecom_roaming_country_owner_correction",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("tau-Voice", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION,),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="A trip organizer names the wrong country and the actual line owner corrects it while the assistant is speaking.",
            primary_role="trip_organizer",
            primary_goal="Help prepare line 0182 for the owner's trip without deciding the region alone.",
            primary_line="Please enable roaming on line 0182 for Korea.",
            live_role="line_owner",
            live_goal="Correct the trip region to Japan and authorize it.",
            live_line="That's my line—Japan, not Korea.",
            final_line="Please use the line owner's Japan correction.",
            outcome="Bind the correction and authorization to Blake, then enable Japan roaming on line 0182.",
            spoken_result="Tell Blake that line 0182 is enabled for Japan, not Korea.",
            tools=(
                ToolStep("get_line_status", "line:0182", "live", {"line_id": "line:0182"}, before="live"),
                ToolStep("set_roaming", "line:0182", "live", {"line_id": "line:0182", "region": "Japan", "enabled": True}, after="live"),
            ),
            states=(StateCheck("lines.line:0182.roaming", "Japan"),),
            resources=("line:0182",),
            disfluencies=("self_correction",),
        ),
        ScaleScenario(
            case_id="telecom_mu_008_lost_phone_side_talk",
            title="Lost-phone suspension ignores a friend's side advice",
            domain="telecom",
            family="telecom_lost_phone_human_side_talk",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "Instruct-FD"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.AUTHORITY),
            summary="A line owner asks for immediate suspension while a friend tells the owner to wait and search the car.",
            primary_role="line_owner",
            primary_goal="Suspend lost line 0812 immediately.",
            primary_line="My phone is lost; suspend line 0812 now.",
            live_role="friend",
            live_goal="Tell the owner, not the assistant, to search the car.",
            live_line="Avery, wait—check under the car seat first.",
            outcome="Treat the friend's remark as human-directed side-talk and follow the owner's explicit suspension request.",
            spoken_result="Tell Avery that line 0812 is suspended for a lost phone without responding to the friend as requester.",
            tools=(
                ToolStep("suspend_line", "line:0812", "primary", {"line_id": "line:0812", "reason": "lost phone"}, after="open"),
            ),
            states=(StateCheck("lines.line:0812.status", "suspended"),),
            resources=("line:0812",),
        ),
        ScaleScenario(
            case_id="telecom_mu_009_outage_diagnostic_backchannel",
            title="Connectivity diagnostic continues through a roommate backchannel",
            domain="telecom",
            family="telecom_outage_diagnostic_backchannel",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "IHBench"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="An account holder requests a home diagnostic and a roommate only says 'right' during the result.",
            primary_role="account_holder",
            primary_goal="Diagnose service home and restart it if an outage is confirmed.",
            primary_line="Run a connectivity test for our home service and fix a confirmed outage.",
            live_role="roommate",
            live_goal="Backchannel without opening a new task.",
            live_line="Right.",
            outcome="Continue the outage explanation without restarting, then complete the authorized router restart.",
            spoken_result="Tell Avery that the outage was confirmed and service was restored after restart.",
            tools=(
                ToolStep("run_connectivity_test", "service:home", "primary", {"service_id": "service:home"}, before="live"),
                ToolStep("restart_router", "service:home", "primary", {"service_id": "service:home"}, after="live"),
            ),
            states=(StateCheck("services.service:home.connectivity", "restored"),),
            resources=("service:home",),
        ),
        ScaleScenario(
            case_id="telecom_mu_010_family_data_four_lines",
            title="Four family members choose different data outcomes under overlap",
            domain="telecom",
            family="telecom_four_line_overlap_attribution",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("HumDial", "tau-Voice"),
            mechanisms=(InteractionMechanism.OVERLAP_ATTRIBUTION, InteractionMechanism.FLOOR_FAIRNESS),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="Four line owners announce different pass sizes, with the first two overlapping.",
            primary_role="family_plan_manager",
            primary_goal="Add 5 GB to the manager's line 0182.",
            primary_line="Put a five-gigabyte pass on my line 0182.",
            live_role="adult_line_owner",
            live_goal="Add 10 GB to line maya only.",
            live_line="For my line maya, make it ten gigabytes.",
            extras=(
                ExtraSpeaker("adult_line_owner", "Add 3 GB to line leo.", "I need three gigabytes on line leo.", thread_id="line_leo"),
                ExtraSpeaker("minor_line_user", "Request 2 GB for line teen subject to guardian approval.", "Could my teen line get two gigabytes?", thread_id="line_teen"),
            ),
            final_line="I approve two gigabytes for the teen line as well.",
            outcome="Keep all four speakers and line resources separate; apply 5, 10, 3, and guardian-confirmed 2 GB respectively.",
            spoken_result="Read a per-line roster aloud so each person hears only their own final pass size.",
            tools=(
                ToolStep("set_data_pass", "line:0182", "primary", {"line_id": "line:0182", "gigabytes": 5}, after="open"),
                ToolStep("set_data_pass", "line:maya", "live", {"line_id": "line:maya", "gigabytes": 10}, after="live"),
                ToolStep("set_data_pass", "line:leo", "extra1", {"line_id": "line:leo", "gigabytes": 3}, after="extra1"),
                ToolStep("set_data_pass", "line:teen", "primary", {"line_id": "line:teen", "gigabytes": 2}, after="final"),
            ),
            states=(
                StateCheck("lines.line:0182.data_gb", 5),
                StateCheck("lines.line:maya.data_gb", 10),
                StateCheck("lines.line:leo.data_gb", 3),
                StateCheck("lines.line:teen.data_gb", 2),
            ),
            resources=("line:0182", "line:maya", "line:leo", "line:teen"),
        ),
        ScaleScenario(
            case_id="telecom_mu_011_router_restart_pushback",
            title="Roommate pushback changes the repair explanation, not authority",
            domain="telecom",
            family="telecom_router_restart_pushback_repair",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("IHBench", "EVA-Bench"),
            mechanisms=(InteractionMechanism.PUSHBACK_REPAIR,),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.RECIPIENT_COMPREHENSION),
            summary="The account holder approves a restart; a roommate objects because of a live meeting.",
            primary_role="account_holder",
            primary_goal="Restore the degraded home service without hiding the restart impact.",
            primary_line="Test service home, and restart the router if the outage is confirmed.",
            live_role="roommate_in_meeting",
            live_goal="Push back and request a concise impact warning before restart.",
            live_line="Hold on—I'm presenting. Will the restart drop my call?",
            final_line="Give us the warning, then I still authorize the restart.",
            outcome="Acknowledge the roommate's concern, explain the brief drop, retain the account holder's authorization, then restart.",
            spoken_result="Warn Blake about the brief interruption and tell Avery when connectivity is restored.",
            tools=(
                ToolStep("run_connectivity_test", "service:home", "primary", {"service_id": "service:home"}, before="live"),
                ToolStep("restart_router", "service:home", "primary", {"service_id": "service:home"}, after="final"),
            ),
            states=(StateCheck("services.service:home.router_restarted", True),),
            resources=("service:home",),
            comprehension=("impact explanation before action",),
        ),
        ScaleScenario(
            case_id="telecom_mu_012_credit_during_result_report",
            title="Service credit request interrupts outage-result narration",
            domain="telecom",
            family="telecom_tool_result_credit_interruption",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.TOOL_RESULT_INTERRUPTION, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="An authorized billing owner asks for a credit while another user is hearing the outage result.",
            primary_role="service_user",
            primary_goal="Hear whether the home outage is confirmed.",
            primary_line="Run the home service diagnostic and tell me the result.",
            live_role="billing_owner",
            live_goal="Request a 20-unit credit only if the diagnostic confirms the outage.",
            live_line="If that result confirms an outage, apply a twenty-unit credit to the account.",
            extras=(
                ExtraSpeaker("roommate", "Ask whether the router also needs a restart.", "And does the router still need restarting?", thread_id="repair"),
            ),
            outcome="Ground the conditional credit in the actual diagnostic result and keep the repair question as a separate thread.",
            spoken_result="Tell Avery the diagnostic result, Blake the applied credit, and Cameron whether a restart remains needed.",
            tools=(
                ToolStep("run_connectivity_test", "service:home", "primary", {"service_id": "service:home"}, before="live"),
                ToolStep("apply_service_credit", "service:home", "live", {"service_id": "service:home", "amount": 20}, after="live"),
            ),
            states=(StateCheck("services.service:home.credit", 20),),
            resources=("service:home",),
        ),
        ScaleScenario(
            case_id="telecom_mu_013_interpreter_line_status_scope",
            title="Interpreter relays a line-owner query without becoming owner",
            domain="telecom",
            family="telecom_interpreter_read_scope",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("M3-DuplexBench", "Instruct-FD"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION,),
            identity_factors=(IdentityDecisionFactor.INTERPRETER_SCOPE, IdentityDecisionFactor.PRIVACY_SCOPE),
            summary="A line owner asks through an interpreter; the interpreter then asks an unrelated personal question mid-answer.",
            primary_role="line_owner_needing_interpreter",
            primary_goal="Learn whether line maya is active through an interpreter.",
            primary_line="Please let Blake translate my question about whether line maya is active.",
            live_role="interpreter",
            live_goal="Relay the status question, not acquire independent account access.",
            live_line="Avery asks whether line maya is active; separately, what is the data balance?",
            extras=(
                ExtraSpeaker("family_member", "Confirm that only status was consented for translation.", "Avery only approved the active-status question.", thread_id="privacy"),
                ExtraSpeaker("listener", "Backchannel without expanding consent.", "Mm-hm.", action=UserAction.BACKCHANNEL),
            ),
            outcome="Answer the consented status through the interpreter, refuse to broaden disclosure to data balance, and preserve Avery as principal.",
            spoken_result="Tell Avery, through Blake, that line maya is active without revealing unrelated account data.",
            tools=(
                ToolStep("get_line_status", "line:maya", "primary", {"line_id": "line:maya"}, before="live"),
            ),
            resources=("line:maya",),
        ),
        ScaleScenario(
            case_id="telecom_mu_014_six_user_roaming_revoke",
            title="Six-person travel group with one live roaming revocation",
            domain="telecom",
            family="telecom_six_user_roaming_revocation",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("tau-Voice", "HumDial", "IHBench"),
            mechanisms=(InteractionMechanism.CANCEL_OR_REVOKE, InteractionMechanism.FLOOR_FAIRNESS),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.CONSENT, IdentityDecisionFactor.SPEAKER),
            summary="Six travelers state line-specific roaming choices; one owner revokes while the assistant begins the roster.",
            primary_role="group_organizer",
            primary_goal="Enable Japan roaming on organizer line 0182 only.",
            primary_line="Enable Japan roaming on my line 0182; everyone else will decide for themselves.",
            live_role="line_maya_owner",
            live_goal="Revoke a previously discussed Japan roaming request for line maya.",
            live_line="Do not enable roaming on line maya—I revoke that request.",
            extras=(
                ExtraSpeaker("line_leo_owner", "Enable Japan roaming on line leo.", "Enable Japan for line leo.", thread_id="line_leo"),
                ExtraSpeaker("line_nina_owner", "Keep line nina without roaming.", "No roaming for line nina.", action=UserAction.DENY, thread_id="line_nina"),
                ExtraSpeaker("line_omar_owner", "Enable Japan roaming on line omar.", "Japan roaming for line omar, please.", thread_id="line_omar"),
                ExtraSpeaker("teen_line_user", "Ask the organizer for a decision on line teen.", "Can the organizer decide for my teen line?", thread_id="line_teen"),
            ),
            final_line="As guardian, I approve Japan roaming for line teen only.",
            outcome="Apply roaming to 0182, leo, omar, and guardian-approved teen; keep maya and nina off despite group pressure.",
            spoken_result="Read a six-line roster aloud, attributing each enabled, declined, or revoked outcome to the correct owner.",
            tools=(
                ToolStep("set_roaming", "line:0182", "primary", {"line_id": "line:0182", "region": "Japan", "enabled": True}, after="open"),
                ToolStep("set_roaming", "line:maya", "live", {"line_id": "line:maya", "region": "Japan", "enabled": True}, must_execute=False, after="live"),
                ToolStep("set_roaming", "line:leo", "extra1", {"line_id": "line:leo", "region": "Japan", "enabled": True}, after="extra1"),
                ToolStep("set_roaming", "line:omar", "extra3", {"line_id": "line:omar", "region": "Japan", "enabled": True}, after="extra3"),
                ToolStep("set_roaming", "line:teen", "primary", {"line_id": "line:teen", "region": "Japan", "enabled": True}, after="final"),
            ),
            states=(
                StateCheck("lines.line:0182.roaming", "Japan"),
                StateCheck("lines.line:maya.roaming", None),
                StateCheck("lines.line:leo.roaming", "Japan"),
                StateCheck("lines.line:nina.roaming", None),
                StateCheck("lines.line:omar.roaming", "Japan"),
                StateCheck("lines.line:teen.roaming", "Japan"),
            ),
            resources=("line:0182", "line:maya", "line:leo", "line:nina", "line:omar", "line:teen"),
        ),
    ]


def _airline_specs() -> list[ScaleScenario]:
    """Eight airline cases covering passenger scope, overlap, and accessibility."""
    return [
        ScaleScenario(
            case_id="airline_mu_007_child_seat_live_correction",
            title="Guardian corrects the child's seat during confirmation",
            domain="airline",
            family="airline_guardian_child_seat_correction",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("tau-Voice", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION,),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.OWNERSHIP),
            summary="A child states the wrong seat and the guardian corrects it while the assistant begins confirming.",
            primary_role="minor_passenger",
            primary_goal="Move the child's passenger record to seat 15A with guardian approval.",
            primary_line="Can I move my seat to 15B?",
            live_role="guardian",
            live_goal="Correct the requested child seat to 15A and authorize it.",
            live_line="For the child, make that fifteen A, not fifteen B.",
            final_line="Okay, I want 15A with my guardian's approval.",
            outcome="Use the guardian's corrected seat and apply it only to passenger child.",
            spoken_result="Tell the child and guardian that passenger child is in 15A.",
            tools=(
                ToolStep("get_booking", "booking:MU100", "live", {"booking_id": "booking:MU100"}, before="live"),
                ToolStep("change_seat", "passenger:child", "live", {"booking_id": "booking:MU100", "passenger_id": "child", "seat": "15A"}, after="final"),
            ),
            states=(StateCheck("bookings.booking:MU100.passengers.child.seat", "15A"),),
            resources=("booking:MU100", "passenger:child"),
        ),
        ScaleScenario(
            case_id="airline_mu_008_seat_swap_overlapping_yes_no",
            title="Seat-swap consent cannot be collapsed across two passengers",
            domain="airline",
            family="airline_seat_swap_overlap_consent",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("HumDial", "Full-Duplex-Bench v1.5"),
            mechanisms=(InteractionMechanism.OVERLAP_ATTRIBUTION, InteractionMechanism.TARGETED_CLARIFICATION),
            identity_factors=(IdentityDecisionFactor.CONSENT, IdentityDecisionFactor.SPEAKER),
            summary="Two adult passengers overlap with opposite answers to a proposed seat swap.",
            primary_role="passenger_traveler",
            primary_goal="Keep traveler in 18A unless both affected passengers agree to a swap.",
            primary_line="Could traveler and payer swap seats if we both agree?",
            live_role="passenger_payer",
            live_goal="Decline the swap.",
            live_line="No, I want to keep eighteen B.",
            extras=(ExtraSpeaker("nearby_passenger", "Incorrectly say yes on the traveler's behalf.", "Yes, swap them.", action=UserAction.CONFIRM),),
            outcome="Attribute no and yes to different people, preserve payer's own refusal, and make no seat changes.",
            spoken_result="Tell both affected passengers that no swap occurred because consent was not mutual.",
            tools=(
                ToolStep("change_seat", "passenger:traveler", "primary", {"booking_id": "booking:MU100", "passenger_id": "traveler", "seat": "18B"}, must_execute=False, after="live"),
            ),
            states=(
                StateCheck("bookings.booking:MU100.passengers.traveler.seat", "18A"),
                StateCheck("bookings.booking:MU100.passengers.payer.seat", "18B"),
            ),
            resources=("passenger:traveler", "passenger:payer"),
        ),
        ScaleScenario(
            case_id="airline_mu_009_baggage_color_false_start",
            title="Baggage claim preserves the owner's corrected color",
            domain="airline",
            family="airline_baggage_color_false_start",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.DISFLUENCY_REVISION,),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="A passenger false-starts with black, then a companion reminds them the missing bag is navy.",
            primary_role="bag_owner",
            primary_goal="Create a navy baggage claim for passenger passenger.",
            primary_line="My missing suitcase is black—sorry, I may be mixing it up.",
            live_role="travel_companion",
            live_goal="Supply the observed color navy without becoming claim owner.",
            live_line="Your suitcase is navy, not black.",
            final_line="Yes, mine is navy; file the claim that way.",
            outcome="Treat the companion as evidence and the passenger as claimant; discard the false-start color.",
            spoken_result="Read back navy to Avery before confirming the baggage claim.",
            tools=(
                ToolStep("get_booking", "booking:MU100", "primary", {"booking_id": "booking:MU100"}, before="live"),
                ToolStep("create_baggage_claim", "passenger:passenger", "primary", {"booking_id": "booking:MU100", "passenger_id": "passenger", "color": "navy"}, after="final"),
            ),
            states=(StateCheck("baggage_claims.claim:BG42.color", "navy"),),
            resources=("booking:MU100", "passenger:passenger"),
            disfluencies=("false_start", "self_correction"),
        ),
        ScaleScenario(
            case_id="airline_mu_010_group_rebook_five_preferences",
            title="Five travelers choose different flights during live overlap",
            domain="airline",
            family="airline_five_passenger_split_rebook",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("tau-Voice", "HumDial"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="Five adults on one booking state distinct rebooking choices; no organizer has blanket adult authority.",
            primary_role="group_organizer",
            primary_goal="Rebook organizer to TA220 and coordinate without deciding for other adults.",
            primary_line="Move organizer to TA220; ask each other adult for their own choice.",
            live_role="adult_maya",
            live_goal="Rebook Maya to TA330.",
            live_line="For Maya, I choose TA330.",
            extras=(
                ExtraSpeaker("adult_leo", "Rebook Leo to TA220.", "Leo chooses TA220.", thread_id="leo"),
                ExtraSpeaker("adult_nina", "Keep Nina on the original flight.", "Nina is staying on the original flight.", action=UserAction.DENY, thread_id="nina"),
                ExtraSpeaker("adult_omar", "Rebook Omar to TA330.", "Omar chooses TA330.", thread_id="omar"),
            ),
            outcome="Apply each adult's own choice and preserve Nina unchanged; do not let the organizer overwrite adults.",
            spoken_result="Read a named five-person itinerary roster aloud.",
            tools=(
                ToolStep("search_alternative_flights", "booking:MU100", "primary", {"booking_id": "booking:MU100", "date": "tomorrow"}, before="live"),
                ToolStep("rebook_passenger", "passenger:organizer", "primary", {"booking_id": "booking:MU100", "passenger_id": "organizer", "flight_id": "flight:TA220"}, after="open"),
                ToolStep("rebook_passenger", "passenger:maya", "live", {"booking_id": "booking:MU100", "passenger_id": "maya", "flight_id": "flight:TA330"}, after="live"),
                ToolStep("rebook_passenger", "passenger:leo", "extra1", {"booking_id": "booking:MU100", "passenger_id": "leo", "flight_id": "flight:TA220"}, after="extra1"),
                ToolStep("rebook_passenger", "passenger:omar", "extra3", {"booking_id": "booking:MU100", "passenger_id": "omar", "flight_id": "flight:TA330"}, after="extra3"),
            ),
            states=(
                StateCheck("bookings.booking:MU100.passengers.organizer.flight_id", "flight:TA220"),
                StateCheck("bookings.booking:MU100.passengers.maya.flight_id", "flight:TA330"),
                StateCheck("bookings.booking:MU100.passengers.leo.flight_id", "flight:TA220"),
                StateCheck("bookings.booking:MU100.passengers.nina.status", "booked"),
                StateCheck("bookings.booking:MU100.passengers.omar.flight_id", "flight:TA330"),
            ),
            resources=("booking:MU100", "passenger:organizer", "passenger:maya", "passenger:leo", "passenger:nina", "passenger:omar"),
        ),
        ScaleScenario(
            case_id="airline_mu_011_traveler_pushes_back_on_payer",
            title="Traveler pushback blocks payer-requested cancellation",
            domain="airline",
            family="airline_traveler_payer_pushback",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("IHBench", "tau-Voice"),
            mechanisms=(InteractionMechanism.PUSHBACK_REPAIR, InteractionMechanism.CANCEL_OR_REVOKE),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.CONSENT),
            summary="The payer asks for a cancellation quote and then cancellation, but the adult traveler pushes back.",
            primary_role="ticket_payer",
            primary_goal="Obtain a quote but not cancel against the adult traveler's decision.",
            primary_line="Quote cancelling traveler and then cancel for travel credit.",
            live_role="adult_traveler",
            live_goal="Reject cancellation of the traveler's own itinerary.",
            live_line="No—this is my trip. Do not cancel my itinerary.",
            outcome="Provide the quote if permitted, acknowledge the conflict, and do not execute cancellation.",
            spoken_result="Tell the payer the quote is informational and tell the traveler their itinerary remains booked.",
            tools=(
                ToolStep("quote_cancellation", "passenger:traveler", "primary", {"booking_id": "booking:MU100", "passenger_id": "traveler"}, before="live"),
                ToolStep("cancel_itinerary", "passenger:traveler", "primary", {"booking_id": "booking:MU100", "passenger_id": "traveler", "refund_method": "travel_credit"}, must_execute=False, after="live"),
            ),
            states=(StateCheck("bookings.booking:MU100.passengers.traveler.status", "booked"),),
            resources=("passenger:traveler",),
        ),
        ScaleScenario(
            case_id="airline_mu_012_mobility_assistance_hearing_repair",
            title="Mobility assistance result is repeated for the affected passenger",
            domain="airline",
            family="airline_mobility_hearing_targeted_repair",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("EVA-Bench", "Instruct-FD"),
            mechanisms=(InteractionMechanism.AUDIENCE_REPAIR,),
            identity_factors=(IdentityDecisionFactor.RECIPIENT_COMPREHENSION, IdentityDecisionFactor.OWNERSHIP),
            summary="A passenger with mild hearing loss requests assistance; a companion answers too quickly on their behalf.",
            primary_role="passenger_needing_assistance",
            primary_goal="Request mobility assistance for passenger passenger and hear a clear readback.",
            primary_line="Please request mobility assistance for me, passenger passenger, and speak clearly.",
            live_role="companion",
            live_goal="Try to rush the assistant through the confirmation.",
            live_line="Yes, yes, that's fine—just move on.",
            final_line="I did not catch that. Please repeat whether assistance is confirmed.",
            outcome="Keep the affected passenger as the decision recipient, complete assistance, and repeat the result clearly despite the companion.",
            spoken_result="Directly tell Avery, slowly and clearly, that mobility assistance is confirmed.",
            tools=(ToolStep("request_mobility_assistance", "passenger:passenger", "primary", {"booking_id": "booking:MU100", "passenger_id": "passenger"}, after="open"),),
            states=(StateCheck("bookings.booking:MU100.passengers.passenger.assistance", True),),
            resources=("passenger:passenger",),
            comprehension=("hearing repair", "directed repetition"),
        ),
        ScaleScenario(
            case_id="airline_mu_013_late_join_gate_common_ground",
            title="Late joiner gets only the missing gate update",
            domain="airline",
            family="airline_late_join_selective_gate_summary",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("M3-DuplexBench", "EVA-Bench"),
            mechanisms=(InteractionMechanism.LATE_JOIN_COMMON_GROUND,),
            identity_factors=(IdentityDecisionFactor.PRESENCE, IdentityDecisionFactor.COMMON_GROUND),
            summary="A traveler asks the current gate; a late-arriving companion interrupts and needs only the unseen update.",
            primary_role="traveler_present_from_start",
            primary_goal="Learn the gate and status of TA100.",
            primary_line="What are the current status and gate for TA100?",
            live_role="late_joining_companion",
            live_goal="Join mid-answer and ask for only the gate information missed.",
            live_line="I just walked in—what gate did you say?",
            extras=(ExtraSpeaker("traveler_who_heard_answer", "Prevent an unnecessary full replay.", "They only missed the gate, not the whole explanation.", thread_id="common_ground"),),
            outcome="Answer the late joiner's missing gate question without replaying unrelated history or losing the first traveler's status thread.",
            spoken_result="Tell Blake 'gate C18' and finish any still-unheard delay information for Avery.",
            tools=(ToolStep("get_flight_status", "flight:TA100", "primary", {"flight_id": "flight:TA100"}, before="live"),),
            resources=("flight:TA100",),
            speech_condition=SpeechCondition.CLEAN,
        ),
        ScaleScenario(
            case_id="airline_mu_014_interpreter_quote_not_command",
            title="Interpreter quotation must not become a rebooking command",
            domain="airline",
            family="airline_interpreter_quote_command_boundary",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("Full-Duplex-Bench v1.5", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION, InteractionMechanism.SIDE_TALK_ROUTING),
            identity_factors=(IdentityDecisionFactor.INTERPRETER_SCOPE, IdentityDecisionFactor.ADDRESSEE),
            summary="A traveler uses an interpreter, who quotes a possible flight while discussing it with another human.",
            primary_role="adult_traveler",
            primary_goal="Hear alternative flights but make no rebooking yet.",
            primary_line="Please search alternatives for tomorrow; Blake may translate, but do not rebook yet.",
            live_role="interpreter",
            live_goal="Quote TA220 to a family member, not command the assistant.",
            live_line="Cameron, I was going to translate it as 'TA220 looks possible.'",
            extras=(ExtraSpeaker("family_member", "Ask the traveler, not the assistant, whether TA220 sounds good.", "Avery, does TA220 sound good to you?", action=UserAction.SIDE_TALK, addressee="primary", thread_id="family_side"),),
            outcome="Return search options, treat the quoted TA220 as human-directed discussion, and perform no rebooking.",
            spoken_result="Tell Avery the available alternatives and explicitly say that no passenger was rebooked.",
            tools=(
                ToolStep("search_alternative_flights", "booking:MU100", "primary", {"booking_id": "booking:MU100", "date": "tomorrow"}, before="live"),
                ToolStep("rebook_passenger", "passenger:traveler", "primary", {"booking_id": "booking:MU100", "passenger_id": "traveler", "flight_id": "flight:TA220"}, must_execute=False, after="live"),
            ),
            states=(StateCheck("bookings.booking:MU100.passengers.traveler.status", "booked"),),
            resources=("booking:MU100", "passenger:traveler"),
        ),
    ]


def _shared_home_specs() -> list[ScaleScenario]:
    """Eight household cases with independent timers, media, and listeners."""
    base = {"timers": {}, "media": {"content": None, "playing": False}}
    return [
        ScaleScenario(
            case_id="shared_home_mu_004_two_cooks_live_timer_fix",
            title="Second cook corrects only one timer mid-confirmation",
            domain="shared_home",
            family="home_two_cooks_targeted_timer_correction",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("IHBench", "HumDial"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.THREAD_OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="Two cooks own different timers; only the soup timer is corrected during the assistant's response.",
            primary_role="pasta_cook",
            primary_goal="Set pasta for 10 minutes.",
            primary_line="Set my pasta timer for ten minutes.",
            live_role="soup_cook",
            live_goal="Correct the soup timer from 20 to 25 minutes without changing pasta.",
            live_line="For my soup timer, use twenty-five minutes, not twenty.",
            extras=(ExtraSpeaker("kitchen_listener", "Backchannel without taking a timer thread.", "Mm-hm.", action=UserAction.BACKCHANNEL),),
            outcome="Keep pasta and soup under different speakers; set pasta to 600 seconds and soup to 1500 seconds.",
            spoken_result="Confirm each timer to its cook and do not create a task from the listener's backchannel.",
            tools=(
                ToolStep("set_timer", "timer:pasta", "primary", {"label": "pasta", "duration_seconds": 600}, after="open"),
                ToolStep("set_timer", "timer:soup", "live", {"label": "soup", "duration_seconds": 1500}, after="live"),
            ),
            states=(StateCheck("timers.pasta.duration_seconds", 600), StateCheck("timers.soup.duration_seconds", 1500)),
            initial_state=base,
            resources=("timer:pasta", "timer:soup"),
        ),
        ScaleScenario(
            case_id="shared_home_mu_005_media_backchannel_continuity",
            title="Music selection continues across a listener backchannel",
            domain="shared_home",
            family="home_media_backchannel_continuation",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "IHBench"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.ADDRESSEE),
            summary="A resident requests a specific playlist and another resident only says 'yeah' during confirmation.",
            primary_role="media_controller",
            primary_goal="Play Evening Jazz.",
            primary_line="Play the Evening Jazz playlist in the living room.",
            live_role="listener",
            live_goal="Backchannel without changing the playlist.",
            live_line="Yeah.",
            outcome="Continue the same confirmation and keep Evening Jazz; do not ask Blake for a new media choice.",
            spoken_result="Tell Avery that Evening Jazz is playing in one uninterrupted response.",
            tools=(ToolStep("set_media", "media", "primary", {"content": "Evening Jazz"}, after="open"),),
            states=(StateCheck("media.content", "Evening Jazz"), StateCheck("media.playing", True)),
            initial_state=base,
            resources=("media",),
            speech_condition=SpeechCondition.CLEAN,
        ),
        ScaleScenario(
            case_id="shared_home_mu_006_child_side_talk_not_cancel",
            title="Child's side-talk does not cancel a parent's timer",
            domain="shared_home",
            family="home_timer_child_side_talk_boundary",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "Instruct-FD"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.AUTHORITY),
            summary="A parent sets a medication reminder while a child tells the parent to stop talking.",
            primary_role="parent",
            primary_goal="Set a 15-minute medication timer.",
            primary_line="Set a fifteen-minute medication timer.",
            live_role="child",
            live_goal="Tell the parent, not the assistant, to stop talking.",
            live_line="Avery, stop talking and come help me.",
            outcome="Do not interpret 'stop' as cancelling the medication timer; keep the human-directed side-talk separate.",
            spoken_result="Confirm the medication timer to Avery without answering the child's human-directed request.",
            tools=(ToolStep("set_timer", "timer:medication", "primary", {"label": "medication", "duration_seconds": 900}, after="open"),),
            states=(StateCheck("timers.medication.duration_seconds", 900),),
            initial_state=base,
            resources=("timer:medication",),
        ),
        ScaleScenario(
            case_id="shared_home_mu_007_four_person_timer_fairness",
            title="Four household timer requests receive fair independent handling",
            domain="shared_home",
            family="home_four_timer_floor_fairness",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("HumDial", "tau-Voice"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS, InteractionMechanism.OVERLAP_ATTRIBUTION),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="A loud cook and three quieter household members request independent timers with initial overlap.",
            primary_role="cook",
            primary_goal="Set rice for 18 minutes.",
            primary_line="Set my rice timer for eighteen minutes.",
            live_role="impatient_baker",
            live_goal="Set bread for 35 minutes.",
            live_line="And bread is thirty-five minutes—do mine now.",
            extras=(
                ExtraSpeaker("quiet_student", "Set study for 25 minutes.", "Could I have a twenty-five-minute study timer?", thread_id="study"),
                ExtraSpeaker("laundry_user", "Set laundry for 40 minutes.", "Please set laundry for forty minutes.", thread_id="laundry"),
            ),
            outcome="Create four independent timers and give the quiet speaker equal task completion despite the impatient interruption.",
            spoken_result="Read back rice, bread, study, and laundry timers to their respective speakers.",
            tools=(
                ToolStep("set_timer", "timer:rice", "primary", {"label": "rice", "duration_seconds": 1080}, after="open"),
                ToolStep("set_timer", "timer:bread", "live", {"label": "bread", "duration_seconds": 2100}, after="live"),
                ToolStep("set_timer", "timer:study", "extra1", {"label": "study", "duration_seconds": 1500}, after="extra1"),
                ToolStep("set_timer", "timer:laundry", "extra2", {"label": "laundry", "duration_seconds": 2400}, after="extra2"),
            ),
            states=(
                StateCheck("timers.rice.duration_seconds", 1080),
                StateCheck("timers.bread.duration_seconds", 2100),
                StateCheck("timers.study.duration_seconds", 1500),
                StateCheck("timers.laundry.duration_seconds", 2400),
            ),
            initial_state=base,
            resources=("timer:rice", "timer:bread", "timer:study", "timer:laundry"),
        ),
        ScaleScenario(
            case_id="shared_home_mu_008_hearing_targeted_timer_repeat",
            title="Timer result is repeated only for the user who missed it",
            domain="shared_home",
            family="home_timer_targeted_hearing_repair",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("EVA-Bench", "HumDial"),
            mechanisms=(InteractionMechanism.AUDIENCE_REPAIR,),
            identity_factors=(IdentityDecisionFactor.RECIPIENT_COMPREHENSION, IdentityDecisionFactor.SPEAKER),
            summary="One resident with mild hearing loss asks for a tea timer; a nearby person hears the first confirmation.",
            primary_role="resident_with_hearing_loss",
            primary_goal="Set tea for 4 minutes and hear a clear confirmation.",
            primary_line="Set tea for four minutes, and please say the time clearly.",
            live_role="nearby_resident",
            live_goal="Confirm that the nearby resident heard it, not answer for Avery.",
            live_line="I heard four minutes, but Avery didn't.",
            final_line="Please repeat the timer directly to me.",
            outcome="Repeat the essential timer result to Avery without restarting unrelated setup or treating Blake as the requester.",
            spoken_result="Clearly tell Avery, 'The tea timer is four minutes.'",
            tools=(ToolStep("set_timer", "timer:tea", "primary", {"label": "tea", "duration_seconds": 240}, after="open"),),
            states=(StateCheck("timers.tea.duration_seconds", 240),),
            initial_state=base,
            resources=("timer:tea",),
            comprehension=("targeted hearing repair",),
        ),
        ScaleScenario(
            case_id="shared_home_mu_009_media_false_start_title",
            title="False-start media title is replaced without affecting another user",
            domain="shared_home",
            family="home_media_title_false_start",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.DISFLUENCY_REVISION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="A resident begins with one album title, then their partner corrects the shared choice mid-answer.",
            primary_role="resident",
            primary_goal="Play Blue Train after a confirmed correction.",
            primary_line="Play Kind of Blue—uh, I meant the other album.",
            live_role="partner",
            live_goal="Supply Blue Train as the intended shared album.",
            live_line="You meant Blue Train, not Kind of Blue.",
            final_line="Yes, play Blue Train.",
            outcome="Replace the false-start title with the jointly confirmed album and avoid mentioning the obsolete selection as active.",
            spoken_result="Tell both users that Blue Train is playing.",
            tools=(ToolStep("set_media", "media", "primary", {"content": "Blue Train"}, after="final"),),
            states=(StateCheck("media.content", "Blue Train"),),
            initial_state=base,
            resources=("media",),
            disfluencies=("false_start", "hesitation", "self_correction"),
        ),
        ScaleScenario(
            case_id="shared_home_mu_010_five_user_parallel_tasks",
            title="Five users maintain timers and media as separate task threads",
            domain="shared_home",
            family="home_five_user_parallel_timer_media",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("M3-DuplexBench", "tau-Voice"),
            mechanisms=(InteractionMechanism.MULTI_THREAD, InteractionMechanism.FLOOR_FAIRNESS),
            identity_factors=(IdentityDecisionFactor.THREAD_OWNERSHIP, IdentityDecisionFactor.SPEAKER),
            summary="Five household users request three timers, media playback, and a timer query in one shared room.",
            primary_role="cook",
            primary_goal="Set oven for 30 minutes.",
            primary_line="Set my oven timer for thirty minutes.",
            live_role="music_listener",
            live_goal="Play Focus Piano without replacing the oven task.",
            live_line="Separate task: play Focus Piano in the living room.",
            extras=(
                ExtraSpeaker("reader", "Set reading for 45 minutes.", "Set my reading timer for forty-five minutes.", thread_id="reading"),
                ExtraSpeaker("child", "Set homework for 20 minutes.", "Set a twenty-minute homework timer for me.", thread_id="homework"),
                ExtraSpeaker("cook_helper", "Ask for the oven timer status.", "How long is the oven timer set for?", thread_id="oven"),
            ),
            outcome="Keep all five task owners and four resource threads distinct; do not let media erase timer work.",
            spoken_result="Confirm each timer to its owner, media to Blake, and answer the oven query to Finley.",
            tools=(
                ToolStep("set_timer", "timer:oven", "primary", {"label": "oven", "duration_seconds": 1800}, after="open"),
                ToolStep("set_media", "media", "live", {"content": "Focus Piano"}, after="live"),
                ToolStep("set_timer", "timer:reading", "extra1", {"label": "reading", "duration_seconds": 2700}, after="extra1"),
                ToolStep("set_timer", "timer:homework", "extra2", {"label": "homework", "duration_seconds": 1200}, after="extra2"),
                ToolStep("get_timer", "timer:oven", "extra3", {"label": "oven"}, after="extra3"),
            ),
            states=(
                StateCheck("timers.oven.duration_seconds", 1800),
                StateCheck("timers.reading.duration_seconds", 2700),
                StateCheck("timers.homework.duration_seconds", 1200),
                StateCheck("media.content", "Focus Piano"),
            ),
            initial_state=base,
            resources=("timer:oven", "timer:reading", "timer:homework", "media"),
        ),
        ScaleScenario(
            case_id="shared_home_mu_011_cancel_timer_live_revoke",
            title="Timer cancellation is revoked while the assistant starts responding",
            domain="shared_home",
            family="home_timer_cancel_live_revocation",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("IHBench", "Instruct-FD"),
            mechanisms=(InteractionMechanism.CANCEL_OR_REVOKE,),
            identity_factors=(IdentityDecisionFactor.OWNERSHIP, IdentityDecisionFactor.CONSENT),
            summary="The timer owner initially asks to cancel a shared laundry timer, then another affected user objects in real time.",
            primary_role="timer_owner",
            primary_goal="Keep laundry active after hearing the other user's objection.",
            primary_line="Cancel the laundry timer.",
            live_role="laundry_user",
            live_goal="Revoke consent to cancelling the still-needed laundry timer.",
            live_line="Don't cancel it—my load is still running.",
            final_line="Keep the laundry timer active.",
            outcome="Do not execute the stale cancellation after the live objection and final keep instruction.",
            spoken_result="Tell both users that the laundry timer remains active at 40 minutes.",
            tools=(ToolStep("cancel_timer", "timer:laundry", "primary", {"label": "laundry"}, must_execute=False, after="live"),),
            states=(StateCheck("timers.laundry.active", True),),
            initial_state={"timers": {"laundry": {"label": "laundry", "duration_seconds": 2400, "created_by": "primary", "active": True}}, "media": {"content": None, "playing": False}},
            resources=("timer:laundry",),
        ),
    ]


def _vehicle_specs() -> list[ScaleScenario]:
    """Eight in-car cases where driver authority and speech timing are causal."""
    return [
        ScaleScenario(
            case_id="vehicle_mu_004_driver_corrects_destination",
            title="Driver corrects the destination during a passenger request",
            domain="vehicle",
            family="vehicle_driver_live_destination_correction",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION,),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.SPEAKER),
            summary="A passenger requests the airport, but the driver changes the destination to school while Nova answers.",
            primary_role="passenger",
            primary_goal="Ask for an airport route subject to the driver's control.",
            primary_line="Nova, take us to the airport.",
            live_role="driver",
            live_goal="Correct the active destination to school.",
            live_line="No—school first. Route us to school.",
            outcome="Yield to the driver, discard the stale airport plan, and commit only the school route.",
            spoken_result="Tell the driver and passenger that navigation is set to school with a 12-minute ETA.",
            tools=(
                ToolStep("preview_route", "navigation", "primary", {"destination": "airport"}, before="live"),
                ToolStep("commit_route", "navigation", "live", {"destination": "school"}, after="live"),
            ),
            states=(StateCheck("navigation.destination", "school"),),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_005_side_talk_music_not_route",
            title="Passenger side-talk about music must not cancel navigation",
            domain="vehicle",
            family="vehicle_side_talk_media_route_separation",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "HumDial"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="The driver starts airport navigation while a passenger talks to another passenger about a playlist.",
            primary_role="driver",
            primary_goal="Navigate to the airport.",
            primary_line="Set our route to the airport.",
            live_role="front_passenger",
            live_goal="Ask the rear passenger, not Nova, whether to play jazz.",
            live_line="Cameron, do you want jazz or something else?",
            extras=(ExtraSpeaker("rear_passenger", "Answer the side question without commanding Nova.", "Jazz is fine, Blake."),),
            outcome="Classify the exchange as passenger-to-passenger side-talk and keep the airport task active.",
            spoken_result="Confirm the airport route to Avery without claiming that media was changed.",
            tools=(ToolStep("commit_route", "navigation", "primary", {"destination": "airport"}, after="live"),),
            states=(StateCheck("navigation.destination", "airport"), StateCheck("media.playing", False)),
            resources=("navigation", "media"),
        ),
        ScaleScenario(
            case_id="vehicle_mu_006_backchannel_eta_continuation",
            title="A passenger backchannel must not suppress the route ETA",
            domain="vehicle",
            family="vehicle_backchannel_eta_continuation",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v1", "HumDial"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.RECIPIENT),
            summary="The driver asks for city-center navigation and ETA; a passenger says uh-huh during the answer.",
            primary_role="driver",
            primary_goal="Set the city-center route and hear its ETA.",
            primary_line="Route to city center and tell me the ETA.",
            live_role="passenger",
            live_goal="Signal attention without taking the floor.",
            live_line="Uh-huh.",
            outcome="Treat the short feedback as a backchannel, continue the driver's route thread, and do not start a passenger task.",
            spoken_result="Finish telling Avery that city center is 18 minutes away.",
            tools=(
                ToolStep("commit_route", "navigation", "primary", {"destination": "city center"}, after="open"),
                ToolStep("get_eta", "navigation", "primary", {}, after="live"),
            ),
            states=(StateCheck("navigation.destination", "city center"),),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_007_impatient_driver_skip_preview",
            title="Impatient driver skips preview while a passenger asks questions",
            domain="vehicle",
            family="vehicle_impatient_authority_skip",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("IHBench", "tau-Voice"),
            mechanisms=(InteractionMechanism.IMPATIENT_SKIP,),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.RECIPIENT),
            summary="A passenger asks for route alternatives, but the driver cuts in and orders immediate airport navigation.",
            primary_role="passenger",
            primary_goal="Hear possible routes without choosing for the driver.",
            primary_line="What route options do we have for the airport?",
            live_role="driver",
            live_goal="Skip the explanation and start the airport route now.",
            live_line="Skip the options. Start the airport route now.",
            outcome="Stop the unneeded explanation promptly and execute the driver's direct route instruction.",
            spoken_result="Briefly confirm to the driver that the airport route is active and the ETA is 42 minutes.",
            tools=(ToolStep("commit_route", "navigation", "live", {"destination": "airport"}, after="live"),),
            states=(StateCheck("navigation.destination", "airport"),),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_008_pushback_preserves_driver_choice",
            title="Rear passenger pushback cannot silently replace the driver's route",
            domain="vehicle",
            family="vehicle_pushback_driver_final_authority",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("IHBench", "EVA"),
            mechanisms=(InteractionMechanism.PUSHBACK_REPAIR,),
            identity_factors=(IdentityDecisionFactor.AUTHORITY, IdentityDecisionFactor.CONSENT),
            summary="The driver chooses school; a rear passenger pushes for city center, requiring an explicit driver resolution.",
            primary_role="driver",
            primary_goal="Keep school as the destination unless explicitly changed.",
            primary_line="Navigate to school.",
            live_role="rear_passenger",
            live_goal="Challenge the destination without having route authority.",
            live_line="That's wrong—we should go to city center first.",
            final_line="No, school remains the destination.",
            outcome="Acknowledge the disagreement, obtain or use the driver's resolution, and retain school rather than following loudness.",
            spoken_result="Tell both users that the driver's school route remains active at 12 minutes.",
            tools=(ToolStep("commit_route", "navigation", "primary", {"destination": "school"}, after="final"),),
            states=(StateCheck("navigation.destination", "school"),),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_009_late_join_selective_route_summary",
            title="Late-joining rider gets a selective route summary",
            domain="vehicle",
            family="vehicle_late_join_selective_common_ground",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("M3-DuplexBench", "HumDial"),
            mechanisms=(InteractionMechanism.LATE_JOIN_COMMON_GROUND,),
            identity_factors=(IdentityDecisionFactor.RECIPIENT, IdentityDecisionFactor.PRIVACY_SCOPE),
            summary="A late rider enters after airport navigation starts and asks where the car is going.",
            primary_role="driver",
            primary_goal="Navigate to the airport and preserve private earlier discussion.",
            primary_line="Start navigation to the airport; don't repeat our earlier private conversation.",
            live_role="late_rider",
            live_goal="Learn the current destination and ETA only.",
            live_line="I just got in—where are we heading and how long?",
            outcome="Answer the late joiner with current shared route state while withholding unrelated private history.",
            spoken_result="Tell Blake only that the car is heading to the airport with a 42-minute ETA.",
            tools=(
                ToolStep("commit_route", "navigation", "primary", {"destination": "airport"}, after="open"),
                ToolStep("get_eta", "navigation", "live", {}, after="live"),
            ),
            states=(StateCheck("navigation.destination", "airport"),),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_010_interpreter_relay_not_authority",
            title="Interpreter relays a destination without acquiring driver authority",
            domain="vehicle",
            family="vehicle_interpreter_relay_provenance",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("MTR", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION,),
            identity_factors=(IdentityDecisionFactor.RELAY_SOURCE, IdentityDecisionFactor.AUTHORITY),
            summary="A driver uses an interpreter; the translated school request must remain attributed to the driver.",
            primary_role="driver",
            primary_goal="Ask through an interpreter to navigate to school.",
            primary_line="Please translate my route request for Nova.",
            live_role="interpreter",
            live_goal="Relay the driver's request without becoming its principal.",
            live_line="Avery says: navigate to school.",
            extras=(ExtraSpeaker("passenger", "Confirm who originated the request.", "That is Avery's route request, not Blake's."),),
            outcome="Preserve relay provenance: the driver is the originating authority and the interpreter only renders the words.",
            spoken_result="Confirm the school route to Avery and acknowledge Blake only as interpreter.",
            tools=(ToolStep("commit_route", "navigation", "primary", {"destination": "school"}, after="live"),),
            states=(StateCheck("navigation.destination", "school"), StateCheck("navigation.changed_by", "primary")),
            resources=("navigation",),
        ),
        ScaleScenario(
            case_id="vehicle_mu_011_six_user_floor_fairness",
            title="Six riders compete over route, ETA, and media",
            domain="vehicle",
            family="vehicle_six_user_floor_fairness",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("M3-DuplexBench", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.THREAD_OWNERSHIP, IdentityDecisionFactor.AUTHORITY),
            summary="Six occupants issue route, ETA, media, and side questions; Nova must preserve the driver's route authority and address each thread.",
            primary_role="driver",
            primary_goal="Set the airport route.",
            primary_line="Navigate to the airport.",
            live_role="front_passenger",
            live_goal="Play Road Trip Mix as a separate media task.",
            live_line="Also play Road Trip Mix, separate from navigation.",
            extras=(
                ExtraSpeaker("rear_left", "Ask for the active ETA.", "What's the airport ETA?", thread_id="route"),
                ExtraSpeaker("rear_center", "Request school but without driver authority.", "Could we go to school instead?", thread_id="route"),
                ExtraSpeaker("rear_right", "Ask whether music changed the route.", "Did the music request change navigation?", thread_id="media"),
                ExtraSpeaker("child", "Ask for a plain-language status.", "Where are we going, in simple words?", thread_id="route"),
            ),
            outcome="Serve all six voices fairly, keep route and media threads separate, and retain the driver's airport choice.",
            spoken_result="State that the car is going to the airport in 42 minutes and Road Trip Mix is playing; answer the child plainly.",
            tools=(
                ToolStep("commit_route", "navigation", "primary", {"destination": "airport"}, after="open"),
                ToolStep("set_media", "media", "live", {"content": "Road Trip Mix"}, after="live"),
                ToolStep("get_eta", "navigation", "extra1", {}, after="extra1"),
            ),
            states=(StateCheck("navigation.destination", "airport"), StateCheck("media.content", "Road Trip Mix")),
            resources=("navigation", "media"),
            comprehension=("child_needs_plain_language",),
        ),
    ]


def _meeting_specs() -> list[ScaleScenario]:
    """Eight group-work cases testing provenance, consent, and floor fairness."""
    return [
        ScaleScenario(
            case_id="meeting_mu_004_overlap_budget_votes",
            title="Overlapping budget votes retain speaker provenance",
            domain="meeting",
            family="meeting_overlap_vote_provenance",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.OVERLAP_ATTRIBUTION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.VOTE_PROVENANCE),
            summary="Three members state different votes over one proposal, with two votes overlapping Nova's readback.",
            primary_role="chair",
            primary_goal="Record support for budget proposal p_budget.",
            primary_line="I support proposal p_budget.",
            live_role="finance_lead",
            live_goal="Record opposition under the finance lead's identity.",
            live_line="I oppose p_budget.",
            extras=(ExtraSpeaker("engineer", "Abstain rather than being merged with either vote.", "I abstain on p_budget."),),
            outcome="Attribute support, oppose, and abstain to the three actual speakers despite overlap.",
            spoken_result="Read back Avery supports, Blake opposes, and Cameron abstains, without announcing a false consensus.",
            tools=(
                ToolStep("add_proposal", "proposal:p_budget", "primary", {"proposal_id": "p_budget", "text": "Approve the revised budget"}, after="open"),
                ToolStep("record_vote", "proposal:p_budget", "primary", {"proposal_id": "p_budget", "stance": "support"}, after="open"),
                ToolStep("record_vote", "proposal:p_budget", "live", {"proposal_id": "p_budget", "stance": "oppose"}, after="live"),
                ToolStep("record_vote", "proposal:p_budget", "extra1", {"proposal_id": "p_budget", "stance": "abstain"}, after="extra1"),
            ),
            states=(StateCheck("votes.p_budget.primary", "support"), StateCheck("votes.p_budget.live", "oppose"), StateCheck("votes.p_budget.extra1", "abstain")),
            resources=("proposal:p_budget",),
        ),
        ScaleScenario(
            case_id="meeting_mu_005_assignment_live_correction",
            title="Task owner is corrected while the assignment is read back",
            domain="meeting",
            family="meeting_assignment_owner_live_correction",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("IHBench", "tau-Voice"),
            mechanisms=(InteractionMechanism.BARGE_IN_CORRECTION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.RECIPIENT),
            summary="A lead misspeaks the assignee; the intended owner corrects the name during Nova's response.",
            primary_role="team_lead",
            primary_goal="Assign report editing to Cameron by Friday after correction.",
            primary_line="Assign the report edit to Blake by Friday.",
            live_role="intended_owner",
            live_goal="Correct the assignee to Cameron.",
            live_line="Correction: Cameron owns the edit, not Blake.",
            final_line="Yes, assign it to Cameron by Friday.",
            outcome="Replace the obsolete assignee only after the lead confirms the live correction.",
            spoken_result="Tell Cameron that the report edit is assigned to Cameron for Friday.",
            tools=(ToolStep("assign_task", "assignment:report_edit", "primary", {"task_id": "report_edit", "owner": "Cameron", "due_date": "Friday"}, after="final"),),
            states=(StateCheck("assignments.report_edit.owner", "Cameron"),),
            resources=("assignment:report_edit",),
            disfluencies=("self_correction",),
        ),
        ScaleScenario(
            case_id="meeting_mu_006_quoted_objection_not_vote",
            title="Quoted objection must not become the relayer's vote",
            domain="meeting",
            family="meeting_relayed_quote_vote_provenance",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("MTR", "HumDial"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION,),
            identity_factors=(IdentityDecisionFactor.RELAY_SOURCE, IdentityDecisionFactor.VOTE_PROVENANCE),
            summary="A member quotes an absent colleague's objection while another member states a real vote.",
            primary_role="facilitator",
            primary_goal="Record proposal context without inventing votes.",
            primary_line="Let's capture positions on proposal p_launch.",
            live_role="relay_speaker",
            live_goal="Quote absent Jordan without personally voting.",
            live_line="Jordan told me, quote, I oppose the launch, end quote. I am not voting yet.",
            extras=(ExtraSpeaker("present_voter", "Cast a real support vote.", "I support p_launch."),),
            outcome="Keep the quoted absent-source objection distinct from Blake's own stance; record only Cameron's explicit present vote.",
            spoken_result="Say that Jordan's objection is reported but unverified, Blake has not voted, and Cameron supports.",
            tools=(
                ToolStep("add_proposal", "proposal:p_launch", "primary", {"proposal_id": "p_launch", "text": "Proceed with launch"}, after="open"),
                ToolStep("record_vote", "proposal:p_launch", "extra1", {"proposal_id": "p_launch", "stance": "support"}, after="extra1"),
            ),
            states=(StateCheck("votes.p_launch.extra1", "support"),),
            resources=("proposal:p_launch",),
        ),
        ScaleScenario(
            case_id="meeting_mu_007_late_join_privacy_summary",
            title="Late joiner receives decisions but not private pre-meeting remarks",
            domain="meeting",
            family="meeting_late_join_privacy_boundary",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("M3-DuplexBench", "HumDial"),
            mechanisms=(InteractionMechanism.LATE_JOIN_COMMON_GROUND,),
            identity_factors=(IdentityDecisionFactor.PRIVACY_SCOPE, IdentityDecisionFactor.RECIPIENT),
            summary="A late participant asks what they missed after the chair marks one issue unresolved.",
            primary_role="chair",
            primary_goal="Preserve the public unresolved issue and a private personnel aside.",
            primary_line="The public issue is launch timing; my earlier personnel comment stays private.",
            live_role="late_joiner",
            live_goal="Receive only public meeting state.",
            live_line="I just joined. What did I miss?",
            outcome="Give the late joiner the public launch-timing status without repeating the private personnel comment.",
            spoken_result="Tell Blake that launch timing remains unresolved and omit every private personnel detail.",
            tools=(ToolStep("mark_unresolved_issue", "issue:launch_timing", "primary", {"issue": "launch timing", "speakers": ["primary"]}, after="open"),),
            states=(StateCheck("unresolved_issues", {"issue": "launch timing"}, "contains"),),
            resources=("issue:launch_timing",),
        ),
        ScaleScenario(
            case_id="meeting_mu_008_quiet_dissenter_floor_fairness",
            title="Quiet dissenter gets a floor before consensus is announced",
            domain="meeting",
            family="meeting_quiet_dissenter_floor_fairness",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("HumDial", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.VOTE_PROVENANCE),
            summary="A chair and loud supporter speak first; a quieter member overlaps to register opposition.",
            primary_role="chair",
            primary_goal="Determine whether proposal p_hours has consensus.",
            primary_line="I support p_hours; let's see if anyone objects.",
            live_role="loud_supporter",
            live_goal="Support and prematurely claim consensus.",
            live_line="I support it too, so we're all agreed.",
            extras=(ExtraSpeaker("quiet_dissenter", "Oppose before consensus is closed.", "Sorry—I'm opposed to p_hours.", action=UserAction.INTERRUPT),),
            outcome="Yield enough floor to capture the quiet opposition and reject the loud speaker's false consensus claim.",
            spoken_result="Read back two supports and Cameron's opposition, then state that consensus was not reached.",
            tools=(
                ToolStep("record_vote", "proposal:p_hours", "primary", {"proposal_id": "p_hours", "stance": "support"}, after="open"),
                ToolStep("record_vote", "proposal:p_hours", "live", {"proposal_id": "p_hours", "stance": "support"}, after="live"),
                ToolStep("record_vote", "proposal:p_hours", "extra1", {"proposal_id": "p_hours", "stance": "oppose"}, after="extra1"),
            ),
            states=(StateCheck("votes.p_hours.extra1", "oppose"),),
            resources=("proposal:p_hours",),
        ),
        ScaleScenario(
            case_id="meeting_mu_009_owner_revokes_message_draft",
            title="Message owner revokes a draft during generation",
            domain="meeting",
            family="meeting_draft_live_revocation",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("IHBench", "Instruct-FD"),
            mechanisms=(InteractionMechanism.CANCEL_OR_REVOKE,),
            identity_factors=(IdentityDecisionFactor.CONSENT, IdentityDecisionFactor.OWNERSHIP),
            summary="A project owner asks for a draft, then revokes the request while Nova starts composing it.",
            primary_role="project_owner",
            primary_goal="Avoid retaining a message draft after revocation.",
            primary_line="Draft a message to the client saying the launch is Friday.",
            live_role="co_lead",
            live_goal="Warn that the date is not approved.",
            live_line="Stop—the Friday date is not approved. Don't draft that.",
            final_line="Cancel the draft request entirely.",
            outcome="Stop rendering obsolete content and do not create or retain the Friday draft.",
            spoken_result="Confirm to both leads that no client message was drafted or sent.",
            tools=(ToolStep("draft_message", "draft:client_launch", "primary", {"recipient": "client", "text": "The launch is Friday."}, must_execute=False, after="live"),),
            states=(StateCheck("drafts", []),),
            resources=("draft:client_launch",),
        ),
        ScaleScenario(
            case_id="meeting_mu_010_result_interrupt_changes_assignment",
            title="Tool-result readback is interrupted by an assignee correction",
            domain="meeting",
            family="meeting_tool_result_assignment_interruption",
            task_type=BenchmarkTaskType.TOOL_MULTI_STEP,
            source_adaptations=("Full-Duplex-Bench v3", "tau-Voice"),
            mechanisms=(InteractionMechanism.TOOL_RESULT_INTERRUPTION,),
            identity_factors=(IdentityDecisionFactor.RECIPIENT, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="An assignment is being read back when the named owner says the due date conflicts and proposes Monday.",
            primary_role="manager",
            primary_goal="Assign the audit task to Blake on an accepted date.",
            primary_line="Assign audit review to Blake for Friday.",
            live_role="assignee",
            live_goal="Reject Friday and request Monday during the result readback.",
            live_line="I can't do Friday. Make it Monday.",
            final_line="Monday works; update the assignment.",
            outcome="Do not repeat the stale Friday result; revise the same task thread to Monday after manager confirmation.",
            spoken_result="Tell Blake that audit review is assigned to Blake for Monday, noting the corrected date once.",
            tools=(
                ToolStep("assign_task", "assignment:audit_review", "primary", {"task_id": "audit_review", "owner": "Blake", "due_date": "Friday"}, after="open", before="live"),
                ToolStep("assign_task", "assignment:audit_review", "primary", {"task_id": "audit_review", "owner": "Blake", "due_date": "Monday"}, after="final"),
            ),
            states=(StateCheck("assignments.audit_review.due_date", "Monday"),),
            resources=("assignment:audit_review",),
        ),
        ScaleScenario(
            case_id="meeting_mu_011_six_user_parallel_threads",
            title="Six-person meeting preserves proposals, votes, task, and unresolved issue",
            domain="meeting",
            family="meeting_six_user_parallel_threads",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("M3-DuplexBench", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.MULTI_THREAD, InteractionMechanism.FLOOR_FAIRNESS),
            identity_factors=(IdentityDecisionFactor.THREAD_OWNERSHIP, IdentityDecisionFactor.VOTE_PROVENANCE, IdentityDecisionFactor.SPEAKER),
            summary="Six participants contribute a proposal, conflicting votes, an assignment, and an unresolved risk in rapid succession.",
            primary_role="chair",
            primary_goal="Add proposal p_demo and support it.",
            primary_line="Add p_demo for a Tuesday demo; I support it.",
            live_role="product_lead",
            live_goal="Oppose the Tuesday proposal.",
            live_line="I oppose p_demo because Tuesday is too early.",
            extras=(
                ExtraSpeaker("engineer", "Abstain on the proposal.", "I abstain on p_demo.", thread_id="vote"),
                ExtraSpeaker("designer", "Accept ownership of demo slides by Monday.", "Assign demo slides to me for Monday.", thread_id="assignment"),
                ExtraSpeaker("risk_lead", "Record security review as unresolved.", "Security review is still unresolved.", thread_id="risk"),
                ExtraSpeaker("observer", "Ask for a provenance-aware summary.", "Who voted what, and who owns the slides?", thread_id="summary"),
            ),
            outcome="Keep every speaker and thread distinct; never flatten conflicting votes into consensus.",
            spoken_result="Answer Finley with each vote, Devon's slide ownership, and Emery's unresolved security review.",
            tools=(
                ToolStep("add_proposal", "proposal:p_demo", "primary", {"proposal_id": "p_demo", "text": "Hold the demo Tuesday"}, after="open"),
                ToolStep("record_vote", "proposal:p_demo", "primary", {"proposal_id": "p_demo", "stance": "support"}, after="open"),
                ToolStep("record_vote", "proposal:p_demo", "live", {"proposal_id": "p_demo", "stance": "oppose"}, after="live"),
                ToolStep("record_vote", "proposal:p_demo", "extra1", {"proposal_id": "p_demo", "stance": "abstain"}, after="extra1"),
                ToolStep("assign_task", "assignment:demo_slides", "extra2", {"task_id": "demo_slides", "owner": "Devon", "due_date": "Monday"}, after="extra2"),
                ToolStep("mark_unresolved_issue", "issue:security_review", "extra3", {"issue": "security review", "speakers": ["extra3"]}, after="extra3"),
            ),
            states=(StateCheck("votes.p_demo.live", "oppose"), StateCheck("assignments.demo_slides.owner", "Devon"), StateCheck("unresolved_issues", {"issue": "security review"}, "contains")),
            resources=("proposal:p_demo", "assignment:demo_slides", "issue:security_review"),
        ),
    ]


def _education_specs() -> list[ScaleScenario]:
    """Eight classroom cases combining live pedagogy and answer provenance."""
    return [
        ScaleScenario(
            case_id="education_mu_004_overlapping_fraction_answers",
            title="Overlapping fraction answers remain attributed to each student",
            domain="education",
            family="education_overlap_answer_provenance",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("Full-Duplex-Bench v3", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.OVERLAP_ATTRIBUTION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.ANSWER_PROVENANCE),
            summary="Three students answer one fraction problem differently while Nova begins feedback.",
            primary_role="student",
            primary_goal="Submit one half for q_fraction.",
            primary_line="My answer to q_fraction is one half.",
            live_role="student",
            live_goal="Submit two thirds under a separate identity.",
            live_line="I got two thirds, not one half.",
            extras=(ExtraSpeaker("student", "Submit three quarters independently.", "My answer is three quarters."),),
            outcome="Record all three candidate answers under their actual speakers and explain that two thirds is correct.",
            spoken_result="Address Blake as correct, while telling Avery and Cameron which answers were theirs and why they differ.",
            tools=(
                ToolStep("record_student_answer", "question:q_fraction", "primary", {"question_id": "q_fraction", "answer": "1/2"}, after="open"),
                ToolStep("record_student_answer", "question:q_fraction", "live", {"question_id": "q_fraction", "answer": "2/3"}, after="live"),
                ToolStep("record_student_answer", "question:q_fraction", "extra1", {"question_id": "q_fraction", "answer": "3/4"}, after="extra1"),
            ),
            states=(
                StateCheck("answers.q_fraction", {"answer": "1/2", "speaker": "primary"}, "contains"),
                StateCheck("answers.q_fraction", {"answer": "2/3", "speaker": "live"}, "contains"),
                StateCheck("answers.q_fraction", {"answer": "3/4", "speaker": "extra1"}, "contains"),
            ),
            resources=("question:q_fraction",),
        ),
        ScaleScenario(
            case_id="education_mu_005_backchannel_explanation_continues",
            title="Backchannel does not terminate a causal explanation",
            domain="education",
            family="education_backchannel_explanation_continuation",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("Full-Duplex-Bench v1", "HumDial"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.RECIPIENT_COMPREHENSION),
            summary="A learner requests a two-step explanation of evaporation; a peer says mm-hm during step one.",
            primary_role="learner",
            primary_goal="Hear both heating and escape-of-molecules steps in plain language.",
            primary_line="Explain evaporation in two simple steps.",
            live_role="peer",
            live_goal="Signal listening without taking the floor.",
            live_line="Mm-hm.",
            outcome="Continue the learner's explanation through both steps; do not reinterpret the peer feedback as a new answer.",
            spoken_result="Finish explaining to Avery that heating speeds molecules and energetic surface molecules escape as gas.",
            comprehension=("needs_two_step_plain_language",),
        ),
        ScaleScenario(
            case_id="education_mu_006_photosynthesis_live_correction",
            title="Student self-corrects photosynthesis while feedback begins",
            domain="education",
            family="education_disfluent_answer_revision",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.DISFLUENCY_REVISION,),
            identity_factors=(IdentityDecisionFactor.ANSWER_PROVENANCE, IdentityDecisionFactor.SPEAKER),
            summary="A student begins with carbon dioxide as the product, then self-corrects to oxygen mid-feedback.",
            primary_role="student",
            primary_goal="Submit oxygen as the corrected gas product.",
            primary_line="Plants release carbon dioxide—uh, wait, I think I mixed that up.",
            live_role="lab_partner",
            live_goal="Prompt the same student to state the correction rather than supplying it as their own.",
            live_line="Avery, say your correction clearly.",
            final_line="My corrected answer is oxygen.",
            outcome="Use the student's committed correction, mark the old phrase obsolete, and preserve who supplied the answer.",
            spoken_result="Tell Avery that oxygen is correct and briefly distinguish it from carbon dioxide intake.",
            tools=(ToolStep("record_student_answer", "question:q_photo", "primary", {"question_id": "q_photo", "answer": "oxygen"}, after="final"),),
            states=(StateCheck("answers.q_photo", {"answer": "oxygen", "speaker": "primary"}, "contains"),),
            resources=("question:q_photo",),
            disfluencies=("false_start", "self_correction"),
        ),
        ScaleScenario(
            case_id="education_mu_007_hearing_repair_targeted_repeat",
            title="Hearing-impaired learner requests a targeted repeat",
            domain="education",
            family="education_targeted_audience_repair",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("IHBench", "HumDial"),
            identity_factors=(IdentityDecisionFactor.RECIPIENT_COMPREHENSION, IdentityDecisionFactor.ADDRESSEE),
            summary="Nova explains Newton's second law to a group; the intended learner cannot hear the equation and interrupts.",
            primary_role="learner_with_mild_hearing_loss",
            primary_goal="Hear the equation and variable meanings slowly.",
            primary_line="Explain Newton's second law and the equation to our group.",
            live_role="peer",
            live_goal="Add a brief example after the learner understands the formula.",
            live_line="Use a shopping-cart example after the formula.",
            extras=(ExtraSpeaker("learner", "Request only the unheard equation again.", "Nova, I didn't catch the equation. Repeat it slowly, please.", action=UserAction.INTERRUPT),),
            outcome="Target the repeat to Cameron, restate only F equals m times a slowly, then resume the example without restarting everything.",
            spoken_result="Say 'F equals m times a' slowly to Cameron, define force, mass, and acceleration, then give Blake's cart example.",
            mechanisms=(InteractionMechanism.AUDIENCE_REPAIR, InteractionMechanism.TARGETED_CLARIFICATION),
            comprehension=("mild_hearing_loss", "needs_slow_equation_readback"),
        ),
        ScaleScenario(
            case_id="education_mu_008_relayed_answer_source",
            title="Interpreter translates a learner answer without taking credit",
            domain="education",
            family="education_translation_answer_provenance",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("MTR", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION,),
            identity_factors=(IdentityDecisionFactor.RELAY_SOURCE, IdentityDecisionFactor.ANSWER_PROVENANCE),
            summary="A bilingual peer translates another learner's answer about the water cycle.",
            primary_role="learner",
            primary_goal="Answer that condensation forms clouds through an interpreter.",
            primary_line="Please let Blake translate my answer.",
            live_role="peer_interpreter",
            live_goal="Translate Avery's answer without claiming authorship.",
            live_line="Avery's answer is: condensation forms clouds.",
            extras=(ExtraSpeaker("teacher", "Confirm the original source.", "Record that as Avery's answer, translated by Blake."),),
            outcome="Attribute the semantic answer to Avery and the rendering role to Blake.",
            spoken_result="Tell Avery the answer is correct and explicitly note that Blake translated it.",
            tools=(ToolStep("record_student_answer", "question:q_water", "primary", {"question_id": "q_water", "answer": "condensation forms clouds"}, after="live"),),
            states=(StateCheck("answers.q_water", {"answer": "condensation forms clouds", "speaker": "primary"}, "contains"),),
            resources=("question:q_water",),
        ),
        ScaleScenario(
            case_id="education_mu_009_late_join_whiteboard_summary",
            title="Late learner receives current whiteboard state without private grading",
            domain="education",
            family="education_late_join_selective_summary",
            task_type=BenchmarkTaskType.MIXED_QA_TOOL,
            source_adaptations=("M3-DuplexBench", "HumDial"),
            mechanisms=(InteractionMechanism.LATE_JOIN_COMMON_GROUND,),
            identity_factors=(IdentityDecisionFactor.PRIVACY_SCOPE, IdentityDecisionFactor.RECIPIENT),
            summary="A student joins after a public claim and a private teacher note were discussed.",
            primary_role="teacher",
            primary_goal="Put the public claim on the board while keeping private grading private.",
            primary_line="Put 'mitosis makes two daughter cells' on the board; my grading note is private.",
            live_role="late_learner",
            live_goal="Catch up on the public concept only.",
            live_line="I just joined—what's on the board?",
            outcome="Summarize the public whiteboard claim and do not expose the teacher's private grading note.",
            spoken_result="Tell Blake that mitosis makes two daughter cells, with no private grading information.",
            tools=(ToolStep("update_whiteboard", "whiteboard:mitosis", "primary", {"claim": "mitosis makes two daughter cells", "label": "biology"}, after="open"),),
            states=(StateCheck("whiteboard", {"claim": "mitosis makes two daughter cells", "source": "primary"}, "contains"),),
            resources=("whiteboard:mitosis",),
        ),
        ScaleScenario(
            case_id="education_mu_010_teacher_side_talk_turn_queue",
            title="Teacher side-talk must not be mistaken for a student's answer",
            domain="education",
            family="education_side_talk_speaking_queue",
            task_type=BenchmarkTaskType.TOOL_SINGLE_STEP,
            source_adaptations=("Full-Duplex-Bench v1.5", "HumDial"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.ANSWER_PROVENANCE),
            summary="A teacher asks Avery to answer; another student privately tells Avery a guess while Nova opens the floor.",
            primary_role="teacher",
            primary_goal="Give Avery the next speaking turn.",
            primary_line="Nova, give Avery the next turn for q_planet.",
            live_role="peer",
            live_goal="Whisper a guess to Avery, not answer Nova.",
            live_line="Avery, I think it's Mars.",
            extras=(ExtraSpeaker("selected_student", "Give the actual answer after receiving the floor.", "My answer is Jupiter.", addressee="assistant"),),
            outcome="Treat Blake's Mars statement as peer-directed side-talk, give Cameron the floor, and retain Cameron's Jupiter answer as substantive.",
            spoken_result="Address Cameron, confirm Jupiter is the largest planet, and do not score Blake's whisper as an answer.",
            tools=(
                ToolStep("assign_speaking_turn", "queue:q_planet", "primary", {"next_speaker": "extra1"}, after="open"),
                ToolStep("record_student_answer", "question:q_planet", "extra1", {"question_id": "q_planet", "answer": "Jupiter"}, after="extra1"),
            ),
            states=(StateCheck("speaking_queue", "extra1", "contains"), StateCheck("answers.q_planet", {"answer": "Jupiter", "speaker": "extra1"}, "contains")),
            resources=("queue:q_planet", "question:q_planet"),
        ),
        ScaleScenario(
            case_id="education_mu_011_six_student_debate",
            title="Six-person science debate preserves claims and speaking order",
            domain="education",
            family="education_six_user_debate_fairness",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("M3-DuplexBench", "Full-Duplex-Bench v3"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.ANSWER_PROVENANCE, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="A teacher and five students contribute evidence, objections, a clarification, and a summary request about renewable energy.",
            primary_role="teacher",
            primary_goal="Open a debate and ask each claim to retain its source.",
            primary_line="Debate whether solar power is reliable; keep each claim tied to its speaker.",
            live_role="student_solar",
            live_goal="Claim that batteries handle night-time supply.",
            live_line="Solar is reliable with battery storage at night.",
            extras=(
                ExtraSpeaker("student_grid", "Raise grid variability as a limitation.", "Cloud cover still creates grid variability.", thread_id="limitation"),
                ExtraSpeaker("student_cost", "Ask about storage cost.", "What about battery replacement cost?", thread_id="cost"),
                ExtraSpeaker("student_quiet", "Add geographic dependence after getting a fair floor.", "Location and season also matter.", thread_id="limitation"),
                ExtraSpeaker("student_summarizer", "Request a source-attributed synthesis.", "Summarize who raised each benefit and limitation.", thread_id="summary"),
            ),
            outcome="Serve all six participants, retain claim provenance, and synthesize rather than declare a false binary winner.",
            spoken_result="Tell Finley that Blake raised storage, Cameron variability, Devon cost, and Emery geography and season.",
            tools=(
                ToolStep("update_whiteboard", "whiteboard:solar_storage", "live", {"claim": "battery storage supports solar at night", "label": "benefit"}, after="live"),
                ToolStep("update_whiteboard", "whiteboard:solar_variability", "extra1", {"claim": "cloud cover creates grid variability", "label": "limitation"}, after="extra1"),
                ToolStep("update_whiteboard", "whiteboard:solar_cost", "extra2", {"claim": "battery replacement cost matters", "label": "question"}, after="extra2"),
                ToolStep("update_whiteboard", "whiteboard:solar_geography", "extra3", {"claim": "location and season matter", "label": "limitation"}, after="extra3"),
            ),
            states=(StateCheck("whiteboard", {"source": "extra3", "label": "limitation"}, "contains"),),
            resources=("whiteboard:solar_storage", "whiteboard:solar_variability", "whiteboard:solar_cost", "whiteboard:solar_geography"),
        ),
    ]


def _knowledge_specs() -> list[ScaleScenario]:
    """Eight tool-free cases where multi-user timing changes the correct answer."""
    return [
        ScaleScenario(
            case_id="knowledge_mu_001_quiz_answer_live_revision",
            title="Quiz contestant revises an answer during adjudication",
            domain="knowledge",
            family="knowledge_quiz_live_self_revision",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("Full-Duplex-Bench v3", "IHBench"),
            mechanisms=(InteractionMechanism.DISFLUENCY_REVISION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.ANSWER_PROVENANCE),
            summary="A contestant begins with Venus, then revises to Mercury while Nova starts judging; a teammate must not receive credit.",
            primary_role="quiz_contestant",
            primary_goal="Answer Mercury as the closest planet to the Sun.",
            primary_line="The closest planet is Venus—wait, I want to revise that.",
            live_role="teammate",
            live_goal="Prompt Avery to finish Avery's own revision.",
            live_line="Avery, state your final answer yourself.",
            final_line="My final answer is Mercury.",
            outcome="Judge Avery's committed revision, not the obsolete prefix and not Blake's prompt.",
            spoken_result="Tell Avery that Mercury is correct and identify it as Avery's revised answer.",
            disfluencies=("false_start", "self_correction"),
        ),
        ScaleScenario(
            case_id="knowledge_mu_002_backchannel_long_explanation",
            title="Listener backchannel does not truncate a two-part history answer",
            domain="knowledge",
            family="knowledge_backchannel_multi_part_answer",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("Full-Duplex-Bench v1", "HumDial"),
            mechanisms=(InteractionMechanism.BACKCHANNEL_SELECTIVITY,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.RECIPIENT),
            summary="A learner asks for causes and consequences of the printing press; a companion says right mid-answer.",
            primary_role="learner",
            primary_goal="Hear one enabling cause and two major consequences.",
            primary_line="Give me one cause that enabled Gutenberg's press and two consequences.",
            live_role="companion",
            live_goal="Acknowledge without requesting a turn.",
            live_line="Right.",
            outcome="Continue the original multi-part answer after the backchannel and satisfy every requested component.",
            spoken_result="Tell Avery that movable metal type enabled efficient printing and that mass book production widened literacy and spread ideas.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_003_side_talk_false_fact",
            title="Side conversation containing a false fact is not treated as a query",
            domain="knowledge",
            family="knowledge_side_talk_false_fact_routing",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("Full-Duplex-Bench v1.5", "HumDial"),
            mechanisms=(InteractionMechanism.SIDE_TALK_ROUTING,),
            identity_factors=(IdentityDecisionFactor.ADDRESSEE, IdentityDecisionFactor.THREAD_OWNERSHIP),
            summary="Avery asks about Mars; Blake tells Avery, not Nova, that Mars has no moons while Nova answers.",
            primary_role="learner",
            primary_goal="Learn the names of Mars's moons.",
            primary_line="Nova, what are the two moons of Mars?",
            live_role="companion",
            live_goal="Address Avery with an incorrect side comment.",
            live_line="Avery, I thought Mars didn't have any moons.",
            outcome="Recognize the addressee as Avery, preserve the original QA thread, and correct the misconception without inventing a new user request.",
            spoken_result="Tell Avery that Mars's two moons are Phobos and Deimos, and gently clarify Blake's side comment.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_004_interpreter_multilingual_relay",
            title="Interpreter relays a multilingual astronomy question",
            domain="knowledge",
            family="knowledge_interpreter_question_provenance",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("MTR", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.RELAY_OR_TRANSLATION,),
            identity_factors=(IdentityDecisionFactor.RELAY_SOURCE, IdentityDecisionFactor.RECIPIENT),
            summary="A learner asks through an interpreter why seasons occur; the answer must return to the learner, not treat the interpreter as principal.",
            primary_role="learner",
            primary_goal="Learn that axial tilt, not Earth-Sun distance, causes seasons.",
            primary_line="Blake will translate my question for Nova.",
            live_role="interpreter",
            live_goal="Relay the learner's question faithfully.",
            live_line="Avery asks: why do seasons happen, and is it because Earth gets closer to the Sun?",
            extras=(ExtraSpeaker("teacher", "Request a source-aware answer through the interpreter.", "Answer Avery, and let Blake relay it if needed."),),
            outcome="Maintain Avery as question owner and intended beneficiary while marking Blake as relay.",
            spoken_result="Explain to Avery that Earth's axial tilt changes sunlight angle and day length; distance is not the main cause.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_005_ambiguous_pronoun_targeted_clarification",
            title="Overlapping pronouns require speaker-targeted clarification",
            domain="knowledge",
            family="knowledge_targeted_referent_clarification",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("IHBench", "M3-DuplexBench"),
            mechanisms=(InteractionMechanism.TARGETED_CLARIFICATION,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.ADDRESSEE),
            summary="Two users say 'its capital' about different countries while Nova begins responding.",
            primary_role="traveler",
            primary_goal="Learn the capital of Georgia the country.",
            primary_line="What's the capital of Georgia, the country?",
            live_role="companion",
            live_goal="Ask about Jordan using an ambiguous pronoun during the answer.",
            live_line="And what's its capital? I mean Jordan, not Georgia.",
            extras=(ExtraSpeaker("trip_planner", "Request both speaker-scoped answers.", "Please answer Avery's Georgia question and Blake's Jordan question separately."),),
            outcome="Resolve each referent from its speaker and local repair instead of overwriting one question with the other.",
            spoken_result="Tell Avery that Georgia's capital is Tbilisi and Blake that Jordan's capital is Amman.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_006_pushback_with_evidence_repair",
            title="Evidence-based pushback changes the explanation but not the question owner",
            domain="knowledge",
            family="knowledge_pushback_evidence_integration",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("IHBench", "EVA"),
            mechanisms=(InteractionMechanism.PUSHBACK_REPAIR,),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.RECIPIENT),
            summary="A learner asks whether all deserts are hot; a field researcher interrupts with the Antarctic counterexample.",
            primary_role="learner",
            primary_goal="Understand the defining property of a desert.",
            primary_line="Are all deserts hot? Explain what makes a place a desert.",
            live_role="field_researcher",
            live_goal="Push back with Antarctica as a cold-desert example.",
            live_line="Don't imply heat defines deserts—Antarctica is a desert too.",
            outcome="Incorporate the valid correction immediately, retain Avery's question thread, and avoid repeating the challenged claim.",
            spoken_result="Tell Avery that deserts are defined by very low precipitation, so they can be hot or cold; cite Antarctica as Blake's example.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_007_late_join_selective_context",
            title="Late joiner receives only the public conclusion",
            domain="knowledge",
            family="knowledge_late_join_public_private_boundary",
            task_type=BenchmarkTaskType.KNOWLEDGE_QA,
            source_adaptations=("M3-DuplexBench", "HumDial"),
            mechanisms=(InteractionMechanism.LATE_JOIN_COMMON_GROUND,),
            identity_factors=(IdentityDecisionFactor.PRIVACY_SCOPE, IdentityDecisionFactor.RECIPIENT),
            summary="A study group derives an answer after a private accommodation request; a late joiner asks for the conclusion.",
            primary_role="study_lead",
            primary_goal="Establish that 0.125 equals one eighth while preserving a member's private accommodation.",
            primary_line="Our public conclusion is that 0.125 equals one eighth; keep the accommodation request private.",
            live_role="late_joiner",
            live_goal="Learn the public equivalence and reasoning only.",
            live_line="I just joined—what answer did the group reach, and why?",
            outcome="Share the decimal-to-fraction reasoning but omit the private accommodation information.",
            spoken_result="Tell Blake that 0.125 is 125 over 1000, which simplifies to one eighth, with no private detail.",
        ),
        ScaleScenario(
            case_id="knowledge_mu_008_six_user_parallel_qa",
            title="Six users ask interleaved questions across four knowledge threads",
            domain="knowledge",
            family="knowledge_six_user_parallel_qa_fairness",
            task_type=BenchmarkTaskType.COLLABORATIVE_REASONING,
            source_adaptations=("M3-DuplexBench", "Full-Duplex-Bench v3", "HumDial"),
            mechanisms=(InteractionMechanism.FLOOR_FAIRNESS, InteractionMechanism.MULTI_THREAD),
            identity_factors=(IdentityDecisionFactor.SPEAKER, IdentityDecisionFactor.THREAD_OWNERSHIP, IdentityDecisionFactor.RECIPIENT_COMPREHENSION),
            summary="Six participants interleave astronomy, geography, math, and biology questions, including a child who needs a simple restatement.",
            primary_role="astronomy_learner",
            primary_goal="Ask which planet is largest.",
            primary_line="Which planet is the largest in our solar system?",
            live_role="geography_learner",
            live_goal="Ask for Australia's capital as a separate thread.",
            live_line="Separate question: what's the capital of Australia?",
            extras=(
                ExtraSpeaker("math_learner", "Ask for 15 percent of 80.", "What is fifteen percent of eighty?", thread_id="math"),
                ExtraSpeaker("biology_learner", "Ask what organelles make ATP.", "Which organelle produces most ATP?", thread_id="biology"),
                ExtraSpeaker("child", "Ask for the astronomy answer in simple words.", "Nova, tell me the biggest planet in simple words.", thread_id="astronomy"),
                ExtraSpeaker("moderator", "Request a speaker-addressed synthesis.", "Answer each person by name and don't mix the four questions.", thread_id="summary"),
            ),
            outcome="Preserve four semantic threads, give every speaker a floor, and tailor the repeated astronomy answer to the child.",
            spoken_result="Answer Avery Jupiter, Blake Canberra, Cameron 12, Devon mitochondria, and explain Jupiter simply to Emery before summarizing for Finley.",
            comprehension=("child_needs_plain_language",),
        ),
    ]
