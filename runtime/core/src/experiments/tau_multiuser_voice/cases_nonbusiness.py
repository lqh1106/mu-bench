"""Twelve non-business multi-user full-duplex cases for v0.2."""

from __future__ import annotations

from typing import Any

from experiments.tau_multiuser_voice.cases import (
    _event,
    _goal,
    _grant,
    _relation,
    _timing,
    _tool,
    _user,
)
from experiments.tau_multiuser_voice.models import (
    CapabilityAssertion,
    CapabilityCategory,
    EnvironmentBinding,
    EnvironmentStateAssertion,
    GoalDisposition,
    GoalRelation,
    GoalRelationType,
    MultiUserVoiceCase,
    PresenceEffect,
    ScriptedEvent,
    TimingRequirement,
    TriggerKind,
    UserAction,
    UserProfile,
)


def _environment(
    domain: str,
    fixture_id: str,
    principals: list[str],
    resources: list[str],
    tool_names: list[str],
    initial_state: dict[str, Any],
    *rules: str,
) -> EnvironmentBinding:
    return EnvironmentBinding(
        domain=domain,
        fixture_id=fixture_id,
        principals=principals,
        resources=resources,
        tool_names=tool_names,
        initial_state=initial_state,
        policy_extension=[
            "Track each speaker, addressee, speech act, and task thread separately.",
            "Do not convert quoted speech, backchannels, or human-directed side-talk into assistant commands.",
            "When overlap creates uncertainty, ask the affected speaker for only the missing information.",
            *rules,
        ],
    )


def _capability(
    assertion_id: str,
    category: CapabilityCategory,
    description: str,
    expected: str,
    *event_ids: str,
    judge: str = "semantic",
) -> CapabilityAssertion:
    return CapabilityAssertion(
        assertion_id=assertion_id,
        category=category,
        description=description,
        expected_behavior=expected,
        source_event_ids=list(event_ids),
        judge=judge,
    )


def _state(
    assertion_id: str,
    path: str,
    expected: Any,
    operator: str = "eq",
) -> EnvironmentStateAssertion:
    return EnvironmentStateAssertion(
        assertion_id=assertion_id,
        path=path,
        operator=operator,
        expected=expected,
    )


def _case(
    *,
    case_id: str,
    title: str,
    summary: str,
    tags: list[str],
    environment: EnvironmentBinding,
    users: list[UserProfile],
    events: list[ScriptedEvent],
    assertions: list[CapabilityAssertion],
    goals: list[GoalRelation] | None = None,
    tools: list | None = None,
    timings: list[TimingRequirement] | None = None,
    states: list[EnvironmentStateAssertion] | None = None,
    success: list[str] | None = None,
) -> MultiUserVoiceCase:
    return MultiUserVoiceCase(
        case_id=case_id,
        title=title,
        summary=summary,
        capability_tags=tags,
        identity_setup=[
            "Every enrolled participant speaks a self-introduction in a distinct benchmark voice unless marked unknown or spoof candidate."
        ],
        permission_setup=[
            "This case primarily evaluates multi-party interaction, not access control; declared tool actions are permitted."
        ],
        environment=environment,
        users=users,
        goal_relations=goals or [],
        events=events,
        expected_tool_actions=tools or [],
        timing_requirements=timings or [],
        capability_assertions=assertions,
        state_assertions=states or [],
        final_success_conditions=success
        or [assertion.expected_behavior for assertion in assertions],
    )


def _shared_home_cases() -> list[MultiUserVoiceCase]:
    cases = []
    cases.append(
        _case(
            case_id="shared_home_mu_001_addressee_switch",
            title="Human-directed side question then assistant-directed timer query",
            summary=(
                "A parent sets a pasta timer. A child first asks the parent for juice, "
                "then directly asks the assistant about the timer."
            ),
            tags=["addressee", "side_talk", "thread_routing", "shared_home"],
            environment=_environment(
                "shared_home",
                "home_kitchen_001",
                ["parent", "child"],
                ["timer:pasta", "media"],
                ["set_timer", "get_timer"],
                {"timers": {}, "media": {"content": None, "playing": False}},
                "The child's juice question is addressed to the parent and is not an assistant request.",
            ),
            users=[
                _user(
                    "parent",
                    "Morgan",
                    "parent",
                    goals=[
                        _goal(
                            "g_pasta_timer",
                            "Create a twelve-minute pasta timer.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[_grant("timer_write", ["set_timer"], ["timer:pasta"])],
                    private_knowledge=["The pasta needs twelve minutes."],
                ),
                _user(
                    "child",
                    "Riley",
                    "child",
                    goals=[
                        _goal(
                            "g_juice_side_talk",
                            "Ask Morgan for juice.",
                            GoalDisposition.IGNORE,
                        ),
                        _goal(
                            "g_timer_query",
                            "Ask the assistant about the pasta timer.",
                            GoalDisposition.FULFILL,
                        ),
                    ],
                    permissions=[_grant("timer_read", ["get_timer"], ["timer:pasta"])],
                    private_knowledge=[],
                    interrupt_tendency=0.7,
                ),
            ],
            goals=[
                _relation("g_pasta_timer", "g_timer_query", GoalRelationType.ALIGNED),
                _relation(
                    "g_pasta_timer",
                    "g_juice_side_talk",
                    GoalRelationType.INDEPENDENT,
                ),
            ],
            events=[
                _event(
                    "e_parent_timer",
                    "parent",
                    UserAction.SPEAK,
                    "Assistant, set a twelve-minute timer for the pasta.",
                    "create timer",
                    thread_id="cooking",
                    speech_act="request",
                ),
                _event(
                    "e_child_side_talk",
                    "child",
                    UserAction.SIDE_TALK,
                    "Mom, can I have some juice?",
                    "human-directed question",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=500,
                    fallback_at_ms=5500,
                    addressee="parent",
                    overlap=True,
                    thread_id="family_side_talk",
                    speech_act="question",
                ),
                _event(
                    "e_child_timer_query",
                    "child",
                    UserAction.SPEAK,
                    "Assistant, how long is left on the pasta timer?",
                    "direct timer query",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_child_side_talk",
                    after_ms=700,
                    thread_id="cooking",
                    speech_act="request",
                ),
            ],
            assertions=[
                _capability(
                    "a_ignore_juice_side_talk",
                    CapabilityCategory.ADDRESSEE,
                    "Do not answer the child while the child addresses the parent.",
                    "The assistant does not respond to the juice question.",
                    "e_child_side_talk",
                ),
                _capability(
                    "a_switch_to_direct_child",
                    CapabilityCategory.RESPONSE_RENDERING,
                    "Respond when the same child explicitly addresses the assistant.",
                    "The timer answer is directed to Riley without answering the old juice question.",
                    "e_child_timer_query",
                ),
                _capability(
                    "a_preserve_cooking_thread",
                    CapabilityCategory.MULTI_THREAD,
                    "Keep the cooking task active across unrelated side-talk.",
                    "The timer query resolves against the existing pasta timer.",
                    "e_parent_timer",
                    "e_child_timer_query",
                ),
            ],
            tools=[
                _tool(
                    "set_timer",
                    "timer:pasta",
                    "parent",
                    allowed=True,
                    must_execute=True,
                    arguments={"label": "pasta", "duration_seconds": 720},
                ),
                _tool(
                    "get_timer",
                    "timer:pasta",
                    "child",
                    allowed=True,
                    must_execute=True,
                    arguments={"label": "pasta"},
                ),
            ],
            timings=[_timing("false_interruptions", "e_child_side_talk", "eq", 0)],
            states=[
                _state("s_pasta_duration", "timers.pasta.duration_seconds", 720),
                _state("s_pasta_active", "timers.pasta.active", True),
            ],
        )
    )

    cases.append(
        _case(
            case_id="shared_home_mu_002_parallel_timers_correction",
            title="Two household timers and a mid-answer correction",
            summary=(
                "Alex starts a pasta timer, Jordan interrupts with an independent "
                "laundry timer, and Alex later corrects only the pasta duration."
            ),
            tags=["multi_thread", "correction", "barge_in", "shared_home"],
            environment=_environment(
                "shared_home",
                "home_timers_002",
                ["alex", "jordan"],
                ["timer:pasta", "timer:laundry"],
                ["set_timer"],
                {"timers": {}, "media": {"content": None, "playing": False}},
                "A correction to one named timer must not alter the other timer.",
            ),
            users=[
                _user(
                    "alex",
                    "Alex",
                    "household_member",
                    goals=[
                        _goal(
                            "g_pasta_12",
                            "Correct the pasta timer from ten to twelve minutes.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[_grant("alex_timer", ["set_timer"], ["timer:pasta"])],
                    private_knowledge=["Pasta needs twelve minutes, not ten."],
                    interrupt_tendency=0.7,
                ),
                _user(
                    "jordan",
                    "Jordan",
                    "household_member",
                    goals=[
                        _goal(
                            "g_laundry_30",
                            "Create a thirty-minute laundry timer.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[
                        _grant("jordan_timer", ["set_timer"], ["timer:laundry"])
                    ],
                    private_knowledge=[],
                    interrupt_tendency=0.9,
                ),
            ],
            goals=[
                _relation("g_pasta_12", "g_laundry_30", GoalRelationType.INDEPENDENT)
            ],
            events=[
                _event(
                    "e_alex_pasta_10",
                    "alex",
                    UserAction.SPEAK,
                    "Set a ten-minute pasta timer.",
                    "initial pasta timer",
                    thread_id="timer_pasta",
                    speech_act="request",
                ),
                _event(
                    "e_jordan_laundry",
                    "jordan",
                    UserAction.INTERRUPT,
                    "Also set a separate thirty-minute laundry timer.",
                    "parallel timer task",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=500,
                    fallback_at_ms=5000,
                    overlap=True,
                    urgency=4,
                    thread_id="timer_laundry",
                    speech_act="request",
                ),
                _event(
                    "e_alex_correct_12",
                    "alex",
                    UserAction.CORRECT,
                    "Correction: make only the pasta timer twelve minutes.",
                    "replace pasta duration",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_jordan_laundry",
                    after_ms=700,
                    urgency=5,
                    thread_id="timer_pasta",
                    speech_act="correction",
                ),
            ],
            assertions=[
                _capability(
                    "a_keep_timer_threads_separate",
                    CapabilityCategory.MULTI_THREAD,
                    "Maintain independent pasta and laundry timer threads.",
                    "Both timers exist and the correction touches only pasta.",
                    "e_alex_pasta_10",
                    "e_jordan_laundry",
                    "e_alex_correct_12",
                ),
                _capability(
                    "a_apply_owner_correction",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Attribute the correction to Alex and replace the obsolete duration.",
                    "The final pasta duration is twelve minutes, not ten.",
                    "e_alex_correct_12",
                ),
                _capability(
                    "a_resume_after_timer_interrupt",
                    CapabilityCategory.MULTI_THREAD,
                    "Resume the pasta thread after processing Jordan's timer.",
                    "The assistant confirms the corrected pasta timer without dropping laundry.",
                    "e_jordan_laundry",
                    "e_alex_correct_12",
                ),
            ],
            tools=[
                _tool(
                    "set_timer",
                    "timer:pasta",
                    "alex",
                    allowed=True,
                    must_execute=True,
                    arguments={"label": "pasta", "duration_seconds": 720},
                ),
                _tool(
                    "set_timer",
                    "timer:laundry",
                    "jordan",
                    allowed=True,
                    must_execute=True,
                    arguments={"label": "laundry", "duration_seconds": 1800},
                ),
            ],
            timings=[
                _timing("thread_resume_latency_ms", "e_alex_correct_12", "le", 1800)
            ],
            states=[
                _state("s_pasta_12", "timers.pasta.duration_seconds", 720),
                _state("s_laundry_30", "timers.laundry.duration_seconds", 1800),
            ],
        )
    )

    cases.append(
        _case(
            case_id="shared_home_mu_003_quiet_speaker_fairness",
            title="Quiet child request survives a dominant adult",
            summary=(
                "A dominant adult requests music and repeatedly talks over a quiet "
                "child who asks for a homework timer. Both independent goals should survive."
            ),
            tags=["fairness", "speaker_starvation", "overlap", "shared_home"],
            environment=_environment(
                "shared_home",
                "home_fairness_003",
                ["adult", "child"],
                ["media", "timer:homework"],
                ["set_media", "set_timer"],
                {"timers": {}, "media": {"content": None, "playing": False}},
                "Do not prioritize a louder or more frequent speaker solely because they dominate the floor.",
            ),
            users=[
                _user(
                    "adult",
                    "Casey",
                    "adult",
                    goals=[
                        _goal(
                            "g_music",
                            "Play jazz.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("media", ["set_media"], ["media"])],
                    private_knowledge=[],
                    patience=0,
                    interrupt_tendency=1.0,
                    conflict_style="impatient",
                ),
                _user(
                    "child",
                    "Jamie",
                    "quiet_child",
                    goals=[
                        _goal(
                            "g_homework_timer",
                            "Create a twenty-minute homework timer.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant("homework_timer", ["set_timer"], ["timer:homework"])
                    ],
                    private_knowledge=[],
                    patience=4,
                    interrupt_tendency=0.05,
                    conflict_style="avoidant",
                ),
            ],
            goals=[
                _relation("g_music", "g_homework_timer", GoalRelationType.INDEPENDENT)
            ],
            events=[
                _event(
                    "e_adult_music",
                    "adult",
                    UserAction.SPEAK,
                    "Play some jazz in the living room.",
                    "media request",
                    overlap=True,
                    urgency=4,
                    thread_id="media",
                    speech_act="request",
                ),
                _event(
                    "e_child_timer",
                    "child",
                    UserAction.SPEAK,
                    "Assistant, could you also set a twenty-minute homework timer?",
                    "quiet timer request",
                    overlap=True,
                    urgency=3,
                    thread_id="homework",
                    speech_act="request",
                ),
                _event(
                    "e_adult_repeat",
                    "adult",
                    UserAction.INTERRUPT,
                    "Make sure it is jazz, not pop.",
                    "dominant follow-up",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=500,
                    fallback_at_ms=5500,
                    overlap=True,
                    urgency=5,
                    thread_id="media",
                    speech_act="correction",
                ),
                _event(
                    "e_child_retry",
                    "child",
                    UserAction.SPEAK,
                    "Please do not forget my homework timer.",
                    "quiet speaker retry",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_adult_repeat",
                    after_ms=600,
                    thread_id="homework",
                    speech_act="repair",
                ),
            ],
            assertions=[
                _capability(
                    "a_no_quiet_speaker_starvation",
                    CapabilityCategory.FAIRNESS,
                    "Do not lose Jamie's request because Casey is louder and interrupts.",
                    "Jamie's timer is acknowledged and completed within the fairness window.",
                    "e_child_timer",
                    "e_adult_repeat",
                    "e_child_retry",
                ),
                _capability(
                    "a_keep_independent_goals",
                    CapabilityCategory.MULTI_THREAD,
                    "Complete media and timer tasks independently.",
                    "Jazz plays and the homework timer exists.",
                    "e_adult_music",
                    "e_child_timer",
                ),
            ],
            tools=[
                _tool(
                    "set_media",
                    "media",
                    "adult",
                    allowed=True,
                    must_execute=True,
                    arguments={"content": "jazz"},
                ),
                _tool(
                    "set_timer",
                    "timer:homework",
                    "child",
                    allowed=True,
                    must_execute=True,
                    arguments={"label": "homework", "duration_seconds": 1200},
                ),
            ],
            timings=[
                _timing("speaker_wait_ms", "e_child_timer", "le", 3500),
                _timing("speaker_starvation_count", "e_child_timer", "eq", 0),
            ],
            states=[
                _state("s_media_jazz", "media.content", "jazz"),
                _state(
                    "s_homework_timer",
                    "timers.homework.duration_seconds",
                    1200,
                ),
            ],
        )
    )
    return cases


def _vehicle_cases() -> list[MultiUserVoiceCase]:
    cases = []
    base_state = {
        "navigation": {
            "destination": None,
            "eta_minutes": None,
            "changed_by": None,
        },
        "media": {"content": None, "playing": False},
        "route_catalog": {"airport": 42, "city center": 18, "school": 12},
    }
    cases.append(
        _case(
            case_id="vehicle_mu_001_route_eta_and_side_media",
            title="Navigation, ETA interruption, and human-directed media side-talk",
            summary=(
                "The driver starts airport navigation, the passenger asks the assistant "
                "for ETA, and a child asks the passenger—not the assistant—to play cartoons."
            ),
            tags=["vehicle", "multi_thread", "addressee", "task_resume"],
            environment=_environment(
                "vehicle",
                "vehicle_trip_001",
                ["driver", "passenger", "child"],
                ["navigation", "media"],
                ["commit_route", "get_eta", "set_media"],
                base_state,
                "The child media request is human-directed and must not alter media state.",
            ),
            users=[
                _user(
                    "driver",
                    "Dana",
                    "driver",
                    goals=[
                        _goal(
                            "g_airport",
                            "Navigate to the airport.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[_grant("route", ["commit_route"], ["navigation"])],
                    private_knowledge=[],
                ),
                _user(
                    "passenger",
                    "Pat",
                    "passenger",
                    goals=[
                        _goal(
                            "g_eta",
                            "Get the active route ETA.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("eta", ["get_eta"], ["navigation"])],
                    private_knowledge=[],
                    interrupt_tendency=0.8,
                ),
                _user(
                    "child",
                    "Sky",
                    "child_passenger",
                    goals=[
                        _goal(
                            "g_cartoon_side",
                            "Ask Pat to play cartoons.",
                            GoalDisposition.IGNORE,
                        )
                    ],
                    permissions=[_grant("media", ["set_media"], ["media"])],
                    private_knowledge=[],
                ),
            ],
            events=[
                _event(
                    "e_driver_airport",
                    "driver",
                    UserAction.SPEAK,
                    "Assistant, start navigation to the airport.",
                    "route request",
                    thread_id="navigation",
                    speech_act="request",
                ),
                _event(
                    "e_passenger_eta",
                    "passenger",
                    UserAction.INTERRUPT,
                    "Assistant, what is our ETA?",
                    "temporary ETA subtask",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=600,
                    fallback_at_ms=5500,
                    overlap=True,
                    thread_id="eta_query",
                    speech_act="request",
                ),
                _event(
                    "e_child_cartoon_side",
                    "child",
                    UserAction.SIDE_TALK,
                    "Pat, can you put cartoons on?",
                    "human-directed media request",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_passenger_eta",
                    after_ms=500,
                    addressee="passenger",
                    thread_id="family_side_talk",
                    speech_act="question",
                ),
                _event(
                    "e_driver_resume",
                    "driver",
                    UserAction.SPEAK,
                    "Continue the airport directions.",
                    "resume navigation",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_child_cartoon_side",
                    after_ms=500,
                    thread_id="navigation",
                    speech_act="request",
                ),
            ],
            assertions=[
                _capability(
                    "a_route_survives_eta",
                    CapabilityCategory.MULTI_THREAD,
                    "Answer the ETA without abandoning active navigation.",
                    "The airport route remains active after the ETA subtask.",
                    "e_driver_airport",
                    "e_passenger_eta",
                    "e_driver_resume",
                ),
                _capability(
                    "a_ignore_child_to_passenger",
                    CapabilityCategory.ADDRESSEE,
                    "Do not act on the child's question to Pat.",
                    "Media remains unchanged and no assistant response targets Sky.",
                    "e_child_cartoon_side",
                ),
            ],
            tools=[
                _tool(
                    "commit_route",
                    "navigation",
                    "driver",
                    allowed=True,
                    must_execute=True,
                    arguments={"destination": "airport"},
                ),
                _tool(
                    "get_eta",
                    "navigation",
                    "passenger",
                    allowed=True,
                    must_execute=True,
                ),
            ],
            timings=[
                _timing("yield_latency_ms", "e_passenger_eta", "le", 800),
                _timing("thread_resume_latency_ms", "e_driver_resume", "le", 1600),
                _timing("false_interruptions", "e_child_cartoon_side", "eq", 0),
            ],
            states=[
                _state("s_airport_active", "navigation.destination", "airport"),
                _state("s_media_unchanged", "media.content", None),
            ],
        )
    )

    cases.append(
        _case(
            case_id="vehicle_mu_002_correction_vs_backchannel",
            title="Driver destination correction overlaps passenger backchannel",
            summary=(
                "The driver initially says city center and corrects to airport while "
                "a passenger says mm-hmm. Only the correction changes navigation."
            ),
            tags=["vehicle", "correction", "backchannel", "overlap_attribution"],
            environment=_environment(
                "vehicle",
                "vehicle_correction_002",
                ["driver", "passenger"],
                ["navigation"],
                ["commit_route"],
                base_state,
                "A backchannel is not a competing destination request.",
            ),
            users=[
                _user(
                    "driver",
                    "Dana",
                    "driver",
                    goals=[
                        _goal(
                            "g_correct_airport",
                            "Correct destination from city center to airport.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[_grant("route", ["commit_route"], ["navigation"])],
                    private_knowledge=[],
                    interrupt_tendency=0.9,
                ),
                _user(
                    "passenger",
                    "Pat",
                    "passenger",
                    goals=[
                        _goal(
                            "g_backchannel",
                            "Signal attention without changing the route.",
                            GoalDisposition.IGNORE,
                        )
                    ],
                    permissions=[],
                    private_knowledge=[],
                ),
            ],
            goals=[
                _relation(
                    "g_correct_airport",
                    "g_backchannel",
                    GoalRelationType.INDEPENDENT,
                )
            ],
            events=[
                _event(
                    "e_driver_city",
                    "driver",
                    UserAction.SPEAK,
                    "Navigate to the city center.",
                    "obsolete destination",
                    thread_id="navigation",
                    speech_act="request",
                ),
                _event(
                    "e_driver_airport_correction",
                    "driver",
                    UserAction.CORRECT,
                    "Correction: airport, not city center.",
                    "destination correction",
                    trigger=TriggerKind.ASSISTANT_PREFIX,
                    pattern="city center",
                    fallback_at_ms=5500,
                    overlap=True,
                    urgency=5,
                    thread_id="navigation",
                    speech_act="correction",
                ),
                _event(
                    "e_passenger_mmhm",
                    "passenger",
                    UserAction.BACKCHANNEL,
                    "Mm-hmm.",
                    "continuer",
                    trigger=TriggerKind.ASSISTANT_PREFIX,
                    pattern="city center",
                    fallback_at_ms=5500,
                    overlap=True,
                    urgency=4,
                    thread_id="navigation",
                    speech_act="backchannel",
                ),
            ],
            assertions=[
                _capability(
                    "a_correction_beats_obsolete_destination",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Apply the driver's correction to the driver's earlier request.",
                    "The active route is airport and city center is obsolete.",
                    "e_driver_city",
                    "e_driver_airport_correction",
                ),
                _capability(
                    "a_backchannel_not_route_command",
                    CapabilityCategory.FLOOR_CONTROL,
                    "Treat the passenger's mm-hmm as a continuer.",
                    "The backchannel neither seizes the floor nor changes destination.",
                    "e_passenger_mmhm",
                ),
            ],
            tools=[
                _tool(
                    "commit_route",
                    "navigation",
                    "driver",
                    allowed=True,
                    must_execute=True,
                    arguments={"destination": "airport"},
                )
            ],
            timings=[
                _timing("yield_latency_ms", "e_driver_airport_correction", "le", 700)
            ],
            states=[_state("s_corrected_route", "navigation.destination", "airport")],
        )
    )

    cases.append(
        _case(
            case_id="vehicle_mu_003_late_joiner_common_ground",
            title="Late passenger asks for selective route recap",
            summary=(
                "A passenger enters after navigation has started and asks for ETA. "
                "The assistant should recognize the new participant and give only needed context."
            ),
            tags=["vehicle", "participant_entry", "common_ground", "rendering"],
            environment=_environment(
                "vehicle",
                "vehicle_late_joiner_003",
                ["driver", "late_passenger"],
                ["navigation"],
                ["commit_route", "get_eta"],
                base_state,
                "Do not assume a newly present passenger heard earlier route details.",
            ),
            users=[
                _user(
                    "driver",
                    "Dana",
                    "driver",
                    goals=[
                        _goal(
                            "g_school_route",
                            "Start navigation to school.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("route", ["commit_route"], ["navigation"])],
                    private_knowledge=[],
                ),
                _user(
                    "late_passenger",
                    "Lee",
                    "late_passenger",
                    goals=[
                        _goal(
                            "g_late_eta",
                            "Learn destination and ETA after joining.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("eta", ["get_eta"], ["navigation"])],
                    private_knowledge=[],
                    present_initially=False,
                ),
            ],
            events=[
                _event(
                    "e_driver_school",
                    "driver",
                    UserAction.SPEAK,
                    "Start navigation to school.",
                    "start route before passenger joins",
                    thread_id="navigation",
                    speech_act="request",
                ),
                _event(
                    "e_passenger_join",
                    "late_passenger",
                    UserAction.JOIN,
                    "Hi, I just got in. Where are we going, and what is the ETA?",
                    "participant entry and recap request",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_driver_school",
                    after_ms=1000,
                    thread_id="navigation",
                    speech_act="join_question",
                    presence_effects=[
                        PresenceEffect(target_user_id="late_passenger", present=True)
                    ],
                ),
            ],
            assertions=[
                _capability(
                    "a_detect_late_participant",
                    CapabilityCategory.PARTICIPANT_SCENE,
                    "Treat Lee as newly present rather than a continuation of Dana.",
                    "Lee's utterance is attributed to a new participant.",
                    "e_passenger_join",
                ),
                _capability(
                    "a_selective_recap",
                    CapabilityCategory.RESPONSE_RENDERING,
                    "Give the late passenger enough route context without replaying the whole conversation.",
                    "The response names school and the ETA once, directed to Lee.",
                    "e_driver_school",
                    "e_passenger_join",
                ),
                _capability(
                    "a_late_joiner_common_ground",
                    CapabilityCategory.USER_MEMORY,
                    "Track what Lee could have heard instead of treating group history as shared memory.",
                    "The response supplies the missing destination and ETA without claiming Lee heard the earlier route request.",
                    "e_driver_school",
                    "e_passenger_join",
                ),
            ],
            tools=[
                _tool(
                    "commit_route",
                    "navigation",
                    "driver",
                    allowed=True,
                    must_execute=True,
                    arguments={"destination": "school"},
                ),
                _tool(
                    "get_eta",
                    "navigation",
                    "late_passenger",
                    allowed=True,
                    must_execute=True,
                ),
            ],
            states=[_state("s_school_route", "navigation.destination", "school")],
        )
    )
    return cases


def _meeting_cases() -> list[MultiUserVoiceCase]:
    cases = []
    initial = {
        "proposals": {},
        "votes": {},
        "assignments": {},
        "unresolved_issues": [],
        "drafts": [],
    }
    cases.append(
        _case(
            case_id="meeting_mu_001_quote_not_command",
            title="Quoted message proposal is not a drafting command",
            summary=(
                "Alice quotes Bob's earlier suggestion to email a client, explicitly "
                "without authorizing a draft; Bob then rejects the quoted suggestion."
            ),
            tags=["meeting", "quotation", "provenance", "false_action"],
            environment=_environment(
                "meeting",
                "meeting_quote_001",
                ["alice", "bob"],
                ["drafts"],
                ["draft_message"],
                initial,
                "Reported or quoted speech is evidence about a stance, not an action request.",
            ),
            users=[
                _user(
                    "alice",
                    "Alice",
                    "team_member",
                    goals=[
                        _goal(
                            "g_quote",
                            "Report Bob's old suggestion without drafting anything.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("draft", ["draft_message"], ["drafts"])],
                    private_knowledge=[],
                ),
                _user(
                    "bob",
                    "Bob",
                    "team_member",
                    goals=[
                        _goal(
                            "g_reject_draft",
                            "Clarify that no client message should be drafted.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[_grant("draft", ["draft_message"], ["drafts"])],
                    private_knowledge=[],
                    interrupt_tendency=0.8,
                ),
            ],
            events=[
                _event(
                    "e_alice_quote",
                    "alice",
                    UserAction.SPEAK,
                    "Bob said yesterday, quote, 'email the client tonight,' but I am only reporting that; do not draft it.",
                    "quotation with explicit non-command",
                    thread_id="client_comms",
                    speech_act="quote",
                ),
                _event(
                    "e_bob_reject",
                    "bob",
                    UserAction.CORRECT,
                    "Right, and I no longer support sending that message.",
                    "current stance correction",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_alice_quote",
                    after_ms=500,
                    thread_id="client_comms",
                    speech_act="stance",
                ),
            ],
            assertions=[
                _capability(
                    "a_quote_not_instruction",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Separate quoted content from Alice's current instruction.",
                    "No draft is created from the quoted sentence.",
                    "e_alice_quote",
                ),
                _capability(
                    "a_bob_current_stance",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Track Bob's current opposition separately from his reported old view.",
                    "A summary says Bob once suggested email but now opposes it.",
                    "e_alice_quote",
                    "e_bob_reject",
                ),
            ],
            states=[_state("s_no_draft", "drafts", [])],
        )
    )

    cases.append(
        _case(
            case_id="meeting_mu_002_assignment_due_date_correction",
            title="Assignee corrects due date during task assignment",
            summary=(
                "A manager assigns a report to Maya for Friday. Maya interrupts to "
                "correct the agreed date to Monday, and the manager confirms."
            ),
            tags=["meeting", "assignment", "correction", "confirmation"],
            environment=_environment(
                "meeting",
                "meeting_assignment_002",
                ["manager", "maya"],
                ["assignment:report"],
                ["assign_task"],
                initial,
                "The final assignment should reflect the explicitly confirmed correction.",
            ),
            users=[
                _user(
                    "manager",
                    "Nora",
                    "manager",
                    goals=[
                        _goal(
                            "g_assign_report",
                            "Assign the report to Maya with the agreed due date.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant("assign", ["assign_task"], ["assignment:report"])
                    ],
                    private_knowledge=[],
                ),
                _user(
                    "maya",
                    "Maya",
                    "assignee",
                    goals=[
                        _goal(
                            "g_correct_monday",
                            "Correct the due date from Friday to Monday.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant("assign", ["assign_task"], ["assignment:report"])
                    ],
                    private_knowledge=["The team agreed on Monday."],
                    interrupt_tendency=0.9,
                ),
            ],
            goals=[
                _relation(
                    "g_assign_report",
                    "g_correct_monday",
                    GoalRelationType.PARTIALLY_ALIGNED,
                )
            ],
            events=[
                _event(
                    "e_manager_friday",
                    "manager",
                    UserAction.SPEAK,
                    "Assign the report to Maya, due Friday.",
                    "obsolete assignment details",
                    thread_id="report_assignment",
                    speech_act="assignment",
                ),
                _event(
                    "e_maya_monday",
                    "maya",
                    UserAction.INTERRUPT,
                    "Correction: we agreed on Monday, not Friday.",
                    "assignee due-date correction",
                    trigger=TriggerKind.ASSISTANT_PREFIX,
                    pattern="Friday",
                    fallback_at_ms=5000,
                    overlap=True,
                    urgency=5,
                    thread_id="report_assignment",
                    speech_act="correction",
                ),
                _event(
                    "e_manager_confirm_monday",
                    "manager",
                    UserAction.CONFIRM,
                    "Yes, Maya is right; make it Monday.",
                    "confirm corrected assignment",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_maya_monday",
                    after_ms=500,
                    urgency=5,
                    thread_id="report_assignment",
                    speech_act="confirmation",
                ),
            ],
            assertions=[
                _capability(
                    "a_assignment_correction_provenance",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Preserve that Maya introduced the correction and Nora confirmed it.",
                    "The assignment history distinguishes correction source and confirmation source.",
                    "e_maya_monday",
                    "e_manager_confirm_monday",
                ),
                _capability(
                    "a_no_obsolete_friday",
                    CapabilityCategory.MULTI_THREAD,
                    "Do not resume the obsolete Friday assignment after interruption.",
                    "Only Monday appears in the final assignment.",
                    "e_manager_friday",
                    "e_maya_monday",
                ),
            ],
            tools=[
                _tool(
                    "assign_task",
                    "assignment:report",
                    "manager",
                    allowed=True,
                    must_execute=True,
                    arguments={
                        "task_id": "report",
                        "owner": "maya",
                        "due_date": "Monday",
                    },
                )
            ],
            timings=[_timing("yield_latency_ms", "e_maya_monday", "le", 700)],
            states=[
                _state("s_report_owner", "assignments.report.owner", "maya"),
                _state("s_report_due", "assignments.report.due_date", "Monday"),
            ],
        )
    )

    cases.append(
        _case(
            case_id="meeting_mu_003_minority_view_preserved",
            title="Majority support does not erase an explicit QA objection",
            summary=(
                "Alice proposes Tuesday launch, Bob supports, and Chen objects because "
                "QA is incomplete. The assistant must not announce false consensus."
            ),
            tags=["meeting", "group_decision", "minority_view", "fairness"],
            environment=_environment(
                "meeting",
                "meeting_consensus_003",
                ["alice", "bob", "chen"],
                ["proposal:launch_tuesday", "votes", "unresolved_issues"],
                ["add_proposal", "record_vote", "mark_unresolved_issue"],
                initial,
                "A two-to-one split with a blocking factual concern is unresolved, not consensus.",
            ),
            users=[
                _user(
                    "alice",
                    "Alice",
                    "proposer",
                    goals=[
                        _goal(
                            "g_launch_tuesday",
                            "Propose a Tuesday launch.",
                            GoalDisposition.CLARIFY,
                        )
                    ],
                    permissions=[
                        _grant(
                            "meeting_write",
                            ["add_proposal", "record_vote", "mark_unresolved_issue"],
                            ["*"],
                        )
                    ],
                    private_knowledge=[],
                ),
                _user(
                    "bob",
                    "Bob",
                    "supporter",
                    goals=[
                        _goal(
                            "g_support_tuesday",
                            "Support Tuesday launch.",
                            GoalDisposition.CLARIFY,
                        )
                    ],
                    permissions=[_grant("vote", ["record_vote"], ["votes"])],
                    private_knowledge=[],
                ),
                _user(
                    "chen",
                    "Chen",
                    "qa_lead",
                    goals=[
                        _goal(
                            "g_qa_objection",
                            "Record that QA is incomplete and oppose Tuesday.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant(
                            "qa_vote",
                            ["record_vote", "mark_unresolved_issue"],
                            ["*"],
                        )
                    ],
                    private_knowledge=["Critical regression tests are incomplete."],
                    patience=3,
                    interrupt_tendency=0.2,
                ),
            ],
            events=[
                _event(
                    "e_alice_proposal",
                    "alice",
                    UserAction.SPEAK,
                    "I propose launching on Tuesday.",
                    "launch proposal",
                    thread_id="launch_decision",
                    speech_act="proposal",
                ),
                _event(
                    "e_bob_support",
                    "bob",
                    UserAction.SPEAK,
                    "I support Tuesday.",
                    "supporting vote",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_alice_proposal",
                    after_ms=400,
                    thread_id="launch_decision",
                    speech_act="stance",
                ),
                _event(
                    "e_chen_oppose",
                    "chen",
                    UserAction.INTERRUPT,
                    "I oppose Tuesday; QA still has incomplete regression tests.",
                    "minority objection",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=600,
                    fallback_at_ms=6500,
                    overlap=True,
                    urgency=5,
                    thread_id="launch_decision",
                    speech_act="stance",
                ),
            ],
            assertions=[
                _capability(
                    "a_preserve_minority_objection",
                    CapabilityCategory.FAIRNESS,
                    "Preserve Chen's substantive objection despite majority support.",
                    "The final summary explicitly attributes the QA objection to Chen.",
                    "e_chen_oppose",
                ),
                _capability(
                    "a_no_false_consensus",
                    CapabilityCategory.GROUP_COLLABORATION,
                    "Do not announce consensus while a blocking objection remains.",
                    "Tuesday launch is marked unresolved, not approved.",
                    "e_alice_proposal",
                    "e_bob_support",
                    "e_chen_oppose",
                ),
                _capability(
                    "a_stance_by_speaker",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Keep each participant's stance separate.",
                    "Alice and Bob support; Chen opposes.",
                    "e_alice_proposal",
                    "e_bob_support",
                    "e_chen_oppose",
                ),
            ],
            tools=[
                _tool(
                    "add_proposal",
                    "proposal:launch_tuesday",
                    "alice",
                    allowed=True,
                    must_execute=True,
                    arguments={"proposal_id": "launch_tuesday"},
                ),
                _tool(
                    "record_vote",
                    "votes",
                    "bob",
                    allowed=True,
                    must_execute=True,
                    arguments={
                        "proposal_id": "launch_tuesday",
                        "stance": "support",
                    },
                ),
                _tool(
                    "record_vote",
                    "votes",
                    "chen",
                    allowed=True,
                    must_execute=True,
                    arguments={
                        "proposal_id": "launch_tuesday",
                        "stance": "oppose",
                    },
                ),
                _tool(
                    "mark_unresolved_issue",
                    "unresolved_issues",
                    "chen",
                    allowed=True,
                    must_execute=True,
                    arguments={"speakers": ["alice", "bob", "chen"]},
                ),
            ],
            states=[
                _state(
                    "s_bob_support",
                    "votes.launch_tuesday.bob",
                    "support",
                ),
                _state(
                    "s_chen_oppose",
                    "votes.launch_tuesday.chen",
                    "oppose",
                ),
                _state("s_unresolved_exists", "unresolved_issues", [], "ne"),
            ],
        )
    )
    return cases


def _education_cases() -> list[MultiUserVoiceCase]:
    cases = []
    initial = {"answers": {}, "whiteboard": [], "speaking_queue": []}
    cases.append(
        _case(
            case_id="education_mu_001_parallel_student_questions",
            title="Two student questions form separate tutoring threads",
            summary=(
                "Student A asks algebra while Student B interrupts with geometry. "
                "The teacher asks the assistant to answer A first and then resume B."
            ),
            tags=["education", "multi_thread", "turn_allocation", "task_resume"],
            environment=_environment(
                "education",
                "class_parallel_001",
                ["teacher", "student_a", "student_b"],
                ["speaking_queue"],
                ["assign_speaking_turn"],
                initial,
                "Answer both questions in the teacher-specified order without merging their content.",
            ),
            users=[
                _user(
                    "teacher",
                    "Ms. Lee",
                    "teacher",
                    goals=[
                        _goal(
                            "g_order_students",
                            "Have the assistant answer A, then B.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant("turns", ["assign_speaking_turn"], ["speaking_queue"])
                    ],
                    private_knowledge=[],
                ),
                _user(
                    "student_a",
                    "Ana",
                    "student",
                    goals=[
                        _goal(
                            "g_algebra",
                            "Ask how to solve 2x+3=11.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[],
                    private_knowledge=[],
                ),
                _user(
                    "student_b",
                    "Ben",
                    "student",
                    goals=[
                        _goal(
                            "g_geometry",
                            "Ask for the area formula of a triangle.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[],
                    private_knowledge=[],
                    interrupt_tendency=0.9,
                ),
            ],
            events=[
                _event(
                    "e_a_algebra",
                    "student_a",
                    UserAction.SPEAK,
                    "How do I solve two x plus three equals eleven?",
                    "algebra question",
                    thread_id="algebra",
                    speech_act="question",
                ),
                _event(
                    "e_b_geometry",
                    "student_b",
                    UserAction.INTERRUPT,
                    "And what is the area formula for a triangle?",
                    "geometry question",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=500,
                    fallback_at_ms=5000,
                    overlap=True,
                    thread_id="geometry",
                    speech_act="question",
                ),
                _event(
                    "e_teacher_order",
                    "teacher",
                    UserAction.SPEAK,
                    "Answer Ana first, then come back to Ben's geometry question.",
                    "explicit turn allocation",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_b_geometry",
                    after_ms=500,
                    urgency=5,
                    thread_id="class_management",
                    speech_act="turn_allocation",
                ),
            ],
            assertions=[
                _capability(
                    "a_separate_math_threads",
                    CapabilityCategory.MULTI_THREAD,
                    "Keep algebra and geometry as separate questions and owners.",
                    "Ana receives algebra help and Ben later receives geometry help.",
                    "e_a_algebra",
                    "e_b_geometry",
                ),
                _capability(
                    "a_follow_teacher_turn_order",
                    CapabilityCategory.FLOOR_CONTROL,
                    "Follow the explicit A-then-B speaking order.",
                    "The assistant answers Ana before resuming Ben.",
                    "e_teacher_order",
                ),
                _capability(
                    "a_resume_geometry_once",
                    CapabilityCategory.MULTI_THREAD,
                    "Resume Ben's suspended question after Ana is complete.",
                    "Ben's geometry question is answered exactly once without redoing algebra.",
                    "e_b_geometry",
                    "e_teacher_order",
                ),
            ],
            tools=[
                _tool(
                    "assign_speaking_turn",
                    "speaking_queue",
                    "teacher",
                    allowed=True,
                    must_execute=True,
                    arguments={"next_speaker": "student_a"},
                )
            ],
            timings=[
                _timing("thread_resume_latency_ms", "e_teacher_order", "le", 2500)
            ],
        )
    )

    cases.append(
        _case(
            case_id="education_mu_002_backchannel_and_substantive_correction",
            title="Student backchannel overlaps another student's correction",
            summary=(
                "During a photosynthesis explanation, Ana says uh-huh while Ben "
                "substantively corrects a claim. Only Ben's content needs discussion."
            ),
            tags=["education", "backchannel", "correction", "speech_act"],
            environment=_environment(
                "education",
                "class_backchannel_002",
                ["teacher", "student_a", "student_b"],
                ["whiteboard"],
                ["update_whiteboard"],
                initial,
                "Do not treat a continuer as an answer or correction.",
            ),
            users=[
                _user(
                    "teacher",
                    "Ms. Lee",
                    "teacher",
                    goals=[
                        _goal(
                            "g_explain_photo",
                            "Explain that plants release oxygen during photosynthesis.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[
                        _grant("whiteboard", ["update_whiteboard"], ["whiteboard"])
                    ],
                    private_knowledge=[],
                ),
                _user(
                    "student_a",
                    "Ana",
                    "student",
                    goals=[
                        _goal(
                            "g_continuer",
                            "Signal attention without taking a turn.",
                            GoalDisposition.IGNORE,
                        )
                    ],
                    permissions=[],
                    private_knowledge=[],
                ),
                _user(
                    "student_b",
                    "Ben",
                    "student",
                    goals=[
                        _goal(
                            "g_oxygen_correction",
                            "Correct a claim that plants release carbon dioxide.",
                            GoalDisposition.FULFILL,
                            priority=5,
                        )
                    ],
                    permissions=[
                        _grant("whiteboard", ["update_whiteboard"], ["whiteboard"])
                    ],
                    private_knowledge=["Plants release oxygen during photosynthesis."],
                ),
            ],
            events=[
                _event(
                    "e_teacher_wrong_claim",
                    "teacher",
                    UserAction.SPEAK,
                    "Plants release carbon dioxide during photosynthesis.",
                    "claim to be corrected",
                    overlap=True,
                    thread_id="photosynthesis",
                    speech_act="explanation",
                ),
                _event(
                    "e_a_uhhuh",
                    "student_a",
                    UserAction.BACKCHANNEL,
                    "Uh-huh.",
                    "continuer",
                    overlap=True,
                    urgency=3,
                    thread_id="photosynthesis",
                    speech_act="backchannel",
                ),
                _event(
                    "e_b_correction",
                    "student_b",
                    UserAction.CORRECT,
                    "Actually, plants release oxygen, not carbon dioxide.",
                    "substantive correction",
                    trigger=TriggerKind.ASSISTANT_SPEECH_START,
                    after_ms=500,
                    fallback_at_ms=5000,
                    overlap=True,
                    urgency=5,
                    thread_id="photosynthesis",
                    speech_act="correction",
                ),
            ],
            assertions=[
                _capability(
                    "a_backchannel_not_answer",
                    CapabilityCategory.FLOOR_CONTROL,
                    "Treat Ana's uh-huh as a continuer, not evidence for the claim.",
                    "Ana does not become the source of either scientific claim.",
                    "e_a_uhhuh",
                ),
                _capability(
                    "a_correction_attributed_to_ben",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Attribute the oxygen correction to Ben.",
                    "The corrected whiteboard claim names Ben as its source.",
                    "e_b_correction",
                ),
            ],
            tools=[
                _tool(
                    "update_whiteboard",
                    "whiteboard",
                    "student_b",
                    allowed=True,
                    must_execute=True,
                    arguments={"claim": "plants release oxygen", "label": "corrected"},
                )
            ],
            timings=[_timing("yield_latency_ms", "e_b_correction", "le", 700)],
            states=[
                _state(
                    "s_corrected_whiteboard",
                    "whiteboard",
                    {"claim": "plants release oxygen", "source": "student_b"},
                    "contains",
                )
            ],
        )
    )

    cases.append(
        _case(
            case_id="education_mu_003_targeted_overlap_repair",
            title="Targeted clarification after overlapping numerical answers",
            summary=(
                "Ana and Ben answer at the same time. Ana's twelve is clear, while "
                "Ben's answer is masked; the assistant should ask only Ben to repeat."
            ),
            tags=["education", "targeted_repair", "overlap", "uncertainty"],
            environment=_environment(
                "education",
                "class_repair_003",
                ["teacher", "student_a", "student_b"],
                ["answers:q1"],
                ["record_student_answer"],
                initial,
                "Preserve clearly heard slots and target clarification only to the uncertain speaker.",
            ),
            users=[
                _user(
                    "teacher",
                    "Ms. Lee",
                    "teacher",
                    goals=[
                        _goal(
                            "g_collect_answers",
                            "Collect separate answers from Ana and Ben.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[],
                    private_knowledge=[],
                ),
                _user(
                    "student_a",
                    "Ana",
                    "student",
                    goals=[
                        _goal(
                            "g_answer_12",
                            "Submit answer twelve.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[
                        _grant("answer", ["record_student_answer"], ["answers:q1"])
                    ],
                    private_knowledge=[],
                ),
                _user(
                    "student_b",
                    "Ben",
                    "student",
                    goals=[
                        _goal(
                            "g_answer_20",
                            "Submit answer twenty after targeted clarification.",
                            GoalDisposition.FULFILL,
                        )
                    ],
                    permissions=[
                        _grant("answer", ["record_student_answer"], ["answers:q1"])
                    ],
                    private_knowledge=[],
                ),
            ],
            events=[
                _event(
                    "e_teacher_prompt",
                    "teacher",
                    UserAction.SPEAK,
                    "Ana and Ben, give me your answers to question one.",
                    "request two answers",
                    thread_id="question_1",
                    speech_act="prompt",
                ),
                _event(
                    "e_a_twelve",
                    "student_a",
                    UserAction.SPEAK,
                    "Twelve.",
                    "clearly heard answer",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_teacher_prompt",
                    after_ms=400,
                    overlap=True,
                    urgency=5,
                    thread_id="question_1_ana",
                    speech_act="answer",
                ),
                _event(
                    "e_b_masked",
                    "student_b",
                    UserAction.SPEAK,
                    "Twenty.",
                    "answer masked by overlap in the SUT mix",
                    trigger=TriggerKind.EVENT_COMPLETED,
                    after_event="e_teacher_prompt",
                    after_ms=400,
                    overlap=True,
                    urgency=5,
                    thread_id="question_1_ben",
                    speech_act="answer_uncertain",
                    notes=[
                        "Audio difficulty metadata masks Ben's vowel while preserving speaker presence."
                    ],
                ),
                _event(
                    "e_b_repeat_twenty",
                    "student_b",
                    UserAction.SPEAK,
                    "I said twenty.",
                    "targeted clarification response",
                    trigger=TriggerKind.ASSISTANT_PREFIX,
                    pattern="Ben",
                    fallback_at_ms=7000,
                    thread_id="question_1_ben",
                    speech_act="repair_answer",
                ),
            ],
            assertions=[
                _capability(
                    "a_preserve_ana_clear_answer",
                    CapabilityCategory.TARGETED_REPAIR,
                    "Do not ask Ana to repeat the clearly heard answer.",
                    "Ana remains recorded as twelve without repetition.",
                    "e_a_twelve",
                ),
                _capability(
                    "a_target_ben_only",
                    CapabilityCategory.TARGETED_REPAIR,
                    "Ask Ben, and only Ben, to repeat the uncertain answer.",
                    "The clarification names Ben and requests only the missing number.",
                    "e_b_masked",
                    "e_b_repeat_twenty",
                ),
                _capability(
                    "a_answer_provenance",
                    CapabilityCategory.DISCOURSE_PROVENANCE,
                    "Store each numerical answer under the correct student.",
                    "Ana maps to twelve and Ben maps to twenty.",
                    "e_a_twelve",
                    "e_b_repeat_twenty",
                ),
            ],
            tools=[
                _tool(
                    "record_student_answer",
                    "answers:q1",
                    "student_a",
                    allowed=True,
                    must_execute=True,
                    arguments={"question_id": "q1", "answer": "twelve"},
                ),
                _tool(
                    "record_student_answer",
                    "answers:q1",
                    "student_b",
                    allowed=True,
                    must_execute=True,
                    arguments={"question_id": "q1", "answer": "twenty"},
                ),
            ],
            timings=[_timing("clarification_latency_ms", "e_b_masked", "le", 1800)],
            states=[
                _state(
                    "s_ana_answer",
                    "answers.q1",
                    {"answer": "twelve", "speaker": "student_a"},
                    "contains",
                ),
                _state(
                    "s_ben_answer",
                    "answers.q1",
                    {"answer": "twenty", "speaker": "student_b"},
                    "contains",
                ),
            ],
        )
    )
    return cases


def build_nonbusiness_cases() -> list[MultiUserVoiceCase]:
    """Build the twelve domain-general cases in stable pack order."""
    return (
        _shared_home_cases() + _vehicle_cases() + _meeting_cases() + _education_cases()
    )
