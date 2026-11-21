"""Data models for the multi-user full-duplex benchmark prototype."""

from __future__ import annotations

import re

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility.
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal stdlib StrEnum compatibility for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class UserAction(StrEnum):
    """Actions a simulated user can propose."""

    WAIT = "wait"
    SPEAK = "speak"
    INTERRUPT = "interrupt"
    BACKCHANNEL = "backchannel"
    CORRECT = "correct"
    CONFIRM = "confirm"
    DENY = "deny"
    SIDE_TALK = "side_talk"
    ASK_OTHER_USER = "ask_other_user"
    DELEGATE = "delegate"
    REVOKE = "revoke"
    JOIN = "join"
    LEAVE = "leave"


class TriggerKind(StrEnum):
    """Observable conditions that can release a scripted event."""

    CONVERSATION_START = "conversation_start"
    ASSISTANT_SPEECH_START = "assistant_speech_start"
    ASSISTANT_SPEECH_STOP = "assistant_speech_stop"
    ASSISTANT_PREFIX = "assistant_prefix"
    EVENT_COMPLETED = "event_completed"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ELAPSED = "elapsed"


class ObservationKind(StrEnum):
    """Events exposed to user sessions without future information."""

    CONVERSATION_START = "conversation_start"
    ASSISTANT_SPEECH_START = "assistant_speech_start"
    ASSISTANT_SPEECH_PROGRESS = "assistant_speech_progress"
    ASSISTANT_SPEECH_STOP = "assistant_speech_stop"
    USER_SPEECH = "user_speech"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CLOCK = "clock"


class GoalDisposition(StrEnum):
    """How the SUT should dispose of one user's goal."""

    FULFILL = "fulfill"
    DENY = "deny"
    CLARIFY = "clarify"
    IGNORE = "ignore"


class GoalRelationType(StrEnum):
    """Relationship between two users' goals."""

    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    CONFLICTING = "conflicting"
    INDEPENDENT = "independent"
    UNAUTHORIZED = "unauthorized"


class CapabilityCategory(StrEnum):
    """Orthogonal multi-user capabilities scored outside tool authorization."""

    PARTICIPANT_SCENE = "participant_scene"
    ADDRESSEE = "addressee"
    FLOOR_CONTROL = "floor_control"
    DISCOURSE_PROVENANCE = "discourse_provenance"
    MULTI_THREAD = "multi_thread"
    TARGETED_REPAIR = "targeted_repair"
    GROUP_COLLABORATION = "group_collaboration"
    FAIRNESS = "fairness"
    RESPONSE_RENDERING = "response_rendering"
    USER_MEMORY = "user_memory"
    SPEAKER_ANCHOR = "speaker_anchor"
    RETURN_CONTINUITY = "return_continuity"


class CaseDifficulty(StrEnum):
    """Case-level reasoning and interaction difficulty."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SpeechCondition(StrEnum):
    """τ-Voice-style speech condition preset."""

    CLEAN = "clean"
    REALISTIC = "realistic"


class ParticipantScale(StrEnum):
    """Factorial participant-count axis for v0.7 scenario variants."""

    SMALL = "small_2_3"
    LARGE = "large_4_7"


class ComprehensionLevel(StrEnum):
    """Whether the intended recipient readily understands a correct answer."""

    STRONG = "strong"
    WEAK = "weak"


class BenchmarkTaskType(StrEnum):
    """Primary task semantics independent of the acoustic interaction pattern."""

    TOOL_SINGLE_STEP = "tool_single_step"
    TOOL_MULTI_STEP = "tool_multi_step"
    KNOWLEDGE_QA = "knowledge_qa"
    COLLABORATIVE_REASONING = "collaborative_reasoning"
    MIXED_QA_TOOL = "mixed_qa_tool"


class InteractionMechanism(StrEnum):
    """Full-duplex behavior that materially changes the correct continuation."""

    BARGE_IN_CORRECTION = "barge_in_correction"
    BACKCHANNEL_SELECTIVITY = "backchannel_selectivity"
    SIDE_TALK_ROUTING = "side_talk_routing"
    TOPIC_SWITCH_RESUME = "topic_switch_resume"
    IMPATIENT_SKIP = "impatient_skip"
    PUSHBACK_REPAIR = "pushback_repair"
    LATE_JOIN_COMMON_GROUND = "late_join_common_ground"
    OVERLAP_ATTRIBUTION = "overlap_attribution"
    DISFLUENCY_REVISION = "disfluency_revision"
    TOOL_RESULT_INTERRUPTION = "tool_result_interruption"
    TARGETED_CLARIFICATION = "targeted_clarification"
    FLOOR_FAIRNESS = "floor_fairness"
    CANCEL_OR_REVOKE = "cancel_or_revoke"
    RELAY_OR_TRANSLATION = "relay_or_translation"
    AUDIENCE_REPAIR = "audience_repair"
    MULTI_THREAD = "multi_thread"


class IdentityDecisionFactor(StrEnum):
    """Why merging every microphone utterance into one user changes the answer."""

    SPEAKER = "speaker"
    ADDRESSEE = "addressee"
    AUTHORITY = "authority"
    OWNERSHIP = "ownership"
    EXPERTISE = "expertise"
    CONSENT = "consent"
    PRESENCE = "presence"
    COMMON_GROUND = "common_ground"
    INTERPRETER_SCOPE = "interpreter_scope"
    PRIVACY_SCOPE = "privacy_scope"
    THREAD_OWNERSHIP = "thread_ownership"
    RECIPIENT_COMPREHENSION = "recipient_comprehension"
    RECIPIENT = "recipient"
    RELAY_SOURCE = "relay_source"
    ANSWER_PROVENANCE = "answer_provenance"
    VOTE_PROVENANCE = "vote_provenance"


class CoverageSpec(BaseModel):
    """Machine-auditable coverage and anti-shortcut rationale for one case."""

    scenario_family: str = "legacy_unspecified"
    task_type: BenchmarkTaskType = BenchmarkTaskType.TOOL_SINGLE_STEP
    source_adaptations: list[str] = Field(default_factory=list)
    interaction_mechanisms: list[InteractionMechanism] = Field(default_factory=list)
    identity_decision_factors: list[IdentityDecisionFactor] = Field(
        default_factory=list
    )
    anti_shortcut_proof: list[str] = Field(default_factory=list)


class ScenarioVariantSpec(BaseModel):
    """Orthogonal variant coordinates for one base-scenario realization."""

    scenario_id: str
    participant_scale: ParticipantScale
    speech_condition: SpeechCondition
    comprehension_level: ComprehensionLevel
    variant_index: int = Field(ge=0, le=7)


class PersonaSpec(BaseModel):
    """Stable behavioral controls for one simulated user."""

    verbosity: Literal["minimal", "brief", "normal", "detailed"] = "normal"
    patience: int = Field(default=2, ge=0, le=4)
    interrupt_tendency: float = Field(default=0.2, ge=0.0, le=1.0)
    conflict_style: Literal[
        "avoidant", "cooperative", "assertive", "impatient", "opportunistic"
    ] = "cooperative"
    behavior_rules: list[str] = Field(default_factory=list)
    accent: str = "General American"
    lexical_style: list[str] = Field(default_factory=list)
    comprehension_profile: Literal[
        "typical",
        "literal",
        "needs_repetition",
        "needs_plain_language",
    ] = "typical"
    hearing_profile: Literal["typical", "mild_loss", "noisy_environment"] = "typical"


class VoiceIdentitySpec(BaseModel):
    """Voice identity configuration retained as benchmark oracle metadata."""

    voice_id: str
    enrollment_status: Literal["enrolled", "unknown", "spoof_candidate"] = "enrolled"
    locale: str = "en-US"
    gain_db: float = 0.0
    accent_label: str = "General American"
    speaking_style: str = "natural conversational speech"


class VoiceEnrollmentProtocol(BaseModel):
    """How participants introduce and bind their voice inside one session."""

    mode: Literal["in_session", "none"] = "in_session"
    transport: Literal["audio_self_identification"] = "audio_self_identification"
    self_identification_template: str = (
        "My name is {display_name}. My participant ID is {user_id}. "
        "My role is {role}. {permission_statement} "
        "Please associate this voice with my identity for this session."
    )
    self_identification_activation_template: str = (
        "{assistant_name}, everyone has now introduced themselves. "
        "Registration is complete. Formal work starts now. "
        "You may respond to requests from this point onward."
    )
    session_preamble_template: str = (
        "You are {assistant_name}, our assistant. We will now introduce the "
        "users one by one. This is the user introduction phase; please listen "
        "and do not respond until we say that formal work has started."
    )
    pre_roll_ms: int = Field(default=400, ge=0)
    inter_speaker_gap_ms: int = Field(default=600, ge=0)
    post_roll_ms: int = Field(default=1200, ge=0)
    silence_required: bool = True

    @property
    def uses_dynamic_control_items(self) -> bool:
        """The spoken-only protocol never injects model-visible text items."""
        return False

    @staticmethod
    def _speakable(value: str) -> str:
        if value == "*":
            return "all applicable resources"
        return value.replace("_", " ").replace("/", " slash ")

    def render_permission_statement(
        self,
        permissions: list[PermissionGrant],
        *,
        oracle_user_id: str,
        participant_aliases: dict[str, str],
    ) -> str:
        """Render initially active grants as speech rather than hidden text."""
        grants = [grant for grant in permissions if grant.active_initially]
        if not grants:
            # This is an ACL statement about restricted tool operations, not
            # a ban on ordinary questions, speaking turns or conversational
            # 'protected floor time'. Keep the same scope for every provider.
            return "I do not have permission to authorize restricted tool operations."
        statements: list[str] = []
        for grant in grants:
            actions = ", ".join(self._speakable(value) for value in grant.actions)
            resources = ", ".join(self._speakable(value) for value in grant.resources)
            sentence = f"I am allowed to authorize the following tools: {actions}, for {resources}"
            if grant.constraints:
                rendered_constraints: list[str] = []
                for constraint in grant.constraints:
                    rendered = constraint
                    for internal_id, public_alias in participant_aliases.items():
                        if internal_id == oracle_user_id:
                            rendered = re.sub(
                                rf"\bthe\s+{re.escape(internal_id)}\s+user's\b",
                                "my",
                                rendered,
                                flags=re.IGNORECASE,
                            )
                        rendered = re.sub(
                            rf"\b{re.escape(internal_id)}\b",
                            public_alias,
                            rendered,
                        )
                    rendered_constraints.append(
                        self._speakable(rendered).rstrip(" .;:")
                    )
                sentence += ", subject to these limits: " + "; ".join(
                    rendered_constraints
                )
            statements.append(sentence + ".")
        return " ".join(statements)

    def render_enrollment_utterance(
        self,
        *,
        user_id: str,
        display_name: str,
        role: str,
        permissions: list[PermissionGrant],
        oracle_user_id: str,
        participant_aliases: dict[str, str],
        assistant_name: str = "Nova",
        include_session_preamble: bool = False,
    ) -> str:
        """Render the only model-audible identity source for one participant."""
        identity = self.self_identification_template.format(
            user_id=user_id,
            display_name=display_name,
            role=self._speakable(role),
            assistant_name=assistant_name,
            permission_statement=self.render_permission_statement(
                permissions,
                oracle_user_id=oracle_user_id,
                participant_aliases=participant_aliases,
            ),
        )
        if not include_session_preamble:
            return identity
        preamble = self.session_preamble_template.format(
            assistant_name=assistant_name
        )
        return f"{preamble} {identity}"

    def render_activation_utterance(
        self,
        *,
        fallback_template: str,
        assistant_name: str,
        user_id: str,
        display_name: str,
        role: str,
    ) -> str:
        """Render the spoken phase transition used by every backend."""
        del fallback_template
        return self.self_identification_activation_template.format(
            assistant_name=assistant_name,
            user_id=user_id,
            display_name=display_name,
            role=role,
        )


class AssistantActivationProtocol(BaseModel):
    """Model-visible invocation between enrollment and formal task time zero."""

    enabled: bool = True
    assistant_name: str = "Nova"
    caller_user_id: str | None = None
    utterance_template: str = (
        "{assistant_name}, registration is complete. Formal work starts now; "
        "you may respond to requests."
    )
    pre_roll_ms: int = Field(default=400, ge=0)
    post_roll_ms: int = Field(default=800, ge=0)
    allow_brief_acknowledgement: bool = True


class IterativeReplayProtocol(BaseModel):
    """Required Probe/Freeze/Final execution contract for every case."""

    enabled: bool = True
    planner_scope: Literal["persistent_session_per_user"] = (
        "persistent_session_per_user"
    )
    event_alignment_ms: int = Field(default=80, gt=0)
    probe_completion_guard_ms: int = Field(default=800, ge=0)
    probe_max_seconds: int = Field(default=75, gt=0)
    probe_max_extensions: int = Field(default=2, ge=0, le=8)
    require_fresh_session_per_probe: bool = True
    require_fresh_final_session: bool = True
    score_final_only: bool = True
    tail_probe_enabled: bool = True
    tail_probe_max_rounds: int = Field(default=2, ge=1, le=8)


class BenchmarkProvenance(BaseModel):
    """Primary benchmark patterns adapted by one μ-Bench case."""

    source_benchmarks: list[str] = Field(default_factory=list)
    source_patterns: list[str] = Field(default_factory=list)
    adaptation_summary: str = ""


class DifficultyProfile(BaseModel):
    """Participant and speech-condition gradient for one case."""

    tier: CaseDifficulty = CaseDifficulty.MEDIUM
    speech_condition: SpeechCondition = SpeechCondition.CLEAN
    participant_band: Literal["small_2_3", "large_4_6", "large_4_7"] = (
        "small_2_3"
    )
    disfluency_types: list[
        Literal["filler", "pause", "hesitation", "false_start", "self_correction"]
    ] = Field(default_factory=list)
    audio_conditions: list[str] = Field(default_factory=list)
    comprehension_challenges: list[str] = Field(default_factory=list)


class GoalSpec(BaseModel):
    """One private user goal and its oracle disposition."""

    goal_id: str
    description: str
    priority: int = Field(ge=1, le=5)
    expected_disposition: GoalDisposition
    success_evidence: list[str] = Field(default_factory=list)


class PermissionGrant(BaseModel):
    """Action- and resource-scoped permission known by the benchmark oracle."""

    grant_id: str
    actions: list[str]
    resources: list[str] = Field(default_factory=lambda: ["*"])
    source: Literal[
        "system_acl",
        "environment_acl",
        "spoken_delegation",
        "spoken_consent",
    ] = "system_acl"
    active_initially: bool = True
    constraints: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """Immutable private configuration for one persistent user session."""

    user_id: str
    display_name: str
    role: str
    persona: PersonaSpec
    goals: list[GoalSpec]
    permissions: list[PermissionGrant]
    private_knowledge: list[str] = Field(default_factory=list)
    voice: VoiceIdentitySpec
    wake_on: list[ObservationKind] = Field(default_factory=list)
    present_initially: bool = True


class UserRuntimeState(BaseModel):
    """Mutable state owned by exactly one user session."""

    user_id: str
    goal_progress: dict[str, str] = Field(default_factory=dict)
    emotion: str = "neutral"
    ignored_count: int = 0
    last_spoke_ms: int | None = None
    currently_speaking: bool = False
    pending_audio_ids: list[str] = Field(default_factory=list)
    heard_committed_prefix: str = ""
    active_grant_ids: set[str] = Field(default_factory=set)
    memory: list[str] = Field(default_factory=list)
    present: bool = True
    active_thread_ids: set[str] = Field(default_factory=set)


class GoalRelation(BaseModel):
    """An explicit relation between two goal ids."""

    left_goal_id: str
    right_goal_id: str
    relation: GoalRelationType


class TriggerSpec(BaseModel):
    """Causal trigger for one required event."""

    kind: TriggerKind
    pattern: str | None = None
    event_id: str | None = None
    tool_name: str | None = None
    after_ms: int = Field(default=0, ge=0)
    fallback_at_ms: int | None = Field(default=None, ge=0)
    # A wall-clock fallback must not manufacture an unintended user/user
    # overlap while the prerequisite utterance is still being played.
    fallback_after_event_id: str | None = None


class GrantEffect(BaseModel):
    """Permission transition caused by a committed spoken event."""

    target_user_id: str
    grant_id: str
    active: bool


class PresenceEffect(BaseModel):
    """Participant entry or exit caused by a committed event."""

    target_user_id: str
    present: bool


class ScriptedEvent(BaseModel):
    """A benchmark-critical event with fixed semantics and flexible timing."""

    event_id: str
    actor_id: str
    action: UserAction
    trigger: TriggerSpec
    intent: str
    addressee: str | None = "assistant"
    fixed_utterance: str | None = None
    fixed_utterance_exact: bool = True
    required: bool = True
    urgency: int = Field(default=3, ge=0, le=5)
    overlap_allowed: bool = False
    max_delay_ms: int = Field(default=3000, ge=0)
    grant_effects: list[GrantEffect] = Field(default_factory=list)
    presence_effects: list[PresenceEffect] = Field(default_factory=list)
    thread_id: str = "main"
    speech_act: str = "statement"
    notes: list[str] = Field(default_factory=list)


class EnvironmentBinding(BaseModel):
    """Domain-neutral binding to a deterministic task environment."""

    domain: str = "retail"
    fixture_id: str | None = None
    principals: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    authoritative_facts: dict[str, Any] = Field(default_factory=dict)
    tau_user_id: str | None = None
    order_ids: list[str] = Field(default_factory=list)
    policy_extension: list[str] = Field(default_factory=list)
    authorization_mode: Literal["audit", "enforced"] = "audit"


class ExpectedToolAction(BaseModel):
    """Oracle expectation for a state-changing or read-only tool action."""

    tool_name: str
    resource_id: str
    actor_id: str
    allowed: bool
    must_execute: bool = False
    argument_subset: dict[str, Any] = Field(default_factory=dict)
    after_event_id: str | None = None
    before_event_id: str | None = None
    before_event_anchor: Literal["start", "end"] = "end"


class AuthorizationDecision(BaseModel):
    """Deterministic action/resource decision from the current grant state."""

    actor_id: str
    action: str
    resource_id: str
    allowed: bool
    matching_grant_ids: list[str] = Field(default_factory=list)
    reason: str


class CapabilityAssertion(BaseModel):
    """Judgeable behavioral requirement not reducible to a tool call."""

    assertion_id: str
    category: CapabilityCategory
    description: str
    expected_behavior: str
    source_event_ids: list[str] = Field(default_factory=list)
    judge: Literal["rule", "semantic", "hybrid"] = "semantic"
    critical: bool = True
    # Diagnostic-only obligations remain reportable without entering the
    # v15 interaction contract.  They are used by recognition-split rescore.
    target_participant_id: str | None = None
    return_event_id: str | None = None
    diagnostic_only: bool = False


class EnvironmentStateAssertion(BaseModel):
    """Deterministic assertion over a final virtual-environment snapshot."""

    assertion_id: str
    path: str
    operator: Literal["eq", "ne", "contains", "not_contains", "exists"]
    expected: Any = None
    critical: bool = True


class TimingRequirement(BaseModel):
    """A measurable full-duplex timing constraint."""

    metric: Literal[
        "yield_latency_ms",
        "resume_latency_ms",
        "thread_resume_latency_ms",
        "clarification_latency_ms",
        "speaker_wait_ms",
        "false_interruptions",
        "speaker_starvation_count",
        "overlap_realized_ms",
        "post_revoke_action_latency_ms",
        "floor_invitation_latency_ms",
        "post_invitation_false_interruptions",
    ]
    event_id: str
    operator: Literal["le", "ge", "eq"]
    threshold: float


class MultiUserVoiceCase(BaseModel):
    """Complete case specification for a multi-user voice episode."""

    task_contract_revision: dict[str, Any] = Field(default_factory=dict)
    case_id: str
    title: str
    summary: str
    capability_tags: list[str]
    identity_setup: list[str]
    permission_setup: list[str]
    benchmark_version: str = "v0.4"
    provenance: BenchmarkProvenance = Field(default_factory=BenchmarkProvenance)
    difficulty: DifficultyProfile = Field(default_factory=DifficultyProfile)
    coverage: CoverageSpec = Field(default_factory=CoverageSpec)
    variant: ScenarioVariantSpec | None = None
    anti_wait_constraints: list[str] = Field(default_factory=list)
    voice_enrollment: VoiceEnrollmentProtocol = Field(
        default_factory=VoiceEnrollmentProtocol
    )
    assistant_activation: AssistantActivationProtocol = Field(
        default_factory=AssistantActivationProtocol
    )
    iterative_replay: IterativeReplayProtocol = Field(
        default_factory=IterativeReplayProtocol
    )
    environment: EnvironmentBinding
    users: list[UserProfile]
    goal_relations: list[GoalRelation]
    events: list[ScriptedEvent]
    expected_tool_actions: list[ExpectedToolAction]
    timing_requirements: list[TimingRequirement]
    final_success_conditions: list[str]
    capability_assertions: list[CapabilityAssertion] = Field(default_factory=list)
    state_assertions: list[EnvironmentStateAssertion] = Field(default_factory=list)
    seed: int = 42

    def public_participant_id(self, oracle_user_id: str) -> str:
        """Return the non-semantic model-facing ID for an oracle participant."""
        for index, user in enumerate(self.users, start=1):
            if user.user_id == oracle_user_id:
                return f"p{index:02d}"
        raise KeyError(f"unknown oracle user id: {oracle_user_id}")

    def public_participant_map(self) -> dict[str, str]:
        """Map oracle user IDs to stable per-case public participant IDs."""
        return {
            user.user_id: self.public_participant_id(user.user_id)
            for user in self.users
        }

    @model_validator(mode="after")
    def validate_references(self) -> MultiUserVoiceCase:
        """Reject dangling ids and ambiguous case definitions."""
        user_ids = [user.user_id for user in self.users]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("user ids must be unique")
        enrolled_voice_ids = [
            user.voice.voice_id
            for user in self.users
            if user.voice.enrollment_status == "enrolled"
        ]
        if len(enrolled_voice_ids) != len(set(enrolled_voice_ids)):
            raise ValueError("enrolled participants must have unique voice ids")
        valid_bands = (
            {"small_2_3"}
            if len(user_ids) <= 3
            else {"large_4_6", "large_4_7"}
        )
        if self.difficulty.participant_band not in valid_bands:
            raise ValueError(
                "difficulty participant_band does not match participant count"
            )
        if not 2 <= len(user_ids) <= 7:
            raise ValueError("multi-user cases require two to seven participants")
        activation_caller = self.assistant_activation.caller_user_id
        if activation_caller is not None and activation_caller not in user_ids:
            raise ValueError("assistant activation references an unknown caller")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event ids must be unique")
        unknown_actors = {event.actor_id for event in self.events} - set(user_ids)
        if unknown_actors:
            raise ValueError(f"events reference unknown actors: {unknown_actors}")
        unknown_effect_targets = {
            effect.target_user_id
            for event in self.events
            for effect in event.grant_effects
        } - set(user_ids)
        if unknown_effect_targets:
            raise ValueError(
                f"grant effects reference unknown users: {unknown_effect_targets}"
            )
        unknown_presence_targets = {
            effect.target_user_id
            for event in self.events
            for effect in event.presence_effects
        } - set(user_ids)
        if unknown_presence_targets:
            raise ValueError(
                f"presence effects reference unknown users: {unknown_presence_targets}"
            )
        grants_by_user = {
            user.user_id: {grant.grant_id for grant in user.permissions}
            for user in self.users
        }
        for event in self.events:
            for effect in event.grant_effects:
                if effect.grant_id not in grants_by_user[effect.target_user_id]:
                    raise ValueError(
                        f"unknown grant {effect.grant_id} for {effect.target_user_id}"
                    )

        goal_ids = {goal.goal_id for user in self.users for goal in user.goals}
        relation_goal_ids = {
            goal_id
            for relation in self.goal_relations
            for goal_id in (relation.left_goal_id, relation.right_goal_id)
        }
        if not relation_goal_ids.issubset(goal_ids):
            raise ValueError("goal relation references an unknown goal")

        referenced_events = {
            referenced_event
            for event in self.events
            for referenced_event in (
                event.trigger.event_id,
                event.trigger.fallback_after_event_id,
            )
            if referenced_event is not None
        }
        if not referenced_events.issubset(set(event_ids)):
            raise ValueError("event trigger references an unknown event")
        temporal_events = {
            event_id
            for expected in self.expected_tool_actions
            for event_id in (expected.after_event_id, expected.before_event_id)
            if event_id is not None
        }
        if not temporal_events.issubset(set(event_ids)):
            raise ValueError("tool expectation references an unknown event")
        assertion_ids = [
            assertion.assertion_id for assertion in self.capability_assertions
        ] + [assertion.assertion_id for assertion in self.state_assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("assertion ids must be unique within a case")
        assertion_event_ids = {
            event_id
            for assertion in self.capability_assertions
            for event_id in assertion.source_event_ids
        }
        if not assertion_event_ids.issubset(set(event_ids)):
            raise ValueError("capability assertion references an unknown event")
        return self


class TranscriptTurn(BaseModel):
    """Only the public committed portion of a conversation."""

    speaker: str
    text: str
    start_ms: int
    end_ms: int


class TimedTranscriptFragment(BaseModel):
    """One probe transcript fragment on the task-relative media clock."""

    timestamp_ms: int = Field(ge=0)
    text: str


class WorldObservation(BaseModel):
    """Observation passed to persistent user sessions.

    Ordinary online simulation uses only the committed fields.  Iterative
    replay additionally fills the completed-probe fields after the unscored
    probe has fully drained, allowing retrospective placement of the next
    frozen user event before a separate Final session starts.
    """

    sim_time_ms: int = Field(ge=0)
    kind: ObservationKind
    assistant_speaking: bool = False
    assistant_committed_prefix: str = ""
    probe_complete_assistant_response: str = ""
    probe_assistant_fragments: list[TimedTranscriptFragment] = Field(
        default_factory=list
    )
    probe_assistant_intervals_ms: list[tuple[int, int]] = Field(
        default_factory=list
    )
    probe_task_media_end_ms: int | None = Field(default=None, ge=0)
    public_transcript: list[TranscriptTurn] = Field(default_factory=list)
    completed_event_ids: set[str] = Field(default_factory=set)
    tool_name: str | None = None
    tool_result: str | None = None
    mentioned_user_ids: set[str] = Field(default_factory=set)
    active_thread_ids: set[str] = Field(default_factory=set)


class UserProposal(BaseModel):
    """Structured proposal emitted by one user session."""

    proposal_id: str
    actor_id: str
    action: UserAction
    utterance: str | None = None
    addressee: str | None = "assistant"
    urgency: int = Field(default=0, ge=0, le=5)
    intent: str = ""
    overlap_allowed: bool = False
    requested_delay_ms: int = Field(default=0, ge=0)
    target_start_ms: int | None = Field(default=None, ge=0)
    max_wait_ms: int = Field(default=3000, ge=0)
    triggered_by_event_id: str | None = None
    required_event: bool = False
    thread_id: str = "main"
    speech_act: str = "statement"


class ScheduledProposal(BaseModel):
    """A proposal accepted onto the shared microphone timeline."""

    proposal: UserProposal
    start_ms: int
    score: float
    mode: Literal["exclusive", "overlap"]


class RejectedProposal(BaseModel):
    """A proposal not selected for the current floor."""

    proposal_id: str
    actor_id: str
    reason: str


class ScheduleDecision(BaseModel):
    """Deterministic floor decision for one simulator update."""

    sim_time_ms: int
    selected: list[ScheduledProposal] = Field(default_factory=list)
    rejected: list[RejectedProposal] = Field(default_factory=list)


class SpokenEventTrace(BaseModel):
    """Oracle speech event retained outside the mono SUT input."""

    event_id: str | None = None
    proposal_id: str
    speaker_id: str
    addressee: str | None
    utterance: str
    start_ms: int
    end_ms: int
    ready_ms: int | None = None
    trigger_path: Literal["causal", "fallback"] | None = None
    trigger_observation_kind: ObservationKind | None = None
    schedule_rejection_count: int = Field(default=0, ge=0)
    sut_attributed_speaker_id: str | None = None
    sut_inferred_addressee: str | None = None
    sut_attribution_source: Literal["explicit", "tool", "heuristic"] | None = None
    thread_id: str = "main"
    speech_act: str = "statement"


class VoiceEnrollmentTrace(BaseModel):
    """One oracle-labeled enrollment utterance on the shared session clock."""

    user_id: str
    display_name: str
    role: str
    voice_id: str
    utterance: str
    start_ms: int
    end_ms: int
    enrollment_status: Literal["enrolled", "unknown", "spoof_candidate"]


class AssistantActivationTrace(BaseModel):
    """Oracle record of the user invocation preceding formal task time zero."""

    assistant_name: str
    caller_user_id: str
    utterance: str
    start_ms: int
    end_ms: int


class SpeakerAttributionTrace(BaseModel):
    """Explicit SUT identity/addressee report for one task utterance."""

    timestamp_ms: int
    task_timestamp_ms: int
    predicted_speaker_id: str
    predicted_addressee: str
    reported_speaker_reference: str | None = None
    reported_addressee_reference: str | None = None
    predicted_thread_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_event_id: str | None = None
    matched_proposal_id: str | None = None


class ToolActionTrace(BaseModel):
    """A SUT tool decision annotated with oracle authorization."""

    tool_name: str
    resource_id: str
    timestamp_ms: int
    result_available_ms: int | None = None
    result_availability_basis: str | None = None
    returned_fact_paths: list[str] = Field(default_factory=list)
    returned_result_sha256: str | None = None
    sut_attributed_actor_id: str | None
    oracle_actor_id: str | None
    oracle_authorized: bool
    executed: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    authorization_evidence: list[str] = Field(default_factory=list)
    caller_id: str | None = None
    authorization_basis: str | None = None
    request_basis_status: str | None = None
    request_basis_original_status: str | None = None
    request_basis_review_reason: str | None = None
    request_basis_routing_policy: str | None = None
    request_basis_evidence: list[str] = Field(default_factory=list)
    caller_provenance_status: str | None = None
    caller_provenance_evidence: list[str] = Field(default_factory=list)


class GoalResult(BaseModel):
    """Observed disposition of one goal after the episode."""

    goal_id: str
    disposition: GoalDisposition
    evidence: list[str] = Field(default_factory=list)
    # None preserves unreviewed legacy semantics. Reviewed evidence failures
    # must never pass merely because a fallback disposition matches the oracle.
    evidence_verified: bool | None = None


class TimingMeasurement(BaseModel):
    """One clock-derived interaction measurement."""

    metric: str
    event_id: str
    value: float


class CapabilityResult(BaseModel):
    """Rule- or judge-produced result for one capability assertion."""

    assertion_id: str
    passed: bool
    predicted: str | None = None
    evidence: list[str] = Field(default_factory=list)
    anchor_kind: str | None = None
    failure_kind: str | None = None
    assistant_source_ids: list[str] = Field(default_factory=list)
    tool_source_ids: list[str] = Field(default_factory=list)
    event_source_ids: list[str] = Field(default_factory=list)


class RecognitionSplitPair(BaseModel):
    """One returning participant's anchor/continuity diagnostic result."""

    target_participant_id: str
    return_event_id: str
    spoken_anchor_assertion_id: str
    behavior_continuity_assertion_id: str
    spoken_anchor_passed: bool | None = None
    behavior_continuity_passed: bool | None = None
    spoken_anchor_anchor_kind: str | None = None
    behavior_continuity_failure_kind: str | None = None


class RecognitionSplitReport(BaseModel):
    """Evaluation-only decomposition of returning-speaker routing."""

    contract_version: str = "mu-bench-recognition-split-v1"
    evaluation_only: bool = True
    pairs: list[RecognitionSplitPair] = Field(default_factory=list)
    spoken_anchor_rate: float | None = None
    behavioral_continuity_rate: float | None = None
    robust_return_routing_rate: float | None = None
    behavior_only_rate: float | None = None
    anchor_only_rate: float | None = None
    neither_rate: float | None = None
    no_assistant_evidence_rate: float | None = None
    explicitation_gap: float | None = None


class MultiUserEpisodeTrace(BaseModel):
    """Trace needed for deterministic and semantic evaluation."""

    case_id: str
    voice_enrollment_mode: Literal["not_run", "in_session"] = "not_run"
    task_start_ms: int = 0
    voice_enrollments: list[VoiceEnrollmentTrace] = Field(default_factory=list)
    enrollment_assistant_response_count: int = 0
    enrollment_response_created_count: int = 0
    assistant_activation: AssistantActivationTrace | None = None
    activation_assistant_response_count: int = 0
    session_wall_duration_ms: float | None = None
    session_media_duration_ms: float | None = None
    max_tick_processing_ms: float | None = None
    stalled_tick_count: int = 0
    interaction_mechanism_realized: bool | None = None
    interaction_realization_evidence: list[str] = Field(default_factory=list)
    speaker_attributions: list[SpeakerAttributionTrace] = Field(default_factory=list)
    spoken_events: list[SpokenEventTrace] = Field(default_factory=list)
    tool_actions: list[ToolActionTrace] = Field(default_factory=list)
    goal_results: list[GoalResult] = Field(default_factory=list)
    timing_measurements: list[TimingMeasurement] = Field(default_factory=list)
    capability_results: list[CapabilityResult] = Field(default_factory=list)
    final_environment_state: dict[str, Any] = Field(default_factory=dict)
    privacy_leaks: list[str] = Field(default_factory=list)
    simulator_violations: list[str] = Field(default_factory=list)
    execution_protocol: str = "legacy_single_session"
    frozen_event_plan_sha256: str | None = None
    probe_count: int = 0


class MultiUserScoreReport(BaseModel):
    business_basis_routing_policy: str | None = None
    business_basis_original_insufficient_calls: list[int] = Field(default_factory=list)
    """Disaggregated task, interaction, and run-validity scores."""

    scoring_contract_version: str = "mu-bench-scoring-v3-mean-interaction"
    evaluation_policy_version: str | None = None
    raw_reference_state_match: bool | None = None
    causal_query_audit: list[dict[str, Any]] = Field(default_factory=list)
    case_id: str
    voice_enrollment_coverage: float | None = None
    enrollment_silence_pass: bool | None = None
    activation_present: bool = False
    activation_assistant_response_count: int = 0
    voice_binding_probe_coverage: float | None = None
    voice_binding_probe_accuracy: float | None = None
    explicit_attribution_rate: float | None = None
    session_wall_duration_ms: float | None = None
    session_media_duration_ms: float | None = None
    timing_clock: Literal["media_clock"] = "media_clock"
    max_tick_processing_ms: float | None = None
    stalled_tick_count: int = 0
    clock_integrity_pass: bool
    wall_timing_eligible: bool
    wall_clock_overrun_ms: float | None = None
    interaction_mechanism_realized: bool | None = None
    required_event_coverage: float
    speaker_attribution_accuracy: float | None
    addressee_accuracy: float | None
    permission_decision_accuracy: float | None
    unauthorized_action_attempts: int
    unauthorized_actions_executed: int
    goal_disposition_accuracy: float
    timing_pass_rate: float | None
    semantic_assertion_accuracy: float | None
    capability_scores: dict[str, float] = Field(default_factory=dict)
    state_assertion_accuracy: float | None
    reference_state_match: bool | None = None
    privacy_leak_count: int
    simulator_valid: bool
    run_valid: bool
    goal_disposition_success: bool
    expected_tools_success: bool
    final_state_success: bool
    caller_provenance_success: bool | None = None
    caller_id_misattributions: int = 0
    authorization_success: bool
    privacy_success: bool
    task_success: bool
    speaker_success: bool
    addressee_success: bool
    attribution_success: bool
    floor_control_score: float | None = None
    timing_score: float | None = None
    interruption_recovery_score: float | None = None
    semantic_routing_score: float | None = None
    # v15 mechanism-level floor diagnostics.  The aggregate floor component
    # remains backward-compatible while making its two incompatible mechanisms
    # independently visible.
    floor_mechanism: str | None = None
    backchannel_policy_score: float | None = None
    fairness_policy_score: float | None = None
    quiet_input_integration_score: float | None = None
    interaction_score: float | None = None
    recognition_split: RecognitionSplitReport = Field(
        default_factory=RecognitionSplitReport
    )
    floor_control_success: bool | None
    timing_success: bool
    interruption_recovery_success: bool | None
    semantic_routing_success: bool | None
    interaction_success: bool | None
    strict_joint_success: bool
    task_failures: list[str] = Field(default_factory=list)
    attribution_failures: list[str] = Field(default_factory=list)
    interaction_failures: list[str] = Field(default_factory=list)
    run_diagnostics: list[str] = Field(default_factory=list)
    # Backward-compatible union for consumers that still display one issue list.
    failures: list[str] = Field(default_factory=list)
