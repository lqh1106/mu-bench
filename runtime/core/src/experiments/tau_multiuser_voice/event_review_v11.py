"""GPT-reviewed event overrides for the deeper μ-Bench case catalog."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.tau_multiuser_voice.models import (
    ComprehensionLevel,
    MultiUserVoiceCase,
    ScriptedEvent,
    TriggerKind,
    TriggerSpec,
    UserAction,
)
from experiments.tau_multiuser_voice.version import (
    WEAK_COMPREHENSION_SCHEDULE_VERSION,
)


DEFAULT_REVIEW_PATH = Path(__file__).with_name("reviewed_event_overrides_v08.json")

# The weak-comprehension event is deliberately a post-resolution audience
# repair.  These constants are a deterministic release policy layered on top
# of the GPT review: they keep the planner window within the declared timing
# contract and guarantee that a model which never emits a final stop event
# cannot leave a required user event pending forever.
WEAK_REPAIR_MAX_DELAY_MS = 2200
WEAK_REPAIR_FALLBACK_AT_MS = 30000


class StrictReviewModel(BaseModel):
    """Ensure the API schema rejects invented review fields."""

    model_config = ConfigDict(extra="forbid")


class ReviewedTrigger(StrictReviewModel):
    """Full trigger returned by the structured event reviewer."""

    kind: TriggerKind
    pattern: str | None
    event_id: str | None
    tool_name: str | None
    after_ms: int = Field(ge=0)
    fallback_at_ms: int | None
    fallback_after_event_id: str | None


class ReviewedEvent(StrictReviewModel):
    """Safe event-contract surface; this is not a generated user turn."""

    event_id: str
    actor_id: str
    action: UserAction
    trigger: ReviewedTrigger
    intent: str = Field(min_length=3)
    addressee: str | None
    fixed_utterance: str | None
    fixed_utterance_exact: bool
    urgency: int = Field(ge=0, le=5)
    overlap_allowed: bool
    max_delay_ms: int = Field(ge=0)
    thread_id: str = Field(min_length=1)
    speech_act: str = Field(min_length=1)
    notes: list[str]


class EventQualityReview(StrictReviewModel):
    """Per-event critique and its corrected event."""

    event_id: str
    verdict: Literal["pass", "revise"]
    realism_score: int = Field(ge=1, le=5)
    multiuser_necessity_score: int = Field(ge=1, le=5)
    full_duplex_necessity_score: int = Field(ge=1, le=5)
    continuity_score: int = Field(ge=1, le=5)
    evaluability_score: int = Field(ge=1, le=5)
    problems: list[str]
    modifications: list[str]
    revised_event: ReviewedEvent

    @property
    def minimum_score(self) -> int:
        return min(
            self.realism_score,
            self.multiuser_necessity_score,
            self.full_duplex_necessity_score,
            self.continuity_score,
            self.evaluability_score,
        )


class CaseQualityReview(StrictReviewModel):
    """Whole-trajectory checks that cannot be judged event by event."""

    realistic_conversation: bool
    sufficient_multi_round_depth: bool
    repeated_speakers_are_stateful: bool
    multiuser_identity_is_decisive: bool
    full_duplex_timing_is_decisive: bool
    tool_permission_consistent: bool
    no_wait_until_end_shortcut: bool
    no_future_information_leakage: bool
    overall_problems: list[str]
    recommended_case_changes: list[str]
    approved: bool


class CaseEventReview(StrictReviewModel):
    """Structured GPT-5.5 result for one concrete case variant."""

    case_id: str
    event_reviews: list[EventQualityReview]
    case_review: CaseQualityReview

    @model_validator(mode="after")
    def unique_event_reviews(self) -> CaseEventReview:
        event_ids = [review.event_id for review in self.event_reviews]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event review ids must be unique")
        return self


def compact_case_payload(case: MultiUserVoiceCase) -> dict:
    """Return the evidence GPT needs without unrelated audio implementation fields."""
    return {
        "review_scope": {
            "stage": "pre-SUT event-contract review",
            "sut_response_available": False,
            "reviewer_must_not_author_final_dialogue": True,
            "probe_responsibility": (
                "After observing a complete actual SUT response, one persistent "
                "user simulator per participant converts the approved event intent "
                "into natural dialogue and selects its exact insertion time."
            ),
        },
        "case_id": case.case_id,
        "title": case.title,
        "summary": case.summary,
        "variant": case.variant.model_dump(mode="json") if case.variant else None,
        "coverage": case.coverage.model_dump(mode="json"),
        "users": [
            {
                "user_id": user.user_id,
                "display_name": user.display_name,
                "role": user.role,
                "persona": user.persona.model_dump(mode="json"),
                "goals": [goal.model_dump(mode="json") for goal in user.goals],
                "permissions": [
                    grant.model_dump(mode="json") for grant in user.permissions
                ],
                "present_initially": user.present_initially,
            }
            for user in case.users
        ],
        "goal_relations": [
            relation.model_dump(mode="json") for relation in case.goal_relations
        ],
        "events": [event.model_dump(mode="json") for event in case.events],
        "environment": {
            "domain": case.environment.domain,
            "tool_names": case.environment.tool_names,
            "resources": case.environment.resources,
            "authoritative_facts": case.environment.authoritative_facts,
            "authorization_mode": case.environment.authorization_mode,
            "policy_extension": case.environment.policy_extension,
        },
        "expected_tool_actions": [
            action.model_dump(mode="json") for action in case.expected_tool_actions
        ],
        "timing_requirements": [
            requirement.model_dump(mode="json")
            for requirement in case.timing_requirements
        ],
        "capability_assertions": [
            assertion.model_dump(mode="json")
            for assertion in case.capability_assertions
        ],
        "final_success_conditions": case.final_success_conditions,
        "anti_wait_constraints": case.anti_wait_constraints,
    }


def validate_review_for_case(
    case: MultiUserVoiceCase,
    review: CaseEventReview,
    *,
    require_approved: bool = True,
) -> list[str]:
    """Reject missing reviews, oracle drift, and structurally invalid revisions."""
    errors: list[str] = []
    if review.case_id != case.case_id:
        errors.append("review case_id mismatch")
    reviews_by_id = {item.event_id: item for item in review.event_reviews}
    if list(reviews_by_id) != [event.event_id for event in case.events]:
        errors.append("review must contain every event exactly once in case order")
    declared: set[str] = set()
    valid_addressees = {user.user_id for user in case.users} | {
        "assistant",
        "group",
    }
    for event in case.events:
        item = reviews_by_id.get(event.event_id)
        if item is None:
            continue
        revised = item.revised_event
        for field in (
            "event_id",
            "actor_id",
            "action",
            "addressee",
            "thread_id",
            "speech_act",
            "fixed_utterance_exact",
            "fixed_utterance",
        ):
            if getattr(revised, field) != getattr(event, field):
                errors.append(f"{event.event_id} changed protected field {field}")
        for field in (
            "kind",
            "pattern",
            "event_id",
            "tool_name",
            "fallback_after_event_id",
        ):
            if getattr(revised.trigger, field) != getattr(event.trigger, field):
                errors.append(
                    f"{event.event_id} changed protected trigger field {field}"
                )
        if event.event_id != case.events[0].event_id and revised.fixed_utterance:
            errors.append(
                f"{event.event_id} authored dialogue before the Probe stage"
            )
        if revised.addressee is not None and revised.addressee not in valid_addressees:
            errors.append(f"{event.event_id} has unknown addressee")
        for dependency in (
            revised.trigger.event_id,
            revised.trigger.fallback_after_event_id,
        ):
            if dependency is not None and dependency not in declared:
                errors.append(
                    f"{event.event_id} trigger depends on non-earlier event {dependency}"
                )
        if require_approved and item.minimum_score < 4:
            errors.append(f"{event.event_id} has a quality score below 4")
        declared.add(event.event_id)
    if require_approved:
        checks = review.case_review
        for field in (
            "realistic_conversation",
            "sufficient_multi_round_depth",
            "repeated_speakers_are_stateful",
            "multiuser_identity_is_decisive",
            "full_duplex_timing_is_decisive",
            "tool_permission_consistent",
            "no_wait_until_end_shortcut",
            "no_future_information_leakage",
            "approved",
        ):
            if not getattr(checks, field):
                errors.append(f"case review failed {field}")
    return errors


def normalize_review_verdicts(
    case: MultiUserVoiceCase,
    review: CaseEventReview,
) -> CaseEventReview:
    """Make verdict describe whether the reviewer actually changed the contract."""
    normalized = review.model_copy(deep=True)
    by_id = {item.event_id: item for item in normalized.event_reviews}
    for event in case.events:
        item = by_id.get(event.event_id)
        if item is None:
            continue
        revised = item.revised_event
        changed = any(
            getattr(revised, field) != getattr(event, field)
            for field in (
                "intent",
                "urgency",
                "overlap_allowed",
                "max_delay_ms",
                "notes",
            )
        ) or any(
            getattr(revised.trigger, field) != getattr(event.trigger, field)
            for field in ("after_ms", "fallback_at_ms")
        )
        if changed and item.verdict == "pass":
            item.verdict = "revise"
            if not item.modifications:
                item.modifications.append(
                    "Changed one or more mutable event-contract fields."
                )
    return normalized


def apply_case_review(
    case: MultiUserVoiceCase,
    review: CaseEventReview,
    *,
    require_approved: bool = True,
) -> MultiUserVoiceCase:
    """Apply a validated review while preserving permission/presence effects."""
    errors = validate_review_for_case(
        case,
        review,
        require_approved=require_approved,
    )
    if errors:
        raise ValueError(f"invalid review for {case.case_id}: " + "; ".join(errors))
    reviews_by_id = {item.event_id: item for item in review.event_reviews}
    revised_events: list[ScriptedEvent] = []
    for event in case.events:
        revised = reviews_by_id[event.event_id].revised_event
        revised_events.append(
            event.model_copy(
                deep=True,
                update={
                    "trigger": TriggerSpec.model_validate(
                        revised.trigger.model_dump(mode="json")
                    ),
                    "intent": revised.intent,
                    "urgency": revised.urgency,
                    "overlap_allowed": revised.overlap_allowed,
                    "max_delay_ms": revised.max_delay_ms,
                    "notes": revised.notes,
                },
            )
        )
    payload = case.model_dump(mode="json")
    payload["events"] = [event.model_dump(mode="json") for event in revised_events]
    return MultiUserVoiceCase.model_validate(payload)


def _configured_review_path() -> Path:
    configured = os.getenv("MU_BENCH_EVENT_OVERRIDES")
    return Path(configured) if configured else DEFAULT_REVIEW_PATH


@lru_cache(maxsize=1)
def load_review_overrides() -> dict[str, CaseEventReview]:
    """Load checked-in reviews unless explicitly disabled for draft export."""
    if os.getenv("MU_BENCH_DISABLE_EVENT_OVERRIDES") == "1":
        return {}
    path = _configured_review_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", raw)
    return {
        case_id: CaseEventReview.model_validate(payload)
        for case_id, payload in cases.items()
    }


def apply_default_review(case: MultiUserVoiceCase) -> MultiUserVoiceCase:
    """Apply one release override when available."""
    review = load_review_overrides().get(case.case_id)
    return case if review is None else apply_case_review(case, review)


def normalize_weak_comprehension_schedule(
    case: MultiUserVoiceCase,
) -> MultiUserVoiceCase:
    """Apply the versioned, deterministic weak-repair timing contract.

    GPT-5.5 is used to critique semantic contracts, but timing safety must not
    depend on a mutable review file.  In particular, a reviewed event may
    retain the generic three-second planner window even though the case scores
    its clarification response with a 2.2-second gate.  The normalizer keeps
    any stricter reviewed value, caps the window at the gate, and supplies a
    finite fallback only when the review accidentally removed one.

    The normalizer does not author dialogue, change the trigger dependency, or
    move the event: weak repair remains after ``e_primary_commit`` and is still
    generated from the complete Probe response by the persistent user session.
    """
    if (
        case.variant is None
        or case.variant.comprehension_level != ComprehensionLevel.WEAK
    ):
        return case

    repair_index = next(
        (
            index
            for index, event in enumerate(case.events)
            if event.event_id == "e_comprehension_repair"
        ),
        None,
    )
    if repair_index is None:
        return case

    event = case.events[repair_index]
    trigger = event.trigger
    intent = event.intent
    # A handful of accepted reviews left only a generic intent.  Make the
    # contract measurable without inventing the user's final words.
    if intent.strip().casefold() == "request a shorter plain-language restatement":
        intent = (
            "request a shorter plain-language restatement of the final outcome "
            "for the primary thread, addressed to the same recipient and "
            "preserving the already established speaker-scoped decision"
        )
    revised_event = event.model_copy(
        deep=True,
        update={
            "max_delay_ms": min(event.max_delay_ms, WEAK_REPAIR_MAX_DELAY_MS),
            "intent": intent,
            "trigger": trigger.model_copy(
                update={
                    "fallback_at_ms": (
                        WEAK_REPAIR_FALLBACK_AT_MS
                        if trigger.fallback_at_ms is None
                        else trigger.fallback_at_ms
                    )
                }
            ),
        },
    )
    if revised_event == event:
        return case
    revised_events = list(case.events)
    revised_events[repair_index] = revised_event
    return case.model_copy(update={"events": revised_events})
