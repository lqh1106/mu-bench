"""Versioned, case-scoped task-input corrections and frozen-input guards.

Legacy public context used user speech. Scene r3 has an explicit pre-session assistant source view.
Unreviewed cases are unchanged. Private fixture values are not auto-broadcast.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

FACT_CONTRACT_VERSION = "mu-bench-task-fact-contract-v1.3.2"
REVIEW_PATH = Path(__file__).with_name("task_contract_reviews_v132.json")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_reviews() -> dict[str, dict]:
    if not REVIEW_PATH.exists():
        return {}
    payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    reviews = payload["reviews"] if isinstance(payload, dict) else payload
    by_id = {item["case_id"]: item for item in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("duplicate reviewed case IDs")
    return by_id


def review_for_case(case: Any) -> dict:
    revision = getattr(case, "task_contract_revision", {})
    if revision.get("review"):
        return deepcopy(revision["review"])
    return deepcopy(load_reviews().get(case.case_id, {}))


def apply_task_review(case: Any, review: dict | None = None) -> Any:
    """Apply explicit, validated overrides without changing unrelated cases."""
    from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
    review = review_for_case(case) if review is None else review
    if not review:
        return case
    if review.get("case_revision_id") == "mu-bench-tool-free-v1":
        if review.get("case_id") != case.case_id or case.environment.tool_names:
            raise ValueError("scene revision only applies to the matching no-tool case")
        if "scene_payload" in review:
            payload = deepcopy(review["scene_payload"])
            if payload["case_id"] != case.case_id or payload["environment"]["tool_names"]:
                raise ValueError("invalid scene payload")
            # No unreviewed speaker identity, seed or acoustic-condition change.
            old_ids = [(u.user_id, u.display_name, u.voice.model_dump(mode="json")) for u in case.users]
            new_ids = [(u["user_id"], u["display_name"], u["voice"]) for u in payload["users"]]
            if old_ids != new_ids or payload["seed"] != case.seed:
                raise ValueError("scene payload altered speaker identity, voice or seed")
            payload["task_contract_revision"] = {
                "case_revision_id": review["case_revision_id"],
                "scene_payload_sha256": digest(payload),
                "review": {k: deepcopy(v) for k, v in review.items() if k != "scene_payload"},
            }
            return MultiUserVoiceCase.model_validate(payload)
        # A materialized r3 case already embeds its scoped review and final payload.
        return case
    if review.get("case_id") != case.case_id or review.get("status") == "blocked":
        raise ValueError(f"unreleased task review: {case.case_id}")
    payload = case.model_dump(mode="json")
    before_identity = [(u["user_id"], u["display_name"], u["voice"]) for u in payload["users"]]
    event_map = {event["event_id"]: event for event in payload["events"]}
    user_map = {user["user_id"]: user for user in payload["users"]}
    goal_map = {goal["goal_id"]: goal for user in payload["users"] for goal in user["goals"]}
    replacements: dict[str, dict[str, str]] = {user_id: {} for user_id in user_map}
    goal_owners = {goal["goal_id"]: user["user_id"] for user in payload["users"] for goal in user["goals"]}
    for patch in review.get("event_overrides", []):
        event_id = patch["event_id"]
        if event_id not in event_map:
            raise ValueError(f"unknown reviewed event: {event_id}")
        allowed = {"intent", "fixed_utterance", "fixed_utterance_exact", "notes", "addressee", "speech_act", "thread_id"}
        if set(patch) - allowed - {"event_id"}:
            raise ValueError(f"unsafe event override: {patch}")
        event_map[event_id].update({k: v for k, v in patch.items() if k != "event_id"})
    for patch in review.get("goal_overrides", []):
        goal_id = patch["goal_id"]
        if goal_id not in goal_map or set(patch) - {"goal_id", "description", "expected_disposition", "success_evidence"}:
            raise ValueError(f"unsafe goal override: {patch}")
        if "description" in patch:
            replacements[goal_owners[goal_id]][goal_map[goal_id]["description"]] = patch["description"]
        goal_map[goal_id].update({k: v for k, v in patch.items() if k != "goal_id"})
    for patch in review.get("user_overrides", []):
        user_id = patch["user_id"]
        if user_id not in user_map or set(patch) - {"user_id", "role", "persona", "private_knowledge"}:
            raise ValueError(f"unsafe user override: {patch}")
        if "private_knowledge" in patch:
            if not isinstance(patch["private_knowledge"], list) or not all(isinstance(x, str) for x in patch["private_knowledge"]):
                raise ValueError("private knowledge must be a list of strings")
            user_map[user_id]["private_knowledge"] = list(patch["private_knowledge"])
        if "role" in patch:
            user_map[user_id]["role"] = patch["role"]
        if "persona" in patch:
            if not isinstance(patch["persona"], dict):
                raise ValueError("persona correction must be a field mapping")
            if set(patch["persona"]) - set(user_map[user_id]["persona"]):
                raise ValueError("unknown persona correction field")
            user_map[user_id]["persona"].update(patch["persona"])
    overrides = review.get("case_overrides", {})
    if set(overrides) - {"summary", "final_success_conditions", "capability_assertions", "identity_setup", "permission_setup"}:
        raise ValueError("case override changes an unreviewed structural field")
    if "capability_assertions" in overrides:
        # v132 task reviews froze the v14 assertion list.  v15 deliberately
        # adds diagnostic recognition-split obligations and replaces the two
        # floor mechanisms' overloaded duplex assertion with mechanism-specific
        # obligations.  Preserve reviewed semantic corrections while carrying
        # those code-audited v15 additions through task review.
        reviewed_assertions = deepcopy(overrides["capability_assertions"])
        code_assertions = payload["capability_assertions"]
        v15_assertion_ids = {
            item["assertion_id"]
            for item in code_assertions
            if (
                item.get("diagnostic_only") is True
                or item["assertion_id"] in {
                    "a_backchannel_non_takeover",
                    "a_backchannel_original_continuation",
                    "a_targeted_floor_invitation",
                    "a_no_premature_closure",
                    "a_quiet_input_integration",
                }
            )
        }
        floor_mechanism = any(
            item["assertion_id"] in {
                "a_backchannel_non_takeover",
                "a_targeted_floor_invitation",
            }
            for item in code_assertions
        )
        reviewed_by_id = {
            item["assertion_id"]: item for item in reviewed_assertions
        }
        if floor_mechanism:
            reviewed_by_id.pop("a_full_duplex_policy", None)
        reviewed_by_id.update(
            {
                item["assertion_id"]: item
                for item in code_assertions
                if item["assertion_id"] in v15_assertion_ids
            }
        )
        before_order = [
            item["assertion_id"] for item in payload["capability_assertions"]
        ]
        overrides = deepcopy(overrides)
        overrides["capability_assertions"] = [
            reviewed_by_id[assertion_id]
            for assertion_id in before_order
            if assertion_id in reviewed_by_id
        ]
    payload.update(overrides)
    # User planner snapshots repeat the private goal description. Correct the
    # exact duplicate as well; do not edit unrelated private knowledge.
    for user in payload["users"]:
        for index, text in enumerate(user["private_knowledge"]):
            for old, new in replacements[user["user_id"]].items():
                if text == old:
                    text = new
            user["private_knowledge"][index] = text
    context = review.get("opening_context", "").strip()
    opener = event_map.get("e_open")
    if context:
        if opener is None or not opener.get("fixed_utterance"):
            raise ValueError("public context requires a fixed opening user request")
        if not opener["fixed_utterance"].startswith(context):
            opener["fixed_utterance"] = context + " " + opener["fixed_utterance"]
        opener["fixed_utterance_exact"] = True
        opener["intent"] = "Read the supplied public task context verbatim, then make the opening request. " + opener["intent"]
    criteria = review.get("goal_criteria", {})
    if set(criteria) != set(goal_map) or any(not isinstance(x, str) or not x.strip() for x in criteria.values()):
        raise ValueError(f"review must cover every goal exactly: {case.case_id}")
    for goal_id, criterion in criteria.items():
        goal_map[goal_id]["success_evidence"] = [criterion]
    facts = review.get("required_answer_facts", [])
    if len({fact["id"] for fact in facts}) != len(facts):
        raise ValueError("duplicate required fact IDs")
    for fact in facts:
        if fact["goal_id"] not in goal_map or not fact.get("description"):
            raise ValueError(f"invalid fact obligation: {fact}")
    result = MultiUserVoiceCase.model_validate(payload)
    before_assertions = [(a.assertion_id, a.category, a.judge, a.critical, a.source_event_ids)
                         for a in case.capability_assertions]
    after_assertions = [(a.assertion_id, a.category, a.judge, a.critical, a.source_event_ids)
                        for a in result.capability_assertions]
    if before_assertions != after_assertions:
        raise ValueError("case review altered assertion identity/category/criticality/source")
    after_identity = [(u.user_id, u.display_name, u.voice.model_dump(mode="json")) for u in result.users]
    if before_identity != after_identity or result.seed != case.seed:
        raise ValueError("review unexpectedly altered identity/voice/seed")
    return result


def input_contract(case: Any, review: dict | None = None) -> dict:
    review = review_for_case(case) if review is None else review
    if not review:
        return {"enabled": False, "case_id": case.case_id}
    return {
        "enabled": True, "version": FACT_CONTRACT_VERSION,
        "case_id": case.case_id, "review_sha256": digest(review),
        "case_sha256": digest(case.model_dump(mode="json")),
        "opening_context": review.get("opening_context", ""),
        "input_transport": "provider_initial_instructions_plus_user_audio" if review.get("initial_assistant_context") else "primary_stable_voice_task_audio",
        "system_prompt_fact_injection": bool(review.get("initial_assistant_context")),
        "initial_assistant_context": review.get("initial_assistant_context", {}),
        "review": review,
    }


def validate_frozen_input(case: Any, events: list[Any], *, contract_path: Path | None = None) -> None:
    """Fail before API/TTS use when a reviewed case reuses stale user inputs."""
    contract = input_contract(case)
    if not contract["enabled"]:
        return
    if contract_path is not None:
        if not contract_path.exists():
            raise ValueError("reused frozen plan lacks case_input_contract.json; rebuild Probes")
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if any(previous.get(k) != contract[k] for k in ("case_id", "review_sha256", "case_sha256")):
            raise ValueError("reused frozen plan has a stale task/input contract; rebuild Probes")
    specs = {event.event_id: event for event in case.events}
    for event in events:
        spec = specs.get(event.event_id)
        if spec is not None and spec.fixed_utterance and spec.fixed_utterance_exact:
            if event.utterance != spec.fixed_utterance:
                raise ValueError(f"frozen {event.event_id} differs from reviewed literal user speech")
    if not any(event.event_id == "e_open" for event in events):
        raise ValueError("reviewed input is missing opening speech")
