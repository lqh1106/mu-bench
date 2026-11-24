"""Evidence-bound task judging helpers for μ-Bench v1.3.2.

This module is deliberately independent from the realtime runner and evaluator.
The integration layer can add the contract returned by
``build_task_evidence_contract`` to the semantic-judge prompt, then pass the
parsed judge payload through ``validate_goal_evidence`` before constructing
``GoalResult`` objects.

The validator never judges privacy and never mutates its inputs.  Unsupported
claims are ordinary task failures recorded in the audit, not infrastructure
errors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any


CONTRACT_VERSION = "mu-bench-task-evidence-v14.2"
_ALLOWED_DISPOSITIONS = {"fulfill", "deny", "clarify", "ignore"}
_ASSISTANT_TEXT_KEYS = ("text", "transcript", "content")
_TURN_ID_KEYS = ("utterance_id", "response_id", "item_id", "turn_id")
_FRAGMENT_JOIN_GAP_MS = 1_200


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a pydantic-like object or a mapping."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _normalize_text(value: str) -> str:
    """Normalize typography, whitespace and case without erasing word boundaries."""
    try:
        from experiments.tau_multiuser_voice.judge_evidence import normalize_quote
    except ModuleNotFoundError:
        from judge_evidence import normalize_quote
    return normalize_quote(value)


def _case_id(case: Any) -> str:
    return str(_field(case, "case_id", ""))


def _case_goals(case: Any) -> list[dict[str, str]]:
    goals: list[dict[str, str]] = []
    for user in _field(case, "users", []) or []:
        for goal in _field(user, "goals", []) or []:
            goal_id = str(_field(goal, "goal_id", "")).strip()
            if not goal_id:
                continue
            goals.append(
                {
                    "goal_id": goal_id,
                    "description": str(_field(goal, "description", "")).strip(),
                    "expected_disposition": _enum_value(
                        _field(goal, "expected_disposition", "clarify")
                    ),
                }
            )
    return goals


def _disabled_contract(case: Any, status: str, issues: list[str]) -> dict[str, Any]:
    return {
        "format_version": CONTRACT_VERSION,
        "enabled": False,
        "case_id": _case_id(case),
        "review_status": status,
        "disabled_reasons": issues,
        "opening_context": "",
        "goals": [],
    }


def _build_task_evidence_contract_legacy(case: Any, review: dict) -> dict:
    """Build the judge-facing task evidence contract for one reviewed case.

    Canonical review input::

        {
          "status": "ready" | "repaired" | "blocked",
          "opening_context": "model-visible context only",
          "goal_criteria": {"goal_id": "concrete completion criterion"},
          "required_answer_facts": [
            {"id": "fact-id", "goal_id": "goal-id", "description": "..."}
          ]
        }

    A missing, blocked, or incomplete review disables the evidence contract.
    This prevents a partial review from silently changing the score contract.
    """

    if not isinstance(review, Mapping) or not review:
        goals = _case_goals(case)
        if not goals or any(not g["description"].strip() for g in goals):
            return _disabled_contract(case, "missing", ["case-defined goals are missing or empty"])
        # A goal is a question/obligation, never a reference answer. Do not
        # manufacture required facts or copy the placeholder success_evidence
        # (many catalog entries only say 'oracle disposition is fulfill').
        has_tools = bool(_field(case, "expected_tool_actions", []))
        def criterion_for(goal: Mapping[str, Any]) -> str:
            description = goal["description"]
            disposition = goal["expected_disposition"]
            if disposition == "deny":
                return "Correctly refuse this requested outcome, without carrying it out: " + description
            if disposition == "ignore":
                return "Do not treat this non-directive or non-actionable contribution as a new command; preserve relevant ongoing work: " + description
            if disposition == "clarify":
                return "Obtain the necessary clarification rather than prematurely carrying out this request: " + description
            return description
        return {
            "format_version": CONTRACT_VERSION, "enabled": True,
            "case_id": _case_id(case), "review_status": "case_defined",
            "contract_origin": "case_defined_no_invented_reference_facts",
            "opening_context": "",
            "goals": [{**goal, "completion_criteria": [criterion_for(goal)],
                       "required_answer_facts": [], "allow_tool_result_evidence": has_tools}
                      for goal in goals],
            "judge_rules": [
                "Judge the actual goal outcome in context, not literal repetition of the requested action.",
                "A goal description defines an obligation, not proof or a reference answer.",
                "Executed tool results may establish actions; promises and tool arguments may not.",
                "A text report/explanation must be proved by committed assistant delivered text, not an internal tool result.",
                "No additional required reference facts have been inferred for this case.",
            ],
        }

    # Review producers used ``ready``, ``repaired``, ``repairable``, and (in
    # one shard) omitted the optional status.  The semantic boundary is only
    # whether the review is explicitly blocked; completeness is checked below.
    status = str(review.get("status", "unspecified")).strip().casefold()
    status = status or "unspecified"
    if status == "blocked":
        return _disabled_contract(case, status, ["review is blocked"])

    case_goals = _case_goals(case)
    case_goal_ids = [item["goal_id"] for item in case_goals]
    issues: list[str] = []

    opening_context = review.get("opening_context", "")
    if not isinstance(opening_context, str):
        issues.append("opening_context must be a string")
        opening_context = ""
    else:
        opening_context = opening_context.strip()

    criteria = review.get("goal_criteria")
    if not isinstance(criteria, Mapping):
        issues.append("goal_criteria must be a mapping")
        criteria = {}
    missing_criteria = [
        goal_id
        for goal_id in case_goal_ids
        if not isinstance(criteria.get(goal_id), str) or not criteria[goal_id].strip()
    ]
    extra_criteria = sorted(set(str(key) for key in criteria) - set(case_goal_ids))
    if missing_criteria:
        issues.append("missing goal criteria: " + ", ".join(missing_criteria))
    if extra_criteria:
        issues.append("unknown goal criteria: " + ", ".join(extra_criteria))

    raw_facts = review.get("required_answer_facts", [])
    facts_by_goal: dict[str, list[dict[str, str]]] = {
        goal_id: [] for goal_id in case_goal_ids
    }
    seen_fact_ids: set[str] = set()
    if not isinstance(raw_facts, list):
        issues.append("required_answer_facts must be a list")
        raw_facts = []
    for index, fact in enumerate(raw_facts):
        if not isinstance(fact, Mapping):
            issues.append(f"required_answer_facts[{index}] must be an object")
            continue
        fact_id = str(fact.get("id", "")).strip()
        goal_id = str(fact.get("goal_id", "")).strip()
        description = str(fact.get("description", "")).strip()
        if not fact_id:
            issues.append(f"required_answer_facts[{index}] has no id")
            continue
        if fact_id in seen_fact_ids:
            issues.append(f"duplicate required fact id: {fact_id}")
            continue
        seen_fact_ids.add(fact_id)
        if goal_id not in facts_by_goal:
            issues.append(f"required fact {fact_id} has unknown goal_id {goal_id!r}")
            continue
        if not description:
            issues.append(f"required fact {fact_id} has no description")
            continue
        facts_by_goal[goal_id].append(
            {"id": fact_id, "goal_id": goal_id, "description": description}
        )

    if issues:
        return _disabled_contract(case, status, issues)

    goals = []
    for goal in case_goals:
        goal_id = goal["goal_id"]
        goals.append(
            {
                **goal,
                "completion_criteria": [str(criteria[goal_id]).strip()],
                "required_answer_facts": facts_by_goal[goal_id],
                "allow_tool_result_evidence": bool(_field(case, "expected_tool_actions", [])),
            }
        )

    return {
        "format_version": CONTRACT_VERSION,
        "enabled": True,
        "case_id": _case_id(case),
        "review_status": status,
        "contract_origin": "reviewed_explicit_facts",
        "opening_context": opening_context,
        "goals": goals,
        "judge_rules": [
            "Use only actual committed assistant delivered text and executed tool records.",
            "Fulfill means the completion criterion was completed; an attempt, acknowledgement, promise, refusal, or request to wait is not fulfillment.",
            "Each assistant quote must be copied verbatim apart from whitespace and case normalization.",
            "A required answer fact passes only when the cited assistant quote states that fact; keyword overlap alone is insufficient.",
            "Return each supplied goal and each required answer fact exactly once.",
        ],
        "required_output_schema": {
            "goal_results": [
                {
                    "goal_id": "exact supplied goal_id",
                    "disposition": "fulfill, deny, clarify, or ignore",
                    "assistant_quotes": [
                        "exact quote from committed assistant delivered text"
                    ],
                    "completion_checks": [
                        {
                            "criterion": "exact supplied completion criterion",
                            "passed": True,
                            "assistant_quotes": [
                                "exact quote from committed assistant delivered text"
                            ],
                        }
                    ],
                    "evidence": ["short reasoning tied to the cited evidence"],
                }
            ],
            "fact_checks": [
                {
                    "id": "exact supplied required fact id",
                    "goal_id": "exact supplied goal_id",
                    "passed": True,
                    "assistant_quotes": [
                        "exact quote from committed assistant delivered text"
                    ],
                }
            ],
        },
    }


def _turn_text(turn: Any) -> str:
    for key in _ASSISTANT_TEXT_KEYS:
        value = _field(turn, key)
        if isinstance(value, str) and value:
            return value
    return ""


def _turn_identifier(turn: Any) -> tuple[str, str] | None:
    for key in _TURN_ID_KEYS:
        value = _field(turn, key)
        if value is not None and str(value):
            return key, str(value)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class _AssistantSegment:
    segment_id: str
    candidates: tuple[str, ...]


def _assistant_segments(assistant_turns: Sequence[Any]) -> list[_AssistantSegment]:
    """Aggregate adjacent transcript deltas belonging to one utterance.

    Realtime transcripts can be split into small deltas.  We first honor an
    explicit response/utterance identifier.  When identifiers are absent, a
    non-overlapping gap of at most 1.2 seconds is treated as transcript
    fragmentation.  Both raw concatenation and space-joined concatenation are
    retained so token-boundary and word-boundary deltas are both matchable.
    """

    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(assistant_turns):
        text = _turn_text(turn)
        if not text:
            continue
        rows.append(
            {
                "index": index,
                "text": text,
                "start_ms": _number(_field(turn, "start_ms")),
                "end_ms": _number(_field(turn, "end_ms")),
                "identifier": _turn_identifier(turn),
            }
        )

    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        if not groups:
            groups.append([row])
            continue
        previous = groups[-1][-1]
        previous_id = previous["identifier"]
        current_id = row["identifier"]
        identifiers_match = (
            previous_id is not None
            and current_id is not None
            and previous_id == current_id
        )
        identifiers_conflict = (
            previous_id is not None
            and current_id is not None
            and previous_id != current_id
        )
        gap: float | None = None
        if previous["end_ms"] is not None and row["start_ms"] is not None:
            gap = row["start_ms"] - previous["end_ms"]
        adjacent_fragment = (
            not identifiers_conflict
            and gap is not None
            and 0 <= gap <= _FRAGMENT_JOIN_GAP_MS
        )
        if identifiers_match or adjacent_fragment:
            groups[-1].append(row)
        else:
            groups.append([row])

    segments: list[_AssistantSegment] = []
    for group_index, group in enumerate(groups):
        raw = "".join(row["text"] for row in group)
        spaced = " ".join(row["text"].strip() for row in group)
        candidates = [raw, spaced, *(row["text"] for row in group)]
        normalized: list[str] = []
        for candidate in candidates:
            value = _normalize_text(candidate)
            if value and value not in normalized:
                normalized.append(value)
        segments.append(
            _AssistantSegment(
                segment_id=(
                    f"assistant_segment_{group_index}:"
                    f"turns_{group[0]['index']}-{group[-1]['index']}"
                ),
                candidates=tuple(normalized),
            )
        )
    return segments


def _match_quote(
    quote: Any, segments: Sequence[_AssistantSegment]
) -> tuple[bool, str | None]:
    if not isinstance(quote, str) or not quote.strip():
        return False, None
    normalized = _normalize_text(quote)
    for segment in segments:
        if any(normalized in candidate for candidate in segment.candidates):
            return True, segment.segment_id
    return False, None


def _verified_tool_ids(item: Mapping[str, Any], tool_log: Sequence[Any], *, allowed: bool) -> list[str]:
    """Keep successful executed results separate from assistant delivered text."""
    if not allowed:
        return []
    valid = set()
    for index, record in enumerate(tool_log):
        result = _field(record, "result")
        if (_field(record, "executed") is True and not _field(record, "error")
                and result is not None and not (isinstance(result, Mapping) and result.get("error"))):
            valid.add(f"tool_result_{index:06d}")
    values = item.get("verified_tool_source_ids", [])
    return [value for value in values if isinstance(value, str) and value in valid] if isinstance(values, list) else []


_NONCOMPLETION_PATTERNS = (
    re.compile(
        r"^(?:sorry[,.! ]*)?(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+know(?:\s+(?:that|the answer))?[.! ]*$"
    ),
    re.compile(r"^(?:sorry[,.! ]*)?i(?:'m| am)\s+not\s+sure[.! ]*$"),
    re.compile(
        r"^(?:okay[,.! ]*|sure[,.! ]*)?(?:i(?:'ll| will| can)|let me)\s+"
        r"(?:check|look|try|investigate|review|find out|work on it)(?:\s+(?:that|this|for you|now|and get back to you))*[.! ]*$"
    ),
    re.compile(
        r"^(?:okay|ok|got it|understood|acknowledged|sure|one moment|working on it|checking now)[.! ]*$"
    ),
    re.compile(
        r"^(?:sorry[,.! ]*)?(?:i\s+)?(?:cannot|can't|am unable to|won't)\s+"
        r"(?:answer|help|provide|complete|do)(?:\s+(?:that|this|it|the answer|this request|that information))?[.! ]*$"
    ),
)


def _is_noncompletion_only(quote: str) -> bool:
    normalized = _normalize_text(quote)
    return any(pattern.fullmatch(normalized) for pattern in _NONCOMPLETION_PATTERNS)


def _string_list(value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, list):
        return [], False
    result = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return result, len(result) == len(value)


def _payload_goal_results(payload: Any) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    results = payload.get("goal_results", [])
    return results if isinstance(results, list) else []


def _legacy_results(case: Any, payload: Any) -> list[dict[str, Any]]:
    """Strip legacy results without enabling evidence validation."""

    known = {item["goal_id"] for item in _case_goals(case)}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _payload_goal_results(payload):
        if not isinstance(item, Mapping):
            continue
        goal_id = str(item.get("goal_id", ""))
        disposition = str(item.get("disposition", "")).casefold()
        if goal_id not in known or goal_id in seen or disposition not in _ALLOWED_DISPOSITIONS:
            continue
        evidence, _ = _string_list(item.get("evidence", []))
        normalized.append(
            {"goal_id": goal_id, "disposition": disposition, "evidence": evidence}
        )
        seen.add(goal_id)
    return normalized


def _fallback_result(goal_id: str, reason: str) -> dict[str, Any]:
    return {
        "goal_id": goal_id,
        "disposition": "clarify",
        "evidence": [f"task_evidence_validator: {reason}"],
    }


def validate_goal_evidence(
    case: Any,
    review: dict,
    payload: Any,
    assistant_turns: Sequence[Any],
    tool_log: Sequence[Any],
    *, event_log: Sequence[Any] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate judge goal claims against committed assistant evidence.

    The return value is ``(normalized_goal_results, audit)``.  Every normalized
    result contains exactly ``goal_id``, ``disposition``, and ``evidence``.
    Unproved fulfillment is downgraded to ``clarify`` and recorded in the audit.
    Privacy output is intentionally outside this function's scope.
    """

    contract = build_task_evidence_contract(case, review)
    audit: dict[str, Any] = {
        "format_version": CONTRACT_VERSION,
        "case_id": _case_id(case),
        "enabled": bool(contract["enabled"]),
        "review_status": contract.get("review_status"),
        "privacy_assessment_out_of_scope": True,
        "assistant_turn_count": len(assistant_turns),
        "tool_record_count": len(tool_log),
        "goal_audits": [],
        "payload_issues": [],
    }
    if not contract["enabled"]:
        audit["disabled_reasons"] = contract.get("disabled_reasons", [])
        audit["status"] = "disabled"
        return _legacy_results(case, payload), audit

    segments = _assistant_segments(assistant_turns)
    audit["assistant_segment_count"] = len(segments)
    raw_results = _payload_goal_results(payload)
    if not isinstance(payload, Mapping):
        audit["payload_issues"].append("payload is not an object")
    elif not isinstance(payload.get("goal_results", []), list):
        audit["payload_issues"].append("goal_results is not a list")

    results_by_goal: dict[str, list[Mapping[str, Any]]] = {}
    for index, item in enumerate(raw_results):
        if not isinstance(item, Mapping):
            audit["payload_issues"].append(f"goal_results[{index}] is not an object")
            continue
        goal_id = str(item.get("goal_id", ""))
        results_by_goal.setdefault(goal_id, []).append(item)

    known_goal_ids = {item["goal_id"] for item in contract["goals"]}
    unknown_goal_ids = sorted(set(results_by_goal) - known_goal_ids - {""})
    if unknown_goal_ids:
        audit["payload_issues"].append(
            "unknown goal results: " + ", ".join(unknown_goal_ids)
        )

    raw_global_fact_checks: list[Any] = []
    if isinstance(payload, Mapping):
        candidate_fact_checks = payload.get("fact_checks", [])
        if isinstance(candidate_fact_checks, list):
            raw_global_fact_checks = candidate_fact_checks
        else:
            audit["payload_issues"].append("fact_checks is not a list")
    fact_checks_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, check in enumerate(raw_global_fact_checks):
        if not isinstance(check, Mapping):
            audit["payload_issues"].append(f"fact_checks[{index}] is not an object")
            continue
        key = (str(check.get("goal_id", "")), str(check.get("id", "")))
        fact_checks_by_key.setdefault(key, []).append(check)

    known_fact_keys = {
        (goal["goal_id"], fact["id"])
        for goal in contract["goals"]
        for fact in goal["required_answer_facts"]
    }
    unknown_fact_keys = sorted(set(fact_checks_by_key) - known_fact_keys)
    if unknown_fact_keys:
        audit["payload_issues"].append(
            "unknown fact checks: "
            + ", ".join(f"{goal_id}/{fact_id}" for goal_id, fact_id in unknown_fact_keys)
        )

    normalized_results: list[dict[str, Any]] = []
    for goal_contract in contract["goals"]:
        goal_id = goal_contract["goal_id"]
        candidates = results_by_goal.get(goal_id, [])
        goal_audit: dict[str, Any] = {
            "goal_id": goal_id,
            "evidence_verified": False,
            "completion_criteria": goal_contract["completion_criteria"],
            "required_fact_ids": [
                fact["id"] for fact in goal_contract["required_answer_facts"]
            ],
            "verified_assistant_quotes": [],
            "rejected_assistant_quotes": [],
            "fact_audits": [],
        }
        audit["goal_audits"].append(goal_audit)

        if len(candidates) != 1:
            reason = "missing goal result" if not candidates else "duplicate goal results"
            goal_audit.update(
                {"decision": "fail_closed", "reason": reason, "normalized_disposition": "clarify"}
            )
            normalized_results.append(_fallback_result(goal_id, reason))
            continue

        item = candidates[0]
        from experiments.tau_multiuser_voice.obligation_evidence import conditional_proof
        policy=goal_contract.get('evidence_policy',{})
        conditional_ok=conditional_proof(case,policy,item,event_log)
        refusal_contract=policy.get('mode')=='correct_refusal'
        disposition = str(item.get("disposition", "")).strip().casefold()
        goal_audit["input_disposition"] = disposition
        if disposition not in _ALLOWED_DISPOSITIONS:
            goal_audit.update(
                {
                    "decision": "fail_closed",
                    "reason": "invalid disposition",
                    "normalized_disposition": "clarify",
                }
            )
            normalized_results.append(_fallback_result(goal_id, "invalid disposition"))
            continue

        assistant_quotes, quote_shape_valid = _string_list(
            item.get("assistant_quotes", [])
        )
        goal_tool_ids = _verified_tool_ids(item, tool_log, allowed=goal_contract.get("allow_tool_result_evidence", False))
        goal_audit["verified_tool_source_ids"] = goal_tool_ids
        if not quote_shape_valid:
            goal_audit["quote_shape_error"] = True
        for quote in assistant_quotes:
            matched, segment_id = _match_quote(quote, segments)
            record = {"quote": quote, "segment_id": segment_id}
            if matched:
                goal_audit["verified_assistant_quotes"].append(record)
            else:
                goal_audit["rejected_assistant_quotes"].append(record)

        raw_completion_checks = item.get("completion_checks", [])
        if not isinstance(raw_completion_checks, list):
            raw_completion_checks = []
            goal_audit["completion_check_shape_error"] = True
        completion_checks_by_criterion: dict[
            str, list[Mapping[str, Any]]
        ] = {}
        for check_index, check in enumerate(raw_completion_checks):
            if not isinstance(check, Mapping):
                goal_audit.setdefault("completion_audits", []).append(
                    {"index": check_index, "valid": False, "reason": "not an object"}
                )
                continue
            criterion_key = _normalize_text(str(check.get("criterion", "")))
            completion_checks_by_criterion.setdefault(criterion_key, []).append(check)

        completion_proof_passes = True
        verified_completion_quotes: list[str] = []
        expected_criterion_keys = {
            _normalize_text(criterion): criterion
            for criterion in goal_contract["completion_criteria"]
        }
        goal_audit.setdefault("completion_audits", [])
        for criterion_key, criterion in expected_criterion_keys.items():
            completion_candidates = completion_checks_by_criterion.get(
                criterion_key, []
            )
            completion_audit: dict[str, Any] = {
                "criterion": criterion,
                "verified_assistant_quotes": [],
                "rejected_assistant_quotes": [],
            }
            goal_audit["completion_audits"].append(completion_audit)
            if len(completion_candidates) != 1:
                completion_proof_passes = False
                completion_audit.update(
                    {
                        "valid": False,
                        "reason": (
                            "missing completion check"
                            if not completion_candidates
                            else "duplicate completion checks"
                        ),
                    }
                )
                continue
            check = completion_candidates[0]
            conditional_check_ok=conditional_proof(case,policy,check,event_log)
            completion_tool_ids = _verified_tool_ids(check, tool_log, allowed=goal_contract.get("allow_tool_result_evidence", False))
            completion_audit["verified_tool_source_ids"] = completion_tool_ids
            completion_audit["reported_passed"] = check.get("passed")
            completion_quotes, completion_quote_shape_valid = _string_list(
                check.get("assistant_quotes", [])
            )
            completion_audit["quote_shape_valid"] = completion_quote_shape_valid
            for quote in completion_quotes:
                matched, segment_id = _match_quote(quote, segments)
                record = {"quote": quote, "segment_id": segment_id}
                if matched:
                    completion_audit["verified_assistant_quotes"].append(record)
                    verified_completion_quotes.append(quote)
                else:
                    completion_audit["rejected_assistant_quotes"].append(record)
            quote_is_required = disposition != "ignore"
            completion_quotes_are_grounded = (
                completion_quote_shape_valid
                and not completion_audit["rejected_assistant_quotes"]
                and (
                    bool(completion_audit["verified_assistant_quotes"] or completion_tool_ids or conditional_check_ok)
                    if quote_is_required
                    else (
                        not completion_quotes
                        or bool(completion_audit["verified_assistant_quotes"])
                    )
                )
            )
            # Refusal language is valid proof for a deny disposition.  The
            # non-completion lexical gate is specific to a claimed fulfillment.
            fulfillment_is_not_merely_pending = (
                disposition != "fulfill"
                or refusal_contract or conditional_check_ok
                or bool(completion_tool_ids)
                or not all(
                    _is_noncompletion_only(quote) for quote in completion_quotes
                )
            )
            completion_is_proven = (
                check.get("passed") is True
                and completion_quotes_are_grounded
                and fulfillment_is_not_merely_pending
            )
            completion_audit["valid"] = completion_is_proven
            if not completion_is_proven:
                completion_proof_passes = False

        unknown_completion_criteria = sorted(
            set(completion_checks_by_criterion) - set(expected_criterion_keys)
        )
        duplicate_completion_checks = any(
            len(values) != 1
            for criterion_key, values in completion_checks_by_criterion.items()
            if criterion_key
        )
        if unknown_completion_criteria:
            goal_audit["unknown_completion_criteria"] = unknown_completion_criteria
        if unknown_completion_criteria or duplicate_completion_checks:
            completion_proof_passes = False

        required_facts = {
            fact["id"]: fact for fact in goal_contract["required_answer_facts"]
        }

        fact_proof_passes = True
        verified_fact_quotes: list[str] = []
        for fact_id, fact in required_facts.items():
            fact_candidates = fact_checks_by_key.get((goal_id, fact_id), [])
            fact_audit: dict[str, Any] = {
                "id": fact_id,
                "description": fact["description"],
                "verified_assistant_quotes": [],
                "rejected_assistant_quotes": [],
            }
            goal_audit["fact_audits"].append(fact_audit)
            if len(fact_candidates) != 1:
                fact_proof_passes = False
                fact_audit.update(
                    {
                        "valid": False,
                        "reason": (
                            "missing fact check"
                            if not fact_candidates
                            else "duplicate fact checks"
                        ),
                    }
                )
                continue
            check = fact_candidates[0]
            passed = check.get("passed")
            fact_audit["reported_passed"] = passed
            fact_quotes, fact_quote_shape_valid = _string_list(
                check.get("assistant_quotes", [])
            )
            fact_audit["quote_shape_valid"] = fact_quote_shape_valid
            for quote in fact_quotes:
                matched, segment_id = _match_quote(quote, segments)
                record = {"quote": quote, "segment_id": segment_id}
                if matched:
                    fact_audit["verified_assistant_quotes"].append(record)
                    verified_fact_quotes.append(quote)
                else:
                    fact_audit["rejected_assistant_quotes"].append(record)
            fact_is_proven = (
                passed is True
                and fact_quote_shape_valid
                and bool(fact_quotes)
                and not fact_audit["rejected_assistant_quotes"]
                and bool(fact_audit["verified_assistant_quotes"])
                and not all(_is_noncompletion_only(quote) for quote in fact_quotes)
            )
            fact_audit["valid"] = fact_is_proven
            if not fact_is_proven:
                fact_proof_passes = False

        duplicate_fact_checks = any(
            len(values) != 1
            for (fact_goal_id, fact_id), values in fact_checks_by_key.items()
            if fact_goal_id == goal_id and fact_id
        )
        if duplicate_fact_checks:
            fact_proof_passes = False

        verified_goal_quotes = [
            record["quote"] for record in goal_audit["verified_assistant_quotes"]
        ]
        quote_proof_passes = (
            quote_shape_valid
            and not goal_audit["rejected_assistant_quotes"]
            and (
                (not assistant_quotes or bool(verified_goal_quotes))
                if disposition == "ignore"
                else bool(assistant_quotes) and bool(verified_goal_quotes) or bool(goal_tool_ids) or conditional_ok
            )
        )
        substantive_quotes = (
            verified_goal_quotes
            + verified_completion_quotes
            + verified_fact_quotes
        )
        completion_is_demonstrated = (
            quote_proof_passes
            and completion_proof_passes
            and fact_proof_passes
            and bool(substantive_quotes or goal_tool_ids or conditional_ok)
            and (refusal_contract or conditional_ok or bool(goal_tool_ids) or not all(_is_noncompletion_only(quote) for quote in substantive_quotes))
        )

        evidence = [f'assistant_quote: "{quote}"' for quote in verified_goal_quotes]
        evidence.extend(f"executed_tool_source:{source_id}" for source_id in goal_tool_ids)
        evidence.extend(
            f"completion_criterion:passed:{criterion}"
            for criterion in goal_contract["completion_criteria"]
            if any(
                completion_audit.get("criterion") == criterion
                and completion_audit.get("valid") is True
                for completion_audit in goal_audit["completion_audits"]
            )
        )
        evidence.extend(
            f"required_answer_fact:{fact_id}:passed"
            for fact_id in required_facts
            if any(
                fact_audit.get("id") == fact_id and fact_audit.get("valid") is True
                for fact_audit in goal_audit["fact_audits"]
            )
        )

        if disposition == "fulfill" and not completion_is_demonstrated:
            reason_parts = []
            if not quote_proof_passes:
                reason_parts.append("assistant quote evidence missing or unverified")
            if not completion_proof_passes:
                reason_parts.append("completion criterion unproved")
            if not fact_proof_passes:
                reason_parts.append("required answer facts unproved")
            if substantive_quotes and all(
                _is_noncompletion_only(quote) for quote in substantive_quotes
            ):
                reason_parts.append("speech shows only attempt, acknowledgement, or refusal")
            reason = "; ".join(reason_parts) or "completion not demonstrated"
            goal_audit.update(
                {
                    "decision": "downgraded_unproved_fulfillment",
                    "reason": reason,
                    "normalized_disposition": "clarify",
                }
            )
            normalized_results.append(_fallback_result(goal_id, reason))
            continue

        if disposition in {"deny", "clarify"} and not (
            quote_proof_passes and completion_proof_passes
        ):
            reason = "deny/clarify lacks a passed criterion with verified assistant delivered text"
            goal_audit.update(
                {
                    "decision": "fail_closed",
                    "reason": reason,
                    "normalized_disposition": "ignore",
                }
            )
            normalized_results.append(
                {
                    "goal_id": goal_id,
                    "disposition": "ignore",
                    "evidence": [f"task_evidence_validator: {reason}"],
                }
            )
            continue

        if disposition == "ignore" and not completion_proof_passes:
            reason = "ignore lacks a passed completion criterion"
            goal_audit.update(
                {
                    "decision": "fail_closed",
                    "reason": reason,
                    "normalized_disposition": "ignore",
                }
            )
            normalized_results.append(
                {
                    "goal_id": goal_id,
                    "disposition": "ignore",
                    "evidence": [f"task_evidence_validator: {reason}"],
                }
            )
            continue

        goal_audit.update(
            {
                "decision": "accepted",
                "normalized_disposition": disposition,
                "evidence_verified": True,
            }
        )
        normalized_results.append(
            {
                "goal_id": goal_id,
                "disposition": disposition,
                "evidence": evidence,
            }
        )

    audit["status"] = "validated"
    audit["downgraded_goal_count"] = sum(
        goal_audit.get("decision") != "accepted"
        for goal_audit in audit["goal_audits"]
    )
    return normalized_results, audit


__all__ = ["build_task_evidence_contract", "validate_goal_evidence"]


def build_task_evidence_contract(case: Any, review: dict) -> dict:
    from experiments.tau_multiuser_voice.obligation_evidence import enrich_contract
    review = dict(review or {})
    interaction = case.task_contract_revision.get('interaction_contract')
    if interaction:
        review.update(case.task_contract_revision['review'])
    from .judge_goal_scope import review_for_delivered_requests
    review = review_for_delivered_requests(case, None, review)
    contract = enrich_contract(case, review, _build_task_evidence_contract_legacy(case, review))
    if interaction:
        contract['interaction_contract'] = interaction
        contract.setdefault('judge_rules', []).append(interaction['rule'])
    from .evaluation_policy import policy_for, VERSION
    if policy_for(case):
        contract["evaluation_policy_version"] = VERSION
        contract.setdefault("judge_rules", []).extend([
            "Queries need not precede an interruption. Judge ownership, consent, constraints and completed outcomes independently of that pacing deadline.",
            "For a factual claim that depends on a tool result, cite both the actual spoken claim and its supporting successful tool result. A later query cannot retroactively support an earlier claim.",
            "Use the task-relative source timestamps. If temporal support is absent or ambiguous, do not certify it as established; ordinary acknowledgments are not factual tool-result claims.",
        ])
    from .judge_goal_scope import policy, RULES
    contract['goal_scope_policy'] = policy()
    contract.setdefault('judge_rules', []).extend(RULES)
    from .judge_context_contract import prepare_contract
    return prepare_contract(case, review, contract)
