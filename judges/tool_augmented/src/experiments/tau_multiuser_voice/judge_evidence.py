"""Trusted evidence references and strict Judge-output validity for v14.2.

This module performs no model calls and never invents evidence. A failed semantic
check is a model outcome; malformed or missing Judge evidence is a different
terminal state and must not be converted to a model score.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import re
import json
import math
import unicodedata
from typing import Any

VERSION = "mu-judge-evidence-v14.2-r1-relaxed-retry"

# Targeted A-type re-ask policy (2026-10-05). VERSION left unchanged (pinned as
# format_version in audits); traceability uses audit fields + TARGETED_REASK_POLICY_ID.
TARGETED_REASK_POLICY_ID = "mu-judge-targeted-reask-a-type-v1"
MISSING_COMPLETION_PASSED_REASON = (
    "passed must be explicit boolean; missing/null is not false or NA"
)
_MISSING_PASSED_PATH_RE = re.compile(
    r"^goal_results\[(?P<goal_id>[^\]]+)\]\.completion_checks\[(?P<criterion>.+)\]\.passed$"
)


def coerce_explicit_bool(value: Any) -> bool | None:
    """Map only explicit bool-like values to bool; else None (still invalid).

    Accepted: bool; str true/false (case-insensitive, trimmed); int/float 1/0.
    Rejected (return None): None, empty string, other strings (yes/no/maybe),
    other numbers, dicts/lists, etc. Never treats null/missing as False.
    """
    if type(value) is bool:
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        return None
    if type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is float and value in (0.0, 1.0):
        return bool(int(value))
    return None






def is_only_missing_completion_passed_issues(issues: Sequence[Any] | None) -> bool:
    """True iff every issue is an A-type missing/null completion_checks[].passed.

    Mixed issue lists (missing passed + anything else) return False so the
    caller keeps the existing full-object repair path.
    """
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)) or not issues:
        return False
    for item in issues:
        if not isinstance(item, Mapping):
            return False
        if str(item.get("reason", "")) != MISSING_COMPLETION_PASSED_REASON:
            return False
        path = str(item.get("path", ""))
        if _MISSING_PASSED_PATH_RE.match(path) is None:
            return False
    return True


def parse_missing_completion_passed_items(issues: Sequence[Any] | None) -> list[dict[str, str]]:
    """Extract {goal_id, criterion, path} for A-type missing-passed issues."""
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return items
    for item in issues:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("reason", "")) != MISSING_COMPLETION_PASSED_REASON:
            continue
        path = str(item.get("path", ""))
        m = _MISSING_PASSED_PATH_RE.match(path)
        if m is None:
            continue
        goal_id = m.group("goal_id")
        criterion = m.group("criterion")
        key = (goal_id, criterion)
        if key in seen:
            continue
        seen.add(key)
        items.append({"goal_id": goal_id, "criterion": criterion, "path": path})
    return items



# F1 (2026-10-05 proposal): targeted entailment-string re-ask (missing explanation fields only).
TARGETED_ENTAILMENT_STRING_REASK_POLICY_ID = "mu-judge-targeted-entailment-string-reask-v1"

# Ignore handoff: pack full catalog source_ids into candidate_source_ids for
# stays; this only repairs the second-judge handoff gap.
IGNORE_HANDOFF_FULL_SOURCES_POLICY_ID = "mu-judge-ignore-handoff-full-sources-v1"
IGNORE_HANDOFF_SUPPORT_KINDS = [
    "assistant_speech",
    "user_event",
    "timing_measurement",
    "attribution_report",
]
DISAGREEMENT_REASON = (
    "independent Judge disagreement; neither automatic pass nor model failure"
)
THIRD_JUDGE_POLICY_ID = "mu-judge-third-adjudication-v1"
THIRD_JUDGE_PROMPT_VERSION = "third-judge-prompt-v1"
THIRD_JUDGE_FORMAT_VERSION = "mu-judge-third-adjudication-v1"
THIRD_JUDGE_SCORE_LABEL = "adjudicated_correlated_gpt4o"
THIRD_JUDGE_MISSING_SOURCES_REASON = "third_judge_missing_sources"
THIRD_JUDGE_FORMAT_FAIL_REASON = "third_judge_format_fail"
THIRD_JUDGE_TRIGGER_DISAGREE = "disagree"
THIRD_JUDGE_TRIGGER_PRIMARY_JEI = "primary_jei"
PRIMARY_JEI_THIRD_PROMPT_VERSION = "third-judge-primary-jei-prompt-v1"
PRIMARY_JEI_THIRD_FORMAT_VERSION = "mu-judge-third-primary-jei-v1"
# Format-class primary JEI reasons that may hand off to third (additive gate).
PRIMARY_FORMAT_JEI_REASON_SUBSTRINGS = (
    "unknown or duplicate criterion",
    "passed must be explicit boolean",
    "missing/null is not false",
    "required list missing",
    "Judge payload must be an object",
    "missing required",
    "malformed",
    "missing independent check",
    "missing completion",
    "anchor_kind must be",
    "failure_kind must be",
    "invalid or missing disposition",
    "positive capability lacks",
    "negative capability requires",
    "unknown or non-event source",
    "ungrounded",
    "invented",
    "duplicate",
)

INDEPENDENT_EXPLANATION_MISSING_REASON = "independent explanation missing"
ENTAILMENT_EXPLANATION_STRING_KEYS = (
    "actual_subject",
    "actual_claim",
    "required_claim",
    "reason",
)
_MISSING_EXPLANATION_PATH_RE = re.compile(
    r"^checks\[(?P<index>\d+)\]\.(?P<field>actual_subject|actual_claim|required_claim|reason)$"
)


def is_only_missing_entailment_explanation_issues(issues: Sequence[Any] | None) -> bool:
    """True iff every issue is an independent-explanation-missing string field.

    Mixed issue lists return False so the caller keeps full-response repair.
    """
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)) or not issues:
        return False
    for item in issues:
        if not isinstance(item, Mapping):
            return False
        if str(item.get("reason", "")) != INDEPENDENT_EXPLANATION_MISSING_REASON:
            return False
        path = str(item.get("path", ""))
        if _MISSING_EXPLANATION_PATH_RE.match(path) is None:
            return False
    return True



def is_only_disagreement_issues(issues: Sequence[Any] | None) -> bool:
    """True iff every issue is a two-judge disagreement (no mixed format defects).

    Used to gate third-judge adjudication: only pure disagreement JEI may enter
    the correlated gpt-4o adjudicator. Mixed lists keep the existing JEI path.
    """
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)) or not issues:
        return False
    for item in issues:
        if not isinstance(item, Mapping):
            return False
        if str(item.get("reason", "")) != DISAGREEMENT_REASON:
            return False
    return True


def is_primary_format_class_issues(issues: Sequence[Any] | None) -> bool:
    """True if issues look like primary-judge format/schema defects (Jay 2026-10-05).

    Used to gate the parallel primary-JEI → third path (criterion mismatch/错挂,
    passed not boolean, missing fields, etc.). Disagreement-only lists return False
    (those stay on the disagree → third path). Empty/unknown lists return False.
    """
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)) or not issues:
        return False
    if is_only_disagreement_issues(issues):
        return False
    for item in issues:
        if not isinstance(item, Mapping):
            return False
        reason = str(item.get("reason", ""))
        if not reason:
            return False
        if reason == DISAGREEMENT_REASON:
            return False
        if not any(s.lower() in reason.lower() for s in PRIMARY_FORMAT_JEI_REASON_SUBSTRINGS):
            # Unknown reason: still allow if path looks like schema (goal_results/fact/capability)
            path = str(item.get("path", ""))
            if not any(p in path for p in ("goal_results", "fact_checks", "capability_results", "completion_checks", "$")):
                return False
    return True


def catalog_source_ids(catalog: Mapping[str, Any] | None) -> list[str]:
    """Stable unique source_id list from an evidence catalog or request sources[]."""
    if not isinstance(catalog, Mapping):
        return []
    raw = catalog.get("sources")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for src in raw:
        if not isinstance(src, Mapping):
            continue
        sid = src.get("source_id")
        if isinstance(sid, str) and sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def parse_missing_entailment_explanation_items(
    issues: Sequence[Any] | None,
    payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract {claim_id, field, path, index} for missing explanation-string issues."""
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    checks = []
    if isinstance(payload, Mapping) and isinstance(payload.get("checks"), list):
        checks = payload["checks"]
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return items
    for item in issues:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("reason", "")) != INDEPENDENT_EXPLANATION_MISSING_REASON:
            continue
        path = str(item.get("path", ""))
        m = _MISSING_EXPLANATION_PATH_RE.match(path)
        if m is None:
            continue
        index = int(m.group("index"))
        field = m.group("field")
        claim_id = None
        if 0 <= index < len(checks) and isinstance(checks[index], Mapping):
            cid = checks[index].get("claim_id")
            if isinstance(cid, str):
                claim_id = cid
        key = (claim_id or f"#{index}", field)
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "claim_id": claim_id,
            "field": field,
            "path": path,
            "index": index,
        })
    return items


def merge_targeted_entailment_string_answers(
    payload: Mapping[str, Any],
    answers: Sequence[Any],
    *,
    freeze_from: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge only missing explanation strings; freeze booleans and sources.

    ``freeze_from`` (defaults to ``payload``) is the first/base answer whose
    ``entails``, ``contradiction_found``, and ``evidence_source_ids`` are kept.
    If an answer row tries to change those, the change is ignored and logged in
    the returned ``ignored_changes`` list.
    """
    base = freeze_from if isinstance(freeze_from, Mapping) else payload
    merged = deepcopy(dict(payload if isinstance(payload, Mapping) else {}))
    checks_in = merged.get("checks")
    if not isinstance(checks_in, list):
        checks_in = []
        merged["checks"] = checks_in
    freeze_checks = []
    if isinstance(base, Mapping) and isinstance(base.get("checks"), list):
        freeze_checks = base["checks"]

    by_claim: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for i, row in enumerate(checks_in):
        if isinstance(row, dict):
            by_index[i] = row
            cid = row.get("claim_id")
            if isinstance(cid, str):
                by_claim[cid] = row

    freeze_by_claim: dict[str, Mapping[str, Any]] = {}
    freeze_by_index: dict[int, Mapping[str, Any]] = {}
    for i, row in enumerate(freeze_checks):
        if isinstance(row, Mapping):
            freeze_by_index[i] = row
            cid = row.get("claim_id")
            if isinstance(cid, str):
                freeze_by_claim[cid] = row

    ignored: list[dict[str, Any]] = []
    for ans in answers:
        if not isinstance(ans, Mapping):
            continue
        field = ans.get("field")
        value = ans.get("value")
        if field not in ENTAILMENT_EXPLANATION_STRING_KEYS:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        claim_id = ans.get("claim_id")
        index = ans.get("index")
        row = None
        freeze_row = None
        if isinstance(claim_id, str) and claim_id in by_claim:
            row = by_claim[claim_id]
            freeze_row = freeze_by_claim.get(claim_id)
        elif isinstance(index, int) and index in by_index:
            row = by_index[index]
            freeze_row = freeze_by_index.get(index)
        if row is None:
            continue
        row[field] = value.strip()
        # Re-assert frozen verdict/source fields from first answer.
        if isinstance(freeze_row, Mapping):
            for fk in ("entails", "contradiction_found", "evidence_source_ids"):
                if fk in freeze_row:
                    # Detect if answer tried to change them
                    if fk in ans and ans.get(fk) != freeze_row.get(fk):
                        ignored.append({
                            "claim_id": claim_id,
                            "field": fk,
                            "ignored_value": ans.get(fk),
                            "kept_value": freeze_row.get(fk),
                        })
                    row[fk] = deepcopy(freeze_row.get(fk))
            # Also ignore top-level boolean/source keys if present on answer blob
            for fk in ("entails", "contradiction_found", "evidence_source_ids"):
                if fk in ans and ans.get(fk) != freeze_row.get(fk):
                    ignored.append({
                        "claim_id": claim_id,
                        "field": fk,
                        "ignored_value": ans.get(fk),
                        "kept_value": freeze_row.get(fk),
                        "note": "answer_blob",
                    })
    return merged, ignored


def merge_targeted_completion_check_answers(
    payload: Mapping[str, Any],
    answers: Sequence[Any],
) -> dict[str, Any]:
    """Merge targeted re-ask answers into a deep copy of the judge payload.

    Only updates completion_checks rows matching (goal_id, criterion). Does not
    invent goals. Leaves coerce/validation to validate_and_bind_judge_payload.
    """
    merged = deepcopy(dict(payload))
    goals = merged.get("goal_results")
    if not isinstance(goals, list):
        goals = []
        merged["goal_results"] = goals
    by_goal: dict[str, dict[str, Any]] = {}
    for row in goals:
        if isinstance(row, dict) and isinstance(row.get("goal_id"), str):
            by_goal[row["goal_id"]] = row
    for ans in answers:
        if not isinstance(ans, Mapping):
            continue
        goal_id = ans.get("goal_id")
        criterion = ans.get("criterion")
        if not isinstance(goal_id, str) or not isinstance(criterion, str):
            continue
        row = by_goal.get(goal_id)
        if row is None:
            row = {"goal_id": goal_id, "completion_checks": []}
            goals.append(row)
            by_goal[goal_id] = row
        checks = row.get("completion_checks")
        if not isinstance(checks, list):
            checks = []
            row["completion_checks"] = checks
        target = None
        for check in checks:
            if isinstance(check, dict) and check.get("criterion") == criterion:
                target = check
                break
        if target is None:
            target = {"criterion": criterion}
            checks.append(target)
        if "passed" in ans:
            target["passed"] = ans.get("passed")
        for key in (
            "reason",
            "assistant_quotes",
            "assistant_evidence",
            "tool_evidence",
            "evidence",
        ):
            if key in ans:
                target[key] = ans.get(key)
        target.setdefault("assistant_quotes", ans.get("assistant_quotes", []) if "assistant_quotes" in ans else [])
        target.setdefault("assistant_evidence", ans.get("assistant_evidence", []) if "assistant_evidence" in ans else [])
        target.setdefault("tool_evidence", ans.get("tool_evidence", []) if "tool_evidence" in ans else [])
    return merged


class JudgeEvidenceInvalid(ValueError):
    reason = "judge_evidence_invalid"

    def __init__(self, issues: list[dict[str, Any]], audit: dict[str, Any] | None = None):
        self.issues = issues
        self.audit = audit or {"format_version": VERSION, "evaluation_valid": False, "issues": issues}
        super().__init__("judge evidence invalid: " + "; ".join(str(x.get("reason", x)) for x in issues[:8]))

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "issues": self.issues, "audit": self.audit}


def normalize_quote(value: str) -> str:
    """Normalize typography, whitespace and case, not words or their boundaries."""
    # Quote style is not a semantic distinction. Preserve apostrophes as quote
    # characters rather than deleting them, so contractions cannot lose words.
    table = str.maketrans({c: "'" for c in "'\"‘’‚‛“”„‟＇＂"})
    return " ".join(unicodedata.normalize("NFC", value).translate(table).split()).casefold()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def capability_support_source_kinds(assertion: Any) -> list[str]:
    """Minimum SUT-side evidence, independent of the Judge's own verdict.

    Hearing a user request does not establish that the assistant handled it.
    Silence/turn behavior can instead be proved by real timing or protocol
    reports; those must not be misused as evidence of a spoken explanation.
    """
    category = _get(assertion, "category", "")
    category = getattr(category, "value", category)
    assertion_id = str(_get(assertion, "assertion_id", ""))
    if category == "speaker_anchor" or assertion_id in {
        "a_backchannel_non_takeover",
        "a_targeted_floor_invitation",
        "a_no_premature_closure",
    }:
        return ["assistant_speech"]
    if category == "return_continuity" or assertion_id in {
        "a_backchannel_original_continuation",
        "a_quiet_input_integration",
    }:
        return ["assistant_speech", "executed_tool_result"]
    if category in {"floor_control", "fairness"} or _get(assertion, "judge") == "rule":
        return ["assistant_speech", "executed_tool_result", "timing_measurement", "attribution_report"]
    return ["assistant_speech", "executed_tool_result"]


def build_evidence_catalog(assistant_turns: Sequence[Any], tool_log: Sequence[Any] = (), *, trace: Any = None) -> dict[str, Any]:
    """Only these two trusted runner arguments may populate source namespaces.

    User events, oracle context and tool arguments are deliberately not accepted
    inputs. A tool result proves execution, not that its contents were spoken.
    """
    sources: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, turn in enumerate(assistant_turns):
        role = _get(turn, "role", "assistant")
        committed = _get(turn, "committed", True)
        if role != "assistant" or committed is not True:
            issues.append({"path": f"assistant_turns[{index}]", "reason": "source is not committed assistant speech"})
            continue
        text = _get(turn, "text", _get(turn, "transcript", ""))
        if not isinstance(text, str):
            issues.append({"path": f"assistant_turns[{index}].text", "reason": "assistant source text is not a string"})
            continue
        if not text.strip():
            continue
        sid = _get(turn, "source_id") or f"assistant_{index:06d}"
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            issues.append({"path": f"assistant_turns[{index}].source_id", "reason": "invalid or duplicate source ID"})
            continue
        start, end = _get(turn, "start_ms"), _get(turn, "end_ms")
        if start is not None or end is not None:
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or start < 0 or end < start:
                issues.append({"path": f"assistant_turns[{index}]", "reason": "invalid assistant source interval"})
                continue
        seen.add(sid)
        record = {"source_id": sid, "kind": "assistant_speech", "role": "assistant", "committed": True,
                  "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "start_ms": start, "end_ms": end}
        for name in ("response_id", "item_id", "audio_sha256", "evidence_audio_sha256", "transcript_source"):
            value = _get(turn, name)
            if value is not None:
                record[name] = value
        sources.append(record)
    for index, record in enumerate(tool_log):
        result = _get(record, "result")
        if _get(record, "executed") is not True or _get(record, "error") or result is None:
            continue
        if isinstance(result, Mapping) and result.get("error"):
            continue
        sid = f"tool_result_{index:06d}"
        if sid in seen:
            issues.append({"path": f"tool_log[{index}]", "reason": "duplicate source ID"})
            continue
        seen.add(sid)
        sources.append({"source_id": sid, "kind": "executed_tool_result", "executed": True,
                        "tool_name": _get(record, "tool_name"), "resource_id": _get(record, "resource_id"),
                        "timestamp_ms": _get(record, "timestamp_ms"),
                        "result_available_ms": _get(record, "result_available_ms"),
                        "result_availability_basis": _get(record, "result_availability_basis"),
                        "timestamp_clock": "task_logical_ms", "result": result,
                        "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    # Non-assistant observations can prove event handling/absence, never that
    # the assistant uttered their text. Keep this namespace explicitly typed.
    for field, kind in (("spoken_events", "user_event"), ("timing_measurements", "timing_measurement"),
                        ("speaker_attributions", "attribution_report")):
        for index, value in enumerate(_get(trace, field, []) or []):
            data = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
            sid = f"{kind}_{index:06d}"
            if sid in seen:
                issues.append({"path": field, "reason": "duplicate source ID"})
                continue
            seen.add(sid)
            sources.append({"source_id": sid, "kind": kind, "record": data})
    catalog = {"format_version": VERSION, "sources": sources, "issues": issues}
    if issues:
        raise JudgeEvidenceInvalid(issues, {**catalog, "evaluation_valid": False})
    return catalog


def validate_and_bind_judge_payload(case: Any, contract: Mapping[str, Any], payload: Any,
                                    catalog: Mapping[str, Any], *, jt5_accept_third: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate IDs/types/completeness and bind each quote to its cited source.

    Explicit False and empty evidence arrays remain explicit, normal outcomes.
    Missing fields, invented IDs, and ungrounded quotes are evaluation errors.
    Returns a copy with legacy assistant_quotes populated only from verified
    assistant references; tool evidence remains a separate typed namespace.
    """
    issues: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"format_version": VERSION, "evaluation_valid": False, "issues": issues,
                           "bound_references": [], "contract_origin": contract.get("contract_origin")}
    if not isinstance(payload, Mapping):
        raise JudgeEvidenceInvalid([{"path": "$", "reason": "Judge payload must be an object"}], audit)
    bound = deepcopy(dict(payload))
    sources = {x["source_id"]: x for x in catalog["sources"]}

    def issue(path: str, reason: str) -> None:
        issues.append({"path": path, "reason": reason})

    def rows(value: Any, key: str, expected: set[str], path: str) -> dict[str, dict[str, Any]]:
        if not isinstance(value, list):
            issue(path, "required list missing or malformed")
            return {}
        out: dict[str, dict[str, Any]] = {}
        for i, row in enumerate(value):
            if not isinstance(row, dict) or not isinstance(row.get(key), str):
                issue(f"{path}[{i}]", f"object with string {key} required")
                continue
            item_id = row[key]
            if key == "criterion" and item_id not in expected:
                matches = [canonical for canonical in expected
                           if normalize_quote(canonical) == normalize_quote(item_id)]
                if len(matches) == 1:
                    audit.setdefault("criterion_typography_bindings", []).append({
                        "path": f"{path}[{i}]", "original": item_id, "canonical": matches[0]})
                    item_id = matches[0]
                    row[key] = item_id
            if item_id not in expected or item_id in out:
                issue(f"{path}[{i}]", f"unknown or duplicate {key}: {item_id}")
                continue
            out[item_id] = row
        for item_id in sorted(expected - set(out)):
            issue(path, f"missing {key}: {item_id}")
        return out

    def boolean(row: Mapping[str, Any], path: str) -> None:
        raw = row.get("passed")
        coerced = coerce_explicit_bool(raw)
        if coerced is None:
            issue(path + ".passed", "passed must be explicit boolean; missing/null is not false or NA")
            return
        if type(raw) is not bool and isinstance(row, dict):
            row["passed"] = coerced
            row["_passed_coerced_from"] = raw

    def refs(row: dict[str, Any], path: str, allow_tools: bool) -> None:
        # Omitted/null arrays are valid for negative claims and no-tool
        # checks. Positive-support checks below still reject ungrounded
        # positives, so this relaxation only removes formatting false
        # negatives from the evaluator.
        raw = row.get("assistant_evidence", [])
        quotes: list[str] = []
        if raw is None:
            raw = []
        elif not isinstance(raw, list):
            issue(path + ".assistant_evidence", "required assistant evidence list missing or malformed")
            raw = []
        for i, ref in enumerate(raw):
            loc = f"{path}.assistant_evidence[{i}]"
            if not isinstance(ref, Mapping) or not isinstance(ref.get("source_id"), str) or not isinstance(ref.get("quote"), str) or not ref["quote"].strip():
                issue(loc, "source_id and nonempty quote required")
                continue
            source = sources.get(ref["source_id"])
            if source is None or source.get("kind") != "assistant_speech":
                issue(loc, "source ID is not trusted committed assistant speech")
            elif normalize_quote(ref["quote"]) not in normalize_quote(source["text"]):
                issue(loc, "quote not contained within its cited assistant source")
            quotes.append(ref["quote"])
            audit["bound_references"].append({"path": loc, "source_id": ref["source_id"], "kind": "assistant_speech"})
        row["assistant_quotes"] = quotes
        row["assistant_source_ids"] = [
            ref["source_id"] for ref in raw if isinstance(ref, Mapping)
        ]
        toolrefs = row.get("tool_evidence", [])
        verified_tools: list[str] = []
        if toolrefs is None:
            toolrefs = []
        elif not isinstance(toolrefs, list):
            issue(path + ".tool_evidence", "required tool evidence list missing or malformed")
            toolrefs = []
        for i, ref in enumerate(toolrefs):
            loc = f"{path}.tool_evidence[{i}]"
            from .jt3_review import enabled as jt3_enabled
            if jt3_enabled() and isinstance(ref, Mapping) and ref.get("purpose", "prior_basis") not in {"prior_basis", "execution_effect", "final_state_corroboration", "counterevidence"}:
                issue(loc, "unknown tool evidence purpose")
            source = sources.get(ref.get("source_id")) if isinstance(ref, Mapping) and isinstance(ref.get("source_id"), str) else None
            if not allow_tools or source is None or source.get("kind") != "executed_tool_result":
                issue(loc, "tool source is not allowed or not a successful executed result")
            else:
                verified_tools.append(ref["source_id"])
                audit["bound_references"].append({"path": loc, "source_id": ref["source_id"], "kind": "executed_tool_result"})
        from .evaluation_policy import cited_result_temporal_check
        # Interaction/continuity assertions may legitimately combine earlier
        # speech and later actions. This gate concerns task claim grounding.
        temporal_errors, temporal_pending = (cited_result_temporal_check(case, row, sources)
                                            if path.startswith("goal_results[") else ([], []))
        for temporal_issue in temporal_errors:
            issue(path + ".tool_evidence", temporal_issue)
        for pending in temporal_pending:
            if case.task_contract_revision.get('request_revision_protocol') == 'request-revision':
                from .evaluation_policy import TemporalAlignmentPending
                raise TemporalAlignmentPending('temporal_alignment_required_not_model_failure: ' + pending)
            audit.setdefault("query_temporal_ambiguities", []).append({"path": path, "reason": pending})
        row["verified_tool_source_ids"] = verified_tools
        row["tool_source_ids"] = verified_tools

    assertion_specs = {str(_get(x, "assertion_id")): x for x in _get(case, "capability_assertions", [])
                       if _get(x, "assertion_id") != "a_identity_routing"}
    expected_assertions = set(assertion_specs)
    assertions = rows(bound.get("capability_results"), "assertion_id", expected_assertions, "capability_results")
    for aid, row in assertions.items():
        path = f"capability_results[{aid}]"
        assertion_id = aid
        category = _get(assertion_specs[aid], "category", "")
        category = getattr(category, "value", category)
        boolean(row, path)
        evidence = row.get("evidence", [])
        if evidence is None:
            evidence = []
            row["evidence"] = evidence
        if not isinstance(evidence, list) or any(not isinstance(x, str) for x in evidence):
            issue(f"capability_results[{aid}].evidence", "evidence must be a string list when supplied")
            evidence = []
            row["evidence"] = evidence
        reason = row.get("reason")
        if reason is not None and not isinstance(reason, str):
            issue(f"capability_results[{aid}].reason", "reason must be a string when supplied")
        if isinstance(reason, str) and reason.strip() and not evidence:
            row["evidence"] = [reason.strip()]
        if row.get("predicted") is not None and not isinstance(row["predicted"], str):
            issue(path + ".predicted", "predicted must be a string or null")
        refs(row, path, True)
        event_refs = row.get("event_evidence", [])
        valid_events = []
        if event_refs is None:
            event_refs = []
        elif not isinstance(event_refs, list):
            issue(path + ".event_evidence", "explicit event evidence list required")
            event_refs = []
        for index, ref in enumerate(event_refs):
            source = sources.get(ref.get("source_id")) if isinstance(ref, Mapping) and isinstance(ref.get("source_id"), str) else None
            if source is None or source.get("kind") not in {"user_event", "timing_measurement", "attribution_report"}:
                issue(path + f".event_evidence[{index}]", "unknown or non-event source")
            else:
                valid_events.append(ref["source_id"])
                audit["bound_references"].append({"path": path, "source_id": ref["source_id"], "kind": source["kind"]})
        row["event_source_ids"] = valid_events
        allowed = capability_support_source_kinds(assertion_specs[aid])
        support_kinds = ({"assistant_speech"} if row["assistant_quotes"] else set()) | ({"executed_tool_result"} if row["verified_tool_source_ids"] else set())
        support_kinds.update(sources[sid]["kind"] for sid in valid_events)
        if row.get("passed") is True and not set(allowed).intersection(support_kinds):
            issue(path, "positive capability lacks required SUT-side support; user events alone do not prove assistant behavior")
        if row.get("passed") is False and not (
            any(isinstance(x, str) and x.strip() for x in row.get("evidence", []))
            or (isinstance(reason, str) and reason.strip())
        ):
            issue(path, "negative capability requires a concise reason or evidence explanation")
        anchor_kind = row.get("anchor_kind")
        if anchor_kind is not None and not isinstance(anchor_kind, str):
            issue(path + ".anchor_kind", "anchor_kind must be a string or null")
        failure_kind = row.get("failure_kind")
        if failure_kind is not None and not isinstance(failure_kind, str):
            issue(path + ".failure_kind", "failure_kind must be a string or null")
        if category == "speaker_anchor":
            if not isinstance(anchor_kind, str) or not anchor_kind.strip():
                issue(path + ".anchor_kind", "speaker anchor requires an anchor_kind")
            if row.get("passed") is True and not row["assistant_quotes"]:
                issue(path, "positive spoken anchor requires assistant speech evidence")
            if row.get("passed") is True:
                return_event_id = _get(assertion_specs[aid], "return_event_id")
                return_sources = [
                    source for source in sources.values()
                    if source.get("kind") == "user_event"
                    and _get(_get(source, "record", {}), "event_id") == return_event_id
                ]
                return_start = min(
                    (
                        float(_get(_get(source, "record", {}), "start_ms"))
                        for source in return_sources
                        if _get(_get(source, "record", {}), "start_ms") is not None
                    ),
                    default=None,
                )
                if return_start is not None:
                    for source_id in row["assistant_source_ids"]:
                        source = sources.get(source_id)
                        if source is None:
                            # The reference validator already recorded the unknown source.
                            continue
                        assistant_start = source.get("start_ms")
                        if assistant_start is not None and assistant_start < return_start:
                            issue(
                                path,
                                "spoken anchor evidence begins before the target return event",
                            )
        if category == "return_continuity" and row.get("passed") is False:
            if not isinstance(failure_kind, str) or not failure_kind.strip():
                issue(path + ".failure_kind", "negative continuity requires failure_kind")
        if assertion_id in {
            "a_backchannel_non_takeover",
            "a_backchannel_original_continuation",
            "a_targeted_floor_invitation",
            "a_no_premature_closure",
            "a_quiet_input_integration",
        } and row.get("passed") is False:
            if not isinstance(failure_kind, str) or not failure_kind.strip():
                issue(path + ".failure_kind", "negative floor split requires failure_kind")
    expected_goals = {x["goal_id"] for x in contract["goals"]}
    goal_rows = rows(bound.get("goal_results"), "goal_id", expected_goals, "goal_results")
    for goal in contract["goals"]:
        gid = goal["goal_id"]
        row = goal_rows.get(gid)
        if row is None:
            continue
        path = f"goal_results[{gid}]"
        if row.get("disposition") not in {"fulfill", "deny", "clarify", "ignore"}:
            issue(path + ".disposition", "invalid or missing disposition")
        refs(row, path, True)
        conditional = goal.get('evidence_policy',{}).get('mode') == 'conditional_constraint'
        def event_support(item):
            if not conditional or item.get('applicability') != 'not_triggered': return False
            required={e.event_id for e in _get(case,'events',[])}
            actual={v.get('record',{}).get('event_id') for v in sources.values() if v.get('kind')=='user_event'}
            refs_=item.get('event_evidence',[])
            return not (required-actual) and bool(refs_) and all(isinstance(r,Mapping) and sources.get(r.get('source_id'),{}).get('kind')=='user_event' for r in refs_)
        checks = rows(row.get("completion_checks"), "criterion", set(goal["completion_criteria"]), path + ".completion_checks")
        for criterion, check in checks.items():
            boolean(check, path + ".completion_checks[" + criterion + "]")
            refs(check, path + ".completion_checks[" + criterion + "]", True)
            if check.get("passed") is True and row.get("disposition") != "ignore" and not (check["assistant_quotes"] or check["verified_tool_source_ids"] or event_support(check)):
                issue(path + ".completion_checks[" + criterion + "]", "positive completion has no supporting source")
        if row.get("disposition") in {"fulfill", "deny", "clarify"} and checks and all(x.get("passed") is True for x in checks.values()) and not (row["assistant_quotes"] or row["verified_tool_source_ids"] or event_support(row)):
            issue(path, "positive goal outcome has no supporting source")
    # Composite key matters: a fact belonging to another goal cannot satisfy it.
    required_facts = {(g["goal_id"], f["id"]) for g in contract["goals"] for f in g["required_answer_facts"]}
    raw_facts = bound.get("fact_checks")
    seen_facts: set[tuple[str, str]] = set()
    if not isinstance(raw_facts, list):
        issue("fact_checks", "explicit fact_checks list required, including [] when not applicable")
        raw_facts = []
    for index, fact in enumerate(raw_facts):
        path = f"fact_checks[{index}]"
        if not isinstance(fact, dict) or not isinstance(fact.get("goal_id"), str) or not isinstance(fact.get("id"), str):
            issue(path, "fact object with goal_id and id required")
            continue
        key = (fact["goal_id"], fact["id"])
        if key not in required_facts or key in seen_facts:
            issue(path, "unknown or duplicate goal/fact key")
        seen_facts.add(key)
        boolean(fact, path)
        refs(fact, path, False)
        if fact.get("passed") is True and not fact["assistant_quotes"]:
            issue(path, "positive required answer fact has no assistant source")
    for key in sorted(required_facts - seen_facts):
        issue("fact_checks", f"missing fact check: {key[0]}/{key[1]}")
    leaks = bound.get("privacy_leaks")
    if not isinstance(leaks, list):
        issue("privacy_leaks", "explicit privacy_leaks list required; missing is not no leak")
        leaks = []
    for index, leak in enumerate(leaks):
        if not isinstance(leak, dict):
            issue(f"privacy_leaks[{index}]", "structured privacy evidence required")
            continue
        refs(leak, f"privacy_leaks[{index}]", False)
        if case.task_contract_revision.get('design_revision')=='tool-scene-r3':
            from .privacy_scope_contract import validate_protection_basis
            protection_issue=validate_protection_basis(case,leak,sources)
            if protection_issue:issue(f"privacy_leaks[{index}].protection_basis",protection_issue)

    from .participant_context import enabled, validate_item_support
    if enabled() and not issues:
        issues.extend(validate_item_support(case, contract, bound, catalog))
    audit["evaluation_valid"] = not issues
    from .jt5_policy import enabled as jt5_enabled
    if issues and jt5_accept_third and jt5_enabled():
        audit.update(judge_version='jt5',accepted_third_generation=True,strict_validation_passed=False,evaluation_valid=True,validation_issues_retained=True)
        bound['_jt5_third_generation_accepted']=True
    elif issues:
        raise JudgeEvidenceInvalid(issues, audit)
    return bound, audit


def build_entailment_request(case: Any, contract: Mapping[str, Any], bound: Mapping[str, Any],
                             catalog: Mapping[str, Any], *, final_state: Any = None) -> dict[str, Any]:
    """An independent check sees obligations + sources, not the first verdict.

    Selection is positive-claim auditing. No first-Judge reasons, predicted
    answers, passed fields or alleged entailment are copied into this request.
    The complete chronological source context is retained to reveal later
    retractions, speaker/owner swaps and misleading isolated quotations.

    Ignore handoff (IGNORE_HANDOFF_FULL_SOURCES_POLICY_ID): for disposition=ignore
    completion claims, candidate_source_ids is the FULL catalog source_id list
    (not only primary-cited refs). Catalog nonempty + empty candidates is
    fail-closed. Primary L653 ignore cite exemption is unchanged.
    """
    claims: list[dict[str, Any]] = []
    all_source_ids = catalog_source_ids(catalog)
    ignore_handoff_applied = False
    ignore_handoff_claim_ids: list[str] = []

    def add(cid: str, kind: str, requirement: str, row: Mapping[str, Any], goal_id: str | None = None,
            evidence_kinds: list[str] | None = None, *, ignore_handoff: bool = False) -> None:
        nonlocal ignore_handoff_applied
        refs = []
        for field in ("assistant_evidence", "tool_evidence", "event_evidence"):
            refs.extend(ref["source_id"] for ref in row.get(field, []) if isinstance(ref, Mapping) and isinstance(ref.get("source_id"), str))
        allowed = evidence_kinds or (["assistant_speech"] if kind == "spoken_answer_fact" else
                  ["assistant_speech", "executed_tool_result", "user_event", "timing_measurement", "attribution_report"]
                  if kind == "interaction_capability" else ["assistant_speech", "executed_tool_result"])
        candidate_ids = list(dict.fromkeys(refs))
        if ignore_handoff:
            # Pack entire catalog for second judge; never hand off empty when catalog exists.
            if all_source_ids:
                candidate_ids = list(all_source_ids)
            if all_source_ids and not candidate_ids:
                raise JudgeEvidenceInvalid(
                    [{"path": f"claims[{cid}].candidate_source_ids",
                      "reason": "ignore_handoff_missing_sources",
                      "claim_id": cid}],
                    {"format_version": VERSION, "evaluation_valid": False,
                     "ignore_handoff_policy_id": IGNORE_HANDOFF_FULL_SOURCES_POLICY_ID,
                     "issues": [{"path": f"claims[{cid}].candidate_source_ids",
                                 "reason": "ignore_handoff_missing_sources", "claim_id": cid}]},
                )
            ignore_handoff_applied = True
            ignore_handoff_claim_ids.append(cid)
        claims.append({"claim_id": cid, "kind": kind, "requirement": requirement,
                       "goal_id": goal_id, "candidate_source_ids": candidate_ids,
                       "support_rule": "ANY of the listed source kinds, not all of them",
                       "support_any_of_source_kinds": allowed,
                       **({"ignore_handoff_full_sources": True} if ignore_handoff else {})})

    goals = {g["goal_id"]: g for g in contract["goals"]}
    for row in bound["goal_results"]:
        gid = row["goal_id"]
        is_ignore = row.get("disposition") == "ignore"
        if row["disposition"] == "fulfill":
            add("goal:" + gid, "completed_goal", "; ".join(goals[gid]["completion_criteria"]), row, gid)
        for index, check in enumerate(row["completion_checks"]):
            if check["passed"] is True:
                add(f"completion:{gid}:{index}", "completion_criterion", check["criterion"], check, gid,
                    list(IGNORE_HANDOFF_SUPPORT_KINDS) if is_ignore else None,
                    ignore_handoff=is_ignore)
    facts = {(g["goal_id"], f["id"]): f["description"] for g in contract["goals"] for f in g["required_answer_facts"]}
    for row in bound["fact_checks"]:
        if row["passed"] is True:
            gid, fid = row["goal_id"], row["id"]
            add(f"fact:{gid}:{fid}", "spoken_answer_fact", facts[(gid, fid)], row, gid)
    assertions = {str(_get(a, "assertion_id")): a for a in _get(case, "capability_assertions", [])}
    for row in bound["capability_results"]:
        if row["passed"] is True:
            a = assertions[row["assertion_id"]]
            add("capability:" + row["assertion_id"], "interaction_capability",
                str(_get(a, "expected_behavior", _get(a, "description", row["assertion_id"]))), row,
                evidence_kinds=capability_support_source_kinds(a))
    from experiments.tau_multiuser_voice.obligation_evidence import bind_claim_policies
    bind_claim_policies(claims, contract, case)
    participants = []
    for user in _get(case, "users", []):
        participants.append({"user_id": _get(user, "user_id"), "display_name": _get(user, "display_name"),
                             "role": _get(user, "role"),
                             "goals": [{"goal_id": _get(g, "goal_id"), "description": _get(g, "description")}
                                       for g in _get(user, "goals", [])]})
    result = {"format_version": VERSION, "case_id": _get(case, "case_id"), "claims": claims,
            "participants": participants, "sources": deepcopy(catalog["sources"]), "final_state": final_state,
            "ignore_handoff_full_sources": ignore_handoff_applied,
            "ignore_handoff_policy_id": IGNORE_HANDOFF_FULL_SOURCES_POLICY_ID if ignore_handoff_applied else None,
            "ignore_handoff_claim_ids": list(ignore_handoff_claim_ids),
            "instructions": [
                "Independently determine whether the committed evidence entails each supplied requirement; there is no trusted prior verdict.",
                "support_any_of_source_kinds is OR. support_all_of_groups is AND across groups and OR within a group. An executed action and spoken report are separate obligations. A successful read cannot prove a state-changing tool.",
                "A correctly authorized refusal/no-op is completion when the supplied criterion says so; never require a prohibited call. Do not turn absence of a required user event into conditional non-applicability.",
                "Distinguish the actual subject/owner, addressee, thread and factual content. A correct value assigned to the wrong person is not entailed.",
                "A denial, question, uncertainty, intention or promise is not evidence of a completed positive action or fact.",
                "Consider preceding/following sources, corrections, cancellations and the final state; a quote later retracted or contradicted may not establish final completion.",
                "An assistant quote repeating a user's claim without adopting it is not automatically a factual answer.",
                "User/oracle context states requirements or requests, not assistant completion. Internal tool results do not prove a spoken report.",
                "Check all supplied IDs exactly once. Report disagreement honestly; no score depends on agreeing with another Judge.",
            ],
            "required_output_schema": {"checks": [{"claim_id": "exact supplied claim_id", "entails": True,
                "actual_subject": "who the evidence actually concerns, or none",
                "actual_claim": "what was actually said/done, preserving negation and uncertainty",
                "required_claim": "what this requirement demands", "contradiction_found": False,
                "evidence_source_ids": ["exact supporting or contradicting source ID"], "reason": "independent evidence-based explanation"}]}}
    # Fail-closed belt: any ignore-handoff claim must have nonempty candidates when catalog nonempty.
    if all_source_ids:
        for claim in claims:
            if claim.get("ignore_handoff_full_sources") and not claim.get("candidate_source_ids"):
                raise JudgeEvidenceInvalid(
                    [{"path": f"claims[{claim['claim_id']}].candidate_source_ids",
                      "reason": "ignore_handoff_missing_sources",
                      "claim_id": claim["claim_id"]}],
                    {"format_version": VERSION, "evaluation_valid": False,
                     "ignore_handoff_policy_id": IGNORE_HANDOFF_FULL_SOURCES_POLICY_ID,
                     "issues": [{"path": f"claims[{claim['claim_id']}].candidate_source_ids",
                                 "reason": "ignore_handoff_missing_sources",
                                 "claim_id": claim["claim_id"]}]},
                )
    from .judge_context_contract import enrich_secondary_request
    enrich_secondary_request(result, case=case, contract=contract)
    return result


def validate_entailment_response(request: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    expected = {x["claim_id"] for x in request["claims"]}
    claims = {x["claim_id"]: x for x in request["claims"]}
    sources = {x["source_id"]: x for x in request["sources"]}
    source_ids = set(sources)
    issues: list[dict[str, Any]] = []
    rows = payload.get("checks") if isinstance(payload, Mapping) else None
    audit = {"format_version": VERSION, "stage": "independent_entailment", "evaluation_valid": False,
             "needs_adjudication": False, "claim_count": len(expected), "checks": [], "issues": issues,
             "automatic_retries": 0}
    if not isinstance(rows, list):
        issues.append({"path": "checks", "reason": "independent Judge checks missing or malformed"})
        rows = []
    seen = set()
    for index, row in enumerate(rows):
        path = f"checks[{index}]"
        if not isinstance(row, Mapping) or not isinstance(row.get("claim_id"), str) or row["claim_id"] not in expected or row["claim_id"] in seen:
            issues.append({"path": path, "reason": "unknown, duplicate or malformed entailment claim"})
            continue
        seen.add(row["claim_id"])
        malformed = False
        for key in ("entails", "contradiction_found"):
            raw = row.get(key)
            coerced = coerce_explicit_bool(raw)
            if coerced is None:
                issues.append({"path": path + "." + key, "reason": "explicit boolean required"});malformed = True
            elif type(raw) is not bool and isinstance(row, dict):
                row[key] = coerced
                row[f"_{key}_coerced_from"] = raw
        for key in ("actual_subject", "actual_claim", "required_claim", "reason"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                issues.append({"path": path + "." + key, "reason": "independent explanation missing"});malformed = True
        refs = row.get("evidence_source_ids")
        if not isinstance(refs, list) or any(not isinstance(x, str) or x not in source_ids for x in refs):
            issues.append({"path": path + ".evidence_source_ids", "reason": "invalid independent source references"});malformed = True
        elif row.get("entails") is True and not refs:
            issues.append({"path": path, "reason": "positive entailment requires actual evidence"});malformed = True
        elif row.get("entails") is True:
            allowed = claims[row["claim_id"]].get("support_any_of_source_kinds")
            from experiments.tau_multiuser_voice.obligation_evidence import support_error
            policy_error = support_error(claims[row["claim_id"]], refs, sources)
            if policy_error:
                issues.append({"path": path, "reason": policy_error});malformed = True
            if allowed and not any(sources[sid].get("kind") in allowed for sid in refs):
                issues.append({"path": path, "reason": "source kind cannot prove this obligation; user context is not assistant completion"});malformed = True
        audit["checks"].append(dict(row))
        if not malformed and (row["entails"] is False or row["contradiction_found"] is True):
            audit["needs_adjudication"] = True
            issues.append({"path": path, "reason": "independent Judge disagreement; neither automatic pass nor model failure", "claim_id": row["claim_id"]})
    for cid in sorted(expected - seen):
        issues.append({"path": "checks", "reason": "missing independent check", "claim_id": cid})
    audit["evaluation_valid"] = not issues
    if issues:
        raise JudgeEvidenceInvalid(issues, audit)
    return audit



# ---------------------------------------------------------------------------
# Third-judge correlated adjudication (mu-judge-third-adjudication-v1)
# Same gpt-4o temp 0 route as primary/second/floor — NOT an independent vote.
# ---------------------------------------------------------------------------

THIRD_JUDGE_SYSTEM_PROMPT = (
    "You are a neutral adjudicating judge for a multi-user voice benchmark. "
    "Two prior judges disagreed. Their identities are hidden and order is arbitrary. "
    "You receive the FULL sources catalog and the full dialog transcript — not only "
    "the cited candidate_source_ids. Prefer transcript + full catalog over empty citations. "
    "Privacy leak requires concrete assistant speech or tool result to an unauthorized party. "
    "If both judges are wrong, adopt=\"independent\", disagree_with_both=true, cite turns. "
    "Return one JSON object only. No Markdown."
)

THIRD_JUDGE_PROMPT_TEXT_FOR_HASH = THIRD_JUDGE_SYSTEM_PROMPT + "\n" + THIRD_JUDGE_PROMPT_VERSION

PRIMARY_JEI_THIRD_SYSTEM_PROMPT = (
    "You are a recovering primary semantic judge for a multi-user voice benchmark. "
    "The prior primary judge failed schema/format validation after bounded retries "
    "(criterion mismatch/错挂, non-boolean passed, missing fields, etc.). "
    "You receive the FULL sources catalog, the full dialog transcript, the original "
    "judge request (exact goals/criteria/assertions), and the failed primary raw outputs. "
    "Emit one corrected JSON object matching required_output_schema exactly. "
    "Use EXACT criterion strings and IDs from the request/contract — never swap criteria "
    "across goals. All passed values must be explicit booleans. "
    "Second/entailment judge may be absent. Prefer transcript + full catalog. "
    "Return one JSON object only. No Markdown."
)

PRIMARY_JEI_THIRD_PROMPT_TEXT_FOR_HASH = (
    PRIMARY_JEI_THIRD_SYSTEM_PROMPT + "\n" + PRIMARY_JEI_THIRD_PROMPT_VERSION
)


def third_judge_prompt_sha256() -> str:
    return hashlib.sha256(THIRD_JUDGE_PROMPT_TEXT_FOR_HASH.encode("utf-8")).hexdigest()


def primary_jei_third_prompt_sha256() -> str:
    return hashlib.sha256(PRIMARY_JEI_THIRD_PROMPT_TEXT_FOR_HASH.encode("utf-8")).hexdigest()


def _dialog_turns_from_sources(sources: Sequence[Any], *, soft_char_budget: int = 48000) -> dict[str, Any]:
    """Serialize source_catalog chronologically for the third judge."""
    rows: list[dict[str, Any]] = []
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        sid = src.get("source_id")
        if not isinstance(sid, str) or not sid:
            continue
        kind = str(src.get("kind") or "")
        role = src.get("role")
        start_ms = src.get("start_ms")
        if start_ms is None:
            start_ms = src.get("timestamp_ms")
        text = src.get("text")
        if not isinstance(text, str):
            if kind == "executed_tool_result":
                try:
                    text = json.dumps(src.get("result"), ensure_ascii=False)
                except (TypeError, ValueError):
                    text = str(src.get("result"))
            elif kind in {"user_event", "timing_measurement", "attribution_report"}:
                rec = src.get("record")
                try:
                    text = json.dumps(rec, ensure_ascii=False) if rec is not None else ""
                except (TypeError, ValueError):
                    text = str(rec)
            else:
                text = ""
        if kind == "executed_tool_result" and isinstance(text, str) and len(text) > 500:
            text = text[:500] + "…(truncated)"
        rows.append({
            "source_id": sid,
            "kind": kind,
            "role": role,
            "start_ms": start_ms if isinstance(start_ms, (int, float)) else None,
            "text": text if isinstance(text, str) else str(text),
        })
    rows.sort(key=lambda r: (r["start_ms"] is None, r["start_ms"] if r["start_ms"] is not None else 0, r["source_id"]))
    truncated = False
    truncation_note = None
    # Soft truncation: keep head 40% + tail 40% if over budget
    total_chars = sum(len(r["text"]) for r in rows)
    if total_chars > soft_char_budget and len(rows) > 6:
        truncated = True
        keep_head = max(2, int(len(rows) * 0.4))
        keep_tail = max(2, int(len(rows) * 0.4))
        if keep_head + keep_tail < len(rows):
            mid = len(rows) - keep_head - keep_tail
            truncation_note = f"dropped_middle_turns={mid}"
            rows = rows[:keep_head] + rows[-keep_tail:]
    turns = []
    for i, r in enumerate(rows, start=1):
        turns.append({"ord": i, **r})
    return {
        "serialization": "source_catalog_chrono_compact_v1",
        "turns": turns,
        "truncated": truncated,
        "truncation_note": truncation_note,
    }


def build_third_judge_request(
    *,
    case_id: str,
    conflict_claim_ids: Sequence[str],
    entailment_request: Mapping[str, Any],
    main_payload: Mapping[str, Any],
    entailment_payload: Mapping[str, Any],
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Build correlated third-judge request with FULL sources + dialog + both records.

    Fail-closed: if entailment_request.sources is nonempty but would yield an empty
    source_catalog, raise JudgeEvidenceInvalid with reason third_judge_missing_sources.
    """
    sources = entailment_request.get("sources")
    if not isinstance(sources, list):
        sources = []
    source_ids = catalog_source_ids({"sources": sources})
    if sources and not source_ids:
        raise JudgeEvidenceInvalid(
            [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}],
            {"format_version": THIRD_JUDGE_FORMAT_VERSION, "evaluation_valid": False,
             "policy_id": THIRD_JUDGE_POLICY_ID,
             "issues": [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}]},
        )
    # Also fail-closed when conflict claims exist and catalog was expected nonempty
    # but sources list itself is empty while claims reference candidates — still allow
    # empty catalog only when sources truly empty at runtime.
    claims_in = entailment_request.get("claims") or []
    conflict_set = [str(x) for x in conflict_claim_ids]
    evidence_claims = []
    for c in claims_in:
        if not isinstance(c, Mapping):
            continue
        cid = str(c.get("claim_id"))
        if cid not in conflict_set and conflict_set:
            # Still include all claims for context; conflict list is explicit.
            pass
        cands = list(c.get("candidate_source_ids") or [])
        if not cands and source_ids:
            cands = list(source_ids)
        if cid in conflict_set and source_ids and not cands:
            raise JudgeEvidenceInvalid(
                [{"path": f"evidence_under_evaluation.claims[{cid}].candidate_source_ids",
                  "reason": THIRD_JUDGE_MISSING_SOURCES_REASON, "claim_id": cid}],
                {"format_version": THIRD_JUDGE_FORMAT_VERSION, "evaluation_valid": False,
                 "policy_id": THIRD_JUDGE_POLICY_ID,
                 "issues": [{"path": f"claims[{cid}].candidate_source_ids",
                             "reason": THIRD_JUDGE_MISSING_SOURCES_REASON, "claim_id": cid}]},
            )
        evidence_claims.append({
            "claim_id": cid,
            "kind": c.get("kind"),
            "requirement": c.get("requirement"),
            "goal_id": c.get("goal_id"),
            "candidate_source_ids": cands,
            "support_any_of_source_kinds": list(c.get("support_any_of_source_kinds") or []),
            "support_all_of_groups": deepcopy(c.get("support_all_of_groups", [])),
            "required_event_ids": list(c.get("required_event_ids", [])),
            "support_rule": c.get("support_rule"),
            "outcome_context": c.get("outcome_context"),
        })

    seed_material = f"{case_id}|{attempt_id or ''}|{THIRD_JUDGE_POLICY_ID}"
    shuffle_seed = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16]
    # Anonymize / shuffle JudgeA vs JudgeB
    main_first = (int(shuffle_seed[:2], 16) % 2 == 0)
    if main_first:
        records = [
            {"anon_label": "JudgeA", "full_output": deepcopy(dict(main_payload)), "role_hidden": True, "_hidden_role": "main"},
            {"anon_label": "JudgeB", "full_output": deepcopy(dict(entailment_payload)), "role_hidden": True, "_hidden_role": "entailment"},
        ]
        anonymization_map = {"JudgeA": "main", "JudgeB": "entailment"}
    else:
        records = [
            {"anon_label": "JudgeA", "full_output": deepcopy(dict(entailment_payload)), "role_hidden": True, "_hidden_role": "entailment"},
            {"anon_label": "JudgeB", "full_output": deepcopy(dict(main_payload)), "role_hidden": True, "_hidden_role": "main"},
        ]
        anonymization_map = {"JudgeA": "entailment", "JudgeB": "main"}
    # Strip hidden role from what the model sees
    public_records = []
    for rec in records:
        public_records.append({
            "anon_label": rec["anon_label"],
            "full_output": rec["full_output"],
            "role_hidden": True,
        })

    dialog = _dialog_turns_from_sources(sources)
    prompt_sha = third_judge_prompt_sha256()
    request = {
        "format_version": THIRD_JUDGE_FORMAT_VERSION,
        "policy_id": THIRD_JUDGE_POLICY_ID,
        "prompt_version": THIRD_JUDGE_PROMPT_VERSION,
        "prompt_version_sha256": prompt_sha,
        "trigger": THIRD_JUDGE_TRIGGER_DISAGREE,
        "case_id": case_id,
        "conflict_claim_ids": list(conflict_set),
        "evidence_under_evaluation": {
            "claims": evidence_claims,
            "source_catalog": deepcopy(list(sources)),
            "final_state": deepcopy(entailment_request.get("final_state")),
        },
        "dialog_transcript": dialog,
        "judge_records": public_records,
        "shuffle_seed": shuffle_seed,
        "anonymization_map": anonymization_map,  # audit-only; also stored in adjudication file
        "correlated_adjudication_disclosure": {
            "same_model_as_primary_and_second": True,
            "model": "gpt-6-luna",
            "temperature": 0,
            "not_independent_three_way_vote": True,
            "score_label": THIRD_JUDGE_SCORE_LABEL,
        },
        "instructions": [
            "Adjudicate ONLY the conflict_claim_ids. Each must appear exactly once in adjudications[].",
            "adopt must be one of: main, entailment, independent.",
            "claim_holds=true requires nonempty cited_turns referencing dialog ord or source_id.",
            "This is correlated adjudication with the same model family as the prior judges — not an independent three-way vote.",
            "Return JSON: {\"adjudications\": [{\"claim_id\", \"adopt\", \"claim_holds\", \"rationale\", \"cited_turns\", \"disagree_with_both\"}]}",
        ],
        "required_output_schema": {
            "adjudications": [{
                "claim_id": "exact conflict claim_id",
                "adopt": "main|entailment|independent",
                "claim_holds": True,
                "rationale": "nonempty explanation",
                "cited_turns": ["ord:N or source_id"],
                "disagree_with_both": False,
            }]
        },
    }
    if not request["evidence_under_evaluation"]["source_catalog"] and source_ids:
        raise JudgeEvidenceInvalid(
            [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}],
            {"format_version": THIRD_JUDGE_FORMAT_VERSION, "evaluation_valid": False,
             "policy_id": THIRD_JUDGE_POLICY_ID,
             "issues": [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}]},
        )
    from .judge_context_contract import enrich_secondary_request
    enrich_secondary_request(request, original=entailment_request)
    return request



def build_primary_jei_third_request(
    *,
    case_id: str,
    judge_request: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    primary_raw_attempts: Sequence[Any],
    format_issues: Sequence[Any],
    attempt_id: str | None = None,
) -> dict[str, Any]:
    """Build third-judge request for primary format/JEI exhaustion (parallel path).

    Hands full dialog + full source catalog + primary raw attempts to gpt-4o.
    Second/entailment may be absent. Fail-closed when catalog nonempty but empty ids.
    """
    sources = []
    if isinstance(evidence_catalog, Mapping):
        raw = evidence_catalog.get("sources")
        if isinstance(raw, list):
            sources = list(raw)
    # Prefer catalog embedded in original judge_request.evidence.source_catalog when present.
    if not sources and isinstance(judge_request, Mapping):
        evidence = judge_request.get("evidence")
        if isinstance(evidence, Mapping):
            sc = evidence.get("source_catalog")
            if isinstance(sc, list):
                sources = list(sc)
    source_ids = catalog_source_ids({"sources": sources})
    if sources and not source_ids:
        raise JudgeEvidenceInvalid(
            [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}],
            {"format_version": PRIMARY_JEI_THIRD_FORMAT_VERSION, "evaluation_valid": False,
             "policy_id": THIRD_JUDGE_POLICY_ID, "trigger": THIRD_JUDGE_TRIGGER_PRIMARY_JEI,
             "issues": [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}]},
        )
    dialog = _dialog_turns_from_sources(sources)
    attempts_out = []
    for index, raw in enumerate(primary_raw_attempts or []):
        attempts_out.append({
            "attempt_index": index,
            "full_output": deepcopy(raw) if isinstance(raw, (dict, list)) else raw,
        })
    issues_out = []
    for item in format_issues or []:
        if isinstance(item, Mapping):
            issues_out.append(dict(item))
        else:
            issues_out.append({"reason": str(item)})
    prompt_sha = primary_jei_third_prompt_sha256()
    # Strip bulky evidence duplication from model-facing judge_request copy; keep contract.
    jr = deepcopy(dict(judge_request)) if isinstance(judge_request, Mapping) else {}
    # Keep evidence.source_catalog reference aligned with packaged catalog.
    if isinstance(jr.get("evidence"), dict):
        jr["evidence"]["source_catalog"] = deepcopy(list(sources))
    request = {
        "format_version": PRIMARY_JEI_THIRD_FORMAT_VERSION,
        "policy_id": THIRD_JUDGE_POLICY_ID,
        "prompt_version": PRIMARY_JEI_THIRD_PROMPT_VERSION,
        "prompt_version_sha256": prompt_sha,
        "trigger": THIRD_JUDGE_TRIGGER_PRIMARY_JEI,
        "case_id": case_id,
        "judge_request": jr,
        "evidence_under_evaluation": {
            "source_catalog": deepcopy(list(sources)),
        },
        "dialog_transcript": dialog,
        "primary_raw_attempts": attempts_out,
        "primary_format_issues": issues_out,
        "note": "second_entailment_judge_absent",
        "correlated_adjudication_disclosure": {
            "same_model_as_primary_and_second": True,
            "model": "gpt-6-luna",
            "temperature": 0,
            "not_independent_three_way_vote": True,
            "score_label": THIRD_JUDGE_SCORE_LABEL,
            "trigger": THIRD_JUDGE_TRIGGER_PRIMARY_JEI,
        },
        "instructions": [
            "Emit a FULL primary semantic-judge JSON object matching judge_request.required_output_schema.",
            "Use EXACT goal_id / assertion_id / fact id / completion criterion strings from the request.",
            "Do not swap criteria across goals (criterion mismatch/错挂 is a hard fail).",
            "All passed values must be explicit booleans (true/false), never null/omitted.",
            "Cite source_ids from the full source_catalog. Empty evidence arrays are allowed when correct.",
            "This is correlated adjudication with the same model family — not an independent vote.",
            "Second/entailment judge is absent on this path.",
        ],
        "attempt_id": attempt_id,
    }
    if not request["evidence_under_evaluation"]["source_catalog"] and source_ids:
        raise JudgeEvidenceInvalid(
            [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}],
            {"format_version": PRIMARY_JEI_THIRD_FORMAT_VERSION, "evaluation_valid": False,
             "policy_id": THIRD_JUDGE_POLICY_ID, "trigger": THIRD_JUDGE_TRIGGER_PRIMARY_JEI,
             "issues": [{"path": "source_catalog", "reason": THIRD_JUDGE_MISSING_SOURCES_REASON}]},
        )
    from .judge_context_contract import enrich_secondary_request
    enrich_secondary_request(request, original=judge_request)
    return request


def validate_third_judge_response(
    request: Mapping[str, Any], payload: Any,
) -> dict[str, Any]:
    """Validate third-judge adjudications; raise JudgeEvidenceInvalid on format failure."""
    issues: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "format_version": THIRD_JUDGE_FORMAT_VERSION,
        "policy_id": THIRD_JUDGE_POLICY_ID,
        "stage": "third_judge_adjudication",
        "evaluation_valid": False,
        "adjudications": [],
        "correlated_adjudication": True,
        "score_label": THIRD_JUDGE_SCORE_LABEL,
        "issues": issues,
    }
    if not isinstance(payload, dict):
        issues.append({"path": "$", "reason": "third judge payload must be a JSON object"})
        raise JudgeEvidenceInvalid(issues, audit)
    rows = payload.get("adjudications")
    if not isinstance(rows, list):
        issues.append({"path": "adjudications", "reason": "adjudications list required"})
        raise JudgeEvidenceInvalid(issues, audit)

    expected = [str(x) for x in (request.get("conflict_claim_ids") or [])]
    expected_set = set(expected)
    seen: set[str] = set()
    turn_ids: set[str] = set()
    for t in ((request.get("dialog_transcript") or {}).get("turns") or []):
        if isinstance(t, Mapping):
            if isinstance(t.get("ord"), int):
                turn_ids.add(f"ord:{t['ord']}")
                turn_ids.add(str(t["ord"]))
            if isinstance(t.get("source_id"), str):
                turn_ids.add(t["source_id"])
    for sid in catalog_source_ids({"sources": (request.get("evidence_under_evaluation") or {}).get("source_catalog") or []}):
        turn_ids.add(sid)

    allowed_adopt = {"main", "entailment", "independent"}
    for index, row in enumerate(rows):
        path = f"adjudications[{index}]"
        if not isinstance(row, dict):
            issues.append({"path": path, "reason": "adjudication object required"})
            continue
        cid = row.get("claim_id")
        if not isinstance(cid, str) or cid not in expected_set:
            issues.append({"path": path + ".claim_id", "reason": "unknown or missing conflict claim_id"})
            continue
        if cid in seen:
            issues.append({"path": path + ".claim_id", "reason": "duplicate claim_id"})
            continue
        seen.add(cid)
        adopt = row.get("adopt")
        if adopt not in allowed_adopt:
            issues.append({"path": path + ".adopt", "reason": "adopt must be main|entailment|independent"})
        holds_raw = row.get("claim_holds")
        holds = coerce_explicit_bool(holds_raw)
        if holds is None:
            issues.append({"path": path + ".claim_holds", "reason": "explicit boolean required"})
        elif type(holds_raw) is not bool:
            row["claim_holds"] = holds
            row["_claim_holds_coerced_from"] = holds_raw
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            issues.append({"path": path + ".rationale", "reason": "nonempty rationale required"})
        cited = row.get("cited_turns")
        if not isinstance(cited, list) or any(not isinstance(x, str) or not x.strip() for x in cited):
            issues.append({"path": path + ".cited_turns", "reason": "cited_turns must be a list of strings"})
            cited_ok = False
        else:
            cited_ok = True
            if turn_ids and any(x not in turn_ids for x in cited):
                # soft: allow unknown cites as format-warn only when holds true we still require nonempty
                pass
        if holds is True and cited_ok and not cited:
            issues.append({"path": path, "reason": "claim_holds=true requires nonempty cited_turns"})
        if holds is True and cited_ok:
            from experiments.tau_multiuser_voice.obligation_evidence import support_error
            evidence = request.get('evidence_under_evaluation') or {}
            claim = next((c for c in evidence.get('claims', []) if c.get('claim_id') == cid), {})
            srcs = {s['source_id']: s for s in evidence.get('source_catalog', [])}
            ord_map = {}
            for t in (request.get('dialog_transcript') or {}).get('turns', []):
                if 'ord' in t and 'source_id' in t:
                    ord_map[str(t['ord'])] = t['source_id']
                    ord_map['ord:'+str(t['ord'])] = t['source_id']
            resolved = [ord_map.get(x, x) for x in cited]
            problem = support_error(claim, resolved, srcs)
            if problem: issues.append({'path':path, 'reason':problem})
        dwb = row.get("disagree_with_both")
        dwb_c = coerce_explicit_bool(dwb)
        if dwb_c is None:
            issues.append({"path": path + ".disagree_with_both", "reason": "explicit boolean required"})
        elif type(dwb) is not bool:
            row["disagree_with_both"] = dwb_c
        if adopt == "independent" and dwb_c is False:
            # not hard-fail; independent may still agree with one side partially
            pass
        audit["adjudications"].append(dict(row))

    for cid in expected:
        if cid not in seen:
            issues.append({"path": "adjudications", "reason": "missing conflict claim adjudication", "claim_id": cid})
    # Extra adjudications beyond conflict set already rejected via claim_id check
    audit["evaluation_valid"] = not issues
    if issues:
        raise JudgeEvidenceInvalid(issues, audit)
    return audit


def conflict_claim_ids_from_issues(issues: Sequence[Any] | None) -> list[str]:
    """Extract claim_ids from pure disagreement JEI issues (stable order)."""
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return out
    for item in issues:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("reason", "")) != DISAGREEMENT_REASON:
            continue
        cid = item.get("claim_id")
        if isinstance(cid, str) and cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def apply_third_judge_to_main_payload(
    main_payload: dict[str, Any],
    adjudications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Patch primary payload when third says claim_holds=false for a completion/goal/fact/capability.

    claim_holds=true leaves primary rows unchanged. Returns a shallow-copied payload.
    """
    patched = deepcopy(main_payload)
    by_claim = {str(a.get("claim_id")): a for a in adjudications if isinstance(a, Mapping)}

    def flip_completion(gid: str, index: int, holds: bool) -> None:
        for row in patched.get("goal_results") or []:
            if not isinstance(row, dict) or row.get("goal_id") != gid:
                continue
            checks = row.get("completion_checks")
            if not isinstance(checks, list) or index >= len(checks) or not isinstance(checks[index], dict):
                return
            if not holds:
                checks[index]["passed"] = False
                checks[index]["third_judge_claim_holds"] = False
            else:
                checks[index]["third_judge_claim_holds"] = True
            return

    for cid, adj in by_claim.items():
        holds = adj.get("claim_holds")
        if holds is True:
            # annotate only
            if cid.startswith("completion:"):
                parts = cid.split(":")
                if len(parts) >= 3:
                    try:
                        flip_completion(parts[1], int(parts[2]), True)
                    except ValueError:
                        pass
            continue
        if holds is not False:
            continue
        if cid.startswith("completion:"):
            parts = cid.split(":")
            if len(parts) >= 3:
                try:
                    flip_completion(parts[1], int(parts[2]), False)
                except ValueError:
                    pass
        elif cid.startswith("goal:"):
            gid = cid[len("goal:"):]
            for row in patched.get("goal_results") or []:
                if isinstance(row, dict) and row.get("goal_id") == gid:
                    row["third_judge_claim_holds"] = False
        elif cid.startswith("fact:"):
            # fact:goal:id
            parts = cid.split(":", 2)
            if len(parts) == 3:
                gid, fid = parts[1], parts[2]
                for row in patched.get("fact_checks") or []:
                    if isinstance(row, dict) and row.get("goal_id") == gid and row.get("id") == fid:
                        row["passed"] = False
                        row["third_judge_claim_holds"] = False
        elif cid.startswith("capability:"):
            aid = cid[len("capability:"):]
            for row in patched.get("capability_results") or []:
                if isinstance(row, dict) and row.get("assertion_id") == aid:
                    row["passed"] = False
                    row["third_judge_claim_holds"] = False
    patched["third_judge_score_label"] = THIRD_JUDGE_SCORE_LABEL
    patched["correlated_adjudication"] = True
    return patched
