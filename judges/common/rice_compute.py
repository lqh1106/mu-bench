#!/usr/bin/env python3
"""Vendored RICE compute for E6 formal post-score (from Survey rice_offline_sidecar_draft.py).

Contract: mu-bench-rice-offline-v1.1-l0-digfix
Public: compute_rice(sealed, overlay), OVERLAY_REL, RICE_VERSION, load_json

Formal online path writes rice_score.json INTO the case dir via formal_post_score_rice.py
(not via this module's CLI main, which still refuses in-tree OUT for offline safety).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

RICE_VERSION = "mu-bench-rice-offline-v1.1-l0-digfix"
OVERLAY_REL = Path("_v15_rescore") / "recognition_floor_split_score.json"

R_CATS = ("participant_scene", "addressee", "discourse_provenance")
C_CATS = ("multi_thread", "targeted_repair", "response_rendering")
E_CATS = ("fairness", "group_collaboration")


def load_json(p: Path) -> Optional[dict]:
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_unit(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("pass_rate", "score", "rate", "success"):
            if k in v:
                return as_unit(v[k])
    return None


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def dig_category_rate(score: dict, cat: str) -> Optional[float]:
    # L0 measured: category floats mostly live in capability_scores
    cap = score.get("capability_scores")
    if isinstance(cap, dict) and cat in cap:
        u = as_unit(cap[cat])
        if u is not None:
            return u
    for bag_key in (
        "assertion_category_rates",
        "category_scores",
        "category_pass_rates",
    ):
        bag = score.get(bag_key)
        if isinstance(bag, dict) and cat in bag:
            u = as_unit(bag[cat])
            if u is not None:
                return u
    diag = score.get("diagnostics")
    if isinstance(diag, dict):
        for bag_key in ("category_pass_rate", "category_scores", "assertion_category_rates"):
            bag = diag.get(bag_key)
            if isinstance(bag, dict) and cat in bag:
                u = as_unit(bag[cat])
                if u is not None:
                    return u
    for list_key in ("assertions", "assertion_results", "assertion_reports"):
        items = score.get(list_key)
        if not isinstance(items, list):
            continue
        matched = []
        for a in items:
            if not isinstance(a, dict):
                continue
            if a.get("category") != cat:
                continue
            passed = a.get("passed", a.get("success", a.get("pass")))
            u = as_unit(passed)
            if u is not None:
                matched.append(u)
        if matched:
            return mean(matched)
    return None


def recognition_split(sealed: dict, overlay: Optional[dict]) -> dict:
    # L0 measured: recognition_split is overlay-only (do not fall back to sealed)
    del sealed  # sealed unused for split by contract
    split: dict = {}
    if overlay:
        o_split = overlay.get("recognition_split")
        if isinstance(o_split, dict):
            for k, v in o_split.items():
                if v is not None:
                    split[k] = v
        for k in ("spoken_anchor_rate", "behavioral_continuity_rate", "robust_return_routing_rate"):
            if overlay.get(k) is not None:
                split[k] = overlay[k]
    return split


def capability_rate(score: dict, *names: str) -> Optional[float]:
    cap = score.get("capability_scores")
    if not isinstance(cap, dict):
        return None
    for n in names:
        if n in cap:
            u = as_unit(cap[n])
            if u is not None:
                return u
    return None


def compute_rice(sealed: dict, overlay: Optional[dict]) -> dict:
    split = recognition_split(sealed, overlay)
    units_used = {"R": [], "C": [], "I": [], "E": []}

    r_units: list[float] = []
    routing_approx = False
    for cat in R_CATS:
        u = dig_category_rate(sealed, cat)
        if u is not None:
            r_units.append(u)
            units_used["R"].append(cat)
    sa = as_unit(split.get("spoken_anchor_rate"))
    if sa is not None:
        r_units.append(sa)
        units_used["R"].append("spoken_anchor_rate")
    if r_units:
        routing_score = mean(r_units)
    else:
        routing_score = as_unit(sealed.get("semantic_routing_score"))
        if routing_score is not None:
            routing_approx = True
            units_used["R"].append("semantic_routing_score(approx)")

    c_units: list[float] = []
    per_c = [(cat, dig_category_rate(sealed, cat)) for cat in C_CATS]
    present_c = [(c, u) for c, u in per_c if u is not None]
    if present_c:
        for c, u in present_c:
            c_units.append(u)
            units_used["C"].append(c)
    else:
        ir = as_unit(sealed.get("interruption_recovery_score"))
        if ir is not None:
            c_units.append(ir)
            units_used["C"].append("interruption_recovery_score")
    bc = as_unit(split.get("behavioral_continuity_rate"))
    if bc is None:
        bc = capability_rate(
            sealed,
            "return_continuity",
            "return_behavior_continuity",
            "behavioral_continuity",
        )
        if bc is not None:
            units_used["C"].append("capability:return_continuity")
    else:
        units_used["C"].append("behavioral_continuity_rate")
    if bc is not None:
        c_units.append(bc)
    continuity_score = mean(c_units)

    i_units: list[float] = []
    isolation_mode = None
    for cat in ("privacy", "authorization", "isolation"):
        u = dig_category_rate(sealed, cat)
        if u is not None:
            i_units.append(u)
            units_used["I"].append(cat)
    if i_units:
        isolation_mode = "assertion_category"
    else:
        for key in ("privacy_success", "authorization_success"):
            if key in sealed and sealed[key] is not None:
                i_units.append(as_unit(sealed[key]))  # type: ignore
                units_used["I"].append(key)
        ua = sealed.get("unauthorized_action_attempts", sealed.get("unauthorized_attempts"))
        if not i_units and ua is not None:
            i_units.append(1.0 if ua == 0 else 0.0)
            units_used["I"].append("unauthorized_action_attempts==0")
        if i_units:
            isolation_mode = "weak_task_boolean_proxy"
    isolation_score = mean(i_units)

    e_units: list[float] = []
    equity_approx = False
    for cat in E_CATS:
        u = dig_category_rate(sealed, cat)
        if u is not None:
            e_units.append(u)
            units_used["E"].append(cat)
    # v1.1 digfix: fairness/quiet policy scores — overlay FIRST, then sealed
    for key in ("fairness_policy_score", "quiet_input_integration_score"):
        u = None
        src = key
        if overlay is not None:
            u = as_unit(overlay.get(key))
            if u is not None:
                src = f"overlay:{key}"
        if u is None:
            u = as_unit(sealed.get(key))
            if u is not None:
                src = f"sealed:{key}"
        if u is not None:
            e_units.append(u)
            units_used["E"].append(src)
    if e_units:
        equity_score = mean(e_units)
    else:
        equity_score = as_unit(sealed.get("floor_control_score"))
        if equity_score is not None:
            equity_approx = True
            units_used["E"].append("floor_control_score(approx)")

    axes = {
        "routing_score": routing_score,
        "continuity_score": continuity_score,
        "isolation_score": isolation_score,
        "equity_score": equity_score,
    }
    interaction = mean(v for v in axes.values() if v is not None)

    return {
        "rice_contract_version": RICE_VERSION,
        **axes,
        "interaction_score_rice": interaction,
        "routing_approx": routing_approx,
        "equity_approx": equity_approx,
        "isolation_mode": isolation_mode,
        "units_used": units_used,
        "timing_excluded": True,
        "overwrite_sealed": False,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: rice_offline_sidecar_draft.py CASE_DIR OUT_JSON", file=sys.stderr)
        return 2
    case_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    # Hard guard: never write inside the L0 case tree
    try:
        out_path.relative_to(case_dir)
        print(f"REFUSE: OUT_JSON must not be under CASE_DIR ({case_dir})", file=sys.stderr)
        return 3
    except ValueError:
        pass
    if out_path.name in {"score.json", "recognition_floor_split_score.json"}:
        print("REFUSE: OUT_JSON collides with sealed names", file=sys.stderr)
        return 3
    if "_v15_rescore" in out_path.parts:
        print("REFUSE: OUT_JSON under _v15_rescore", file=sys.stderr)
        return 3

    sealed_path = case_dir / "score.json"
    sealed = load_json(sealed_path)
    if sealed is None:
        print(f"missing {sealed_path}", file=sys.stderr)
        return 1
    overlay_path = case_dir / OVERLAY_REL
    overlay = load_json(overlay_path)
    out = compute_rice(sealed, overlay)
    out["model"] = case_dir.parent.name
    out["case_id"] = case_dir.name
    out["source_score"] = str(sealed_path)
    out["source_rescore_overlay"] = str(overlay_path) if overlay is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
