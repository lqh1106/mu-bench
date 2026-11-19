"""Shared frozen-plan path resolution for direct and unified run layouts."""

from __future__ import annotations

from pathlib import Path


def resolve_frozen_plan_path(root: str | Path, case_id: str) -> Path:
    """Resolve one immutable plan from either a direct or sharded run root.

    Direct runners write ``<root>/<case_id>/frozen_event_plan.json``.  The
    unified coordinator writes ``<root>/worker_XX/<case_id>/...``.  Supporting
    both layouts lets a reference run produced by either entrypoint become the
    shared plan source without copying or editing artifacts.
    """

    plan_root = Path(root).resolve()
    direct = plan_root / case_id / "frozen_event_plan.json"
    if direct.exists():
        return direct
    sharded = sorted(
        plan_root.glob(f"worker_*/{case_id}/frozen_event_plan.json")
    )
    if len(sharded) == 1:
        return sharded[0]
    if len(sharded) > 1:
        raise ValueError(
            f"shared frozen plan is ambiguous for {case_id}: "
            + ", ".join(str(path) for path in sharded)
        )
    raise FileNotFoundError(
        f"no frozen_event_plan.json for {case_id} below {plan_root}"
    )
