#!/usr/bin/env python3
"""B2 offline: attempt-local floor A + write_rice + stratum. A_OUT only. Never migrate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="B2 attempt-local RICE fill (no migrate)")
    p.add_argument("--attempt-case-dir", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Hard refuse formal post-score path
    if os.environ.get("MUBENCH_FORMAL_POST_SCORE") == "1":
        print(json.dumps({"status": "refused_formal_post_score"}, ensure_ascii=False, indent=2))
        return 2

    runtime = args.runtime_root.resolve()
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from formal_post_score_rice import run_attempt_local_rice

    res = run_attempt_local_rice(
        args.attempt_case_dir,
        runtime_root=runtime,
        bridge=args.bridge,
        env_file=args.env,
        model=args.model,
        dry_run=args.dry_run,
    )
    # Always try receipt on attempt
    try:
        from formal_post_score_rice import _atomic_write_json
        _atomic_write_json(Path(args.attempt_case_dir) / "_attempt_local_rice_receipt.json", res)
    except Exception as exc:
        res["receipt_write_error"] = str(exc)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    ok = res.get("status") in {
        "ok",
        "rice_ok_floor_failed",
        "dry_run",
        "skipped_missing_env",
        "skipped_formal_post_score",
    }
    return 0 if ok or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
