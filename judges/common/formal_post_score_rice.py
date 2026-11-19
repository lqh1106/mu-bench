#!/usr/bin/env python3
"""E6 formal post-score: MUST-2 migrate attempts→flat FORMAL, then floor (path A) + RICE.

Locks:
  MUST-1 overlay → formal_case/_v15_rescore/recognition_floor_split_score.json
  MUST-2 attempts first; after score seal, atomic migrate/hardlink into FORMAL/{model}/{case}/
  MUST-3 online formal only rescore_recognition_floor_split.py (semantic_judge)
Never mutate sealed score.json / traj. After migrate+RICE receipt exists, may delete played/segment wav (PPO) and mono session stems user_mixed_native/assistant_native (+ closed *_8k), and audio/oracle/* reference stems; keep only audio/both.wav (Jay 2026-10-05). Failures do not roll back score. No stock backfill for mono stems.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

def _resolve_floor_scratch_root(formal_case_dir: Path) -> Path:
    """F5: floor scratch on shared storage — never bare /tmp.

    Order:
      1) MUBENCH_FLOOR_SCRATCH_ROOT (explicit shared dir)
      2) TMPDIR/TEMP/TMP if set and NOT under /tmp
      3) formal_case_dir/_floor_scratch (alongside the attempt/formal case)

    Raises RuntimeError if the only available default would be /tmp.
    """
    candidates: list[Path] = []
    env_root = os.environ.get("MUBENCH_FLOOR_SCRATCH_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for key in ("TMPDIR", "TEMP", "TMP"):
        val = os.environ.get(key)
        if val:
            candidates.append(Path(val))
    candidates.append(Path(formal_case_dir) / "_floor_scratch")

    def _is_forbidden_tmp(p: Path) -> bool:
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        parts = resolved.as_posix()
        # allow /tmp only if it is somehow a bind-mount elsewhere — still forbid
        return parts == "/tmp" or parts.startswith("/tmp/")

    for cand in candidates:
        if _is_forbidden_tmp(cand):
            continue
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    raise RuntimeError(
        "F5: refusing to use /tmp for floor scratch; set MUBENCH_FLOOR_SCRATCH_ROOT "
        "or TMPDIR to a shared-storage path"
    )


OVERLAY_NAME = "recognition_floor_split_score.json"
OVERLAY_REL = Path("_v15_rescore") / OVERLAY_NAME
RICE_NAME = "rice_score.json"
PLAYED_SEGMENT_DELETE_RECEIPT = "_played_segment_delete_receipt.json"
MONO_STEM_DELETE_RECEIPT = "_mono_stem_delete_receipt.json"

# Exact relative paths only. Never both.wav / oracle.
# native: open-weight finals. 8k: closed-source Grok/OpenAI finals.
_MONO_SESSION_STEM_RELS = (
    "audio/user_mixed_native.wav",
    "audio/assistant_native.wav",
    "audio/user_mixed_8k.wav",
    "audio/assistant_8k.wav",
)

# Relative path prefixes / name patterns for PPO played-segment audio only.
# Do NOT delete audio/both.wav here. Mono stems and audio/oracle/* are deleted by separate functions after score+RICE.
_PLAYED_SEGMENT_DIR_PREFIXES = (
    "played_prefix_observer/",
    "played_segments/",
    "ppo_played/",
    "ppo_segments/",
)
_PLAYED_SEGMENT_NAME_PREFIXES = (
    "played_prefix_",
    "played_segment_",
    "segment_played_",
)
_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".webm"}


def _is_played_segment_audio(rel: Path) -> bool:
    """True only for segment-level already-played audio artifacts."""
    if rel.suffix.lower() not in _AUDIO_SUFFIXES:
        return False
    s = rel.as_posix()
    low = s.lower()
    for pref in _PLAYED_SEGMENT_DIR_PREFIXES:
        if low.startswith(pref) or f"/{pref}" in f"/{low}":
            return True
    # Top-level case cache: segments/<id>/segment.wav only (exact 3-part rel).
    # Do NOT broad-match every wav under segments/ (no nested extras / non-segment names).
    parts = rel.parts
    if (
        len(parts) == 3
        and parts[0] == "segments"
        and parts[2] == "segment.wav"
    ):
        return True
    name = rel.name.lower()
    return any(name.startswith(p) for p in _PLAYED_SEGMENT_NAME_PREFIXES)


def delete_played_segment_audio(
    *roots: Path,
    dry_run: bool = False,
    require_rice: bool = True,
) -> dict[str, Any]:
    """Delete played/segment wav under case roots after score+RICE exist.

    Keeps score/RICE/trace/_v15_rescore/ORIG and audio/both.wav (oracle is handled by delete_case_oracle).
    Safe to call on formal and attempt dirs. Hardlink pairs: unlink EACH directory entry
    (do not skip attempt path just because formal path already consumed the inode).
    """
    cleaned_roots = [str(Path(r)) for r in roots if Path(r).is_dir()]
    deleted: list[dict[str, Any]] = []
    per_root: dict[str, dict[str, Any]] = {}

    rice_ok = any((Path(r) / RICE_NAME).is_file() for r in roots if Path(r).is_dir())
    score_ok = any((Path(r) / "score.json").is_file() for r in roots if Path(r).is_dir())
    if require_rice and not rice_ok:
        return {
            "status": "skipped_no_rice",
            "deleted_count": 0,
            "bytes": 0,
            "paths": [],
            "per_root": {},
        }
    if not score_ok:
        return {
            "status": "skipped_no_score",
            "deleted_count": 0,
            "bytes": 0,
            "paths": [],
            "per_root": {},
        }

    total_bytes = 0
    # Dedupe by absolute path only. NEVER skip a distinct path that shares an inode
    # with an already-unlinked formal hardlink — that left attempt wavs at nlink=1.
    seen_path: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        root_key = str(root)
        root_deleted: list[dict[str, Any]] = []
        root_bytes = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if not _is_played_segment_audio(rel):
                continue
            # never touch sealed / overlay / ORIG markers even if misnamed
            parts = set(rel.parts)
            if "score.json" in parts or RICE_NAME in parts or "_v15_rescore" in parts:
                continue
            if any(part.startswith("ORIG") for part in rel.parts):
                continue
            abs_s = str(path)
            if abs_s in seen_path:
                continue
            seen_path.add(abs_s)
            st = path.stat()
            entry = {
                "path": abs_s,
                "rel": rel.as_posix(),
                "bytes": int(st.st_size),
                "root": root_key,
                "nlink_before": int(st.st_nlink),
                "inode": int(st.st_ino),
            }
            if not dry_run:
                path.unlink(missing_ok=True)
            deleted.append(entry)
            root_deleted.append(entry)
            total_bytes += int(st.st_size)
            root_bytes += int(st.st_size)
        if dry_run:
            root_status = "dry_run_deleted" if root_deleted else "noop_no_matches"
        else:
            root_status = "deleted" if root_deleted else "noop_no_matches"
        per_root[root_key] = {
            "status": root_status,
            "deleted_count": len(root_deleted),
            "bytes": root_bytes,
            "paths": [d["rel"] for d in root_deleted[:50]],
            "paths_truncated": len(root_deleted) > 50,
        }

    any_deleted = any(v["deleted_count"] > 0 for v in per_root.values())
    if dry_run:
        status = "dry_run_deleted" if any_deleted else "noop_no_matches"
    else:
        status = "deleted" if any_deleted else "noop_no_matches"
    receipt = {
        "status": status,
        "deleted_count": len(deleted),
        "bytes": total_bytes,
        "paths": [d["rel"] for d in deleted[:50]],
        "paths_truncated": len(deleted) > 50,
        "roots": cleaned_roots,
        "per_root": per_root,
    }
    # write receipt on formal-looking root (prefer root that has rice_score)
    if not dry_run:
        for root in roots:
            root = Path(root)
            if (root / RICE_NAME).is_file():
                _atomic_write_json(root / PLAYED_SEGMENT_DELETE_RECEIPT, receipt)
                receipt["receipt"] = str(root / PLAYED_SEGMENT_DELETE_RECEIPT)
                break
    return receipt



_ORACLE_NAME = re.compile(r"(?:mixed_user|[A-Za-z0-9_]+_stem)\.wav")


def delete_case_oracle(*roots: Path, dry_run: bool = False, require_score: bool = True) -> dict[str, Any]:
    """After score seal, remove only this case's audio/oracle reference stems.

    The directory must be exactly <case>/audio/oracle. Only regular files named
    mixed_user.wav or <speaker>_stem.wav are unlinked. both.wav is not touched.
    """
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    total_bytes = 0
    seen_inode: set[tuple[int, int]] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            continue
        if require_score and not (root / "score.json").is_file():
            skipped.append({"root": str(root), "reason": "no_score"})
            continue
        audio = root / "audio"
        oracle = audio / "oracle"
        if not oracle.exists():
            continue
        if audio.is_symlink() or oracle.is_symlink() or not oracle.is_dir():
            skipped.append({"root": str(root), "reason": "symlink_or_not_dir"})
            continue
        try:
            resolved_root = root.resolve()
            resolved = oracle.resolve()
        except OSError:
            skipped.append({"root": str(root), "reason": "resolve_failed"})
            continue
        if resolved != resolved_root / "audio" / "oracle" or resolved.parent != resolved_root / "audio":
            skipped.append({"root": str(root), "reason": "resolved_outside_case_audio"})
            continue
        for child in list(oracle.iterdir()):
            if child.is_symlink() or not child.is_file():
                skipped.append({"path": str(child), "reason": "not_regular_file"})
                continue
            if _ORACLE_NAME.fullmatch(child.name) is None:
                skipped.append({"path": str(child), "reason": "unexpected_name"})
                continue
            try:
                if child.resolve().parent != resolved:
                    skipped.append({"path": str(child), "reason": "resolved_outside_oracle"})
                    continue
                st = child.stat()
            except OSError:
                skipped.append({"path": str(child), "reason": "stat_failed"})
                continue
            key = (st.st_dev, st.st_ino)
            if not dry_run:
                child.unlink()
            deleted.append({"path": str(child), "bytes": int(st.st_size)})
            if key not in seen_inode:
                total_bytes += int(st.st_size)
            seen_inode.add(key)
        if not dry_run:
            try:
                oracle.rmdir()
            except OSError:
                skipped.append({"root": str(oracle), "reason": "oracle_not_empty"})
    return {
        "status": "dry_run" if dry_run else "deleted",
        "deleted_count": len(deleted),
        "bytes": total_bytes,
        "skipped": skipped,
    }


def delete_mono_session_stems(
    *roots: Path,
    dry_run: bool = False,
    require_rice: bool = True,
) -> dict[str, Any]:
    """After score+RICE: drop user/assistant mono finals; keep audio/both.wav (oracle handled by delete_case_oracle).

    Live formal_post hook only — no stock backfill script. Does not touch score/RICE/trace/ORIG
    or played-segment rules. Hardlink pairs: unlink EACH directory entry (formal + attempt).
    """
    cleaned_roots = [str(Path(r)) for r in roots if Path(r).is_dir()]
    rice_ok = any((Path(r) / RICE_NAME).is_file() for r in roots if Path(r).is_dir())
    score_ok = any((Path(r) / "score.json").is_file() for r in roots if Path(r).is_dir())
    if require_rice and not rice_ok:
        return {"status": "skipped_no_rice", "deleted_count": 0, "bytes": 0, "paths": [], "per_root": {}}
    if not score_ok:
        return {"status": "skipped_no_score", "deleted_count": 0, "bytes": 0, "paths": [], "per_root": {}}

    deleted: list[dict[str, Any]] = []
    per_root: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    seen_path: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        root_key = str(root)
        root_deleted: list[dict[str, Any]] = []
        root_bytes = 0
        # refuse if both.wav missing — keep monos rather than strand the case with no session audio
        both = root / "audio" / "both.wav"
        if not both.is_file():
            per_root[root_key] = {
                "status": "skipped_no_both",
                "deleted_count": 0,
                "bytes": 0,
                "paths": [],
            }
            continue
        for rel_s in _MONO_SESSION_STEM_RELS:
            path = root / rel_s
            if not path.is_file():
                continue
            # hard safety: never match both/oracle by typo
            rel = path.relative_to(root)
            if rel.as_posix() == "audio/both.wav" or "oracle" in rel.parts:
                continue
            abs_s = str(path)
            if abs_s in seen_path:
                continue
            seen_path.add(abs_s)
            st = path.stat()
            entry = {
                "path": abs_s,
                "rel": rel.as_posix(),
                "bytes": int(st.st_size),
                "root": root_key,
                "nlink_before": int(st.st_nlink),
                "inode": int(st.st_ino),
            }
            if not dry_run:
                path.unlink(missing_ok=True)
            deleted.append(entry)
            root_deleted.append(entry)
            total_bytes += int(st.st_size)
            root_bytes += int(st.st_size)
        if dry_run:
            root_status = "dry_run_deleted" if root_deleted else "noop_no_matches"
        else:
            root_status = "deleted" if root_deleted else "noop_no_matches"
        per_root[root_key] = {
            "status": root_status,
            "deleted_count": len(root_deleted),
            "bytes": root_bytes,
            "paths": [d["rel"] for d in root_deleted],
        }

    any_deleted = any(v.get("deleted_count", 0) > 0 for v in per_root.values())
    if dry_run:
        status = "dry_run_deleted" if any_deleted else "noop_no_matches"
    else:
        status = "deleted" if any_deleted else "noop_no_matches"
    receipt = {
        "status": status,
        "deleted_count": len(deleted),
        "bytes": total_bytes,
        "paths": [d["rel"] for d in deleted],
        "roots": cleaned_roots,
        "per_root": per_root,
        "kept": ["audio/both.wav", "audio/oracle/*"],
    }
    if not dry_run:
        for root in roots:
            root = Path(root)
            if (root / RICE_NAME).is_file():
                _atomic_write_json(root / MONO_STEM_DELETE_RECEIPT, receipt)
                receipt["receipt"] = str(root / MONO_STEM_DELETE_RECEIPT)
                break
    return receipt



# Files we never overwrite on conflict with different bytes
SEALED_NAMES = {"score.json", "trace.json"}


def _load_json(p: Path) -> Optional[dict]:
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def resolve_formal_case_dir(
    attempt_case_dir: Path,
    *,
    formal_root: Path,
    model: str,
    case_id: Optional[str] = None,
) -> Path:
    cid = case_id or attempt_case_dir.name
    # supervisor attempts often name dirs like "{idx}_{case_id}"
    if "_" in cid and case_id is None:
        # prefer score.json case_id when present
        sealed = _load_json(attempt_case_dir / "score.json") or {}
        cid = str(sealed.get("case_id") or cid)
        if cid.startswith(tuple(f"{i}_" for i in range(10))) and sealed.get("case_id"):
            cid = str(sealed["case_id"])
        elif sealed.get("case_id"):
            cid = str(sealed["case_id"])
    return (formal_root / model / cid).resolve()


def _same_file_bytes(a: Path, b: Path) -> bool:
    if not a.is_file() or not b.is_file():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    return a.read_bytes() == b.read_bytes()


def migrate_attempt_to_formal(
    attempt_case_dir: Path,
    formal_case_dir: Path,
    *,
    migrate_mode: str = "hardlink",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy/hardlink attempt tree into flat formal case dir. Never rewrite differing sealed score."""
    attempt_case_dir = attempt_case_dir.resolve()
    formal_case_dir = formal_case_dir.resolve()
    score_src = attempt_case_dir / "score.json"
    if not score_src.is_file():
        return {"status": "error", "error": "missing attempt score.json"}

    dest_score = formal_case_dir / "score.json"
    if dest_score.is_file() and not _same_file_bytes(score_src, dest_score):
        return {
            "status": "error",
            "error": "formal score.json exists with different content; refuse overwrite",
            "formal_score": str(dest_score),
        }

    if dry_run:
        return {
            "status": "dry_run_migrate",
            "attempt": str(attempt_case_dir),
            "formal": str(formal_case_dir),
            "mode": migrate_mode,
        }

    formal_case_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    skipped = 0
    for root, dirs, files in os.walk(attempt_case_dir):
        rel_root = Path(root).relative_to(attempt_case_dir)
        # skip ephemeral
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in files:
            src = Path(root) / name
            dst = formal_case_dir / rel_root / name
            if dst.exists():
                if _same_file_bytes(src, dst) or (src.samefile(dst) if dst.exists() else False):
                    skipped += 1
                    continue
                if name in SEALED_NAMES:
                    return {
                        "status": "error",
                        "error": f"refuse overwrite differing sealed {name}",
                        "dst": str(dst),
                    }
                # non-sealed conflict: leave existing
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if migrate_mode == "hardlink":
                try:
                    os.link(src, dst)
                    linked += 1
                    continue
                except OSError:
                    pass
            shutil.copy2(src, dst)
            copied += 1

    return {
        "status": "migrated",
        "attempt": str(attempt_case_dir),
        "formal": str(formal_case_dir),
        "linked": linked,
        "copied": copied,
        "skipped": skipped,
        "mode": migrate_mode,
    }


DIGFIX_CONTRACT = "mu-bench-rice-offline-v1.1-l0-digfix"
PARTIAL_COHORTS = frozenset({"option1_partial", "option1", "partial"})


def overlay_provides_digfix_units(overlay: Optional[dict]) -> bool:
    """True only when overlay can supply digfix recognition units (not mock / empty)."""
    if not isinstance(overlay, dict):
        return False
    if overlay.get("mock") is True:
        return False
    split = overlay.get("recognition_split")
    split = split if isinstance(split, dict) else {}
    for key in (
        "spoken_anchor_rate",
        "behavioral_continuity_rate",
        "robust_return_routing_rate",
    ):
        if overlay.get(key) is not None or split.get(key) is not None:
            return True
    return bool(split)


def is_partial_rice_receipt(rice: dict) -> bool:
    cohort = str(rice.get("cohort") or rice.get("receipt_tag") or "").strip().lower()
    if cohort in PARTIAL_COHORTS or "option1_partial" in cohort:
        return True
    if rice.get("rice_completeness") in {"incomplete", "partial"}:
        # incomplete alone is not enough to label option1, but must not skip
        pass
    return False


def assess_comparable_to_l0_digfix(overlay: Optional[dict], rice_axes: dict) -> bool:
    """Set comparable only for real digfix overlay units; never for mock/partial."""
    del rice_axes  # reserved for future axis-presence gates
    return overlay_provides_digfix_units(overlay)


def _should_skip_digfix(formal_case_dir: Path) -> bool:
    """Skip recompute ONLY for complete digfix. option1_partial / mock / incomplete never skip."""
    overlay_path = formal_case_dir / OVERLAY_REL
    rice = _load_json(formal_case_dir / RICE_NAME)
    if not overlay_path.is_file() or not isinstance(rice, dict):
        return False
    if is_partial_rice_receipt(rice):
        return False
    if rice.get("comparable_to_l0_digfix") is not True:
        return False
    if rice.get("rice_completeness") != "complete":
        return False
    if rice.get("rice_contract_version") != DIGFIX_CONTRACT:
        return False
    # Survey pin: source_rescore_overlay must point at this case's _v15_rescore overlay
    src_ov = rice.get("source_rescore_overlay")
    if not src_ov:
        return False
    try:
        if Path(src_ov).resolve() != overlay_path.resolve():
            return False
    except OSError:
        return False
    overlay = _load_json(overlay_path)
    if not overlay_provides_digfix_units(overlay):
        return False
    return True


def run_floor_path_a(
    formal_case_dir: Path,
    *,
    runtime_root: Path,
    bridge: Path,
    env_file: Path,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """MUST-3 path A only. Remap OUT into formal_case/_v15_rescore/ (MUST-1)."""
    formal_case_dir = formal_case_dir.resolve()
    score = formal_case_dir / "score.json"
    dest = formal_case_dir / OVERLAY_REL
    err_path = formal_case_dir / "_v15_rescore" / "_floor_overlay_error.json"

    if os.environ.get("E6_SMOKE_MOCK_FLOOR") == "1":
        if dry_run:
            return {"status": "dry_run_floor_mock", "dest": str(dest)}
        mock = {
            "evaluation_only": True,
            "mock": True,
            "recognition_split": {
                "spoken_anchor_rate": 0.5,
                "behavioral_continuity_rate": 0.5,
                "robust_return_routing_rate": 0.5,
            },
            "fairness_policy_score": 0.5,
            "quiet_input_integration_score": 0.5,
            "case_id": (_load_json(score) or {}).get("case_id"),
        }
        _atomic_write_json(dest, mock)
        return {"status": "floor_mock", "dest": str(dest)}

    if dry_run:
        return {"status": "dry_run_floor", "dest": str(dest)}

    script = runtime_root / "rescore_recognition_floor_split.py"
    if not script.is_file():
        payload = {"error": "missing rescore_recognition_floor_split.py", "script": str(script)}
        _atomic_write_json(err_path, payload)
        return {"status": "floor_error", **payload}

    # F5: persist floor scratch under shared storage (never /tmp); keep for forensics.
    scratch_root = _resolve_floor_scratch_root(formal_case_dir)
    out_root = Path(tempfile.mkdtemp(prefix="e6_floor_", dir=str(scratch_root)))
    scratch_meta = {
        "scratch_root": str(scratch_root),
        "out_root": str(out_root),
        "policy": "mu-floor-scratch-shared-v1",
    }
    try:
        _atomic_write_json(out_root / "_scratch_meta.json", scratch_meta)
    except Exception:
        pass
    cmd = [
        sys.executable,
        str(script),
        "--score-json",
        str(score),
        "--output-root",
        str(out_root),
        "--bridge",
        str(bridge),
        "--env",
        str(env_file),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(runtime_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        payload = {"error": str(exc), "error_type": type(exc).__name__, **scratch_meta}
        _atomic_write_json(err_path, payload)
        return {"status": "floor_error", **payload}

    produced = out_root / model / formal_case_dir.name / OVERLAY_NAME
    # also accept any recognition_floor_split_score.json under out_root
    if not produced.is_file():
        found = list(out_root.rglob(OVERLAY_NAME))
        produced = found[0] if found else produced

    # Keep a stable pointer under the formal/attempt case for forensics.
    persist_link = formal_case_dir / "_floor_scratch" / "latest"
    try:
        persist_link.parent.mkdir(parents=True, exist_ok=True)
        marker = {
            **scratch_meta,
            "returncode": proc.returncode,
            "produced": str(produced) if produced.is_file() else None,
        }
        _atomic_write_json(persist_link.parent / "latest_receipt.json", marker)
    except Exception:
        pass

    if proc.returncode != 0 or not produced.is_file():
        payload = {
            "error": "rescore path A failed",
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "expected": str(produced),
            **scratch_meta,
        }
        _atomic_write_json(err_path, payload)
        return {"status": "floor_error", **payload}

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    shutil.copy2(produced, tmp)
    os.replace(tmp, dest)
    return {"status": "floor_ok", "dest": str(dest), "src_tmp": str(produced), **scratch_meta}


def write_rice(formal_case_dir: Path, *, model: str, dry_run: bool = False) -> dict[str, Any]:
    from rice_compute import compute_rice, load_json, OVERLAY_REL as _OR  # noqa: F401

    formal_case_dir = formal_case_dir.resolve()
    sealed_path = formal_case_dir / "score.json"
    overlay_path = formal_case_dir / OVERLAY_REL
    sealed = load_json(sealed_path)
    if sealed is None:
        return {"status": "rice_error", "error": "missing formal score.json"}
    overlay = load_json(overlay_path)
    out = compute_rice(sealed, overlay)
    out["model"] = model
    out["case_id"] = sealed.get("case_id") or formal_case_dir.name
    out["source_score"] = str(sealed_path)
    out["source_rescore_overlay"] = str(overlay_path) if overlay is not None else None
    comparable = assess_comparable_to_l0_digfix(overlay, out)
    out["comparable_to_l0_digfix"] = bool(comparable)
    out["rice_completeness"] = "complete" if comparable else "incomplete"
    if overlay is not None and overlay.get("mock") is True:
        out["cohort"] = "smoke_mock"
        out["comparable_to_l0_digfix"] = False
        out["rice_completeness"] = "incomplete"
    rice_path = formal_case_dir / RICE_NAME
    if dry_run:
        return {"status": "dry_run_rice", "path": str(rice_path), "preview": out}
    _atomic_write_json(rice_path, out)
    return {"status": "rice_ok", "path": str(rice_path), "rice_completeness": out["rice_completeness"]}



def post_score_status(formal_complete: bool, floor: Any, rice: Any, *, dry_run: bool = False) -> str:
    """Status label only (ISSUE-1, PM 2026-10-05 17:35). No scoring logic.

    ok                        score migrated AND floor_ok AND RICE complete
    floor_failed              score migrated, floor did not produce an overlay (floor_error / other)
    score_ok_rice_incomplete  score migrated, floor ok (or smoke mock) but RICE missing/incomplete
    incomplete                score not migrated (unchanged meaning)
    dry_run_score_ok          dry run: floor/RICE not executed
    """
    if not formal_complete:
        return "incomplete"
    if dry_run:
        return "dry_run_score_ok"
    floor_status = floor.get("status") if isinstance(floor, dict) else None
    rice_complete = (
        isinstance(rice, dict)
        and rice.get("status") == "rice_ok"
        and rice.get("rice_completeness") == "complete"
    )
    if floor_status == "floor_ok" and rice_complete:
        return "ok"
    if floor_status not in {"floor_ok", "floor_mock"}:
        return "floor_failed"
    return "score_ok_rice_incomplete"




def derive_incompleteness_stratum(score: dict[str, Any]) -> str:
    """B2 stratum for rice_score.json. Prefer B1 field; else termination; soft cap flag."""
    raw = score.get("incompleteness_stratum")
    if raw in ("logical_duration_cap", "proposal_contract_exhausted", "none"):
        return str(raw)
    ts = score.get("termination_state")
    if ts == "completed_logical_limit":
        return "logical_duration_cap"
    if ts == "completed_proposal_contract_exhausted":
        return "proposal_contract_exhausted"
    # Critic soft: honor logical_duration_cap_applied when termination missing / other
    if score.get("logical_duration_cap_applied") is True:
        return "logical_duration_cap"
    return "none"


def stamp_stratum_into_rice(attempt_dir: Path, score: dict[str, Any]) -> dict[str, Any]:
    """Post-write_rice stamp. Does not change write_rice signature. Never raises."""
    rice_path = Path(attempt_dir) / RICE_NAME
    if not rice_path.is_file():
        return {"status": "stamp_skipped_no_rice"}
    rice = _load_json(rice_path)
    if not isinstance(rice, dict):
        return {"status": "stamp_error", "error": "rice_unreadable"}
    stratum = derive_incompleteness_stratum(score if isinstance(score, dict) else {})
    rice["incompleteness_stratum"] = stratum
    for k in (
        "events_unreached_due_to_logical_cap",
        "events_unreached_due_to_proposal_contract",
        "required_events_unreached_due_to_cap",
    ):
        if isinstance(score, dict) and k in score:
            rice[k] = score[k]
    _atomic_write_json(rice_path, rice)
    return {"status": "stamp_ok", "incompleteness_stratum": stratum}


def run_attempt_local_rice(
    attempt_dir: Path,
    *,
    runtime_root: Path,
    bridge: Path,
    env_file: Path,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """B2: attempt-local floor A + write_rice + stratum stamp. Never migrate. Never raises."""
    receipt: dict[str, Any] = {
        "status": "hook_error",
        "attempt": str(attempt_dir),
        "floor": None,
        "rice": None,
        "stratum": None,
        "formal_post_score_env": os.environ.get("MUBENCH_FORMAL_POST_SCORE"),
        "migrated": False,
        "knife": "B2_attempt_local_rice",
    }
    try:
        if os.environ.get("MUBENCH_FORMAL_POST_SCORE") == "1":
            receipt["status"] = "skipped_formal_post_score"
            return receipt

        attempt_dir = Path(attempt_dir)
        runtime_root = Path(runtime_root)
        bridge = Path(bridge) if bridge is not None else Path()
        env_file = Path(env_file) if env_file is not None else Path()
        model = str(model or "")

        missing: list[str] = []
        if not model:
            missing.append("MUBENCH_MODEL")
        if not runtime_root.is_dir():
            missing.append("MUBENCH_RUNTIME_ROOT")
        if not bridge.is_file():
            missing.append("MUBENCH_BRIDGE")
        if not env_file.is_file():
            missing.append("MUBENCH_ENV")
        if not (attempt_dir / "score.json").is_file():
            missing.append("score.json")
        if missing:
            receipt["status"] = "skipped_missing_env"
            receipt["missing"] = missing
            return receipt

        if dry_run:
            receipt["status"] = "dry_run"
            receipt["floor"] = {"status": "dry_run_floor"}
            receipt["rice"] = {"status": "dry_run_rice"}
            return receipt

        score = _load_json(attempt_dir / "score.json") or {}
        floor: dict[str, Any]
        try:
            floor = run_floor_path_a(
                attempt_dir,
                runtime_root=runtime_root,
                bridge=bridge,
                env_file=env_file,
                model=model,
                dry_run=False,
            )
        except Exception as floor_exc:
            floor = {
                "status": "floor_error",
                "error": str(floor_exc),
                "error_type": type(floor_exc).__name__,
            }
        receipt["floor"] = floor

        # Prefer still calling write_rice even after floor_error (incomplete OK).
        rice: dict[str, Any]
        try:
            rice = write_rice(attempt_dir, model=model, dry_run=False)
        except Exception as rice_exc:
            rice = {
                "status": "rice_error",
                "error": str(rice_exc),
                "error_type": type(rice_exc).__name__,
            }
        receipt["rice"] = rice

        stamp: dict[str, Any]
        try:
            stamp = stamp_stratum_into_rice(attempt_dir, score)
        except Exception as stamp_exc:
            stamp = {
                "status": "stamp_error",
                "error": str(stamp_exc),
                "error_type": type(stamp_exc).__name__,
            }
        receipt["stratum"] = stamp
        if isinstance(stamp, dict) and stamp.get("incompleteness_stratum"):
            receipt["incompleteness_stratum"] = stamp["incompleteness_stratum"]

        if rice.get("status") == "rice_ok":
            if floor.get("status") in {"floor_ok", "floor_mock"}:
                receipt["status"] = "ok"
            else:
                receipt["status"] = "rice_ok_floor_failed"
        elif floor.get("status") == "floor_error":
            receipt["status"] = "floor_error"
        else:
            receipt["status"] = "rice_error"
        return receipt
    except Exception as exc:
        receipt["status"] = "hook_error"
        receipt["error"] = str(exc)
        receipt["error_type"] = type(exc).__name__
        receipt["traceback"] = traceback.format_exc()[-4000:]
        return receipt


def cleanup_attempt_seal_audio(
    attempt_case_dir: Path,
    *,
    dry_run: bool = False,
    require_rice: bool = False,
    delete_oracle: bool = False,
) -> dict[str, Any]:
    """Cleanup-only after attempt score seal: played + mono on attempt dir; never migrate.

    For A-class / no POST_SCORE runs: gate on score.json + audio/both.wav.
    Default require_rice=False (Jay 2026-10-09 A attempt policy).
    Does NOT write into FORMAL flat. Does NOT call migrate / floor / rice.
    Oracle kept unless delete_oracle=True.
    """
    root = Path(attempt_case_dir)
    if not root.is_dir():
        return {"status": "error", "error": "not_a_dir", "attempt": str(root)}
    if not (root / "score.json").is_file():
        return {"status": "skipped_no_score", "attempt": str(root)}
    if not (root / "audio" / "both.wav").is_file():
        return {"status": "skipped_no_both", "attempt": str(root)}
    if require_rice and not (root / RICE_NAME).is_file():
        return {"status": "skipped_no_rice", "attempt": str(root)}

    played = delete_played_segment_audio(root, dry_run=dry_run, require_rice=require_rice)
    mono = delete_mono_session_stems(root, dry_run=dry_run, require_rice=require_rice)
    oracle: dict[str, Any] = {"status": "kept_by_policy"}
    if delete_oracle:
        oracle = delete_case_oracle(root, dry_run=dry_run, require_score=True)

    if not dry_run:
        try:
            _atomic_write_json(root / PLAYED_SEGMENT_DELETE_RECEIPT, played)
            played["receipt"] = str(root / PLAYED_SEGMENT_DELETE_RECEIPT)
        except Exception as exc:
            played["receipt_write_error"] = str(exc)
        try:
            _atomic_write_json(root / MONO_STEM_DELETE_RECEIPT, mono)
            mono["receipt"] = str(root / MONO_STEM_DELETE_RECEIPT)
        except Exception as exc:
            mono["receipt_write_error"] = str(exc)

    return {
        "status": "ok",
        "attempt": str(root),
        "played_segment_delete": played,
        "mono_stem_delete": mono,
        "oracle_delete": oracle,
        "require_rice": require_rice,
        "migrated": False,
    }


def run_after_score_seal(
    attempt_case_dir: Path,
    *,
    formal_case_dir: Path,
    runtime_root: Path,
    model: str,
    bridge: Optional[Path] = None,
    env_file: Optional[Path] = None,
    dry_run: bool = False,
    migrate_mode: str = "hardlink",
) -> dict[str, Any]:
    """Orchestrate MUST-2 migrate → MUST-1/3 floor → RICE. Never raises to caller."""
    try:
        attempt_case_dir = Path(attempt_case_dir)
        formal_case_dir = Path(formal_case_dir)
        runtime_root = Path(runtime_root)
        if not (attempt_case_dir / "score.json").is_file():
            return {"status": "error", "error": "missing attempt score.json"}

        mig = migrate_attempt_to_formal(
            attempt_case_dir,
            formal_case_dir,
            migrate_mode=migrate_mode,
            dry_run=dry_run,
        )
        if mig.get("status") == "error":
            return {"status": "migrate_error", "migrate": mig}

        if _should_skip_digfix(formal_case_dir):
            played = delete_played_segment_audio(
                formal_case_dir,
                attempt_case_dir,
                dry_run=dry_run,
                require_rice=True,
            )
            mono = delete_mono_session_stems(
                formal_case_dir,
                attempt_case_dir,
                dry_run=dry_run,
                require_rice=True,
            )
            oracle = delete_case_oracle(formal_case_dir, attempt_case_dir, dry_run=dry_run)
            return {
                "status": "skip_complete_digfix",
                "migrate": mig,
                "played_segment_delete": played,
                "mono_stem_delete": mono,
                "oracle_delete": oracle,
                "formal": str(formal_case_dir),
            }

        bridge_p = Path(bridge) if bridge else runtime_root / "bridge.json"
        env_p = Path(env_file) if env_file else runtime_root / "env"
        # common ops names; caller should pass real paths
        floor = run_floor_path_a(
            formal_case_dir,
            runtime_root=runtime_root,
            bridge=bridge_p,
            env_file=env_p,
            model=model,
            dry_run=dry_run,
        )
        rice = write_rice(formal_case_dir, model=model, dry_run=dry_run)
        formal_complete = mig.get("status") in {"migrated", "dry_run_migrate"} and (
            dry_run or (formal_case_dir / "score.json").is_file()
        )
        played = {"status": "skipped_no_rice"}
        mono = {"status": "skipped_no_rice"}
        if (formal_case_dir / RICE_NAME).is_file() or (
            isinstance(rice, dict) and rice.get("status") in {"rice_ok", "dry_run_rice"}
        ):
            played = delete_played_segment_audio(
                formal_case_dir,
                attempt_case_dir,
                dry_run=dry_run,
                require_rice=not dry_run,
            )
            mono = delete_mono_session_stems(
                formal_case_dir,
                attempt_case_dir,
                dry_run=dry_run,
                require_rice=not dry_run,
            )
            oracle = delete_case_oracle(formal_case_dir, attempt_case_dir, dry_run=dry_run)
        else:
            oracle = {"status": "skipped_no_rice"}
        return {
            "status": post_score_status(formal_complete, floor, rice, dry_run=dry_run),
            "formal_complete": bool(formal_complete),
            "migrate": mig,
            "floor": floor,
            "rice": rice,
            "played_segment_delete": played,
            "mono_stem_delete": mono,
            "oracle_delete": oracle,
            "formal": str(formal_case_dir),
            "attempt": str(attempt_case_dir),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc()[-4000:],
        }


if __name__ == "__main__":
    # minimal CLI for Ops smoke
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--attempt-case-dir", type=Path, required=False)
    p.add_argument("--formal-case-dir", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=False)
    p.add_argument("--model", required=False)
    p.add_argument("--bridge", type=Path)
    p.add_argument("--env", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--migrate-mode", default="hardlink", choices=("hardlink", "copy"))
    p.add_argument(
        "--delete-played-segments-only",
        action="store_true",
        help="Only delete played/segment wav under --formal-case-dir (and optional attempt); requires score+RICE.",
    )
    p.add_argument(
        "--delete-mono-stems-only",
        action="store_true",
        help="Only delete audio/user_mixed_native.wav + audio/assistant_native.wav (keep both); requires score+RICE. Live hook smoke — not for stock backfill.",
    )
    args = p.parse_args()
    if args.delete_played_segments_only and args.delete_mono_stems_only:
        raise SystemExit("pass only one of --delete-played-segments-only / --delete-mono-stems-only")
    if args.delete_played_segments_only or args.delete_mono_stems_only:
        roots = [args.formal_case_dir]
        if args.attempt_case_dir:
            roots.append(args.attempt_case_dir)
        fn = delete_mono_session_stems if args.delete_mono_stems_only else delete_played_segment_audio
        result = fn(*roots, dry_run=args.dry_run, require_rice=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("status") in {"deleted", "dry_run_deleted", "skipped_no_rice"} else 1)
    if args.attempt_case_dir is None or args.runtime_root is None:
        raise SystemExit("--attempt-case-dir and --runtime-root required unless --delete-played-segments-only")
    result = run_after_score_seal(
        args.attempt_case_dir,
        formal_case_dir=args.formal_case_dir,
        runtime_root=args.runtime_root,
        model=args.model,
        bridge=args.bridge,
        env_file=args.env,
        dry_run=args.dry_run,
        migrate_mode=args.migrate_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("status") in {"ok", "skip_complete_digfix", "dry_run_migrate"} or args.dry_run else 1)
