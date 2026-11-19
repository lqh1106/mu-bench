"""ATC presentation of the existing auditable RICE component scores."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math

ATC_VERSION = "mu-bench-atc-v1"


def from_rice(rice: dict) -> dict:
    """A=R, T=Continuity, Coordination=(I+E)/2; absent E -> I."""
    def unit(name):
        value = rice.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Metric must be a number or null: " + name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Metric outside [0,1]: " + name)
        return float(value)

    r, t, i, e = (unit(k) for k in (
        "routing_score", "continuity_score", "isolation_score", "equity_score"))
    coordination = None if i is None else i if e is None else (i + e) / 2
    components = {"isolation_score": i, "equity_score": e}
    result = {
        "atc_contract_version": ATC_VERSION,
        "attribution_score": r,
        "threading_score": t,
        "coordination_score": coordination,
        "coordination_components": components,
        "coordination_components_used": [] if i is None else ["I"] if e is None else ["I", "E"],
        "coordination_denominator": 0 if i is None else 1 if e is None else 2,
        "equity_applicable": e is not None,
        "mapping": {"A": "R", "T": "Continuity", "C": "(I+E)/2 if E is applicable; otherwise I"},
        "source_rice_contract_version": rice.get("rice_contract_version"),
        "source_components": {"R": r, "Continuity": t, "I": i, "E": e},
        "source_units_used": rice.get("units_used", {}),
        "missing_values_zero_imputed": False,
    }
    for key in ("measurement_track", "voice_comparable", "source_score", "timing_excluded",
                "routing_approx", "equity_approx", "isolation_mode"):
        if key in rice:
            result[key] = rice[key]
    return result


def write_atc(case_dir: Path, rice: dict) -> dict:
    """Write a derived artifact; never alter semantic scores or source RICE."""
    result = from_rice(rice)
    source = Path(case_dir) / "rice_score.json"
    result["source_rice_file"] = "rice_score.json"
    result["source_rice_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = Path(case_dir) / "atc_score.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return result
