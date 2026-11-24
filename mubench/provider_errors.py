"""Recognize provider admission blocks without retaining credentials or raw logs."""
from pathlib import Path
import json
import re

_HTTP_402 = re.compile(r'\bhttp(?:/\d(?:\.\d)?)?(?:\s+error)?\s*[:=]?\s*402\b', re.I)
_MARKERS = (
    'invalid_api_key', 'payment_required', 'insufficient_quota',
    'credit_balance_exhausted', 'spending-limit', 'billing_hard_limit_reached',
    'insufficient credits', 'exceeded your current quota',
)

def admission_block_reason(message):
    text = str(message)
    if _HTTP_402.search(text):
        return 'http_402'
    lower = text.lower()
    return next((marker for marker in _MARKERS if marker in lower), None)

def attempt_admission_block(attempt, result, error):
    reason = admission_block_reason(error)
    if reason:
        return reason
    for name in ('runner_error.json', 'completion_error.json'):
        path = Path(result) / name
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        reason = admission_block_reason(json.dumps(payload))
        if reason:
            return reason
    try:
        with (Path(attempt) / 'case.log').open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 32768))
            reason = admission_block_reason(stream.read().decode(errors='replace'))
    except OSError:
        return None
    return reason
