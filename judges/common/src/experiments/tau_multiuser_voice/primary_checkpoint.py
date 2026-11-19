"""Reuse a validated primary review for the same case and committed evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from .judge_evidence import JudgeEvidenceInvalid, validate_and_bind_judge_payload
from .judge_goal_scope import SYSTEM_RULE

CHECKPOINT_VERSION = 'mu-primary-review-checkpoint-v1-20261117'

def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def context_fingerprint(request, catalog):
    return fingerprint({'request': request, 'catalog': catalog, 'goal_scope_system_rule': SYSTEM_RULE,
                        'judge_model': 'gpt-6-luna'})


def save_primary_checkpoint(audit_dir, request, catalog, primary, validation, write):
    if audit_dir is None:
        return
    write(Path(audit_dir) / 'primary_review_checkpoint.json', {
        'version': CHECKPOINT_VERSION,
        'stage': 'primary_validated',
        'context_sha256': context_fingerprint(request, catalog),
        'primary_sha256': fingerprint(primary),
        'primary': primary,
        'validation': validation,
    })


def candidate_directories(audit_dir, resume_dirs):
    roots = [Path(audit_dir)] if audit_dir is not None else []
    roots.extend(Path(path) for path in resume_dirs)
    if audit_dir is not None and Path(audit_dir).name.startswith('semantic_judge_retry_'):
        roots.append(Path(audit_dir).parent)
    candidates = set()
    for root in roots:
        candidates.add(root)
        candidates.update(path for path in root.glob('semantic_judge_retry_*') if path.is_dir())
    return sorted((path for path in candidates if path.is_dir()),
                  key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_completed_primary(case, contract, request, catalog, audit_dir, resume_dirs=()):
    """Return the latest matching valid primary, including pre-checkpoint audit records."""
    expected_context = context_fingerprint(request, catalog)
    for directory in candidate_directories(audit_dir, resume_dirs):
        checkpoint = _read(directory / 'primary_review_checkpoint.json')
        candidates = []
        if checkpoint and checkpoint.get('version') == CHECKPOINT_VERSION:
            primary = checkpoint.get('primary')
            if (checkpoint.get('stage') == 'primary_validated'
                    and checkpoint.get('context_sha256') == expected_context
                    and checkpoint.get('primary_sha256') == fingerprint(primary)):
                candidates.append(('primary_review_checkpoint.json', primary))
        # An older full-review request proves that its primary stage reached the reviewer.
        saved_request = _read(directory / 'semantic_judge_request.json')
        validation = _read(directory / 'judge_evidence_audit.json') or {}
        if (saved_request == request and validation.get('evaluation_valid') is True
                and (directory / 'semantic_verifier_request.json').is_file()):
            primary = _read(directory / 'semantic_primary_response.json')
            if primary is not None:
                candidates.append(('semantic_primary_response.json', primary))
            else:
                review_audit = _read(directory / 'full_review_audit.json') or {}
                if not review_audit.get('completed'):
                    repair = _read(directory / 'semantic_judge_repair_success.json') or {}
                    if repair.get('repair_attempt'):
                        name = 'semantic_judge_response_repaired_attempt_' + str(repair['repair_attempt']) + '.json'
                        candidates.append((name, _read(directory / name)))
                    candidates.append(('semantic_judge_response.json', _read(directory / 'semantic_judge_response.json')))
        for name, primary in candidates:
            if not isinstance(primary, dict):
                continue
            try:
                bound, validation = validate_and_bind_judge_payload(case, contract, deepcopy(primary), catalog)
            except (JudgeEvidenceInvalid, KeyError, TypeError, ValueError, AttributeError, IndexError):
                continue
            if not validation.get('evaluation_valid'):
                continue
            provenance = {'policy': CHECKPOINT_VERSION, 'stage': 'primary_validated',
                          'source': str(directory / name), 'context_sha256': expected_context,
                          'primary_api_calls': 0, 'next_stage': 'independent_full_review'}
            return bound, validation, provenance
    return None
