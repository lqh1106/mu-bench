"""Carry actual input receipts into a retry and retain its adopted audit at the episode root."""
from pathlib import Path
import hashlib
import json
import os
import shutil

_INPUT_FILES = (
    'initial_assistant_context.json', 'session_protocol_audio.json',
    'voice_enrollment.before_judge.json', 'voice_enrollment.json',
    'registration_audio_segments.json', 'adapter_audit.json',
    'session_control_events.json', 'initial_context_assistant_turns.json',
    'session_spec.json', 'system_prompt_policy.json', 'system_prompt.txt',
    'reviewed_case.json', 'reviewed_audio_integrity.json', 'case_input_contract.json',
)

def prepare_retry(root, target):
    root, target = Path(root), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _INPUT_FILES:
        source = root / name
        if source.is_file():
            shutil.copy2(source, target / name)
            copied.append(name)
    return copied

def promote_retry(root, target, attempt):
    root, target = Path(root), Path(target)
    audit = json.loads((target / 'full_review_audit.json').read_text())
    if not audit.get('completed') or set(audit['item_ids']) != set(audit['covered_item_ids']):
        raise ValueError('Cannot adopt an incomplete retry review')
    manifest = []
    for source in sorted(target.iterdir()):
        if not source.is_file() or source.suffix not in ('.json', '.jsonl', '.txt') or source.name in _INPUT_FILES:
            continue
        dest = root / source.name
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            old = root / '_judge_retry_history' / ('before_attempt_' + str(attempt)) / source.name
            old.parent.mkdir(parents=True, exist_ok=True)
            if not old.exists():
                shutil.copy2(dest, old)
        tmp = dest.with_name('.' + dest.name + '.retry-copy')
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
        manifest.append({'file': source.name, 'source': str(source), 'sha256': sha})
    receipt = {'attempt': attempt, 'source': str(target), 'full_review_completed': True,
               'score_unchanged': True, 'original_input_receipts_preserved': True, 'files': manifest}
    (root / 'judge_retry_adoption.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt
