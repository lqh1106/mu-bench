"""Pin runtime implementation independently from model-independent case data."""
import os,hashlib,json
from pathlib import Path
from ablation_core import digest
def scoring_version(runtime):
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads(Path(os.environ['MUBENCH_RUNTIME_MANIFEST']).read_text())
    actual={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in manifest['files']}
    if actual!=manifest['files'] or digest(actual)!=manifest['scoring_sha256']:
        raise ValueError('Runtime implementation differs from its compiled fingerprint')
    return {k:manifest[k] for k in ('scoring_sha256','task_success_policy')}
