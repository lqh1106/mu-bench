"""Validated access to the complete public case files."""
from pathlib import Path
import os,json
from .models import MultiUserVoiceCase
class CaseCatalog:
    @classmethod
    def load_default(cls):
        root=Path(os.environ['MUBENCH_CASE_DIR'])
        result=[MultiUserVoiceCase.model_validate(json.loads(p.read_text())) for p in sorted(root.glob('*.json'))]
        if not result:raise ValueError('Case directory contains no cases')
        return result
    @classmethod
    def get(cls,case_id):
        p=Path(os.environ['MUBENCH_CASE_DIR'])/(case_id+'.json')
        if not p.is_file():raise KeyError(case_id)
        c=MultiUserVoiceCase.model_validate(json.loads(p.read_text()))
        if c.case_id!=case_id:raise ValueError('Case ID/file mismatch')
        return c
