"""Billing classification only; no bundled credentials or account pool."""
import json
def quota_evidence(payload=None):
 text=json.dumps(payload or {},ensure_ascii=False).lower()
 return 'payment_required' if any(x in text for x in ('payment_required','insufficient_balance','insufficient balance','balance exhausted')) else None
def exception_quota_evidence(exc):
 code=getattr(getattr(exc,'response',None),'status_code',None) or getattr(exc,'status_code',None)
 return 'payment_required' if code==402 else None
