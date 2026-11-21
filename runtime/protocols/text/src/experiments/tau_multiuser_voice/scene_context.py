"""Only the explicitly authored, pre-session source view is model-visible.

Never derive this view from environment.authoritative_facts, goals, events, ACL
maps, or the case review. Providers receive identical semantic content through
this common prompt builder. Native audio-only tracks must not silently omit it.
"""
import hashlib,json
VERSION='mu-bench-tool-augmented-v1'
def get_initial_context(case):
 review=case.task_contract_revision.get('review',{})
 if review.get('case_revision_id') not in {VERSION,'mu-bench-tool-augmented-v1','mu-bench-tool-augmented-v2','mu-bench-tool-augmented-v3','mu-bench-tool-augmented-v4','mu-bench-tool-augmented-v5'}:return {}
 ctx=review.get('initial_assistant_context',{})
 if not ctx:return {}
 assert ctx['available_at']=='before_registration'
 assert ctx['contains_future_event_oracle'] is False
 assert ctx['contains_internal_participant_mapping'] is False
 return ctx

def context_section(case):
 ctx=get_initial_context(case)
 if not ctx:return ''
 # Ordinary source data, not an oracle response or instructions for handling future turns.
 return '\n\nInformation available to the assistant before this conversation\nRead-only service or shared-document snapshot:\n'+ctx['content']+'\n'

def verify_transport(case, *, openweight=False):
 ctx=get_initial_context(case)
 return {'version':case.task_contract_revision.get('review',{}).get('case_revision_id',VERSION),'required':bool(ctx),'transport':('post_registration_pre_task_tts' if openweight else 'provider_initial_instructions') if ctx else 'none','content_sha256':hashlib.sha256(ctx.get('content','').encode()).hexdigest(),'future_event_oracle_sent':False,'user_readout_substitution':False}
