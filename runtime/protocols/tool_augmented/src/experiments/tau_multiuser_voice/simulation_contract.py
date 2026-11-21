"""Tool event semantics; the tool-free simulator stays on its own contract."""
from pathlib import Path
_ROOT=next(p for p in Path(__file__).resolve().parents if (p/'mubench/config.py').exists())
_core=_ROOT/'runtime/core/src/experiments/tau_multiuser_voice/simulation_contract.py'
exec(compile(_core.read_text(),str(_core),'exec'),globals())
_base_compile=compile_event_contract
_base_validate=validate_candidate
_base_writer=writer_prompt
_base_critic=critique_candidate

def compile_event_contract(case,event):
 out=_base_compile(case,event)
 if case.task_contract_revision.get('design_revision') in {'tool-scene-r3','tool-scene-r4','tool-scene-r5'}:
  out['immutable_meaning']=case.task_contract_revision['event_semantics'].get(event.event_id,{})
 return out

def writer_prompt(context,*,alignment_ms):
 out=json.loads(_base_writer(context,alignment_ms=alignment_ms))
 meaning=context['event_contract'].get('immutable_meaning')
 if meaning:out['rules'].extend(meaning['requirements'])
 return json.dumps(out,ensure_ascii=False)

def validate_candidate(raw,**kwargs):
 out=_base_validate(raw,**kwargs)
 context=kwargs['context'];case=kwargs['case'];event=kwargs['event']
 if case.task_contract_revision.get('design_revision') in {'tool-scene-r3','tool-scene-r4','tool-scene-r5'} and case.case_id.startswith('telecom_02_lost_phone_number_fix__') and event.event_id in {'e_primary_reconcile','e_primary_commit','e_live_followup'}:
  value=out.utterance.casefold().replace('’',"'")
  # A narrow observed inversion, never a general ban on initial incorrect speech.
  if re.search(r'(?:^|[.!?;]\s+)(?:please\s+)?(?:keep|leave|retain)\s+(?:line\s+)?0182(?:[’\']s)?(?:\s+suspension)?\s+(?:in place|suspended|switched off)',value):
   raise ValueError('core target inverted: 0182 must be active after the correction')
 return out

def critique_candidate(backend,*,context,candidate):
 meaning=context['event_contract'].get('immutable_meaning')
 if not meaning:return _base_critic(backend,context=context,candidate=candidate)
 class ContractBackend:
  def generate_json(self,messages):
   from copy import deepcopy
   messages=deepcopy(messages)
   messages[0]['content'] += ' Check every immutable_meaning requirement explicitly under event_speech_act_satisfied. Reject altered core target states, requests replaced with announcements, preservation of a known mistaken state, and a mandatory extra consent stage not authored for this event. A truthful acknowledgment is permitted only when the actual desired effect is already complete; a mistaken prior effect is not completion. Return the same required checks; never rewrite the candidate.'
   return backend.generate_json(messages)
 return _base_critic(ContractBackend(),context=context,candidate=candidate)

# Tail planner intents instruct the selected user generator; they are not
# already-delivered utterances addressed to the assistant.
_base_closure_critic=critique_closure

def critique_closure(backend, *, context, decision):
 class ClosureIntentBackend:
  def generate_json(self,messages):
   from copy import deepcopy
   messages=deepcopy(messages)
   messages[0]['content'] += ' decision.intent is a semantic instruction for the selected USER generator, not a verbatim utterance already addressed to the assistant. Interpret "state your preference" as asking the selected user to express their OWN preference, not asking the assistant to invent it. Still reject actual assistant-answer substitution or an invented task. A review request may be complete with unresolved proposals when the user explicitly asked to keep them provisional; do not require final agreement merely because the assistant offers optional next steps.'
   return backend.generate_json(messages)
 return _base_closure_critic(ClosureIntentBackend(),context=context,decision=decision)
