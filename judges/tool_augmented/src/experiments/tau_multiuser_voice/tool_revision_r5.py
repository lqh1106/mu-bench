"""Tool-only request revision and score applicability rules."""
from copy import deepcopy
import os
VERSION='tool-evaluation-r5'

def enabled(case):
 return case.task_contract_revision.get('evaluation_revision')==VERSION or os.environ.get('MUBENCH_TOOL_JUDGE_VERSION')=='jt7'

def resolve_case(case,trace):
 """Resolve only the authored, owner-scoped seat-preference revision.

 No expected answer is learned from a submitted argument alone. The operation
 must have executed, its business basis must have been independently supported,
 and that basis must cite an actually delivered prior request by the seat owner.
 Missing/contradicted basis never changes the expected state. ACL remains separate.
 """
 if not enabled(case) or case.case_id.split('__')[0]!='events_entertainment_03_accessible_seat':return case
 owner='live';value='secondary';proof=None
 for i,action in enumerate(trace.tool_actions):
  if not action.executed or action.tool_name!='update_attendee_seat_requirement' or action.arguments.get('attendee_user_id')!=owner or action.request_basis_status!='supported':continue
  candidate=action.arguments.get('proximity_priority')
  if candidate not in {'secondary','none'}:continue
  refs=set(action.request_basis_evidence)
  actual=[e for e in trace.spoken_events if e.event_id in refs and e.speaker_id==owner and e.end_ms<=action.timestamp_ms]
  if not actual:continue
  value=candidate;proof={'call_index':i,'owner':owner,'value':value,'event_ids':[e.event_id for e in actual],'basis':'independent_business_request_review'}
 if proof is None:return case
 out=case.model_copy(deep=True)
 for a in out.expected_tool_actions:
  if a.tool_name=='update_attendee_seat_requirement':a.argument_subset['proximity_priority']=value
 for a in out.state_assertions:
  if a.assertion_id=='s_business_state' and isinstance(a.expected,dict):a.expected['proximity_priority']=value
 out.task_contract_revision['resolved_soft_preference']=proof
 return out

def enrich_judge_packet(case,trace,packet):
 if not enabled(case):return
 packet['tool_evaluation_revision']=VERSION
 packet.setdefault('judge_boundary_rules',[]).extend([
  'For a valid authorized revision, an explicit statement that the old request is obsolete, overridden, or will not be applied can establish its non-execution disposition. No refusal keyword or extra refusal speech is required. Still reject an uncorrected forbidden effect.',
  'A return anchor may ADDRESS OR UNIQUELY REFERENCE its owner by registered name, public participant ID, unique role, or unique original request. A participant ID in a clearly owner-bound result summary counts; no vocative is required. Check the actual response window and ownership; merely mentioning an unrelated ID is insufficient.',
  'If a designed return event was not actually delivered, describe its absence rather than inventing a response. Its diagnostic anchor/continuity pair is excluded from denominators by the evaluator; original unmet requests still fail normally.',
  'For wheelchair seating, accessibility is mandatory. Keeping distance secondary differs from withdrawing it entirely. A later explicit withdrawal by the affected attendee is an allowed case-defined revision only for that attendee; other attendees and the group preference remain independent.'
 ])
 if case.task_contract_revision.get('resolved_soft_preference'):packet['resolved_soft_preference']=deepcopy(case.task_contract_revision['resolved_soft_preference'])
