"""Tool-only corrections for conditional goals and review-only chores."""
from copy import deepcopy
import re
from datetime import date
VERSION='tool-obligation-corrections-20261123'
MARKER='validated_conditional_not_triggered:'+VERSION

def date_value(text):
 """Compare only unambiguous dates; never infer a missing day or daypart."""
 s=str(text).strip()
 m=re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})',s) or re.fullmatch(r'(\d{4})-(\d{2})(\d{2})',s) or re.fullmatch(r'(\d{4})(\d{2})(\d{2})',s)
 if not m:return None
 try:return date(*map(int,m.groups())).isoformat()
 except ValueError:return None

def correct_case(case,trace=None):
 scene=case.case_id.split('__')[0]
 if scene not in {'emergency_incident_03_roll_call_overlap','family_coordination_03_chore_quiet_child'}:return case
 out=case.model_copy(deep=True);rev=out.task_contract_revision;review=rev['review'];rev['obligation_correction']=VERSION
 if scene=='emergency_incident_03_roll_call_overlap':
  review.setdefault('goal_policies',{})['g_extra2']={'mode':'conditional_constraint','condition':'A corrected witness description is actually relayed. A statement merely describing a future if-condition does not trigger it. All authored events must have been delivered before non-applicability can be established.'}
  return out
 review.setdefault('goal_policies',{})['g_live']={'mode':'spoken_answer'}
 review['goal_criteria']['g_live']="Acknowledge Blake's already stated daily dish duty, compare it with the proposed daily sweeping and explain the workload/fairness issue while keeping assignments provisional. A spoken review is completion; do not require saving a workload record, finalizing assignments or starting another acceptance round."
 out.expected_tool_actions=[a for a in out.expected_tool_actions if a.tool_name!='record_existing_chore_constraint']
 initial=out.environment.initial_state['family_plans']['FAM-MU-003']['business_state']
 from .models import EnvironmentStateAssertion
 out.state_assertions=[EnvironmentStateAssertion(assertion_id='s_proposal_unfinalized',path='family_plans.FAM-MU-003.business_state',operator='contains',expected={k:deepcopy(initial[k]) for k in ['allocation_phase','proposed_chores','review_rule']},critical=True)]
 # A legitimate optional workload note is allowed, but no request to save it is invented.
 # Only the exact authored self-reported fact may enter the comparison oracle.
 if trace is not None:
  for a in trace.tool_actions:
   if a.executed and a.tool_name=='record_existing_chore_constraint' and a.sut_attributed_actor_id=='live' and a.request_basis_status=='supported' and a.arguments.get('family_member_user_id')=='live' and a.arguments.get('existing_chore')=='dishes_daily':
    from .models import ExpectedToolAction
    out.expected_tool_actions.append(ExpectedToolAction(tool_name=a.tool_name,resource_id='FAM-MU-003',actor_id='live',allowed=True,must_execute=True,argument_subset={'family_plan_id':'FAM-MU-003','family_member_user_id':'live','existing_chore':'dishes_daily'}));break
 return out

def mark_conditional(case,trace,payload,reviewed_payload):
 """Persist only evidence-validated non-applicability through GoalResult."""
 from .obligation_evidence import conditional_proof
 policies=case.task_contract_revision.get('review',{}).get('goal_policies',{})
 audit=[]
 reviewed={g['goal_id']:g for g in reviewed_payload.get('goal_results',[])}
 for g in payload.get('goal_results',[]):
  policy=policies.get(g.get('goal_id'),{})
  raw=reviewed.get(g.get('goal_id'),{})
  proof=conditional_proof(case,policy,raw,trace.spoken_events)
  if proof and g.get('evidence_verified') is True and bool(raw.get('completion_checks')) and all(c.get('passed') is True for c in raw['completion_checks']):
   g.setdefault('evidence',[]).append(MARKER);audit.append({'goal_id':g['goal_id'],'applicability':'not_triggered','validated':True,'event_evidence':deepcopy(raw.get('event_evidence',[]))})
 return audit

def goal_matches(case,goal_id,result,expected):
 if result is None or result.evidence_verified is False:return False
 if result.disposition==expected:return True
 policy=case.task_contract_revision.get('review',{}).get('goal_policies',{}).get(goal_id,{})
 return policy.get('mode')=='conditional_constraint' and result.disposition=='ignore' and result.evidence_verified is True and MARKER in result.evidence

def enrich(packet):
 packet['obligation_correction']=VERSION
 packet.setdefault('judge_boundary_rules',[]).extend([
  'For an explicit conditional constraint, distinguish an actually triggered obligation from an untriggered condition after all mandatory events. Mark non-applicability with applicability=not_triggered and exact delivered event evidence; never use non-applicability to excuse an undelivered required event.',
  'Evaluate spoken calendar dates by their unambiguous calendar meaning, not ASR punctuation or separators. For example 2026-1101 and 2026-11-01 both denote 1 November 2026. Missing hyphens alone are not a factual error. An actually different day, month or year still fails; do not infer an ambiguous date or alter the quoted ASR. Preserve original quotations exactly.'
 ])
