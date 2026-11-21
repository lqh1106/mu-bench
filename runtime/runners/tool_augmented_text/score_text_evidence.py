import bootstrap
"""Shared scoring entry for live runs and one bounded evidence-only completion."""
from pathlib import Path
import json,os
from score_policy import separate_authorization
from run_tau_mu_voice_realtime import run_semantic_judge,measure_floor_split_timings
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,path)

def score_evidence(api,case,trace,turns,logs,out,resume_dirs=()):
 caps,goals,leaks=run_semantic_judge(api,{'GPT_URL':'private','GPT_KEY':'private'},case,trace,turns,logs,audit_dir=out,resume_dirs=resume_dirs)
 trace.capability_results=caps;trace.goal_results=goals;trace.privacy_leaks=leaks
 measure_floor_split_timings(case,trace,turns)
 report=separate_authorization(MultiUserEvaluator().evaluate(case,trace).model_dump(mode='json'))
 report.update(case_version='mu-bench-tool-augmented-v1',request_revision_protocol='request-revision',caller_protocol='explicit-caller',caller_prompt_policy='caller-field-definition',judge_protocol='tool_full_source_review',text_model='qwen3.8-flash',judge_model='gpt-6-luna',simulator_model='gpt-6-luna',simulator_participantlist_label='participant_context_enabled',measurement_track='text-logical',text_source_identity_provided=True,acoustic_condition_applied=False,voice_score_eligible=False,source_audio_created=False)
 if os.environ.get('MUBENCH_TOOL_JUDGE_VERSION') in {'jt3','jt4','jt5','jt6','jt7'}:
  report.update(case_version=case.task_contract_revision['review'].get('case_revision_id','mu-bench-tool-augmented-v2'),judge_protocol=os.environ['MUBENCH_TOOL_JUDGE_VERSION'],judge_protocol_label=os.environ['MUBENCH_TOOL_JUDGE_VERSION'],tool_prechecks='dual-independent-with-arbitration',evidence_roles_version='jt3')
  unresolved=[a for a in trace.tool_actions if a.caller_provenance_status=='insufficient_evidence']
  if unresolved:report.update(caller_provenance_success=None,authorization_success=None,authorization_review_unresolved=True)
 marker=Path(out)/'simulator_exhausted_forced_eval.json'
 if marker.exists():report.update(truncated_simulator_exhaustion=True,simulator_termination_audit=str(marker),normal_conversation_completion_claimed=False)
 if case.task_contract_revision.get('design_revision') in {'tool-scene-r3','tool-scene-r4','tool-scene-r5'}:
  delivered={e.event_id for e in trace.spoken_events}
  missing=[e.event_id for e in case.events if e.required and e.event_id not in delivered]
  report.update(case_input_delivery_complete=not missing,missing_required_input_events=missing,
    full_case_comparable=not missing and not marker.exists(),evaluation_contract='tool-scoring-r5' if __import__('os').environ.get('MUBENCH_TOOL_JUDGE_VERSION')=='jt7' else 'tool-scoring-r4' if os.environ.get('MUBENCH_TOOL_JUDGE_VERSION')=='jt6' else 'tool-scoring-r3',simulator_semantics_contract='tool-event-meaning-r3')
 write(out/'score.json',report);write(out/'trace.after_judge.json',trace.model_dump(mode='json'))
 return report
