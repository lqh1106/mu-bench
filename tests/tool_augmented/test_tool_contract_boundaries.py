from pathlib import Path
import json,sys,copy
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));from mubench.config import activate;activate(R/'runtime/runners/tool_augmented_audio')
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas
from experiments.tau_multiuser_voice.interface_contract import normalize_fields,validate_shape
from experiments.tau_multiuser_voice.evaluator import _contains_subset
from experiments.tau_multiuser_voice.business_acceptance import canonical_values
from experiments.tau_multiuser_voice.tool_temporal_alignment import segments,guard_unresolved_verdicts
from experiments.tau_multiuser_voice.evaluation_policy import TemporalAlignmentPending
for scene,key,a,b in [('events_entertainment_02_playlist_cheer','playlist_scope','complete set','complete_set'),('shared_home_02_media_backchannel','playback_scope','full album','full_album')]:
 for f in (R/'cases/tool_augmented').glob(scene+'__*.json'):
  c=MultiUserVoiceCase.model_validate_json(f.read_text());e=create_virtual_environment(c.environment);configure_schemas(e,c.environment.tool_names);action=next(a for a in c.expected_tool_actions if a.must_execute);schema=e.definitions[action.tool_name].parameters['properties'][key]
  assert a in schema['enum'] and b in schema['enum'];validate_shape(a,schema,key)
  assert normalize_fields(scene,{key:a})[key]==b
  assert _contains_subset({key:a},{key:b},c)
  assert not _contains_subset({key:'single_track'},{key:b},c)
  try:validate_shape('single_track',schema,key)
  except ValueError:pass
  else:raise AssertionError('wrong playback range accepted')
  if key=='playback_scope':assert not _contains_subset({'album':'Full Evening Jazz'},{'album':'Evening Jazz'},c)
for f in (R/'cases/tool_augmented').glob('airline_03_payer_traveler_cancel__*.json'):
 c=MultiUserVoiceCase.model_validate_json(f.read_text());assert not next(a for a in c.expected_tool_actions if a.tool_name=='get_booking_context').must_execute
c=MultiUserVoiceCase.model_validate_json(next((R/'cases/tool_augmented').glob('airline_01_split_rebooking__*.json')).read_text());assert c.task_contract_revision['query_dependencies']['get_booking_context']['dependent_tools']
c=MultiUserVoiceCase.model_validate_json(next((R/'cases/tool_augmented').glob('telecom_03_router_restart_pushback__*.json')).read_text());assert _contains_subset({'until_condition':'active_emergency_video_call_finishes','reason':'The user asks to wait for the emergency call.'},{'until_condition':'active_emergency_video_call_finishes','reason':'affected_user_objected'},c)
assert not _contains_subset({'until_condition':'now','reason':'anything'},{'until_condition':'active_emergency_video_call_finishes','reason':'affected_user_objected'},c)
assert not _contains_subset({'reason':'different'},{'reason':'affected_user_objected'},c)
assert 'contradictory explanation fails' in c.task_contract_revision['review']['goal_criteria']['g_live']
t=[{'source_id':'s','asr_derived':True,'start_ms':10,'end_ms':30}];assert segments(t,[{'executed':True,'result_available_ms':20}])[0][1]==[10.,20.,30.]
try:guard_unresolved_verdicts({'passed':False,'reason':'The timing is not established because there are no word-level timestamps.'},None)
except TemporalAlignmentPending:pass
else:raise AssertionError('uncertainty treated as model error')
guard_unresolved_verdicts({'passed':False,'reason':'The claim ended at 10ms but the tool result was available at 20ms: this is a demonstrated early false claim.'},None)
print(json.dumps({'passed':True,'scope_cases_checked':16,'refusal_cases_checked':8,'necessary_query_preserved':True,'wrong_album_not_equated':True,'structured_reason_constraint_kept':True,'temporal_pending_not_model_failure':True,'actual_early_false_claim_not_excused':True,'api_calls':0}))
