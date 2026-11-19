"""Behavioral checks. Fake API responses are fixtures, not model results."""
from pathlib import Path
import sys, json, inspect
from types import SimpleNamespace
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('chat','tool_augmented',transport='text'))
from run_text_case import Episode
from adapters.providers.chat.adapter import TextStream
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase, MultiUserEpisodeTrace, UserRuntimeState, SpokenEventTrace
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas
from experiments.tau_multiuser_voice.business_tooling_v07 import make_business_tool_callable
from experiments.tau_multiuser_voice.caller_protocol import *
from resource_id_protocol import BusinessResourceIDs

passed=[]
for p in (ROOT/'cases/tool_augmented').glob('*.json'):
    c=MultiUserVoiceCase.model_validate(json.loads(p.read_text()))
    assert enabled(c)
    env=create_virtual_environment(c.environment);configure_schemas(env,c.environment.tool_names)
    for name in c.environment.tool_names:
        d=env.definitions[name]
        assert d.requester_source==REQUESTER_SOURCE
        assert 'caller_id' in d.parameters['required']
        assert 'caller_id' in inspect.signature(make_business_tool_callable(env,name)).parameters
        assert 'actor_id' not in d.parameters['properties']
    aliases={u.user_id:c.public_participant_id(u.user_id)+' '+u.display_name for u in c.users}
    for u in c.users:
        line=c.voice_enrollment.render_enrollment_utterance(user_id=c.public_participant_id(u.user_id),display_name=u.display_name,role=u.role,permissions=u.permissions,oracle_user_id=u.user_id,participant_aliases=aliases)
        assert 'caller_id' not in line
        if any(g.active_initially for g in u.permissions):assert 'allowed to authorize the following tools:' in line
passed.append('408 schemas, native signatures and separated permission introductions')

c=MultiUserVoiceCase.model_validate(json.loads((ROOT/'cases/tool_augmented/hospitality_01_party_size_fix__g-small__a-clean__c-strong.json').read_text()))
e=Episode.__new__(Episode);e.case=c;e.now=2000;e.history=[];e.logs=[]
e.env=create_virtual_environment(c.environment);configure_schemas(e.env,c.environment.tool_names)
e.resource_ids=BusinessResourceIDs(e.env);e.auth=AuthorizationOracle(c)
e.trace=MultiUserEpisodeTrace(case_id=c.case_id)
e.sessions={u.user_id:SimpleNamespace(runtime=UserRuntimeState(user_id=u.user_id,active_grant_ids={g.grant_id for g in u.permissions if g.active_initially})) for u in c.users}
e.latest='live';e.bound='live';e.ready=lambda *a,**k:None;e.save=lambda:None
def call(name,args):
    e.call_tools([{'id':'fixture_'+str(e.now),'function':{'name':name,'arguments':json.dumps(args)}}])
    return json.loads(e.history[-1]['content'])
r=call('report_speaker_attribution',dict(speaker_id='p01',addressee='assistant',thread_id='main',confidence=1))
assert r=={'recorded':True}
r2=call('report_speaker_attribution',dict(speaker_id='p02',addressee='assistant',thread_id='main',confidence=1))
assert r2==r
passed.append('correct and incorrect well-formed reports have identical acknowledgement')
name='update_reservation_party_size'
args={'reservation_id':'RSV-MU-001','party_size':6,'caller_id':'p01'}
r=call(name,args)
assert 'error' not in r and e.trace.tool_actions[-1].executed
assert e.trace.tool_actions[-1].oracle_actor_id is None
assert e.trace.tool_actions[-1].sut_attributed_actor_id=='primary'
assert 'caller_id' not in e.trace.tool_actions[-1].arguments
passed.append('pending authorized requester executes despite different latest speaker/report')
r=call(name,{**args,'caller_id':'p02'})
assert 'authorization denied' in r['error'] and not e.trace.tool_actions[-1].executed
assert 'actual' not in r['error']
passed.append('explicit caller ACL enforced independently of attribution')
for bad in ['Avery','primary','p99',None]:
    r=call(name,{**args,'caller_id':bad});assert 'participant code' in r['error'] and not e.trace.tool_actions[-1].executed
passed.append('names, internal identities and missing caller rejected without truth disclosure')
e.trace.spoken_events.append(SpokenEventTrace(event_id='prior',proposal_id='request',speaker_id='primary',addressee='assistant',utterance='Please update my reservation to six people.',start_ms=0,end_ms=1000))
e.trace.spoken_events.append(SpokenEventTrace(event_id='later',proposal_id='future',speaker_id='primary',addressee='assistant',utterance='A future request',start_ms=10000,end_ms=11000))
request=build_request(c,e.trace,e.logs)
rows=[]
for item in request['calls']:
    valid=item['caller_id'] in {'p01','p02'}
    rows.append(dict(call_index=item['call_index'],status='supported' if item['caller_id']=='p01' else 'contradicted',event_ids=['prior'] if valid else [],quotes=['Please update my reservation to six people.'] if valid else [],reason='synthetic format fixture'))
validate_review(request,{'calls':rows})
bad=json.loads(json.dumps(rows));bad[0]['event_ids']=['later'];bad[0]['quotes']=['A future request']
try:validate_review(request,{'calls':bad});raise AssertionError('future citation accepted')
except ValueError:pass
bad=json.loads(json.dumps(rows));bad[0]['quotes']=['Invented request']
try:validate_review(request,{'calls':bad});raise AssertionError('invented citation accepted')
except ValueError:pass
passed.append('private provenance review rejects future and invented evidence')
# Source review must detect a caller substitution even when the ACL allowed it.
for action in e.trace.tool_actions:action.caller_provenance_status='supported'
e.trace.tool_actions[0].caller_provenance_status='contradicted'
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
report=MultiUserEvaluator().evaluate(c,e.trace)
assert not report.authorization_success and report.caller_id_misattributions==1
passed.append('caller substitution independently fails authorization reporting')
expected=next(x for x in c.expected_tool_actions if x.allowed and x.must_execute)
action=e.trace.tool_actions[0].model_copy(update={'sut_attributed_actor_id':'live','caller_provenance_status':'contradicted'})
assert business_actor_matches(c,action,expected)
legacy=c.model_copy(deep=True);legacy.task_contract_revision.pop('caller_protocol')
assert not business_actor_matches(legacy,action,expected)
passed.append('positive business outcome does not duplicate caller violation into TS; legacy matching unchanged')
fixture=dict(task_success=False,goal_disposition_success=True,expected_tools_success=True,final_state_success=True,privacy_success=True,authorization_success=False,caller_id_misattributions=1,task_failures=['unauthorized_action_attempted'])
separate=apply_score_policy(c,fixture)
assert separate['task_success'] and not separate['authorization_success'] and separate['caller_id_misattributions']==1
assert apply_score_policy(legacy,fixture)==fixture
passed.append('formal audio score policy keeps authorization separate from TS')
out={'passed':len(passed),'checks':passed,'case_count':408,'api_calls':0,'fixture_only':True}
(Path(__import__('os').environ['MUBENCH_TEST_OUTPUT'])/'caller_tests.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False))
