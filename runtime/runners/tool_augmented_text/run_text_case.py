import bootstrap
"""Isolated tool-augmented/tool-review pilot. No audio, ASR or voice score fabrication."""
from pathlib import Path
import os,sys,json,time,math,hashlib,argparse,traceback
HERE=Path(__file__).resolve().parent

sys.path.append(str(HERE.parents[3]/'third_party/tau2/src'))
os.environ['MUBENCH_JUDGE_PROTOCOL']='tool_full_source_review'
from adapters.providers.chat.adapter import API,JSONBackend,TextStream,token_cost
from score_policy import separate_authorization
from run_tau_mu_voice_realtime import build_system_prompt,normalize_user_reference,normalize_addressee_reference,normalize_tool_arguments,infer_resource_id,run_semantic_judge,measure_floor_split_timings
from experiments.tau_multiuser_voice.models import *
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas,public_result
from experiments.tau_multiuser_voice.caller_protocol import resolve_caller, REQUESTER_SOURCE, VERSION as CALLER_VERSION
from experiments.mu_bench.logical_time.online_users import OnlineUserSession,OrderedOnlineEventGraph,OnlineClosurePlanner,LogicalPauseLedger
from experiments.mu_bench.logical_time.pacing import build_case_pacing
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from resource_id_protocol import BusinessResourceIDs
from experiments.tau_multiuser_voice.scene_context import context_section

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,p)

class Episode:
    def __init__(self,cid,output):
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=False)
        self.case=MultiUserVoiceCase.model_validate(json.loads((Path(os.environ['MUBENCH_CASE_DIR'])/(cid+'.json')).read_text()))
        self.api_calls=[];self.api=API(json.loads(Path(os.environ['MUBENCH_CONFIG_FILE']).read_text()),self.api_calls)
        self.backend=JSONBackend(self.api);self.now=0;self.fractional_ms=0.;self.prefix='';self.bound=None;self.latest=None
        self.timeline=[];self.turns=[];self.logs=[];self.stream_audit=[];self.pause=LogicalPauseLedger();self.response_in_flight=False
        self.pacing_plan=build_case_pacing(self.case);self.graph=OrderedOnlineEventGraph(self.case,pacing_plan=self.pacing_plan)
        self.sessions={u.user_id:OnlineUserSession(case=self.case,profile=u,backend=self.backend,timing_alignment_ms=200) for u in self.case.users}
        for session in self.sessions.values():session.event_pacing_rules=self.graph.pacing.rules
        write(self.out/'event_pacing_plan.json',self.pacing_plan.to_dict())
        self.env=create_virtual_environment(self.case.environment);configure_schemas(self.env,self.case.environment.tool_names);self.resource_ids=BusinessResourceIDs(self.env)
        self.auth=AuthorizationOracle(self.case);self.trace=MultiUserEpisodeTrace(case_id=cid,execution_protocol='mu-text-logical-tool-v2',interaction_mechanism_realized=False)
        self.closure=OnlineClosurePlanner(self.backend)
        self.history=[{'role':'system','content':build_system_prompt(self.case,self.env,None,set(self.case.environment.tool_names)).replace(context_section(self.case),'')+self.resource_ids.prompt_suffix()+'\nText-track transport: registration and user speech are delivered as text. The registered-source header identifies the public source, not a grant. Only delivered assistant text counts as communication. Continue from the actually delivered prefix after an interruption; do not claim discarded text was read. Acoustic recognition and physical overlap are not measured. This is a single shared conversation: naming a recipient does not create a private channel.'}]
        aliases={u.user_id:self.case.public_participant_id(u.user_id)+' '+u.display_name for u in self.case.users}
        intros=[]
        for i,u in enumerate(self.case.users):
            if u.voice.enrollment_status!='enrolled':continue
            text=self.case.voice_enrollment.render_enrollment_utterance(user_id=self.case.public_participant_id(u.user_id),display_name=u.display_name,role=u.role,permissions=u.permissions,oracle_user_id=u.user_id,participant_aliases=aliases,assistant_name=self.case.assistant_activation.assistant_name,include_session_preamble=i==0)
            self.history.append({'role':'user','content':'['+u.display_name+'] '+text})
            intros.append({'user_id':u.user_id,'participant_id':self.case.public_participant_id(u.user_id),'display_name':u.display_name,'voice_sample_text':text,'transport':'registration_text'})
        ctx=self.case.task_contract_revision['review']['initial_assistant_context']
        self.history.append({'role':'system','content':ctx['content']})
        caller=self.case.users[0];activation=self.case.voice_enrollment.render_activation_utterance(fallback_template=self.case.assistant_activation.utterance_template,assistant_name=self.case.assistant_activation.assistant_name,user_id=caller.user_id,display_name=caller.display_name,role=caller.role)
        self.history.append({'role':'user','content':'['+caller.display_name+'] '+activation})
        self.trace.assistant_activation=AssistantActivationTrace(assistant_name=self.case.assistant_activation.assistant_name,caller_user_id=caller.user_id,utterance=activation,start_ms=0,end_ms=0)
        write(self.out/'registration_text.json',{'intros':intros,'activation':activation,'transport':'text','audio_created':False})
        # Compatibility read location for V4 provenance; bodies are actual sent text.
        write(self.out/'voice_enrollment.before_judge.json',{'measurement_track':'text-logical','oracle_audio_bindings':intros,'assistant_activation':self.trace.assistant_activation.model_dump(mode='json')})
        write(self.out/'initial_assistant_context.json',{'required':True,'content_sha256':hashlib.sha256(ctx['content'].encode()).hexdigest(),'transport':'text_post_registration_pre_task','delivery_status':'text_submitted'})
        write(self.out/'reviewed_case.json',self.case.model_dump(mode='json'))
        write(self.out/'text_transport_marker.json',{'label':'text-name-labeled','speaker_header':'registered_display_name','speakers':[u.display_name for u in self.case.users],'public_ids_still_used_in_business_parameters':True,'audio_or_voice_identity_test':False,'source_audio_created':False,'caller_prompt_policy':'caller-field-definition'})
        self.tools=[{'type':'function','function':{'name':t,'description':self.env.definitions[t].description,'parameters':self.env.definitions[t].parameters}} for t in self.case.environment.tool_names]
        self.tools.append({'type':'function','function':{'name':'report_speaker_attribution','description':'Report the public participant and addressee of the latest delivered user utterance before responding or business calls.','parameters':{'type':'object','properties':{'speaker_id':{'type':'string'},'addressee':{'type':'string'},'thread_id':{'type':'string'},'confidence':{'type':'number'}},'required':['speaker_id','addressee','thread_id','confidence'],'additionalProperties':False}}})
        self.stream=TextStream(self.api,self.history,self.tools)

    def obs(self,kind=ObservationKind.CLOCK,**extra):
        return WorldObservation(sim_time_ms=self.now,kind=kind,assistant_speaking=extra.pop('assistant_speaking',False),assistant_committed_prefix=self.prefix,public_transcript=[TranscriptTurn(**r) for r in self.timeline if r['end_ms']<=self.now],completed_event_ids=self.graph.completed_event_ids,**extra)

    def ready(self,kind=ObservationKind.CLOCK,**extra):
        speaking=extra.get('assistant_speaking',False);played=extra.pop('_played_ms',0)
        self.graph.pacing.observe_media(self.now,speaking,self.response_in_flight,played_duration_ms=played)
        rows=self.graph.ready_events(self.obs(kind,**extra));return rows[0] if rows else None

    def deliver(self,event):
        # Same OnlineUserSession writer/critic and candidate cap; only no synthesis.
        self.now=int(math.ceil(self.now/200)*200)
        session=self.sessions[event.actor_id]
        if event.action==UserAction.JOIN and not session.runtime.present:
            session.runtime.present=True;session.record_presence(True,self.now)
        with self.pause.planning(self.now,kind='user_planner_and_critic_no_tts'):
            proposal=session.propose(self.obs(),event)
        self.now+=proposal.requested_delay_ms
        start=self.now;duration=max(200,int(math.ceil(sum(token_cost(c) for c in proposal.utterance)/4*1000/200)*200))
        self.now+=duration;self.bound=None;self.latest=event.actor_id
        display_name=session.profile.display_name
        self.history.append({'role':'user','content':'['+display_name+'] '+proposal.utterance})
        self.timeline.append({'speaker':event.actor_id,'text':proposal.utterance,'start_ms':start,'end_ms':self.now})
        self.trace.spoken_events.append(SpokenEventTrace(event_id=event.event_id,proposal_id=proposal.proposal_id,speaker_id=event.actor_id,addressee=proposal.addressee,utterance=proposal.utterance,start_ms=start,end_ms=self.now,ready_ms=start,thread_id=event.thread_id,speech_act=event.speech_act,trigger_path=self.graph.trigger_path(event.event_id) or 'causal'))
        for effect in event.grant_effects:
            rt=self.sessions[effect.target_user_id].runtime
            if effect.active:rt.active_grant_ids.add(effect.grant_id)
            else:rt.active_grant_ids.discard(effect.grant_id)
        for effect in event.presence_effects:
            s=self.sessions[effect.target_user_id];s.runtime.present=effect.present;s.record_presence(effect.present,self.now)
        self.graph.pacing.observe_media(self.now,False,False,played_duration_ms=0)
        self.graph.mark_completed(event.event_id,self.now);self.save()

    def call_tools(self,calls):
        for call in calls:
            name=call['function']['name'];raw=json.loads(call['function']['arguments']);self.now+=200
            if name=='report_speaker_attribution':
                actor=normalize_user_reference(self.case,str(raw.get('speaker_id','')));addressee=normalize_addressee_reference(self.case,str(raw.get('addressee','')))
                valid=actor in set(self.sessions)|{'unknown'} and addressee in set(self.sessions)|{'assistant','group','unknown'}
                self.trace.speaker_attributions.append(SpeakerAttributionTrace(timestamp_ms=self.now,task_timestamp_ms=self.now,predicted_speaker_id=actor,predicted_addressee=addressee,reported_speaker_reference=raw.get('speaker_id'),reported_addressee_reference=raw.get('addressee'),predicted_thread_id=raw.get('thread_id','main'),confidence=max(0,min(1,float(raw.get('confidence',0))))))
                result={'recorded':True} if valid else {'error':'unknown participant code or addressee'}
            else:
                actor,caller_id,args,caller_error=resolve_caller(self.case,raw)
                args=normalize_tool_arguments(name,args,case=self.case);args,resolution,parse_error=self.resource_ids.resolve(name,args);rid=infer_resource_id(self.case,name,actor,args);authorized=False;executed=False;error=None
                if name not in self.case.environment.tool_names:error='unknown business tool'
                elif caller_error:error=caller_error
                elif actor not in self.sessions:error='unknown caller participant code'
                elif parse_error:error=parse_error['code']+': '+parse_error['message']
                else:
                    decision=self.auth.decide(actor_id=actor,action=name,resource_id=rid,runtime=self.sessions[actor].runtime);authorized=decision.allowed
                    if not authorized:error='authorization denied: '+decision.reason
                if error:result={'error':error}
                else:
                    try:result=self.env.invoke(name,actor,**args);executed=True
                    except (ValueError,TypeError,KeyError) as e:error=str(e);result={'error':error}
                row={'timestamp_ms':self.now,'result_available_ms':self.now,'result_availability_basis':'explicit_text_tool_delivery','tool_name':name,'caller_id':caller_id,'authorization_basis':REQUESTER_SOURCE,'actor_id':actor,'oracle_actor_id':None,'resource_id':rid,'arguments':args,'executed':executed,'result':result,'model_visible_result':public_result(self.case,name,result)}
                self.logs.append(row)
                self.trace.tool_actions.append(ToolActionTrace(tool_name=name,resource_id=rid,timestamp_ms=self.now,result_available_ms=self.now,result_availability_basis='explicit_text_tool_delivery',sut_attributed_actor_id=actor or None,oracle_actor_id=None,caller_id=caller_id,authorization_basis=REQUESTER_SOURCE,oracle_authorized=authorized,executed=executed,arguments=args))
            self.history.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(public_result(self.case,name,result),ensure_ascii=False)})
            # No future event content is shown before the actual result exists.
            if name!='report_speaker_attribution' and result.get('error') is None:
                event=self.ready(ObservationKind.TOOL_RESULT,tool_name=name,tool_result=json.dumps(result))
                if event is not None:self.pending=event
        self.save()

    def response(self):
        self.pending=None;started=self.now;local_start=started;last_char_end=started;first=True
        def char(ch,prefix):
            nonlocal first,last_char_end
            self.prefix=prefix
            self.fractional_ms+=token_cost(ch)*1000/4
            delta=int(self.fractional_ms);self.fractional_ms-=delta;self.now+=delta
            last_char_end=self.now
            kind=ObservationKind.ASSISTANT_SPEECH_START if first else ObservationKind.CLOCK;first=False
            event=self.ready(kind,assistant_speaking=True,_played_ms=delta)
            if event is not None:self.pending=event;return True
            return False
        self.response_in_flight=True
        try:out=self.stream.response(char,self.call_tools)
        finally:
            self.response_in_flight=False
            self.graph.pacing.observe_media(self.now,False,False,played_duration_ms=0)
        if out['prefix']:
            row={'speaker':'assistant','text':out['prefix'],'start_ms':local_start,'end_ms':last_char_end}
            self.timeline.append(row);self.turns.append({'role':'assistant','text':out['prefix'],'start_ms':local_start,'end_ms':last_char_end,'committed':True,'transcript_source':'actual_delivered_chat_text','source_id':'assistant_text_'+str(len(self.turns)).zfill(6)})
        if out.get('interrupted') and self.pending is not None:
            self.trace.interaction_mechanism_realized=True
        self.stream_audit.append({'at_ms':started,**out,'tokenizer':'estimated_cjk1_latin_char_quarter','estimated_tokens_per_logical_second':4})
        if self.pending:self.deliver(self.pending)
        elif out['prefix']:
            event=self.ready(ObservationKind.ASSISTANT_SPEECH_STOP)
            if event:self.deliver(event)
        self.save()
        return bool(out['prefix'] or self.pending),out.get('finish_reason')=='tool_calls'

    def save(self):
        self.trace.final_environment_state=self.env.snapshot()
        for name,obj in [('final_trace.before_judge.json',self.trace.model_dump(mode='json')),('assistant_turns.before_judge.json',self.turns),('tool_log.before_judge.json',self.logs),('history.json',self.history),('stream_audit.json',self.stream_audit),('pause_ledger.json',self.pause.records),('api_usage.json',self.api_calls),('event_pacing_audit.json',self.graph.pacing_audit)]:write(self.out/name,obj)
        for uid,s in self.sessions.items():write(self.out/'planner_sessions'/(uid+'.json'),{'messages':s.messages,'contract_audit':s.contract_audit})

    def interact(self):
        self.deliver(self.ready(ObservationKind.CONVERSATION_START))
        quiet=0
        for index in range(100):
            previous_user_count=len(self.trace.spoken_events)
            progressed,tools=self.response()
            # A newly delivered interruption gets an immediate model turn.
            # Do not advance idle time and let a fallback impersonate silence
            # while the client has not even called the model for that utterance.
            if tools or len(self.trace.spoken_events)>previous_user_count:continue
            if len(self.graph.completed_event_ids | self.graph._unrealized_ids)>=len(self.case.events):break
            event=self.ready()
            if event:self.deliver(event);continue
            # Advance only the explicit logical clock to a scheduled/fallback event.
            found=False
            for _ in range(200):
                self.now+=200;event=self.ready()
                if event:self.deliver(event);found=True;break
            if not found:
                quiet+=1
                if quiet>=2:break
            if self.now>=1800000:break
        for round_index in range(2):
            write(self.out/'closure_observations'/(str(round_index)+'.json'),
                {'logical_ms': self.now, 'played_timeline': self.timeline})
            try:
                with self.pause.planning(self.now,'closure_no_tts'):
                    decision=self.closure.decide_online(case=self.case,observation=self.obs(),spoken_events=self.trace.spoken_events,round_index=round_index,played_timeline=self.timeline)
            finally:
                write(self.out/'closure_audit'/(str(round_index)+'.json'),self.closure.audit_log)
            write(self.out/('closure_'+str(round_index)+'.json'),{'decision':decision.model_dump(mode='json'),'audit':self.closure.audit_log})
            if decision.action=='end':break
            event=self.closure.to_event(decision,round_index,self.now);self.graph.register_event(event);self.deliver(event)
            for _ in range(16):
                _,tools=self.response()
                if not tools:break
        self.save()

    def run(self):
        from experiments.tau_multiuser_voice.simulation_contract import SimulatorInvalid
        try:
            self.interact()
        except SimulatorInvalid as exc:
            if exc.reason_code not in {'proposal_contract_exhausted', 'closure_contract_exhausted'}:
                raise
            # Candidates rejected by the simulator are not part of the dialogue.
            # Score the actual delivered prefix, with coverage and truncation disclosed.
            self.save()
            write(self.out/'simulator_exhausted_forced_eval.json', {
                'reason_code': exc.reason_code, 'failed_event_id': exc.event_id,
                'audit': exc.as_dict(), 'forced_eval': True,
                'unaccepted_candidate_delivered': False,
                'required_event_coverage_preserved': True,
                'normal_completion_claimed': False})
        return self.evaluate()

    def evaluate(self):
        self.save();write(self.out/'phase.json',{'phase':'judge','at':time.time()})
        from experiments.tau_multiuser_voice.attribution import apply_attribution_evidence
        apply_attribution_evidence(self.trace,[{'start_ms':t['start_ms'],'end_ms':t['end_ms']} for t in self.turns],self.logs)
        from score_text_evidence import score_evidence
        try:
            report=score_evidence(self.api,self.case,self.trace,self.turns,self.logs,self.out)
        finally:
            self.save()
        write(self.out/'phase.json',{'phase':'complete','at':time.time()})
        return {'case_id':self.case.case_id,'task_success':report['task_success'],'authorization_success':report['authorization_success'],'events':len(self.trace.spoken_events),'tool_calls':len(self.logs),'assistant_turns':len(self.turns)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case-id',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    try:print(json.dumps(Episode(a.case_id,a.output).run()))
    except Exception as e:
        write(Path(a.output)/'error.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'not_model_failure':True,'simulator_audit':e.as_dict() if hasattr(e,'as_dict') else None})
        raise
