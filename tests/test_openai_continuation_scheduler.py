import asyncio,json
from types import SimpleNamespace
from adapters.providers.openai.adapter import LogicalTimeOpenAIAdapter
from tau2.voice.audio_native.openai.events import parse_realtime_event
class Provider:
 def __init__(self):self.outputs=[];self.creates=[]
 async def send_tool_result(self,*args):self.outputs.append(args)
 async def request_response(self,eid):self.creates.append(eid)
def adapter():
 a=LogicalTimeOpenAIAdapter(tick_duration_ms=200);a._provider=Provider();return a
async def event(a,type,**kw):
 raw={'type':type,**kw};e=parse_realtime_event(raw);await a._process_event(SimpleNamespace(events=[],vad_events=[],tool_calls=[]),e)
async def output(a,ids=('one',)):
 a._pending_tool_results.extend((i,'{}',True,False) for i in ids);await a._flush_pending_tool_results();a._continuation_not_before=0
async def ack(a,cid):await event(a,'conversation.item.created',item={'id':'item_'+cid,'type':'function_call_output','call_id':cid,'output':'{}'})
async def created(a,rid):await event(a,'response.created',response={'id':rid})
async def done(a,rid):await event(a,'response.done',response={'id':rid,'status':'completed'})
async def main():
 a=adapter();await created(a,'old');await output(a,('one','two'));await ack(a,'one');await a._maybe_request_continuation();assert not a.provider.creates;await ack(a,'two');await a._maybe_request_continuation();assert not a.provider.creates;await done(a,'old');await a._maybe_request_continuation();await a._maybe_request_continuation();assert len(a.provider.creates)==1 and all(x[2] is False for x in a.provider.outputs)
 a=adapter();await output(a);await event(a,'conversation.item.added',item={'id':'new','type':'function_call_output','call_id':'one','output':'{}'});await a._maybe_request_continuation();assert not a.provider.creates
 await event(a,'conversation.item.done',item={'id':'new','type':'function_call_output','call_id':'one','output':'{}'});await a._maybe_request_continuation();assert len(a.provider.creates)==1 and not a._tool_output_unacked
 a=adapter();await output(a);await ack(a,'one');await created(a,'auto');await done(a,'auto');await a._maybe_request_continuation();assert not a.provider.creates and a._continuation_covered_epoch==1
 a=adapter();await output(a);await created(a,'early_auto');await ack(a,'one');assert a._continuation_covered_epoch==0;await done(a,'early_auto');await a._maybe_request_continuation();assert len(a.provider.creates)==1
 eid=a.provider.creates[-1];await event(a,'error',error={'code':'conversation_already_has_active_response','event_id':eid,'message':'Conversation already has an active response in progress: racing_response. Wait.'});await a._maybe_request_continuation();assert len(a.provider.creates)==1;await done(a,'racing_response');await a._maybe_request_continuation();assert len(a.provider.creates)==2 and len(a.provider.outputs)==1
 try:await event(a,'error',error={'code':'conversation_already_has_active_response','event_id':'unrelated','message':'active'})
 except RuntimeError:pass
 else:raise AssertionError('Unrelated errors swallowed')
 a=adapter();await output(a);await ack(a,'one');a._continuation_not_before=float('inf');await a._maybe_request_continuation();assert not a.provider.creates;a._continuation_not_before=0;a._paused=True;await a._maybe_request_continuation();assert not a.provider.creates;a._paused=False;a._pending_truncate_acks['item']=[100];await a._maybe_request_continuation();assert not a.provider.creates;a._pending_truncate_acks.clear();await a._maybe_request_continuation();assert len(a.provider.creates)==1
 a=adapter();a._pending_tool_results.append(('one','{}',False,False));await a._flush_pending_tool_results();await ack(a,'one');await a._maybe_request_continuation();assert not a.provider.creates
 a=adapter();await output(a);await ack(a,'one');await a._maybe_request_continuation();eid=a.provider.creates[-1];await created(a,'auto_won');await event(a,'error',error={'code':'conversation_already_has_active_response','event_id':eid,'message':'Conversation already has an active response in progress: auto_won. Wait.'});await done(a,'auto_won');await a._maybe_request_continuation();assert len(a.provider.creates)==1
 print(json.dumps({'passed':True,'checks':['all_outputs_once','coalesced_create','active_response_wait','server_ack_wait','auto_response_covers_outputs','early_auto_not_coverage','identified_collision_retry_without_replay','unknown_error_remains_fatal','debounce','pause','truncate_ack','no_continuation_flag','auto_manual_race_no_duplicate'],'network_calls':0}))
asyncio.run(main())
