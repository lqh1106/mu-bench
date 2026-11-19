"""MOSS serial PCM playback with causal, acknowledged auxiliary context.
Cloud inference is not paused. Native stops remain diagnostic evidence.
"""
import asyncio,json,time
from collections import deque
from adapters.providers.moss.initial_context import MossNTAdapter
from adapters.providers.moss.base import MossProtocolError

INSTRUCTIONS=('This is a turn-based audio conversation. Let each speaker finish before replying. '
 'When auxiliary records are supplied, trusted_routing identifies the current audio speaker and request ownership; '
 'structured_user_input describes only that speaker\'s already-spoken input, not a solution or permission grant. '
 'Records split into numbered parts form one JSON record; combine them in order. '
 'Resolve requests and permissions yourself. Do not read auxiliary record formatting aloud.')

class ControlSocket:
 def __init__(self,raw,adapter):self.raw=raw;self.adapter=adapter
 def __getattr__(self,name):return getattr(self.raw,name)
 def __aiter__(self):return self
 async def __anext__(self):
  packet=await self.raw.__anext__()
  if isinstance(packet,str):
   obj=json.loads(packet);action=obj.get('action');d=obj.get('data') or {};a=self.adapter
   if action=='session.thinking.appended':
    eid=d.get('client_event_id');f=a._pending_context.get(eid)
    if f is not None and not f.done():f.set_result(True)
   elif action=='error':
    err=d.get('error') or {};f=a._pending_context.get(err.get('client_event_id'))
    if f is not None and not f.done():f.set_exception(MossProtocolError('context update rejected: '+str(err.get('code'))))
   elif action=='response.audio.stop':
    k=(d.get('run_id'),d.get('turn_id'))
    completed=k in a._turn_complete_sent
    a._audit('serial_native_stop',during_user_audio=a._ab_active,task_phase=a._ab_in_task,
      playback_already_completed=completed,run_id=k[0],turn_id=k[1],
      played_audio_bytes=a._ab_played.get(k,0),cloud_interruption_disabled=False)
    if a._ab_in_task and a._ab_active and a._ab_played.get(k,0)>0 and not completed:
     raise MossProtocolError('serial condition violated: previously played reply interrupted during user audio')
  return packet

class MossAblationAdapter(MossNTAdapter):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self._ab_active=False;self._ab_start=None;self._ab_end=None
  self._pending_context={};self._ab_sequence=0;self._ab_played={};self._ab_hold_peak=0;self._ab_in_task=True
 def _audit(self,kind,**kw):
  if kind=='moss_session_started':
   kw['custom_prompt']=True;kw['custom_prompt_scope']='turn_auxiliary_records_and_pre_session_source';kw['ablation_enabled']=True
  result=super()._audit(kind,**kw)
  if kind=='moss_session_started':
   self.audit_records[-1].update(ablation_enabled=True,custom_prompt_scope='turn_auxiliary_records_and_pre_session_source')
  return result
 async def _send(self,action,**kw):
  if action=='session.update':
   session=dict(kw.get('session') or {})
   if INSTRUCTIONS not in session.get('instructions',''):
    session['instructions']=(session.get('instructions','')+'\n\n'+INSTRUCTIONS).strip()
   kw['session']=session
  return await super()._send(action,**kw)
 async def _connect_once(self):
  await super()._connect_once();self._ws=ControlSocket(self._ws,self)
  self._audit('moss_ablation_transport',serial_playback=True,cloud_generation_frozen=False,
    native_interrupt_disable_supported=False,context_api='session.thinking.append',
    instructions=INSTRUCTIONS,custom_tools_registered=False,initial_source=getattr(self,'_initial_source',{}))
 def set_ablation_input(self,active,start_payload=None,end_payload=None):
  self._ab_active=bool(active);self._ab_start=start_payload;self._ab_end=end_payload
 async def _context(self,payload):
  if payload is None:return
  raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'));parts=[raw[i:i+1750] for i in range(0,len(raw),1750)]
  for i,part in enumerate(parts):
   self._ab_sequence+=1;eid=f'ablation-{self._ab_sequence}'
   content=f'Auxiliary record {payload["event_id"]}, part {i+1}/{len(parts)}: '+part
   if len(content)>2000:raise MossProtocolError('context record part too long')
   f=asyncio.get_running_loop().create_future();self._pending_context[eid]=f
   try:
    await self._send('session.thinking.append',event_id=eid,content=content)
    await asyncio.wait_for(f,20)
   finally:self._pending_context.pop(eid,None)
   self._audit('ablation_context_ack',event_id=payload['event_id'],context_kind=payload['kind'],part=i+1,parts=len(parts),client_event_id=eid,payload=payload if i==0 else None)
 async def _tick(self,pcm):
  self._last_uploaded_input_bytes=0
  self._raise();await self._flush_completed()
  if self.provider_limit_reached and not self._audio:raise MossDurationLimitReached('connection_duration_limit')
  await self._context(self._ab_start);await self._context(self._ab_end)
  self._ab_start=self._ab_end=None
  step=640
  # 20ms uplink frames, including silence. No accelerated uploads or accumulated replay.
  if len(pcm)%step:raise MossProtocolError('Input is not aligned to 20ms PCM frames')
  for i in range(0,len(pcm),step):
   if self.provider_limit_reached:break
   started=time.monotonic()
   try:await self._ws.send(pcm[i:i+step])
   except Exception as exc:
    if self._capture_close(exc):break
    raise
   self._last_uploaded_input_bytes+=len(pcm[i:i+step])
   await asyncio.sleep(max(0,.020-(time.monotonic()-started)))
  self._input_frontier_ms=self._clock_ms+self._last_uploaded_input_bytes/32
  self._raise()
  wanted=self.output_sample_rate*2*self.tick_ms//1000;out=bytearray()
  while not self._ab_active and self._audio and len(out)<wanted:
   c,data,pos=self._audio.popleft();k=(c['run_id'],c['turn_id'])
   if k in self._stopped:continue
   n=min(wanted-len(out),len(data)-pos);out.extend(data[pos:pos+n]);pos+=n
   self._ab_played[k]=self._ab_played.get(k,0)+n
   self._partial[k]=pos
   if pos==len(data):
    self._acks.append(c);self._partial[k]=0
   else:self._audio.appendleft((c,data,pos))
  held=sum(len(b)-p for _,b,p in self._audio);self._ab_hold_peak=max(self._ab_hold_peak,held)
  if held>64*2**20:raise MossProtocolError('serial held audio limit')
  self._audit('serial_playback_tick',user_active=self._ab_active,played_bytes=len(out),held_bytes=held)
  events=list(self._events);self._events.clear()
  return bytes(out).ljust(wanted,b'\0'),events
