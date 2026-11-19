"""Native MOSS-Live transport for the existing causal PCM/played-audio runner.

No output transcript is supplied to the simulator. Only returned PCM counts as
heard; acknowledgements are sent on the following tick, after that PCM was
committed by the caller. Every case creates a new session-scoped memory.
"""
from __future__ import annotations
import asyncio,audioop,inspect,json,socket,threading,time
from pathlib import Path
from types import SimpleNamespace
from collections import deque
from adapters.providers.moss.billing import quota_evidence,exception_quota_evidence

class MossProtocolError(RuntimeError):pass
class MossDurationLimitReached(Exception):
 """User-authorized normal provider end, separate from transport failures."""
 pass
class MossBillingFailure(MossProtocolError):
 def __init__(self,reason):super().__init__(reason);self.reason=reason

class MossLiveLogicalAdapter:
 def __init__(self,endpoint_file,*,tick_duration_ms,seed,max_media_ms,input_sample_rate=16000):
  self.endpoint=json.loads(Path(endpoint_file).read_text())
  self._pool=None  # One explicitly configured key per lane; never rotate mid-session.
  self.tick_ms=tick_duration_ms;self.input_sample_rate=input_sample_rate
  self.bytes_per_tick=input_sample_rate*2*self.tick_ms//1000
  self.output_sample_rate=48000;self.seed=seed;self.max_media_ms=max_media_ms
  self.provider=SimpleNamespace(session_id=None);self.is_connected=False
  self.audit_records=[];self.paused=False;self._clock_ms=0;self._in_state=None
  self._audio=deque();self._contexts=deque();self._events=deque();self._acks=deque()
  self._audio_done=set();self._turn_complete_sent=set();self._stopped=set();self._active=set()
  self._partial={};self._fatal=None;self._ws=None;self._receiver=None
  self.diagnostic_transcripts=[]
  self._provider_end=None;self._connection_started=None;self._last_uploaded_input_bytes=0;self._input_frontier_ms=0
  self.loop=asyncio.new_event_loop();self.thread=threading.Thread(target=self.loop.run_forever,daemon=True)
  self.thread.start()
 def _audit(self,kind,**kw):self.audit_records.append(dict(kind=kind,logical_ms=self._clock_ms,wall_unix=time.time(),**kw))
 @property
 def provider_limit_reached(self):return self._provider_end is not None
 def _capture_close(self,exc=None):
  received=getattr(exc,'rcvd',None)
  code=getattr(received,'code',None) or getattr(self._ws,'close_code',None)
  reason=getattr(received,'reason',None) or getattr(self._ws,'close_reason','')
  self._audit('moss_connection_closed',close_code=code,close_reason=reason,
   actual_elapsed_seconds=time.monotonic()-self._connection_started if self._connection_started is not None else None,
   peer_initiated=getattr(exc,'rcvd_then_sent',None))
  if code==1000 and reason=='connection_duration_limit':
   self._provider_end=dict(close_code=code,close_reason=reason,actual_elapsed_seconds=time.monotonic()-self._connection_started if self._connection_started is not None else None)
   self._active.clear()
   return True
  return False
 def _call(self,coro,timeout=60):return asyncio.run_coroutine_threadsafe(coro,self.loop).result(timeout)
 def _raise(self):
  if self._fatal is not None:raise MossProtocolError(self._fatal)
 def _tunnel(self):
  from urllib.parse import urlsplit
  target=urlsplit(self.endpoint['url']);host=target.hostname;port=target.port or 443
  proxy_host=self.endpoint.get('proxy_host')
  if not proxy_host:return socket.create_connection((host,port),timeout=20)
  sock=socket.create_connection((proxy_host,self.endpoint.get('proxy_port',3128)),timeout=20)
  header=f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n'
  if self.endpoint.get('proxy_credential_file'):
   header+='Proxy-Authorization: Bearer '+Path(self.endpoint['proxy_credential_file']).read_text().strip()+'\r\n'
  sock.sendall((header+'\r\n').encode());reply=b''
  while b'\r\n\r\n' not in reply:
   part=sock.recv(1)
   if not part:raise MossProtocolError('Proxy closed during CONNECT')
   reply+=part
   if len(reply)>16384:raise MossProtocolError('Proxy header too large')
  if b' 200 ' not in reply.split(b'\r\n',1)[0]:
   sock.close();raise MossProtocolError('Proxy CONNECT was rejected')
  return sock
 async def _send(self,action,**kw):
  try:await self._ws.send(json.dumps(dict(action=action,**kw)))
  except Exception as exc:
   if self._capture_close(exc):raise MossDurationLimitReached('connection_duration_limit while sending '+action) from exc
   raise
 async def _connect(self):
  if self._pool is None:
   raw=Path(self.endpoint['credential_file']).read_text().strip()
   self._key=raw.split('=',1)[1] if raw.startswith('MOSS_API_KEY=') else raw
   return await self._connect_once()
  while True:
   self._key_index,self._key=self._pool.current()
   try:
    for attempt in range(4):
     try:return await self._connect_once()
     except Exception as transient:
      response=getattr(transient,'response',None)
      status=getattr(response,'status_code',None) or getattr(transient,'status_code',None)
      if status not in [429,502,503,504] or attempt==3:raise
      self._audit('moss_handshake_retry',http_status=status,attempt=attempt+1,wait_seconds=2**attempt)
      if self._ws is not None:await self._ws.close();self._ws=None
      await asyncio.sleep(2**attempt)
   except Exception as exc:
    reason=exc.reason if isinstance(exc,MossBillingFailure) else exception_quota_evidence(exc)
    if not reason:raise
    if self._ws is not None:await self._ws.close();self._ws=None
    self._pool.exhaust(self._key_index,reason)
    self._audit('moss_credential_rotated_before_media',previous_index=self._key_index,reason=reason)
 def _session_config(self):return {}
 async def _on_protocol_event(self,action,data):pass
 async def _on_session_configured(self,data):pass
 async def _connect_once(self):
  import websockets
  sock=await asyncio.to_thread(self._tunnel)
  header_arg='additional_headers' if 'additional_headers' in inspect.signature(websockets.connect).parameters else 'extra_headers'
  self._ws=await websockets.connect(self.endpoint['url'],sock=sock,
      **{header_arg:{'Authorization':'Bearer '+self._key}},open_timeout=30,
      close_timeout=5,max_size=16*1024*1024,ping_interval=20,ping_timeout=30)
  self._connection_started=time.monotonic()
  await self._send('attach_session',memory_scope='session')
  for expected in ['session_attached','session.updated']:
   msg=await asyncio.wait_for(self._ws.recv(),30)
   if not isinstance(msg,str):raise MossProtocolError('Unexpected media during setup')
   obj=json.loads(msg);data=obj.get('data') or {}
   reason=quota_evidence(payload=data) if obj.get('action')=='error' else None
   if reason:raise MossBillingFailure(reason)
   if obj.get('action')!=expected:raise MossProtocolError('Unexpected setup event: '+str(obj.get('action')))
   if expected=='session_attached':
    if data.get('memory_scope')!='session' or not data.get('session_id'):
     raise MossProtocolError('Fresh session memory was not acknowledged')
    self.provider.session_id=data['session_id']
    # Native no-tool model track: retain the official persona and instructions.
    await self._send('session.update',session=self._session_config())
   elif data.get('error') not in ['',None]:raise MossProtocolError('MOSS session configuration rejected')
   else:await self._on_session_configured(data)
  await self._send('mic_state',enabled=True)
  self._receiver=asyncio.create_task(self._receive())
  self._audit('moss_session_started',session_id=self.provider.session_id,memory_scope='session',custom_prompt=False,
              credential_index=getattr(self,'_key_index',None))
 def connect(self,system_prompt,tools,modality='audio'):
  if self.is_connected:raise MossProtocolError('Session already connected')
  if system_prompt or tools:raise MossProtocolError('This native track uses official defaults and no tools')
  self._call(self._connect());self.is_connected=True
 async def _flush_completed(self):
  if self.provider_limit_reached:return
  while self._acks:
   c=self._acks.popleft();await self._send('playback.chunk.completed',**c)
   self._audit('moss_chunk_playback_ack',**c)
  queued={(c['run_id'],c['turn_id']) for c,_,_ in self._audio}
  for k in sorted(self._audio_done-self._turn_complete_sent-self._stopped):
   if k not in queued:
    self._turn_complete_sent.add(k)
    await self._send('playback.completed',run_id=k[0],turn_id=k[1])
    self._audit('moss_turn_playback_ack',run_id=k[0],turn_id=k[1])
 async def _receive(self):
  try:
   async for packet in self._ws:
    if isinstance(packet,bytes):
     if not self._contexts:raise MossProtocolError('Audio arrived without response.audio.context')
     c=self._contexts.popleft();k=(c['run_id'],c['turn_id'])
     if len(packet)%2:raise MossProtocolError('Odd PCM S16LE audio byte count')
     if k not in self._stopped:self._audio.append((c,packet,0))
     continue
    obj=json.loads(packet);a=obj.get('action');d=obj.get('data') or {}
    await self._on_protocol_event(a,d)
    self._events.append(SimpleNamespace(type='moss.'+str(a)))
    safe={k:d[k] for k in ['run_id','turn_id','chunk_id','status'] if k in d}
    self._audit('moss_wire_event',action=a,**safe)
    if a=='response.audio.context':
     c={k:d.get(k) for k in ['chunk_id','run_id','turn_id']}
     if not all(isinstance(v,str) and v for v in c.values()):raise MossProtocolError('Missing audio identity')
     self._contexts.append(c)
     if (c['run_id'],c['turn_id']) not in self._stopped:self._active.add((c['run_id'],c['turn_id']))
    elif a in ['input.transcript.done','output.transcript.done']:
     self.diagnostic_transcripts.append(dict(action=a,text=d.get('text',''),run_id=d.get('run_id'),turn_id=d.get('turn_id')))
    elif a=='response.audio.done':self._audio_done.add((d['run_id'],d['turn_id']))
    elif a=='response.audio.stop':
     k=(d['run_id'],d['turn_id']);self._stopped.add(k);self._active.discard(k)
     # Whole chunks already heard must be acknowledged before the partial prefix.
     await self._flush_completed()
     played_ms=self._partial.get(k,0)//96
     self._audio=deque(x for x in self._audio if (x[0]['run_id'],x[0]['turn_id'])!=k)
     await self._send('playback.stopped',run_id=k[0],turn_id=k[1],played_audio_ms=played_ms)
     self._audit('moss_playback_stopped',run_id=k[0],turn_id=k[1],played_audio_ms=played_ms)
    elif a=='response.ended':
     if d.get('status') not in ['completed','aborted','failed']:raise MossProtocolError('Unknown terminal status')
     k=(d['run_id'],d['turn_id']);self._active.discard(k)
     if d['status']=='failed':
      reason=quota_evidence(payload=d)
      if reason and getattr(self,'_pool',None):self._pool.exhaust(self._key_index,reason)
      detail=json.dumps(d,ensure_ascii=False).replace(self._key,'[REDACTED]')
      self._audit('moss_failed_response_details',provider_payload=json.loads(detail))
      if reason:raise MossBillingFailure(reason)
      # A failed response is one turn, not a closed conversation.
      self._audit('moss_failed_turn_continued',run_id=k[0],turn_id=k[1])
    elif a=='error':
     reason=quota_evidence(payload=d)
     if reason and getattr(self,'_pool',None):self._pool.exhaust(self._key_index,reason)
     raise MossProtocolError('MOSS server error: '+str((d.get('error') or {}).get('code','unknown')))
  except asyncio.CancelledError:pass
  except Exception as exc:
   if isinstance(exc,(MossProtocolError,json.JSONDecodeError)) or not self._capture_close(exc):
    detail=type(exc).__name__+': '+str(exc)
    self._fatal=self._pool.redact(detail) if getattr(self,'_pool',None) else detail.replace(self._key,'[REDACTED]')
  finally:
   if self.is_connected and self._fatal is None and not self.provider_limit_reached:
    if not self._capture_close():self._fatal='MOSS connection closed during an active case; code='+str(getattr(self._ws,'close_code',None))+'; reason='+str(getattr(self._ws,'close_reason',''))
 async def _tick(self,pcm):
  self._last_uploaded_input_bytes=0
  self._raise();await self._flush_completed()
  if self.provider_limit_reached and not self._audio:raise MossDurationLimitReached('connection_duration_limit')
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
  while self._audio and len(out)<wanted:
   c,data,pos=self._audio.popleft();k=(c['run_id'],c['turn_id'])
   if k in self._stopped:continue
   n=min(wanted-len(out),len(data)-pos);out.extend(data[pos:pos+n]);pos+=n
   self._partial[k]=pos
   if pos==len(data):
    self._acks.append(c);self._partial[k]=0
   else:self._audio.appendleft((c,data,pos))
  events=list(self._events);self._events.clear()
  return bytes(out).ljust(wanted,b'\0'),events
 def run_tick(self,user_audio,tick_number):
  if self.paused or not self.is_connected:raise MossProtocolError('Tick while paused or disconnected')
  if len(user_audio)!=self.bytes_per_tick:raise MossProtocolError('Wrong input tick size')
  pcm=user_audio
  if self.input_sample_rate!=16000:
   pcm,self._in_state=audioop.ratecv(pcm,2,1,self.input_sample_rate,16000,self._in_state)
  wanted=16000*2*self.tick_ms//1000;delta=wanted-len(pcm)
  if delta<0 or delta>8:raise MossProtocolError('Unexpected resampling drift')
  pcm=pcm.ljust(wanted,b'\0')
  played,events=self._call(self._tick(pcm))
  committed=len(user_audio)*self._last_uploaded_input_bytes//max(1,len(pcm));committed-=committed%2
  self._clock_ms=(tick_number+1)*self.tick_ms
  self._audit('moss_media_tick',input_pcm_bytes=len(pcm),played_pcm_bytes=len(played),output_rate=48000)
  return SimpleNamespace(events=events,tool_calls=[],vad_events=[],was_truncated=False,committed_input_pcm=user_audio[:committed].ljust(len(user_audio),b'\0'),
      truncated_audio_bytes=0,proportional_transcript='',get_played_agent_audio=lambda:played)
 def pause_logical_clock(self,reason,logical_time_ms):
  if self.paused or self._clock_ms!=logical_time_ms:raise MossProtocolError('Pause at wrong media frontier')
  self.paused=True;self._audit('planning_pause',reason=reason,server_generation_frozen=False,barrier='logical_media_only')
 def resume_logical_clock(self,reason,logical_time_ms):
  if not self.paused or self._clock_ms!=logical_time_ms:raise MossProtocolError('Resume at wrong media frontier')
  self.paused=False
  if self.provider_limit_reached:raise MossDurationLimitReached('connection_duration_limit during planning')
 async def _drain_closed(self):
  assert self.provider_limit_reached
  output=[];identities=[]
  while self._audio:
   c,data,pos=self._audio.popleft()
   if (c['run_id'],c['turn_id']) in self._stopped:continue
   output.append(data[pos:]);identities.append(dict(**c,remaining_bytes=len(data)-pos))
  pcm=b''.join(output)
  self._audit('moss_duration_limit_output_drained',remaining_pcm_bytes=len(pcm),chunks=identities,
   acknowledgements_not_sent_after_closed_connection=True)
  return pcm
 def drain_closed_audio(self):return self._call(self._drain_closed())
 def is_output_drained(self):return not self._audio and not self._active
 def send_tool_result(self,*a,**kw):raise MossProtocolError('Tools are outside the approved tool-free cohort')
 async def _disconnect(self):
  if self._ws is not None:
   if not self._fatal and not self.provider_limit_reached:await self._flush_completed()
   self.is_connected=False
   await self._ws.close()
  if self._receiver is not None:
   self._receiver.cancel()
   try:await self._receiver
   except asyncio.CancelledError:pass
 def disconnect(self):
  try:self._call(self._disconnect(),timeout=15)
  finally:self.is_connected=False;self.loop.call_soon_threadsafe(self.loop.stop)
 def force_abort(self,reason='supervisor_cleanup'):
  self.disconnect();return dict(aborted=True,reason=reason)
