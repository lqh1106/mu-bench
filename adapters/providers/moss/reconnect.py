"""Reconnect a physical MOSS socket without changing the logical case/session."""
from __future__ import annotations
import asyncio,inspect,json,time,uuid
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
from adapters.providers.moss.base import (
    MossLiveLogicalAdapter as LegacyAdapter, MossProtocolError,
    MossDurationLimitReached, MossBillingFailure,
)

PROTOCOL='moss-same-session-reconnect-v1'
class MossResumeError(MossProtocolError):pass

def duration_close(ws,exc=None):
    received=getattr(exc,'rcvd',None)
    code=getattr(received,'code',None) or getattr(ws,'close_code',None)
    reason=getattr(received,'reason',None) or getattr(ws,'close_reason','')
    return code==1000 and reason=='connection_duration_limit'

class ResumingSocket:
    def __init__(self,raw,adapter,dial=None):
        self.raw=raw;self.read_raw=raw;self.adapter=adapter
        self.dial=dial or self._dial;self.lock=asyncio.Lock()
        self.closing=False;self.reconnect_count=0;self.started=time.monotonic()
        self.max_reconnects=16
    @property
    def close_code(self):return getattr(self.raw,'close_code',None)
    @property
    def close_reason(self):return getattr(self.raw,'close_reason','')
    async def _dial(self):
        import websockets
        a=self.adapter;raw=None
        try:
            sock=await asyncio.to_thread(a._tunnel)
            header='additional_headers' if 'additional_headers' in inspect.signature(websockets.connect).parameters else 'extra_headers'
            raw=await websockets.connect(a.endpoint['url'],sock=sock,
                **{header:{'Authorization':'Bearer '+a._key}},open_timeout=30,
                close_timeout=5,max_size=16*1024*1024,ping_interval=20,ping_timeout=30)
            original=a.provider.session_id
            await raw.send(json.dumps(dict(action='attach_session',session_id=original,memory_scope='session')))
            for expected in ['session_attached','session.updated']:
                packet=await asyncio.wait_for(raw.recv(),20)
                if not isinstance(packet,str):raise MossResumeError('Unexpected media during reconnect setup')
                obj=json.loads(packet);d=obj.get('data') or {}
                if obj.get('action')!=expected:
                    error=(d.get('error') or {})
                    code=error.get('code') if isinstance(error,dict) else str(error)
                    raise MossResumeError('Reconnect setup rejected: '+str(obj.get('action'))+' / '+str(code))
                if expected=='session_attached':
                    if d.get('session_id')!=original or d.get('memory_scope')!='session':
                        raise MossResumeError('Reconnect did not acknowledge the original session and memory scope')
                    await raw.send(json.dumps(dict(action='session.update',session=a._session_config())))
                elif d.get('error') not in ['',None]:raise MossResumeError('Reconnect configuration was rejected')
                else:await a._on_session_configured(d)
            await raw.send(json.dumps(dict(action='mic_state',enabled=True)))
            return raw
        except BaseException:
            if raw is not None:
                try:await asyncio.wait_for(raw.close(),6)
                except BaseException:pass
            raise
    async def _replace(self,closed):
        async with self.lock:
            if self.raw is not closed:return
            if self.closing:raise MossResumeError('Reconnect requested during explicit disconnect')
            if self.reconnect_count>=self.max_reconnects:raise MossResumeError('Bounded session reconnect budget exhausted')
            start=time.monotonic();a=self.adapter
            a._audit('moss_transport_limit_reconnect_begin',session_id=a.provider.session_id,
                connection_index=self.reconnect_count,actual_elapsed_seconds=start-self.started,
                close_code=1000,close_reason='connection_duration_limit',user=a._resume_user)
            async def bounded_dial():
                for n in range(4):
                    try:return await self.dial()
                    except Exception as exc:
                        status=getattr(getattr(exc,'response',None),'status_code',None) or getattr(exc,'status_code',None)
                        if status not in [429,502,503,504] or n==3:raise
                        a._audit('moss_resume_handshake_retry',attempt=n+1,http_status=status,wait_seconds=2**n)
                        await asyncio.sleep(2**n)
            try:new=await asyncio.wait_for(bounded_dial(),120)
            except Exception as exc:
                a._audit('moss_session_reconnect_failed',error_type=type(exc).__name__,session_id=a.provider.session_id)
                raise MossResumeError('Same-session reconnect failed: '+type(exc).__name__) from exc
            self.raw=new;self.reconnect_count+=1;self.started=time.monotonic()
            a._audit('moss_session_reconnected',session_id=a.provider.session_id,
                connection_index=self.reconnect_count,reconnect_seconds=self.started-start,
                memory_scope='session',same_credential=True,logical_clock_reset=False,
                input_replay=False,protocol=PROTOCOL)
    async def send(self,data):
        while True:
            raw=self.raw
            try:return await raw.send(data)
            except Exception as exc:
                if not duration_close(raw,exc):raise
                await self._replace(raw)
    async def recv(self):
        while True:
            raw=self.read_raw
            try:return await raw.recv()
            except Exception as exc:
                if not duration_close(raw,exc):raise
                await self._replace(raw)
                # Drain the old socket before this boundary. Any context left
                # here never received a PCM packet, so it cannot be played.
                a=self.adapter
                orphaned=list(a._contexts);a._contexts.clear()
                interrupted=list(a._active);a._active.clear()
                a._audit('moss_connection_read_boundary',session_id=a.provider.session_id,
                    connection_index=self.reconnect_count,unreceived_contexts=orphaned,
                    interrupted_active_turns=interrupted,
                    received_audio_preserved=True,acknowledgements_preserved=True)
                self.read_raw=self.raw
    def __aiter__(self):return self
    async def __anext__(self):
        try:return await self.recv()
        except Exception:
            if self.closing:raise StopAsyncIteration
            raise
    async def close(self):
        self.closing=True
        await self.raw.close()
        if self.read_raw is not self.raw:
            try:await self.read_raw.close()
            except Exception:pass

class MossLiveLogicalAdapter(LegacyAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._resume_user='mubench-'+uuid.uuid4().hex
        u=urlsplit(self.endpoint['url']);query=dict(parse_qsl(u.query));query['user']=self._resume_user
        self.endpoint['url']=urlunsplit((u.scheme,u.netloc,u.path,urlencode(query),u.fragment))
    @property
    def provider_limit_reached(self):return False
    def _capture_close(self,exc=None):
        # Reconnect failure is infrastructure failure, never an accepted early
        # provider-duration end, even if the old raw socket closed with 1000.
        self._audit('moss_unrecovered_transport_close',error_type=type(exc).__name__ if exc else None)
        return False
    def _call(self,coro,timeout=180):return super()._call(coro,timeout)
    async def _connect_once(self):
        await super()._connect_once()
        self._ws=ResumingSocket(self._ws,self)
        self._audit('moss_session_resume_protocol',protocol=PROTOCOL,
            session_id=self.provider.session_id,user=self._resume_user,
            memory_scope='session',max_reconnects=16)
    async def _disconnect(self):
        ws=self._ws
        if isinstance(ws,ResumingSocket):ws.closing=True
        try:
            if ws is not None and not self._fatal:
                try:await self._flush_completed()
                except Exception as exc:self._audit('moss_disconnect_ack_interrupted',error_type=type(exc).__name__)
        finally:
            self.is_connected=False
            if ws is not None:await ws.close()
            if self._receiver is not None:
                self._receiver.cancel()
                try:await self._receiver
                except asyncio.CancelledError:pass
