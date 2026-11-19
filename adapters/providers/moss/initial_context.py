"""Full-duplex MOSS, unchanged native playback/reconnect, initial source only."""
from adapters.providers.moss.reconnect import MossLiveLogicalAdapter
class MossNTAdapter(MossLiveLogicalAdapter):
 def set_initial_source(self,context):self._initial_source=dict(context or {})
 async def _send(self,action,**kw):
  if action=='session.update':
   source=getattr(self,'_initial_source',{}).get('content','')
   if source:
    session=dict(kw.get('session') or {})
    section='Information available to the assistant before this conversation. Read-only service or shared-document snapshot:\n'+source
    if section not in session.get('instructions',''):session['instructions']=(session.get('instructions','')+'\n\n'+section).strip()
    kw['session']=session
  return await super()._send(action,**kw)
 def _audit(self,kind,**kw):
  if kind=='moss_session_started':
   kw['custom_prompt']=bool(getattr(self,'_initial_source',{}))
   kw['custom_prompt_scope']='tool_free_read_only_initial_source_only'
   kw['case_protocol']='tool-free';kw['ablation_enabled']=False
  return super()._audit(kind,**kw)
