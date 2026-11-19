"""Private MiniCPM ablation: native mixed prefill, same persistent session."""
import hashlib,json,uuid
from experiments.mu_bench.logical_time.openweight_bridge import OpenweightLogicalAdapter
INSTRUCTIONS = "This is a turn-based audio conversation. Let each speaker finish before replying. trusted_routing identifies the current speaker, addressee and request owner, without granting permissions. structured_user_input is a faithful parse of already-spoken user input, not an answer. Resolve requests, constraints and permissions yourself. Do not read record formatting aloud."
class MiniAblationAdapter(OpenweightLogicalAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._ab_active=False;self._ab_start=self._ab_end=None;self._held=bytearray();self._context_seq=0
    def connect(self,system_prompt,tools,modality='audio'):
        super().connect(system_prompt,tools,modality)
        self._append({'kind':'ablation_protocol','instructions':INSTRUCTIONS})
    def _append(self,payload):
        if payload is None:return
        self._context_seq+=1
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        result=self.client._post('/sessions/'+self.provider.session_id+'/append_context',
            {'payload':payload,'logical_ms':self._status['logical_ms'],'request_id':'aux-'+str(self._context_seq)})
        assert result['queued'] and result['payload_sha256']==digest and result['logical_ms']==self._status['logical_ms']
        self.audit_records.append({'type':'ablation_context_queued','payload':payload,**result})
    def set_ablation_input(self,active,start_payload=None,end_payload=None):
        self._ab_active=bool(active);self._ab_start=start_payload;self._ab_end=end_payload
    def run_tick(self,user_audio,tick_number):
        self._append(self._ab_start);self._append(self._ab_end);self._ab_start=self._ab_end=None
        result=super().run_tick(user_audio,tick_number)
        pcm=result.get_played_agent_audio();n=len(pcm)
        if self._ab_active:
            if self._held or any(pcm):self._held.extend(pcm)
            played=bytes(n)
        elif self._held:
            if any(pcm):self._held.extend(pcm)
            played=bytes(self._held[:n]).ljust(n,b'\0');del self._held[:n]
        else:played=pcm
        if len(self._held)>64*1024*1024:raise RuntimeError('serial output hold exceeded 64MiB')
        self.audit_records.append({'type':'serial_pcm_playback','logical_ms':self._status['logical_ms'],
            'user_active':self._ab_active,'held_bytes':len(self._held),'native_bytes':n,'played_sha256':hashlib.sha256(played).hexdigest()})
        result.get_played_agent_audio=lambda:played
        return result
    def is_output_drained(self):
        return not self._held and super().is_output_drained()
