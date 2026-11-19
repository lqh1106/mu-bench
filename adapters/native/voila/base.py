"""Real Voila autonomous streaming adapter, versioned debug implementation.

Only already received audio is encoded. The non-causal codec uses a bounded
past-only context; 1s blocks add explicit adapter latency. This adapter must
pass a single-case pilot before any formal batch is authorized by its gate.
"""
from __future__ import annotations
import sys,time,queue,threading,pickle,json,math
from pathlib import Path
from collections import deque
import numpy as np
import torch

class VoilaBackend:
    def __init__(self,args):
        self.root=Path(args.debug_root);self.repo_path=Path(args.repo_path)
        sys.path.insert(0,str(self.repo_path))
        from transformers import AutoTokenizer
        from model import VoilaAutonomousModel
        from voila_tokenizer import VoilaTokenizer
        self.model=VoilaAutonomousModel.from_pretrained(str(args.model_path),torch_dtype=torch.bfloat16,attn_implementation='sdpa',use_cache=True).cuda().eval()
        self.tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
        self.codec=VoilaTokenizer(model_path=str(self.root/'Voila-Tokenizer'),device='cuda')
        refs=pickle.load(open(self.repo_path/'examples/character_ref_emb_demo.pkl','rb'))
        self.ref=torch.tensor(np.asarray(refs['Homer Simpson']),dtype=torch.float32,device='cuda')[None,:]
        self.ref_mask=torch.tensor([1],device='cuda')
        self.num_codebooks=self.model.config.num_codebooks;self.codebook_size=self.model.config.codebook_size
        self.AUDIO_MIN_TOKEN_ID=self.model.config.min_audio_token_id;self.AUDIO_MAX_TOKEN_ID=self.model.config.max_audio_token_id
        self.ASSISTANT_TOKEN_ID=self.model.config.assistant_token_id
        self.frame_samples=640;self.sample_rate=16000;self.block_frames=25;self.context_frames=75
        self._input=queue.Queue();self._output=deque();self._error=None;self._gen_thread=None
        self._stream_ended=False;self._closed=False;self._last_text='';self._text_queue=[]
        self._raw=[];self._received_frames=0;self._encoded_frames=0;self._consumed_frames=0
        self._output_codes=[];self._pending_codes=[];self._pending_flags=[];self._speaking=False;self._pending_end=False
        self._cond=threading.Condition();self._decoded_frames=0;self._model_steps=0;self._silent=0;self._audit=[]
        self.max_media_ms=1830000;self._max_tokens=100000
        print(json.dumps(dict(event='voila_real_weights_loaded',num_codebooks=self.num_codebooks,audio_min_token_id=self.AUDIO_MIN_TOKEN_ID,input_block_ms=1000,codec_context_ms=3000,causal_input=True)),flush=True)

    def configure(self,seed,max_media_ms):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);self.max_media_ms=max_media_ms
        self._max_tokens=math.ceil(max_media_ms/40)+32768

    def start_stream(self):
        if self._gen_thread is not None:return
        prefix='<SYSTEM><stream_chat><chat_audio_output><au_chat_ref_start><au_chat_ref><au_chat_ref_end></SYSTEM><|HUMAN|>'
        ids=self.tokenizer.encode(prefix,add_special_tokens=False)
        inputs=torch.tensor([[ids]*self.num_codebooks],device='cuda').transpose(1,2)
        inputs=torch.cat([inputs,inputs],dim=2)
        owner=self
        class Streamer:
            def put(self,value):owner._on_token(value)
            def end(self):owner._on_end()
        def inputs_iter():
            while not self._closed:
                try:
                    item=self._input.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:return
                self._consumed_frames+=1
                yield item
        def generate():
            try:
                with torch.inference_mode():
                    self.model.run_generate(input_ids=inputs,input_generator=inputs_iter(),ref_embs=self.ref,ref_embs_mask=self.ref_mask,max_new_tokens=self._max_tokens,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,streamer=Streamer(),llm_audio_token_id=self.model.config.audio_token_id,min_audio_token_id=self.AUDIO_MIN_TOKEN_ID,llm_assistant_token_id=self.ASSISTANT_TOKEN_ID,temperature=0.2,top_k=50,audio_temperature=0.8,audio_top_k=50)
            except BaseException as exc:self._error=exc
            finally:
                with self._cond:self._stream_ended=True;self._cond.notify_all()
        self._gen_thread=threading.Thread(target=generate,daemon=True);self._gen_thread.start()

    def _on_token(self,value):
        item=value.reshape(-1,self.num_codebooks)[-1].tolist();t=item[0];self._model_steps+=1
        if self.AUDIO_MIN_TOKEN_ID<=t<=self.AUDIO_MAX_TOKEN_ID:
            self._output_codes.append([(int(token)-self.AUDIO_MIN_TOKEN_ID)%self.codebook_size for token in item])
            self._pending_codes.append(self._output_codes[-1]);self._pending_flags.append(dict(is_listen=not self._speaking,end_of_turn=self._pending_end))
            self._pending_end=False
            if len(self._pending_codes)>=self.block_frames:self._decode_pending()
        elif t==self.ASSISTANT_TOKEN_ID:self._speaking=True
        elif t==self.tokenizer.eos_token_id:self._speaking=False;self._pending_end=True
        elif t not in [self.tokenizer.pad_token_id]:
            with self._cond:self._text_queue.append(int(t))

    def _decode_pending(self):
        n=len(self._pending_codes)
        if not n:return
        context=self._output_codes[-self.context_frames:]
        with torch.inference_mode():
            codes=torch.tensor(context,device='cuda').T
            wave=self.codec.decode(codes).detach().float().cpu().numpy()[-n*self.frame_samples:]
        if len(wave)!=n*self.frame_samples:raise RuntimeError('decoded codec frames do not match native 40ms geometry')
        pcm=np.clip(wave*32768,-32768,32767).astype('<i2')
        with self._cond:
            for index,flags in enumerate(self._pending_flags):self._output.append((pcm[index*640:(index+1)*640].tobytes(),flags))
            self._decoded_frames+=n;self._pending_codes=[];self._pending_flags=[];self._cond.notify_all()

    def _on_end(self):
        self._decode_pending()
        with self._cond:self._stream_ended=True;self._cond.notify_all()

    def step_pcm(self,pcm16):
        if len(pcm16)!=self.frame_samples*2:raise ValueError('Voila adapter requires exactly one 40ms PCM16 frame')
        if self._error is not None:raise RuntimeError('real Voila generation failed') from self._error
        if self._stream_ended and not self._closed:raise RuntimeError('Voila generator ended before session finish')
        self.start_stream();self._raw.append(np.frombuffer(pcm16,dtype='<i2').astype(np.float32)/32768);self._received_frames+=1
        if len(self._raw)>self.context_frames:self._raw=self._raw[-self.context_frames:]
        if self._received_frames%self.block_frames==0:
            # Encodec is non-causal: encode only a prefix that has already arrived.
            # Preserve a past context, and emit only the current one-second block.
            wav=np.concatenate(self._raw)
            with torch.inference_mode():codes=self.codec.encode(wav,16000)[:,-self.block_frames:]
            assert codes.shape==(self.num_codebooks,self.block_frames)
            target=self._encoded_frames+self.block_frames
            for i in range(self.block_frames):
                token=codes[:,i]+self.AUDIO_MIN_TOKEN_ID+torch.arange(self.num_codebooks,device='cuda')*self.codebook_size
                self._input.put(token)
            self._encoded_frames=target
            deadline=time.monotonic()+120
            with self._cond:
                while self._decoded_frames<target and self._error is None and not self._stream_ended:
                    remain=deadline-time.monotonic()
                    if remain<=0:raise TimeoutError('Voila generation did not finish the current audio block within 120s')
                    self._cond.wait(timeout=min(1,remain))
            if self._error is not None:raise RuntimeError('real Voila generation failed') from self._error
            if self._decoded_frames<target:raise RuntimeError('Voila stream stopped before completing the admitted block')
            self._audit.append(dict(received_frames=self._received_frames,encoded_frames=self._encoded_frames,consumed_frames=self._consumed_frames,decoded_frames=self._decoded_frames,model_steps=self._model_steps,queued_audio_frames=len(self._output)))
        with self._cond:
            text_tokens=self._text_queue;self._text_queue=[]
        self._last_text=self.tokenizer.decode(text_tokens) if text_tokens else ''
        if self._output:out,flags=self._output.popleft()
        else:out=b'\x00\x00'*640;flags=dict(is_listen=True,end_of_turn=False)
        # Silence below threshold describes native state; it never replaces generated audio.
        samples=np.frombuffer(out,dtype='<i2').astype(np.float32)
        self._silent=self._silent+1 if float(np.sqrt(np.mean(samples**2)))<80 else 0
        if self._silent>=50:flags.update(is_listen=True,end_of_turn=True)
        return out,dict(flags,status='real_model_stream',model_steps=self._model_steps,input_buffering_ms=1000)

    def finish(self):
        self._closed=True;self._input.put(None)
        if self._gen_thread:self._gen_thread.join(timeout=15)
        path=self.root/'last_stream_audit.json'
        path.write_text(json.dumps(dict(received_frames=self._received_frames,encoded_frames=self._encoded_frames,decoded_frames=self._decoded_frames,model_steps=self._model_steps,input_buffer_ms=1000,codec_past_context_ms=3000,future_audio_access=False,blocks=self._audit),indent=2)+'\n')
        if self._gen_thread and self._gen_thread.is_alive():raise RuntimeError('Voila generation thread did not exit after finish')
