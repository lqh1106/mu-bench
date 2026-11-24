"""Real played-audio evidence alignment; never generate SUT dialogue."""
from pathlib import Path
from copy import deepcopy
import json,hashlib,wave,re
from .evaluation_policy import TemporalAlignmentPending
VERSION='tool-contract-boundaries-r2-short-asr'
SHORT_ALIGNMENT_LIMIT_MS=1000.0
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def segments(turns,tool_log):
    edges=sorted({float(r['result_available_ms']) for r in tool_log if r.get('executed') and r.get('result_available_ms') is not None})
    return [(t,sorted({float(t['start_ms']),float(t['end_ms']),*[e for e in edges if t['start_ms']<e<t['end_ms']]})) for t in turns if t.get('asr_derived') and isinstance(t.get('start_ms'),(int,float)) and isinstance(t.get('end_ms'),(int,float))]
def short_unintelligible_result(exc,clip,part,duration_ms,previous_audits):
    """Accept a documented subsecond unknown, without inventing transcript text."""
    from experiments.mu_bench.logical_time.audible_asr_v3 import AudibleASRError
    if not (isinstance(exc,AudibleASRError) and str(exc)=='asr_all_segments_unintelligible'
            and 0<duration_ms<SHORT_ALIGNMENT_LIMIT_MS):return None
    audits=set((part/'audible_asr_v3').glob('invocation_*/audit.json'))-previous_audits
    for path in sorted(audits,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        audit=json.loads(path.read_text());groups=audit.get('segments',[])
        if (audit.get('error')!='asr_all_segments_unintelligible'
                or audit.get('source_wav_sha256')!=hashlib.sha256(clip.read_bytes()).hexdigest()
                or not groups or any(g.get('status')!='unintelligible' for g in groups)):continue
        return {'assistant_text':'','assistant_fragments':[],'asr_derived':True,
                'asr_status':'unintelligible','asr_groups':groups,'asr_audit_path':str(path),
                'subsecond_unintelligible_accepted':True,'clip_duration_ms':duration_ms,
                'acceptance_policy':'temporal-subsecond-unintelligible-v1',
                'transcript_content_unknown':True}
    return None
def prepare_turns(bridge,values,trace,turns,tool_log,audit_dir):
    plans={t['source_id']:cuts for t,cuts in segments(turns,tool_log) if len(cuts)>2}
    if not plans:return turns
    if audit_dir is None:raise TemporalAlignmentPending('temporal_alignment_required: no audit destination')
    root=Path(audit_dir);path=next((r/'audio/both.wav' for r in [root,*list(root.parents)[:2]] if (r/'audio/both.wav').is_file()),None)
    if path is None:
        write(root/'temporal_review_pending.json',{'status':'needs_audio_alignment','model_failure':False,'source_ids':list(plans),'reason':'actual played waveform missing'})
        raise TemporalAlignmentPending('temporal_alignment_required: actual played waveform missing')
    from experiments.mu_bench.logical_time.audible_asr_v3 import transcribe_response
    import audioop
    dest=root/'temporal_alignment';dest.mkdir(exist_ok=True)
    results=[];audits=[];before=hashlib.sha256(path.read_bytes()).hexdigest()
    with wave.open(str(path),'rb') as src:
        if src.getnchannels()!=2 or src.getsampwidth()!=2:raise TemporalAlignmentPending('temporal_alignment_required: expected actual stereo PCM16')
        rate=src.getframerate();start=float(trace.task_start_ms)
        for t in turns:
            cuts=plans.get(t.get('source_id'))
            if not cuts:results.append(deepcopy(t));continue
            for j,(a,b) in enumerate(zip(cuts,cuts[1:])):
                key=hashlib.sha256((t['source_id']+repr((a,b))+before).encode()).hexdigest()[:24];part=dest/key;part.mkdir(exist_ok=True);clip=part/'played_assistant.wav'
                aa=round((start+a)*rate/1000);bb=round((start+b)*rate/1000)
                if aa<0 or bb>src.getnframes():raise TemporalAlignmentPending('temporal_alignment_required: crop outside waveform')
                src.setpos(aa);pcm=audioop.tomono(src.readframes(bb-aa),2,0,1)
                with wave.open(str(clip),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
                duration_ms=(bb-aa)*1000/rate
                previous_audits=set((part/'audible_asr_v3').glob('invocation_*/audit.json'))
                try:
                    recognized=transcribe_response({'assistant_wav':str(clip),'assistant_intervals':[{'start_ms':0,'end_ms':(bb-aa)*1000/rate}]},0,part,bridge,values)
                except Exception as exc:
                    recognized=short_unintelligible_result(exc,clip,part,duration_ms,previous_audits)
                    if recognized is None:
                        write(root/'temporal_review_pending.json',{'status':'needs_audio_alignment','model_failure':False,'source_id':t['source_id'],'reason':'ASR evidence preparation failed','exception_type':type(exc).__name__})
                        raise TemporalAlignmentPending('temporal_alignment_required: ASR unavailable; retained actual clip') from exc
                write(part/'asr_result.json',recognized);record={'original_source_id':t['source_id'],'source_id':'assistant_aligned_'+key,'start_ms':a,'end_ms':b,'clip':str(clip),'audio_sha256':hashlib.sha256(clip.read_bytes()).hexdigest(),'text':recognized['assistant_text']};audits.append(record)
                accepted_unknown=recognized.get('subsecond_unintelligible_accepted',False)
                record.update(asr_status=recognized.get('asr_status','transcribed' if recognized['assistant_text'] else 'no_speech'),subsecond_unintelligible_accepted=accepted_unknown,clip_duration_ms=duration_ms)
                if recognized['assistant_text'] or accepted_unknown:
                    item=deepcopy(t);item.update(source_id=record['source_id'],original_source_id=t['source_id'],start_ms=a,end_ms=b,text=record['text'],audio_sha256=record['audio_sha256'],source_context_start_ms=start+a,source_context_end_ms=start+b,word_timestamps_available=False,asr_alignment_source=str(clip),asr_status=record['asr_status'],transcript_content_unknown=accepted_unknown);results.append(item)
    write(root/'evaluation_assistant_turns.json',results);write(root/'temporal_alignment_receipt.json',{'version':VERSION,'source_waveform_sha256':before,'assistant_channel':1,'original_turns_preserved':True,'no_generated_text_used':True,'word_timestamps_invented':False,'segments':audits})
    return results

def guard_unresolved_verdicts(payload,path):
    """Do not convert explicitly unresolved ordering into a binary model failure."""
    pending=[]
    def visit(x):
        if isinstance(x,dict):
            reason=str(x.get('reason',''))
            if x.get('passed') is False and re.search(r'(timing|temporal|word.level|before.{0,25}claim|preced.{0,25}(claim|speech))',reason,re.I) and re.search(r'(not (established|clear)|cannot (establish|determine)|unresolved|ambiguous|no word.level|uncertain|does not establish)',reason,re.I):pending.append({'criterion':x.get('criterion',x.get('assertion_id')),'reason':reason})
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(payload)
    if pending:
        if path is not None:write(Path(path)/'temporal_review_pending.json',{'status':'needs_audio_alignment','model_failure':False,'items':pending})
        raise TemporalAlignmentPending('temporal_alignment_required: final verdict contains unresolved ordering, not a demonstrated model error')
