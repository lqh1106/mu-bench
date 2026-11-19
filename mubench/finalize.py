"""Seal one completed episode without changing raw semantic scores."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,wave

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def finalize(case,rt,envfile,model,formal,transport,cleanup=False):
    from mubench.config import activate
    activate(rt)
    score=json.loads((case/'score.json').read_text())
    if not isinstance(score.get('task_success'),bool):raise ValueError('No task verdict')
    if transport=='audio':
        with wave.open(str(case/'audio/both.wav'),'rb') as w:
            if w.getnchannels()!=2 or w.getnframes()==0:raise ValueError('Final stereo waveform is incomplete')
    # The complete current judge/evaluator already contains recognition_split,
    # fairness and quiet-input units; reuse it without a second semantic judgment.
    from rice_compute import compute_rice
    rice=compute_rice(score,score)
    rice.update(measurement_track='audio-logical' if transport=='audio' else 'text-logical',
                voice_comparable=transport=='audio',source_score='score.json')
    (case/'rice_score.json').write_text(json.dumps(rice,indent=2)+'\n')
    rice=json.loads((case/'rice_score.json').read_text())
    if any(rice.get(k) is None for k in ['routing_score','continuity_score','isolation_score']):raise ValueError('Required RICE axis missing')
    from mubench.metrics import write_atc
    atc=write_atc(case,rice)
    if any(atc[k] is None for k in ('attribution_score','threading_score','coordination_score')):raise ValueError('Required ATC axis missing')
    if formal.exists():raise FileExistsError('Never overwrite an existing published result')
    tmp=formal.with_name('.'+formal.name+'.publishing');tmp.parent.mkdir(parents=True,exist_ok=True)
    def link_or_copy(src,dst):
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
        return dst
    shutil.copytree(case,tmp,copy_function=link_or_copy,ignore=shutil.ignore_patterns('private.env','private_text.json','*.lock','_floor_scratch'))
    # Hardlinks preserve one physical copy on one filesystem. All retained provenance remains accessible.
    receipt={'complete':True,'transport':transport,'task_success':score['task_success'],'source':str(case),'score_sha256':digest(tmp/'score.json'),'rice_sha256':digest(tmp/'rice_score.json'),'atc_sha256':digest(tmp/'atc_score.json'),'audio_sha256':digest(tmp/'audio/both.wav') if transport=='audio' else None}
    (tmp/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');os.replace(tmp,formal)
    if cleanup and transport=='audio':
        from formal_post_score_rice import cleanup_attempt_seal_audio
        # Existing whitelisted cleanup removes played fragments/mono stems/oracle stems,
        # only after full score/RICE/stereo evidence is present. JSON audits survive.
        result=cleanup_attempt_seal_audio(case,require_rice=True,delete_oracle=True)
        (case/'cleanup_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        cleanup_attempt_seal_audio(formal,require_rice=True,delete_oracle=True)
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--env',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--transport',choices=('audio','text'),required=True);p.add_argument('--cleanup',action='store_true');a=p.parse_args();print(json.dumps(finalize(a.case_dir,a.runtime,a.env,a.model,a.formal,a.transport,a.cleanup)))
