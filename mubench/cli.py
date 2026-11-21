"""Portable CLI. Every episode runs in an isolated interpreter/profile."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,time,fcntl,hashlib
from .config import ROOT,MODELS,ARMS,load,profile,environment

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def selected(subset,ids):
    cases=sorted((ROOT/'cases'/subset).glob('*.json'))
    if ids:cases=[p for p in cases if p.stem in ids or p.stem.split('__')[0] in ids]
    if not cases:raise ValueError('No matching cases')
    return cases

def run_one(cfg,subset,case,arm,slot):
    sut=cfg['sut'];provider=sut['provider'];name=sut['model'];transport=sut.get('transport','audio')
    rt=profile(provider,subset,arm,transport);env=environment(rt,subset,arm,provider)
    runroot=Path(cfg.get('output_root','outputs')).expanduser().resolve()/name/subset/(arm or 'baseline')
    lockfile=runroot/'claims'/(case.stem+'.lock');lockfile.parent.mkdir(parents=True,exist_ok=True)
    with lockfile.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'case_id':case.stem,'status':'claimed_elsewhere'}
        published=runroot/'formal'/case.stem
        if (published/'release_receipt.json').is_file():return {'case_id':case.stem,'status':'already_complete'}
        attempts=runroot/'runs'/case.stem;attempts.mkdir(parents=True,exist_ok=True)
        number=1+max([int(p.name.split('_')[-1]) for p in attempts.glob('attempt_*') if p.name.split('_')[-1].isdigit()]+[0])
        attempt=attempts/f'attempt_{number:03d}';attempt.mkdir()
        text=cfg['text'];speech=cfg.get('speech',{});env['MUBENCH_TEXT_MODEL']=text.get('model','gpt-6-luna')
        env['MUBENCH_TEXT_BASE_URL']=text.get('direct_base_url','')
        values={'GPT_URL':text['base_url'],'GPT_KEY':text['api_key']}
        for p in ('GPT','GROK','GEMINI','SEEDUPLEX'):
            values['REALTIME_'+p+'_URL']=sut.get('url','')
            values['REALTIME_'+p+'_KEY']=sut.get('api_key','')
        envfile=attempt/'private.env';envfile.write_text('\n'.join(k+'='+str(v) for k,v in values.items())+'\n');envfile.chmod(0o600)
        env['MUBENCH_CONFIG_FILE']=str(attempt/'private_text.json')
        write(env['MUBENCH_CONFIG_FILE'],dict(base_url=text.get('direct_base_url') or text['base_url'],api_key=text['api_key'],sut_model=name,simulator_model=text.get('model','gpt-6-luna'),judge_model=text.get('model','gpt-6-luna'),timeout_seconds=cfg.get('api_timeout_seconds',720),https_proxy=text.get('https_proxy')))
        Path(env['MUBENCH_CONFIG_FILE']).chmod(0o600)
        if transport=='text':
            if subset!='tool_augmented':raise ValueError('Text track is the tool-augmented diagnostic track')
            result=attempt/'evidence'
            args=[sys.executable,str(rt/'run_text_case.py'),'--case-id',case.stem,'--output',str(result)]
        else:
            result=attempt/'output'/case.stem
            entry='run_audio_ablation.py' if arm else 'run_mu_logical_voice.py'
            # MOSS ablation has its own transport entry, preserving acknowledged source records.
            if arm and provider not in ('moss','native'):raise ValueError('This backend lacks an ablation auxiliary-context transport')
            args=[sys.executable,str(rt/entry),'--env',str(envfile),'--bridge',str(ROOT/'runtime/core/run_tau_voice_realtime_debug_20260909.py'),'--tau2-src',str(ROOT/'third_party/tau2/src'),'--output-dir',str(attempt/'output'),'--case-id',case.stem,'--realtime-model',name,'--planner-model',text.get('model','gpt-6-luna'),'--tts-backend','cosyvoice_http','--tts-endpoint-file',speech['tts_endpoint'],'--tts-cache-dir',str(attempt/'tts_cache'),'--tts-run-namespace','mu-'+hashlib.sha256(str(attempt).encode()).hexdigest()[:18],'--enrollment-transport','audio_self_identification','--planner-observation','audible_asr','--logical-max-seconds','1800','--bilateral-silence-seconds','600','--final-drain-seconds','30']
            if arm:args+=['--ablation-arm',arm]
            if provider=='moss' and subset=='tool_augmented':
                endpoints=sut.get('endpoints') or [sut['endpoint_file']]
                args+=['--realtime-provider','moss','--moss-endpoint',endpoints[slot%len(endpoints)]]
            elif provider in ('native','moss'):
                endpoints=sut.get('endpoints') or [sut['endpoint_file']]
                args+=['--openweight-endpoint',endpoints[slot%len(endpoints)]]
            else:args+=['--realtime-provider',{'openai':'openai','xai':'xai','gemini':'gemini','seeduplex':'seeduplex'}[provider]]
        if provider=='seeduplex' and transport=='audio' and sut.get('voice'):
            args+=['--seeduplex-voice',sut['voice']]
        started=time.time();status='technical_error'
        try:
            with (attempt/'case.log').open('ab') as log:
                p=subprocess.run(args,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=cfg.get('case_timeout_seconds',6000))
            if not (result/'score.json').is_file():raise RuntimeError('No score: retained evidence can be inspected or evaluated offline')
            # Shared recognition/floor split and RICE; no synthetic audio or substituted transcript.
            final=[sys.executable,str(ROOT/'mubench/finalize.py'),'--case-dir',str(result),'--runtime',str(rt),'--env',str(envfile),'--model',name,'--formal',str(published),'--transport',transport]
            if cfg.get('auto_cleanup',True):final+=['--cleanup']
            subprocess.run(final,env=env,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            status='complete'
            return {'case_id':case.stem,'status':status,'task_success':json.loads((published/'score.json').read_text())['task_success']}
        except Exception as exc:
            write(attempt/'error.json',{'type':type(exc).__name__,'message':str(exc),'model_failure':False})
            return {'case_id':case.stem,'status':status,'error_type':type(exc).__name__}
        finally:
            write(attempt/'run_receipt.json',{'case_id':case.stem,'subset':subset,'transport':transport,'model':name,'profile':rt.relative_to(ROOT).as_posix(),'status':status,'started':started,'ended':time.time(),'evidence_dir':str(result),'config_secrets_removed':True})
            envfile.unlink(missing_ok=True);Path(env['MUBENCH_CONFIG_FILE']).unlink(missing_ok=True)

def run(args):
    cfg=load(args.config);cases=selected(args.subset,args.case);sut=cfg['sut']
    if args.workers<1:raise ValueError('workers must be positive')
    if sut['provider']=='moss' and args.workers>len(sut.get('endpoints') or [sut['endpoint_file']]):raise ValueError('MOSS requires one independent key endpoint per lane')
    # Each lane holds its own endpoint/key; it never changes credentials inside a case.
    buckets=[cases[i::args.workers] for i in range(args.workers)]
    def lane(index):
        results=[]
        for case in buckets[index]:
            result=run_one(cfg,args.subset,case,args.ablation,index);print(json.dumps(result),flush=True);results.append(result)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=[r for chunk in pool.map(lane,range(args.workers)) for r in chunk]
    write(Path(cfg.get('output_root','outputs'))/'latest_run_summary.json',rows)
    return int(any(r['status']=='technical_error' for r in rows))

def main():
    p=argparse.ArgumentParser(prog='mubench');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('run');q.add_argument('--config',required=True);q.add_argument('--subset',choices=('tool_free','tool_augmented'),required=True);q.add_argument('--case',action='append');q.add_argument('--workers',type=int,default=1);q.add_argument('--ablation',choices=ARMS)
    q=sub.add_parser('validate');q.add_argument('--assets',action='store_true')
    q=sub.add_parser('model-config');q.add_argument('--model',choices=MODELS,required=True);q.add_argument('--paths',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--config',required=True);q.add_argument('--case-dir',required=True);q.add_argument('--subset',choices=('tool_free','tool_augmented'),required=True);q.add_argument('--provider',default='native');q.add_argument('--transport',choices=('audio','text'),default='audio');q.add_argument('--ablation',choices=ARMS)
    a=p.parse_args()
    if a.command=='run':return run(a)
    if a.command=='validate':
        subprocess.run([sys.executable,str(ROOT/'scripts/validate_release.py')]+(['--assets'] if a.assets else []),check=True);return 0
    if a.command=='model-config':
        from .model_config import generate
        generate(a.model,load(a.paths),Path(a.output));return 0
    if a.command=='evaluate':
        rt=profile(a.provider,a.subset,a.ablation,a.transport);env=environment(rt,a.subset,a.ablation,a.provider)
        subprocess.run([sys.executable,str(ROOT/'mubench/evaluate.py'),'--config',a.config,'--case-dir',a.case_dir,'--runtime',str(rt),'--transport',a.transport],env=env,cwd=ROOT,check=True);return 0
if __name__=='__main__':raise SystemExit(main())
