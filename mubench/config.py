from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
MODELS=('minicpm','moshi','freeze_omni','flm_audio','voila','raon','personaplex','lychee')
ARMS=('serial','serial_identity','serial_identity_semantics')
def load(path):
    cfg=json.loads(Path(path).read_text())
    def resolve(x):
        if isinstance(x,dict):
            if set(x)=={'env'}:
                value=os.environ.get(x['env'])
                if not value:raise ValueError('Missing environment variable: '+x['env'])
                return value
            return {k:resolve(v) for k,v in x.items()}
        if isinstance(x,list):return [resolve(v) for v in x]
        if isinstance(x,str):return os.path.expandvars(x).replace('${REPO_ROOT}',str(ROOT))
        return x
    return resolve(cfg)
def profile(provider,subset,arm=None,transport='audio'):
    if subset=='tool_augmented' and arm:raise ValueError('Ablations apply to tool-free cases')
    if arm:return ROOT/'runtime/runners/ablations'
    return ROOT/'runtime/runners'/('tool_free' if subset=='tool_free' else 'tool_augmented_'+transport)

def source_paths(rt):
    name=Path(rt).name
    tool=name.startswith('tool_augmented')
    text=name.endswith('_text')
    ablation=name=='ablations'
    paths=[ROOT/'compat',Path(rt)]
    if text:paths += [ROOT/'runtime/protocols/text/src',ROOT/'judges/text/src']
    if ablation:paths += [ROOT/'runtime/protocols/ablations/src',ROOT/'judges/ablations/src']
    if tool:paths += [ROOT/'runtime/protocols/tool_augmented/src',ROOT/'judges/tool_augmented/src']
    paths += [ROOT/'judges/common/src',ROOT/'runtime/core/src',
              ROOT/'judges/tool_augmented'/('text' if text else 'audio') if tool else ROOT/'judges/tool_free',
              ROOT/'judges/common',ROOT/'runtime/core',ROOT/'runtime/ablations',
              ROOT/'third_party/tau2/src',ROOT]
    return [p for p in paths if p.is_dir()]

def environment(rt,subset,arm=None,provider=None):
    name=Path(rt).name
    provider=provider or os.environ.get('MUBENCH_PROVIDER','native')
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
      MUBENCH_CASE_DIR=str(ROOT/'cases'/subset),MUBENCH_RUNTIME_ROOT=str(rt),
      MUBENCH_PROVIDER=provider,MUBENCH_TRANSPORT='text' if name.endswith('_text') else 'audio',
      TAU2_SRC=str(ROOT/'third_party/tau2/src'),
      MUBENCH_JUDGE_PROTOCOL='tool_full_source_review' if subset=='tool_augmented' else 'full_source_review',
      MUBENCH_ATTEMPT_SEAL_CLEANUP='0',MUBENCH_FORMAL_POST_SCORE='0',MUBENCH_ATTEMPT_LOCAL_RICE='0')
    if subset=='tool_augmented':
        env.update(MUBENCH_TOOL_JUDGE_VERSION='jt5',MUBENCH_TOOL_CASE_MECHANISM='tr3')
    else:
        env.pop('MUBENCH_TOOL_JUDGE_VERSION',None)
        env.pop('MUBENCH_TOOL_CASE_MECHANISM',None)
    if arm:
        env['MUBENCH_ABLATION_ARM']=arm
        env['MUBENCH_JUDGE_PROTOCOL']='ablation_'+arm+'_review'
        env['MUBENCH_ABLATION_CASE_DIR']=str(ROOT/'cases/ablations')
        env['MUBENCH_RUNTIME_MANIFEST']=str(ROOT/'runtime/fingerprints'/('moss.json' if provider=='moss' else 'minicpm.json'))
    env['PYTHONPATH']=os.pathsep.join(map(str,source_paths(rt)))
    return env

def activate(rt):
    import sys
    rt=Path(rt)
    subset='tool_augmented' if rt.name.startswith('tool_augmented') else 'tool_free'
    arm=os.environ.get('MUBENCH_ABLATION_ARM') if rt.name=='ablations' else None
    env=environment(rt,subset,arm)
    os.environ.update(env)
    for p in reversed(source_paths(rt)):
        value=str(p)
        if value in sys.path:sys.path.remove(value)
        sys.path.insert(0,value)
