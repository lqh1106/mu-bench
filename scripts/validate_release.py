"""CPU-only validation of the public datasets and runtime contracts."""
from pathlib import Path
import argparse,json,sys,hashlib,subprocess,os,wave
ROOT=Path(__file__).resolve().parents[1]
def check(subset):
    sys.path.insert(0,str(ROOT))
    from mubench.config import activate,profile
    activate(profile('native',subset))
    from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
    from experiments.tau_multiuser_voice.fact_contract import input_contract
    from experiments.tau_multiuser_voice.fact_contract import review_for_case
    from experiments.tau_multiuser_voice.task_evidence import build_task_evidence_contract
    from experiments.tau_multiuser_voice.participant_context import table
    paths=sorted((ROOT/'cases'/subset).glob('*.json'));expected=392 if subset=='tool_free' else 408
    assert len(paths)==expected,(subset,len(paths))
    scenes=set();ids=set()
    for p in paths:
        c=MultiUserVoiceCase.model_validate(json.loads(p.read_text()));assert c.case_id==p.stem and c.case_id not in ids;ids.add(c.case_id)
        assert bool(c.environment.tool_names)==(subset=='tool_augmented')
        ctx=c.task_contract_revision['review'].get('initial_assistant_context',{})
        if ctx:assert ctx['contains_future_event_oracle'] is False
        contract=build_task_evidence_contract(c,review_for_case(c));assert contract['enabled'],p.name
        reg=table(c);assert len(reg['participants'])==len(c.users)
        assert len({u['model_visible_id'] for u in reg['participants']})==len(c.users)
        scenes.add(c.variant.scenario_id)
        if subset=='tool_augmented':
            from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
            from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas
            from experiments.tau_multiuser_voice.caller_protocol import enabled
            assert enabled(c)
            env=create_virtual_environment(c.environment);configure_schemas(env,c.environment.tool_names)
            for name in c.environment.tool_names:
                schema=env.definitions[name].parameters
                assert 'caller_id' in schema['required'],(p.name,name)
                assert 'not a display name' in schema['properties']['caller_id']['description']
    return dict(subset=subset,cases=len(paths),scenes=len(scenes),schema=True,typed_evidence=True,participant_namespaces=True)
def assets():
    sys.path.insert(0,str(ROOT))
    from mubench.config import activate,profile
    activate(profile('native','tool_free'))
    from experiments.tau_multiuser_voice.voice_catalog import VoiceCatalogManifest,GlobalVoiceBindingManifest
    p=ROOT/'assets/voices/catalog/voice_catalog.json';v=VoiceCatalogManifest.load(p);b=GlobalVoiceBindingManifest.load(ROOT/'assets/voices/bindings.json')
    assert len(v.voices)==3200 and len(b.bindings)==3200
    for row in v.voices:
        clip=(p.parent/row.reference_audio_path).resolve();assert clip.is_relative_to(ROOT/'assets/voices')
        assert hashlib.sha256(clip.read_bytes()).hexdigest()==row.source_sha256,clip.name
        with wave.open(str(clip),'rb') as w:assert w.getnchannels()==1 and w.getnframes()>0
    return dict(voices=3200,bindings=3200,all_audio_hashes=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subset',choices=('tool_free','tool_augmented'));p.add_argument('--assets',action='store_true');a=p.parse_args()
    if a.subset:print(json.dumps(check(a.subset)));raise SystemExit(0)
    results=[]
    for subset in ('tool_free','tool_augmented'):
        result=subprocess.run([sys.executable,__file__,'--subset',subset],capture_output=True,text=True,check=True);results.append(json.loads(result.stdout))
    if a.assets:results.append(assets())
    print(json.dumps({'passed':True,'results':results},indent=2))
