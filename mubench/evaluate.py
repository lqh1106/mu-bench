"""Resume judging from preserved complete evidence; never replay the tested model."""
from pathlib import Path
import argparse,json,sys,os,time
from mubench.config import load,ROOT
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--transport',required=True);a=p.parse_args()
from mubench.config import activate
activate(a.runtime)
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from run_tau_mu_voice_realtime import run_semantic_judge,measure_floor_split_timings,load_bridge
case=MultiUserVoiceCase.model_validate(json.loads((a.case_dir/'reviewed_case.json').read_text()))
trace=MultiUserEpisodeTrace.model_validate(json.loads((a.case_dir/'final_trace.before_judge.json').read_text()))
turns=json.loads((a.case_dir/'assistant_turns.before_judge.json').read_text());logs=json.loads((a.case_dir/'tool_log.before_judge.json').read_text())
text=load(a.config)['text'];bridge=load_bridge(ROOT/'runtime/core/run_tau_voice_realtime_debug_20260909.py')
out=a.case_dir/('judge_resume_'+str(time.time_ns()));out.mkdir()
# The source guard also reads actual sent enrollment and auxiliary audit files.
import shutil
for n in ('reviewed_case.json','initial_assistant_context.json','voice_enrollment.before_judge.json','voice_enrollment.json','adapter_audit.json','ablation_aux_delivery.json','ablation_manifest.json','ticks.json','final_trace.before_judge.json'):
 if (a.case_dir/n).is_file():shutil.copyfile(a.case_dir/n,out/n)
values={'GPT_URL':text['base_url'],'GPT_KEY':text['api_key']}
caps,goals,leaks=run_semantic_judge(bridge,values,case,trace,turns,logs,audit_dir=out,resume_dirs=(a.case_dir,))
trace.capability_results=caps;trace.goal_results=goals;trace.privacy_leaks=leaks;measure_floor_split_timings(case,trace,turns)
score=MultiUserEvaluator().evaluate(case,trace).model_dump(mode='json')
# Shared evaluator already applies the public authorization-separated contract.
score['judge_protocol']=os.environ.get('MUBENCH_TOOL_JUDGE_VERSION',os.environ['MUBENCH_JUDGE_PROTOCOL']);score['evidence_only_resume']=True
(out/'score.json').write_text(json.dumps(score,ensure_ascii=False,indent=2)+'\n');(out/'trace.after_judge.json').write_text(trace.model_dump_json(indent=2)+'\n')
from rice_compute import compute_rice
from mubench.metrics import write_atc
rice=compute_rice(score,score)
rice.update(measurement_track='audio-logical' if a.transport=='audio' else 'text-logical',
            voice_comparable=a.transport=='audio',source_score='score.json')
(out/'rice_score.json').write_text(json.dumps(rice,indent=2)+'\n')
write_atc(out,rice)
print(json.dumps({'score':str(out/'score.json'),'atc':str(out/'atc_score.json'),'rice':str(out/'rice_score.json'),'sut_replayed':False,'task_success':score['task_success']}))
