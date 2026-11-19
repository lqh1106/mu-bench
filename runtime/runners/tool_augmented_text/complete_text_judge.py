import bootstrap
from pathlib import Path
import sys,json,argparse,os,shutil,traceback,time
HERE=Path(__file__).resolve().parent

os.environ['MUBENCH_JUDGE_PROTOCOL']='tool_full_source_review'
from adapters.providers.chat.adapter import API
from score_text_evidence import score_evidence,write
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace
def main(path, completion_name="judge_completion_001"):
 if (path/'score.json').exists():return {'already_scored':True}
 assert json.loads((path/'phase.json').read_text())['phase']=='judge'
 assert completion_name.startswith('judge_completion_') and completion_name[len('judge_completion_'):].isdigit()
 out=path/completion_name;out.mkdir(exist_ok=False)
 for name in ('reviewed_case.json','initial_assistant_context.json','voice_enrollment.before_judge.json','registration_text.json','text_transport_marker.json','caller_provenance_audit.json','business_request_basis_audit.json','history.json','simulator_exhausted_forced_eval.json'):
  if (path/name).exists():shutil.copyfile(path/name,out/name)
 case=MultiUserVoiceCase.model_validate(json.loads((path/'reviewed_case.json').read_text()))
 trace=MultiUserEpisodeTrace.model_validate(json.loads((path/'final_trace.before_judge.json').read_text()))
 turns=json.loads((path/'assistant_turns.before_judge.json').read_text());logs=json.loads((path/'tool_log.before_judge.json').read_text())
 usage=[];api=API(json.loads(Path(os.environ['MUBENCH_CONFIG_FILE']).read_text()),usage)
 try:
  score=score_evidence(api,case,trace,turns,logs,out,(path,*sorted(path.glob('judge_completion_[0-9]*'))))
  score['evidence_only_completion']=True;score['judge_completion_path']=str(out)
  write(path/'score.json',score);write(path/'phase.json',{'phase':'complete','at':time.time(),'evidence_only_completion':True})
  return {'task_success':score['task_success'],'scored':True,'sut_replayed':False}
 except Exception as e:
  write(out/'error.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'not_model_failure':True});raise
 finally:write(out/'api_usage.json',usage)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--completion-id',default='judge_completion_001');a=ap.parse_args();print(json.dumps(main(Path(a.output),a.completion_id)))
