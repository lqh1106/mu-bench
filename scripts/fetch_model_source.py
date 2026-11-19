"""Fetch upstream source with a recorded commit; leaves benchmark sources intact."""
from pathlib import Path
import argparse,subprocess,json
URLS={'minicpm':'https://github.com/OpenBMB/MiniCPM-o.git','personaplex':'https://github.com/NVIDIA/personaplex.git','moshi':'https://github.com/kyutai-labs/moshi.git','raon':'https://github.com/krafton-ai/Raon-Speech.git','lychee':'https://github.com/HITsz-TMG/Lychee-FD.git','freeze_omni':'https://github.com/VITA-MLLM/Freeze-Omni.git','flm_audio':'https://github.com/cofe-ai/flm-audio.git','voila':'https://github.com/maitrix-org/Voila.git','tts':'https://github.com/FunAudioLLM/CosyVoice.git'}
p=argparse.ArgumentParser();p.add_argument('--model',choices=URLS,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--revision');a=p.parse_args()
if a.destination.exists():raise SystemExit('Choose a fresh destination; existing source is never overwritten')
subprocess.run(['git','clone','--recursive',URLS[a.model],str(a.destination)],check=True)
if a.revision:subprocess.run(['git','-C',str(a.destination),'checkout',a.revision],check=True)
subprocess.run(['git','-C',str(a.destination),'submodule','update','--init','--recursive'],check=True)
commit=subprocess.check_output(['git','-C',str(a.destination),'rev-parse','HEAD'],text=True).strip()
receipt={'source_url':URLS[a.model],'commit':commit,'model':a.model,'path':str(a.destination.resolve())};(a.destination/'mubench_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
