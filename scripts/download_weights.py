"""Download external model assets; checkpoint licenses remain upstream-owned."""
from pathlib import Path
import argparse,json,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
SPECS={
 'minicpm':'OpenBMB/MiniCPM-o-4_5','personaplex':'nv-community/personaplex-7b-v1',
 'moshi':'kyutai/moshika-pytorch-bf16','raon':'KRAFTON/Raon-SpeechChat-9B',
 'lychee':'HIT-TMG/Lychee-FD','freeze_omni':'VITA-MLLM/Freeze-Omni',
 'flm_audio':'CofeAI/FLM-Audio','voila':'maitrix-org/Voila-autonomous-preview',
 'tts':'FunAudioLLM/Fun-CosyVoice3-0.5B-2512','asr':'Systran/faster-whisper-large-v3',
 'voila_tokenizer':'maitrix-org/Voila-Tokenizer','qwen2':'Qwen/Qwen2-7B-Instruct','token2wav':'stepfun-ai/Step-Audio-2-mini'}
p=argparse.ArgumentParser();p.add_argument('--model',choices=SPECS,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--revision');a=p.parse_args()
from huggingface_hub import snapshot_download
revision=a.revision or ('a49141e28b3d9c947cf9aa5314431e1b11cbd2f5' if a.model=='moshi' else 'main')
path=Path(snapshot_download(SPECS[a.model],revision=revision,local_dir=str(a.destination)))
# Preserve runtime Python compatibility code from the tested integration.
overlay=ROOT/'adapters/native'/a.model/'checkpoint_code'
if overlay.is_dir():
 for f in overlay.glob('*.py'):shutil.copyfile(f,path/f.name)
print(json.dumps({'model':SPECS[a.model],'requested_revision':revision,'path':str(path),'compatibility_code_applied':overlay.is_dir()}))
