from pathlib import Path
from types import SimpleNamespace
import importlib.util,os,time
SOURCE=Path(__file__).with_name('base.py')
spec=importlib.util.spec_from_file_location('_tool_free_voila_pinned',SOURCE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
class VoilaBackend(base.VoilaBackend):
 def __init__(self,args):
  values=dict(vars(args));root=Path(values['debug_root'])/('session_'+str(os.getpid())+'_'+str(time.time_ns()));root.mkdir(parents=True)
  asset=Path(values['codec_path']).resolve()
  if not asset.is_dir():raise FileNotFoundError('Voila codec_path is missing')
  (root/'Voila-Tokenizer').symlink_to(asset,target_is_directory=True)
  values['debug_root']=str(root);super().__init__(SimpleNamespace(**values))
