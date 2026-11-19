from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mubench.config import activate
activate(Path(__file__).resolve().parent)
