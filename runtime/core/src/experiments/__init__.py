from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))
"""Isolated benchmark experiment package."""
