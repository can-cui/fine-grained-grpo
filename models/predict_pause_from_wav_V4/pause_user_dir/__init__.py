"""Explicitly register the Fairseq dataset, criterion, model, and task from V4 source files."""

import importlib.util
import sys
from pathlib import Path

import fairseq.data
import fairseq.optim


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FAIRSEQ_SOURCE_ROOT = PACKAGE_ROOT / "iflytek-tts-exp_fbsong" / "fairseq"


def _load_source_module(module_name, relative_path, parent_package=None, attribute=None):
    """Load a module by file path without relying on the current Fairseq package __path__."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    source_path = FAIRSEQ_SOURCE_ROOT / relative_path
    if not source_path.is_file():
        raise FileNotFoundError("V4 custom Fairseq module does not exist: {}".format(source_path))
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not create a loader for the V4 module: {}".format(source_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if parent_package is not None and attribute is not None:
        setattr(parent_package, attribute, module)
    return module


# The task imports fairseq.data.pause_lmdb_dataset, so the V4 dataset must first be
# injected into the fairseq.data namespace loaded by the current training environment.
_load_source_module(
    "fairseq.data.pause_lmdb_dataset",
    "data/pause_lmdb_dataset.py",
    parent_package=fairseq.data,
    attribute="pause_lmdb_dataset",
)

# The remaining modules only need their source registration decorators to run. Use
# pause_user_dir submodule names so the installed Fairseq package in the company
# environment does not need to be modified.
_load_source_module(
    "pause_user_dir.pause_bce_loss",
    "criterions/pause_bce_loss.py",
)
_load_source_module(
    "pause_user_dir.pause_from_wav",
    "models/pause_from_wav.py",
)
_load_source_module(
    "pause_user_dir.pause_adam",
    "optim/pause_adam.py",
    parent_package=fairseq.optim,
    attribute="pause_adam",
)
_load_source_module(
    "pause_user_dir.pause_prediction",
    "tasks/pause_prediction.py",
)
