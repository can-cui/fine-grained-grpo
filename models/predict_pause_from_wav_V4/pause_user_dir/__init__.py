"""从 V4 源文件显式注册 Fairseq dataset、criterion、model 和 task。"""

import importlib.util
import sys
from pathlib import Path

import fairseq.data
import fairseq.optim


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FAIRSEQ_SOURCE_ROOT = PACKAGE_ROOT / "iflytek-tts-exp_fbsong" / "fairseq"


def _load_source_module(module_name, relative_path, parent_package=None, attribute=None):
    """按文件路径加载模块，避免依赖当前 Fairseq 包的 __path__。"""
    if module_name in sys.modules:
        return sys.modules[module_name]
    source_path = FAIRSEQ_SOURCE_ROOT / relative_path
    if not source_path.is_file():
        raise FileNotFoundError("V4 自定义 Fairseq 模块不存在：{}".format(source_path))
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError("无法为 V4 模块创建加载器：{}".format(source_path))
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


# task 会导入 fairseq.data.pause_lmdb_dataset，因此必须先把 V4 dataset
# 注入当前训练环境实际加载的 fairseq.data 命名空间。
_load_source_module(
    "fairseq.data.pause_lmdb_dataset",
    "data/pause_lmdb_dataset.py",
    parent_package=fairseq.data,
    attribute="pause_lmdb_dataset",
)

# 其余模块只需执行源码中的注册装饰器；使用 pause_user_dir 子模块名，
# 不要求修改公司环境中已安装 Fairseq 的包目录。
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
