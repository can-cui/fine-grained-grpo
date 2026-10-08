#!/usr/bin/env python3
"""确认从 V1 补齐的 13 个公共 Fairseq 运行时文件均为非零文件。"""

import argparse
import json
from pathlib import Path


RUNTIME_FILES = (
    "iflytek-tts-exp_fbsong/train.py",
    "iflytek-tts-exp_fbsong/fairseq/checkpoint_utils.py",
    "iflytek-tts-exp_fbsong/fairseq/options.py",
    "iflytek-tts-exp_fbsong/fairseq/trainer.py",
    "iflytek-tts-exp_fbsong/fairseq/models/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/criterions/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/criterions/fairseq_criterion.py",
    "iflytek-tts-exp_fbsong/fairseq/data/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/data/fairseq_dataset.py",
    "iflytek-tts-exp_fbsong/fairseq/tasks/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/tasks/fairseq_task.py",
    "iflytek-tts-exp_fbsong/fairseq/optim/adam.py",
    "iflytek-tts-exp_fbsong/fairseq/optim/lr_scheduler/fixed_schedule.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    args = parser.parse_args()
    root = Path(args.package_root).resolve()
    missing = []
    for relative_path in RUNTIME_FILES:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise RuntimeError(
            "公共 Fairseq 运行时尚未补齐；请在 variation 目录运行 "
            "bash copy_v1_runtime_files_to_v2_v5.sh。缺失/零字节：\n" + "\n".join(missing)
        )
    print(json.dumps({"ok": True, "package_root": str(root), "files": len(RUNTIME_FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
