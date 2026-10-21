#!/usr/bin/env python3
"""Confirm that all 13 shared Fairseq runtime files copied from V1 are non-empty."""

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
            "Shared Fairseq runtime files are incomplete; run "
            "bash copy_v1_runtime_files_to_v2_v5.sh in the variation directory. Missing/empty files:\n" + "\n".join(missing)
        )
    print(json.dumps({"ok": True, "package_root": str(root), "files": len(RUNTIME_FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
