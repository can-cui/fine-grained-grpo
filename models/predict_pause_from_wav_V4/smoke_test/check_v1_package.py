#!/usr/bin/env python3
"""检查 predict_pause_from_wav_V4 内网运行包是否完整且 Python 文件可解析。"""

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path


REQUIRED_NONEMPTY_FILES = (
    "README.md",
    "sh01_prepare_pause_training_data.sh",
    "sh02_train_pause_from_wav.sh",
    "sh03_infer_pause_from_wav.sh",
    "sh06_scan_pause_thresholds.sh",
    "sh07_evaluate_pause_checkpoints.sh",
    "ky_pause_from_wav_train.sh",
    "run_pause_from_wav_train_ky.sh",
    "scripts/prepare_pause_training_data.py",
    "scripts/prepare_pause_shared_data.py",
    "scripts/prepare_pause_shared_data_v2.py",
    "scripts/prepare_pause_label_comparison.py",
    "scripts/validate_pause_lmdb.py",
    "scripts/validate_shared_pause_splits.py",
    "scripts/validate_runtime_files.py",
    "scripts/scan_pause_thresholds.py",
    "scripts/summarize_pause_checkpoint_series.py",
    "scripts/export_wav2vec_torchscript.py",
    "pause_user_dir/__init__.py",
    "iflytek-tts-exp_fbsong/train.py",
    "iflytek-tts-exp_fbsong/infer_pause_from_wav.py",
    "iflytek-tts-exp_fbsong/fairseq/checkpoint_utils.py",
    "iflytek-tts-exp_fbsong/fairseq/options.py",
    "iflytek-tts-exp_fbsong/fairseq/trainer.py",
    "iflytek-tts-exp_fbsong/fairseq/models/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/models/pause_from_wav.py",
    "iflytek-tts-exp_fbsong/fairseq/criterions/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/criterions/fairseq_criterion.py",
    "iflytek-tts-exp_fbsong/fairseq/criterions/pause_bce_loss.py",
    "iflytek-tts-exp_fbsong/fairseq/data/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/data/fairseq_dataset.py",
    "iflytek-tts-exp_fbsong/fairseq/data/pause_lmdb_dataset.py",
    "iflytek-tts-exp_fbsong/fairseq/tasks/__init__.py",
    "iflytek-tts-exp_fbsong/fairseq/tasks/fairseq_task.py",
    "iflytek-tts-exp_fbsong/fairseq/tasks/pause_prediction.py",
    "iflytek-tts-exp_fbsong/fairseq/optim/adam.py",
    "iflytek-tts-exp_fbsong/fairseq/optim/pause_adam.py",
    "iflytek-tts-exp_fbsong/fairseq/optim/lr_scheduler/fixed_schedule.py",
    "smoke_test/README.md",
    "smoke_test/run_pause_v4_smoke_test.sh",
    "smoke_test/ky_pause_from_wav_smoke_test.sh",
    "smoke_test/run_pause_from_wav_smoke_test_ky.sh",
    "smoke_test/check_v1_package.py",
    "smoke_test/build_smoke_manifest.py",
    "smoke_test/validate_smoke_outputs.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wav2vec-jit-path", required=True)
    parser.add_argument("--wav2vec-meta-path", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--check-fairseq-registration", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.package_root).resolve()
    report_path = Path(args.report).resolve()
    missing = []
    empty = []
    syntax_errors = []
    registration_error = ""
    registration = {}

    for relative in REQUIRED_NONEMPTY_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size == 0:
            empty.append(relative)
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as exc:
                syntax_errors.append({"path": relative, "error": str(exc)})

    manifest = Path(args.manifest)
    if not manifest.is_file() or manifest.stat().st_size == 0:
        missing.append("外部正式数据 manifest: {}".format(manifest))

    jit_path = Path(args.wav2vec_jit_path)
    meta_path = Path(args.wav2vec_meta_path)
    asset_exists = jit_path.is_file() and jit_path.stat().st_size > 0
    meta_exists = meta_path.is_file() and meta_path.stat().st_size > 0
    asset_status = "ready" if asset_exists and meta_exists else "will_export"
    if asset_exists != meta_exists:
        asset_status = "incomplete"
        missing.append("wav2vec 永久资产必须模型和元数据同时存在或同时缺失")
    elif asset_exists:
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                asset_metadata = json.load(handle)
            if not (
                asset_metadata.get("format_version") == 3
                and asset_metadata.get("trace_device") == "cuda"
            ):
                asset_status = "will_reexport_cuda_v3"
        except (OSError, UnicodeError, json.JSONDecodeError):
            asset_status = "will_reexport_cuda_v3"

    # 在内网训练环境中真实导入 user-dir，防止只通过 AST、却没有进入
    # Fairseq 注册表的问题拖到正式训练阶段才暴露。
    if args.check_fairseq_registration and not missing and not empty and not syntax_errors:
        try:
            root_string = str(root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            importlib.import_module("pause_user_dir")
            from fairseq import criterions, models, optim, tasks

            compatibility_parser = argparse.ArgumentParser(add_help=False)
            tasks.TASK_REGISTRY["pause_prediction"].add_args(compatibility_parser)
            compatibility_args = compatibility_parser.parse_args(["smoke_data"])

            registration = {
                "task_pause_prediction": "pause_prediction" in tasks.TASK_REGISTRY,
                "expected_arch": args.expected_arch in models.ARCH_MODEL_REGISTRY,
                "criterion_pause_bce_loss": (
                    "pause_bce_loss" in criterions.CRITERION_REGISTRY
                ),
                "train_py_bert_pretrain_compat": (
                    hasattr(compatibility_args, "bert_pretrain")
                    and compatibility_args.bert_pretrain == ""
                ),
                "trainer_buffer_size_compat": (
                    hasattr(compatibility_args, "buffer_size")
                    and compatibility_args.buffer_size == 0
                ),
                "trainer_grouped_shuffling_compat": (
                    hasattr(compatibility_args, "grouped_shuffling")
                    and compatibility_args.grouped_shuffling is False
                ),
                "trainer_random_shuffle_data_compat": (
                    hasattr(compatibility_args, "random_shuffle_data")
                    and compatibility_args.random_shuffle_data is False
                ),
            }
            if args.expected_arch == "pause_from_wav_w2v_unfreeze_last2":
                registration["optimizer_pause_adam"] = "pause_adam" in optim.OPTIMIZER_REGISTRY
            if not all(registration.values()):
                registration_error = "Fairseq 自定义注册表缺项：{}".format(registration)
        except Exception as exc:
            registration_error = "{}: {}".format(type(exc).__name__, exc)

    report = {
        "ok": not missing and not empty and not syntax_errors and not registration_error,
        "package_root": str(root),
        "expected_arch": args.expected_arch,
        "required_file_count": len(REQUIRED_NONEMPTY_FILES),
        "missing": missing,
        "empty": empty,
        "python_syntax_errors": syntax_errors,
        "fairseq_registration": registration,
        "fairseq_registration_error": registration_error,
        "wav2vec_asset_status": asset_status,
        "wav2vec_jit_path": str(jit_path),
        "wav2vec_meta_path": str(meta_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("实验包不完整；请按上方清单补齐后重跑")


if __name__ == "__main__":
    main()
