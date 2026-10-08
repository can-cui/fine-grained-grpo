#!/usr/bin/env python3
"""评测已保存 checkpoint 在六个正式 split 上的句级停顿序列准确率。"""

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fairseq import checkpoint_utils, utils


SPLITS = (
    "valid_m2_manual",
    "valid_m2_mfa",
    "valid_lmdb_mfa",
    "test_m2_manual",
    "test_m2_mfa",
    "test_lmdb_mfa",
)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--wav2vec-jit-path", required=True)
    parser.add_argument("--wav2vec-meta-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_epoch_checkpoints(checkpoint_root):
    checkpoints = []
    for path in checkpoint_root.iterdir():
        if not path.is_file():
            continue
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(
            "未找到已完成的 epoch checkpoint（期望 checkpointN.pt）：{}".format(checkpoint_root)
        )
    return checkpoints


def move_net_input(net_input, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in net_input.items()
    }


def evaluate_split(model, task, split, args, device):
    task.load_dataset(split)
    dataset = task.dataset(split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collater,
    )
    correct_utterances = 0
    utterances = 0
    with torch.no_grad():
        for batch in loader:
            output = model(**move_net_input(batch["net_input"], device))
            predictions = (torch.sigmoid(output["pause_logits"]).cpu() >= args.threshold)
            targets = batch["target"].bool()
            valid_mask = batch["valid_mask"].bool()
            valid_counts = valid_mask.long().sum(dim=1)
            if torch.any(valid_counts == 0):
                bad_indices = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
                raise ValueError("{} 出现没有有效句中词位置的样本：{}".format(split, bad_indices))
            sentence_correct = ((predictions == targets) | ~valid_mask).all(dim=1)
            correct_utterances += int(sentence_correct.long().sum().item())
            utterances += int(sentence_correct.numel())
    if utterances == 0:
        raise ValueError("{} 没有可评测句子".format(split))
    return {
        "correct_utterances": correct_utterances,
        "utterances": utterances,
        "sentence_accuracy": correct_utterances / utterances,
    }


def load_existing_rows(path, threshold, overwrite):
    if overwrite or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("threshold") != threshold or payload.get("splits") != list(SPLITS):
        raise ValueError(
            "已有统计文件的阈值或 split 与当前不同；请更换输出目录或使用 --overwrite"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("已有统计文件 rows 非法：{}".format(path))
    return rows


def best_by_split(rows):
    result = {}
    for split in SPLITS:
        candidates = [row for row in rows if row["split"] == split]
        if candidates:
            result[split] = max(
                candidates,
                key=lambda row: (
                    row["sentence_accuracy"],
                    row["correct_utterances"],
                    -row["epoch"],
                ),
            )
    return result


def write_reports(output_root, threshold, candidate_checkpoints, rows):
    rows.sort(key=lambda row: (row["epoch"], SPLITS.index(row["split"])))
    completed_names = sorted({row["checkpoint"] for row in rows})
    candidate_names = [path.name for _epoch, path in candidate_checkpoints]
    payload = {
        "format_version": 1,
        "metric": "sentence_accuracy_exact_pause_sequence",
        "definition": "同一句所有 valid_mask=1 的词位置，预测 0/1 序列与标签完全一致才记为正确",
        "threshold": threshold,
        "splits": list(SPLITS),
        "candidate_checkpoints": candidate_names,
        "completed_checkpoints": completed_names,
        "complete": set(candidate_names).issubset(set(completed_names)),
        "best_by_split": best_by_split(rows),
        "rows": rows,
    }
    json_path = output_root / "pause_checkpoint_sentence_accuracy.json"
    tsv_path = output_root / "pause_checkpoint_sentence_accuracy.tsv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = (
        "epoch",
        "checkpoint",
        "split",
        "threshold",
        "correct_utterances",
        "utterances",
        "sentence_accuracy",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, tsv_path, payload


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold 必须在 [0, 1] 内")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 wav2vec TorchScript 资产为 CUDA 版本，句级评测必须使用 GPU")
    checkpoint_root = args.checkpoint_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError("checkpoint-root 不存在：{}".format(checkpoint_root))
    for split in SPLITS:
        if not (data_root / split).is_dir() or not (data_root / "{}.key".format(split)).is_file():
            raise FileNotFoundError("缺少正式 split：{}/{}".format(data_root, split))
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = list_epoch_checkpoints(checkpoint_root)
    json_path = output_root / "pause_checkpoint_sentence_accuracy.json"
    rows = load_existing_rows(json_path, args.threshold, args.overwrite)
    expected_keys = {(row["checkpoint"], row["split"]) for row in rows}

    utils.import_user_module(args)
    device = torch.device("cuda")
    for epoch, checkpoint in checkpoints:
        missing_splits = [
            split for split in SPLITS if (checkpoint.name, split) not in expected_keys
        ]
        if not missing_splits:
            print("跳过已统计 checkpoint：{}".format(checkpoint.name))
            continue
        print("评测 {}：{}".format(checkpoint.name, ",".join(missing_splits)))
        models, _checkpoint_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint)],
            arg_overrides={
                "data": str(data_root),
                "wav2vec_jit_path": args.wav2vec_jit_path,
                "wav2vec_meta_path": args.wav2vec_meta_path,
            },
        )
        if len(models) != 1:
            raise ValueError("{} 未加载到唯一模型".format(checkpoint))
        model = models[0].to(device).eval()
        for split in missing_splits:
            metrics = evaluate_split(model, task, split, args, device)
            row = {
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "split": split,
                "threshold": args.threshold,
                **metrics,
            }
            rows.append(row)
            expected_keys.add((checkpoint.name, split))
        json_path, tsv_path, payload = write_reports(output_root, args.threshold, checkpoints, rows)
        print(json.dumps({
            "checkpoint": checkpoint.name,
            "completed_checkpoints": payload["completed_checkpoints"],
            "statistics_json": str(json_path),
            "statistics_tsv": str(tsv_path),
        }, ensure_ascii=False))
        del model, models, task
        torch.cuda.empty_cache()

    json_path, tsv_path, payload = write_reports(output_root, args.threshold, checkpoints, rows)
    print(json.dumps({
        "ok": True,
        "statistics_json": str(json_path),
        "statistics_tsv": str(tsv_path),
        "best_by_split": payload["best_by_split"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
