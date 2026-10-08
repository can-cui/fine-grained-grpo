#!/usr/bin/env python3
"""校验极小数据训练、checkpoint、推理和指标是否形成完整闭环。"""

import argparse
import json
import math
import re
from pathlib import Path

import torch


SPLITS = ("train", "valid", "test")
METRIC_KEYS = ("accuracy", "precision", "recall", "f1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("{}:{} JSON 损坏: {}".format(path, line_number, exc))
    return rows


def checkpoint_keys(path):
    state = torch.load(str(path), map_location="cpu")
    model_state = state.get("model", state) if isinstance(state, dict) else state
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint 不含可识别的 model state_dict")
    return list(model_state.keys())


def main():
    args = parse_args()
    errors = []
    details = {}
    selection = load_json(Path(args.selection_report))
    data_root = Path(args.data_root)

    all_ids = set()
    split_rows = {}
    for split in SPLITS:
        manifest_path = data_root / (split + ".jsonl")
        try:
            rows = load_jsonl(manifest_path)
            split_rows[split] = rows
            expected = selection["splits"][split]
            ids = [str(row["utt_id"]) for row in rows]
            if len(rows) != int(expected["count"]):
                errors.append("{} 数量不符: actual={} expected={}".format(
                    split, len(rows), expected["count"]
                ))
            if set(ids) != set(expected["utt_ids"]):
                errors.append("{} utt_id 与抽样预期不一致".format(split))
            overlap = all_ids.intersection(ids)
            if overlap:
                errors.append("split 间 utt_id 交叉: {}".format(sorted(overlap)))
            all_ids.update(ids)
            positive = sum(int(row["positive_target_count"]) > 0 for row in rows)
            negative = sum(int(row["positive_target_count"]) == 0 for row in rows)
            if positive == 0 or negative == 0:
                errors.append("{} 未同时包含正、负停顿样本".format(split))
            details[split] = {
                "count": len(rows),
                "positive_utterances": positive,
                "negative_utterances": negative,
                "word_count": sum(int(row["word_count"]) for row in rows),
                "valid_target_count": sum(int(row["valid_target_count"]) for row in rows),
            }
        except Exception as exc:
            errors.append("读取 {} 数据清单失败: {}".format(split, exc))

    checkpoint_dir = Path(args.checkpoint_dir)
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("缺少或为空: {}".format(path))
    best_path = checkpoint_dir / "checkpoint_best.pt"
    if best_path.is_file() and best_path.stat().st_size > 0:
        try:
            keys = checkpoint_keys(best_path)
            if not any(key.startswith("wav2vec.") for key in keys):
                errors.append("checkpoint state_dict 不含 wav2vec.*")
            if not any(key.startswith("pause_head.") for key in keys):
                errors.append("checkpoint state_dict 不含 pause_head.*")
            details["checkpoint_parameter_count"] = len(keys)
        except Exception as exc:
            errors.append("读取 checkpoint_best.pt 失败: {}".format(exc))

    train_log = Path(args.train_log)
    if not train_log.is_file() or train_log.stat().st_size == 0:
        errors.append("训练日志不存在或为空: {}".format(train_log))
    else:
        log_text = train_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"traceback\s*\(most recent call last\)", log_text, re.I):
            errors.append("训练日志包含 Traceback")
        if re.search(r"\b(?:loss|nll_loss)\b\s*[=: ]+\s*(?:nan|inf|-inf)\b", log_text, re.I):
            errors.append("训练日志包含非有限 loss")
        if re.search(r"batch contains no valid pause targets|empty batch", log_text, re.I):
            errors.append("训练日志包含空 batch/无有效目标错误")

    inference_dir = Path(args.inference_dir)
    predictions_path = inference_dir / "test_pause_predictions.jsonl"
    tsv_path = inference_dir / "test_pause_predictions.tsv"
    metrics_path = inference_dir / "test_pause_metrics.json"
    for path in (predictions_path, tsv_path, metrics_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("推理输出不存在或为空: {}".format(path))

    if predictions_path.is_file() and predictions_path.stat().st_size > 0 and "test" in details:
        try:
            predictions = load_jsonl(predictions_path)
            if len(predictions) != details["test"]["count"]:
                errors.append("推理句数不等于 test 句数")
            prediction_ids = {str(row["utt_id"]) for row in predictions}
            expected_ids = {str(row["utt_id"]) for row in split_rows.get("test", [])}
            if prediction_ids != expected_ids:
                errors.append("推理 utt_id 与 test split 不一致")
            word_rows = [word for row in predictions for word in row.get("words", [])]
            valid_rows = sum(int(word.get("valid", 0)) for word in word_rows)
            if len(word_rows) != details["test"]["word_count"]:
                errors.append("逐词推理结果数不等于 test 总词数")
            if valid_rows != details["test"]["valid_target_count"]:
                errors.append("推理 valid=1 行数不等于 test 有效目标数")
            details["inference_word_rows"] = len(word_rows)
            details["inference_valid_rows"] = valid_rows
        except Exception as exc:
            errors.append("读取推理 JSONL 失败: {}".format(exc))

    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            metrics = load_json(metrics_path)
            for key in METRIC_KEYS:
                value = float(metrics[key])
                if not math.isfinite(value):
                    errors.append("指标 {} 不是有限数".format(key))
            details["metrics"] = metrics
        except Exception as exc:
            errors.append("读取推理指标失败: {}".format(exc))

    report = {"ok": not errors, "errors": errors, "details": details}
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text_lines = [
        "音频停顿预测 V4 极小数据端到端烟雾测试",
        "=" * 68,
        "结论: {}".format("SMOKE_TEST_PASS" if report["ok"] else "SMOKE_TEST_FAIL"),
        "",
    ]
    if errors:
        text_lines.append("失败项:")
        text_lines.extend("- " + value for value in errors)
    else:
        text_lines.extend([
            "train/valid/test 数量、平衡性及互斥性通过",
            "checkpoint_best.pt 与 checkpoint_last.pt 检查通过",
            "checkpoint 包含 wav2vec.* 与 pause_head.*",
            "训练日志、逐词推理输出及有限指标检查通过",
        ])
    output_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
