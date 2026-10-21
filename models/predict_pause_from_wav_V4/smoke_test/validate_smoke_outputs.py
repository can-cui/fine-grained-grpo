#!/usr/bin/env python3
"""Validate the complete loop for tiny-data training, checkpointing, inference, and metrics."""

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
                    raise ValueError("{}:{} invalid JSON: {}".format(path, line_number, exc))
    return rows


def checkpoint_keys(path):
    state = torch.load(str(path), map_location="cpu")
    model_state = state.get("model", state) if isinstance(state, dict) else state
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint does not contain a recognized model state_dict")
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
                errors.append("{} count mismatch: actual={} expected={}".format(
                    split, len(rows), expected["count"]
                ))
            if set(ids) != set(expected["utt_ids"]):
                errors.append("{} utt_id values do not match the expected sample".format(split))
            overlap = all_ids.intersection(ids)
            if overlap:
                errors.append("Overlapping utt_id values across splits: {}".format(sorted(overlap)))
            all_ids.update(ids)
            positive = sum(int(row["positive_target_count"]) > 0 for row in rows)
            negative = sum(int(row["positive_target_count"]) == 0 for row in rows)
            if positive == 0 or negative == 0:
                errors.append("{} does not contain both positive and negative pause samples".format(split))
            details[split] = {
                "count": len(rows),
                "positive_utterances": positive,
                "negative_utterances": negative,
                "word_count": sum(int(row["word_count"]) for row in rows),
                "valid_target_count": sum(int(row["valid_target_count"]) for row in rows),
            }
        except Exception as exc:
            errors.append("Failed to read {} data manifest: {}".format(split, exc))

    checkpoint_dir = Path(args.checkpoint_dir)
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("Missing or empty: {}".format(path))
    best_path = checkpoint_dir / "checkpoint_best.pt"
    if best_path.is_file() and best_path.stat().st_size > 0:
        try:
            keys = checkpoint_keys(best_path)
            if not any(key.startswith("wav2vec.") for key in keys):
                errors.append("Checkpoint state_dict does not contain wav2vec.*")
            if not any(key.startswith("pause_head.") for key in keys):
                errors.append("Checkpoint state_dict does not contain pause_head.*")
            details["checkpoint_parameter_count"] = len(keys)
        except Exception as exc:
            errors.append("Failed to read checkpoint_best.pt: {}".format(exc))

    train_log = Path(args.train_log)
    if not train_log.is_file() or train_log.stat().st_size == 0:
        errors.append("Training log is missing or empty: {}".format(train_log))
    else:
        log_text = train_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"traceback\s*\(most recent call last\)", log_text, re.I):
            errors.append("Training log contains a Traceback")
        if re.search(r"\b(?:loss|nll_loss)\b\s*[=: ]+\s*(?:nan|inf|-inf)\b", log_text, re.I):
            errors.append("Training log contains a non-finite loss")
        if re.search(r"batch contains no valid pause targets|empty batch", log_text, re.I):
            errors.append("Training log contains an empty-batch/no-valid-target error")

    inference_dir = Path(args.inference_dir)
    predictions_path = inference_dir / "test_pause_predictions.jsonl"
    tsv_path = inference_dir / "test_pause_predictions.tsv"
    metrics_path = inference_dir / "test_pause_metrics.json"
    for path in (predictions_path, tsv_path, metrics_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("Inference output is missing or empty: {}".format(path))

    if predictions_path.is_file() and predictions_path.stat().st_size > 0 and "test" in details:
        try:
            predictions = load_jsonl(predictions_path)
            if len(predictions) != details["test"]["count"]:
                errors.append("Number of inference sentences does not match the test set")
            prediction_ids = {str(row["utt_id"]) for row in predictions}
            expected_ids = {str(row["utt_id"]) for row in split_rows.get("test", [])}
            if prediction_ids != expected_ids:
                errors.append("Inference utt_id values do not match the test split")
            word_rows = [word for row in predictions for word in row.get("words", [])]
            valid_rows = sum(int(word.get("valid", 0)) for word in word_rows)
            if len(word_rows) != details["test"]["word_count"]:
                errors.append("Number of token-level inference results does not match the total test word count")
            if valid_rows != details["test"]["valid_target_count"]:
                errors.append("Number of inference rows with valid=1 does not match the number of valid test targets")
            details["inference_word_rows"] = len(word_rows)
            details["inference_valid_rows"] = valid_rows
        except Exception as exc:
            errors.append("Failed to read inference JSONL: {}".format(exc))

    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            metrics = load_json(metrics_path)
            for key in METRIC_KEYS:
                value = float(metrics[key])
                if not math.isfinite(value):
                    errors.append("Metric {} is not finite".format(key))
            details["metrics"] = metrics
        except Exception as exc:
            errors.append("Failed to read inference metrics: {}".format(exc))

    report = {"ok": not errors, "errors": errors, "details": details}
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text_lines = [
        "Audio Pause Prediction V4 Tiny-Data End-to-End Smoke Test",
        "=" * 68,
        "Result: {}".format("SMOKE_TEST_PASS" if report["ok"] else "SMOKE_TEST_FAIL"),
        "",
    ]
    if errors:
        text_lines.append("Failures:")
        text_lines.extend("- " + value for value in errors)
    else:
        text_lines.extend([
            "train/valid/test counts, balance, and exclusivity checks passed",
            "checkpoint_best.pt and checkpoint_last.pt checks passed",
            "Checkpoint contains wav2vec.* and pause_head.*",
            "Training log, token-level inference output, and finite-metric checks passed",
        ])
    output_txt.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("\n".join(text_lines))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
