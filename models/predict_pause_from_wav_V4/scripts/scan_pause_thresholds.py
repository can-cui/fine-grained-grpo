#!/usr/bin/env python3
"""在 valid 上扫描停顿阈值，并把所选阈值应用到对应的 test 标签口径。"""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path


METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold-start", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--threshold-end", type=Decimal, default=Decimal("0.99"))
    parser.add_argument("--threshold-step", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--baseline-threshold", type=Decimal, default=Decimal("0.50"))
    parser.add_argument(
        "--split-pairs",
        nargs="+",
        default=(
            "valid_m2_manual:test_m2_manual",
            "valid_m2_mfa:test_m2_mfa",
            "valid_lmdb_mfa:test_lmdb_mfa",
        ),
        help="每项格式为 valid_split:test_split；阈值只在左侧 valid 上选择",
    )
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def make_thresholds(start, end, step):
    if start < 0 or end > 1 or start > end or step <= 0:
        raise ValueError("阈值范围必须满足 0 <= start <= end <= 1 且 step > 0")
    thresholds = []
    value = start
    while value <= end:
        thresholds.append(value)
        value += step
    if not thresholds:
        raise ValueError("阈值列表为空")
    return thresholds


def load_predictions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"utt_id", "word_index", "probability", "target", "valid"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"预测 TSV 缺少字段：{path}")
        for line_number, row in enumerate(reader, 2):
            try:
                probability = float(row["probability"])
                target = int(row["target"])
                valid = int(row["valid"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: 数值字段非法") from exc
            if not 0.0 <= probability <= 1.0 or target not in (0, 1) or valid not in (0, 1):
                raise ValueError(f"{path}:{line_number}: probability/target/valid 越界")
            if valid:
                rows.append((probability, bool(target)))
    if not rows:
        raise ValueError(f"没有有效评测位置：{path}")
    return rows


def calculate_metrics(rows, threshold):
    threshold = float(threshold)
    tp = fp = fn = tn = 0
    for probability, target in rows:
        prediction = probability >= threshold
        if prediction and target:
            tp += 1
        elif prediction:
            fp += 1
        elif target:
            fn += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "threshold": threshold,
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_best(rows):
    # F1 优先；完全并列时依次偏向更高 recall、precision 和更高阈值。
    return max(
        rows,
        key=lambda row: (row["f1"], row["recall"], row["precision"], row["threshold"]),
    )


def write_tsv(path, rows, extra_fields=()):
    fields = [*extra_fields, "threshold", *METRIC_FIELDS]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    thresholds = make_thresholds(
        args.threshold_start, args.threshold_end, args.threshold_step
    )
    if not 0 <= args.baseline_threshold <= 1:
        raise ValueError("baseline-threshold 必须在 [0, 1] 内")

    inference_root = Path(args.inference_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    output = {
        "inference_root": str(inference_root),
        "threshold_start": float(args.threshold_start),
        "threshold_end": float(args.threshold_end),
        "threshold_step": float(args.threshold_step),
        "baseline_threshold": float(args.baseline_threshold),
        "selection_rule": "max_f1_then_recall_then_precision_then_threshold",
        "split_pairs": {},
    }

    seen_valid = set()
    for pair in args.split_pairs:
        parts = pair.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"split-pairs 格式非法：{pair}")
        valid_split, test_split = parts
        if valid_split in seen_valid:
            raise ValueError(f"valid split 重复：{valid_split}")
        seen_valid.add(valid_split)
        valid_path = inference_root / valid_split / f"{valid_split}_pause_predictions.tsv"
        test_path = inference_root / test_split / f"{test_split}_pause_predictions.tsv"
        if not valid_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"缺少 {valid_split}/{test_split} 的逐词预测；请先运行 sh03："
                f"{valid_path}, {test_path}"
            )
        valid_predictions = load_predictions(valid_path)
        test_predictions = load_predictions(test_path)
        scan_rows = [calculate_metrics(valid_predictions, value) for value in thresholds]
        best_valid = choose_best(scan_rows)
        baseline_valid = calculate_metrics(valid_predictions, args.baseline_threshold)
        selected_test = calculate_metrics(test_predictions, Decimal(str(best_valid["threshold"])))
        baseline_test = calculate_metrics(test_predictions, args.baseline_threshold)

        scan_path = output_root / f"{valid_split}_threshold_scan.tsv"
        write_tsv(scan_path, scan_rows)
        output["split_pairs"][valid_split] = {
            "valid_split": valid_split,
            "test_split": test_split,
            "valid_prediction_count": len(valid_predictions),
            "test_prediction_count": len(test_predictions),
            "scan_tsv": str(scan_path),
            "best_valid": best_valid,
            "baseline_valid": baseline_valid,
            "test_at_valid_best_threshold": selected_test,
            "baseline_test": baseline_test,
        }
        for dataset, selection, metrics in (
            ("valid", "valid_best", best_valid),
            ("valid", "baseline", baseline_valid),
            ("test", "valid_best", selected_test),
            ("test", "baseline", baseline_test),
        ):
            summary_rows.append(
                {
                    "valid_split": valid_split,
                    "test_split": test_split,
                    "dataset": dataset,
                    "selection": selection,
                    **metrics,
                }
            )

    json_path = output_root / "pause_threshold_scan_summary.json"
    tsv_path = output_root / "pause_threshold_scan_summary.tsv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(
        tsv_path,
        summary_rows,
        extra_fields=("valid_split", "test_split", "dataset", "selection"),
    )
    print(json.dumps({"ok": True, "summary_json": str(json_path), "summary_tsv": str(tsv_path), **output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
