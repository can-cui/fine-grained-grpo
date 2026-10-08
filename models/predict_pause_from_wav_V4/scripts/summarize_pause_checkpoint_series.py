#!/usr/bin/env python3
"""汇总多个 epoch checkpoint 的三组 valid/test 指标并生成折线图。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_PAIRS = (
    ("valid_m2_manual", "test_m2_manual"),
    ("valid_m2_mfa", "test_m2_mfa"),
    ("valid_lmdb_mfa", "test_lmdb_mfa"),
)
SPLITS = tuple(split_name for pair in SPLIT_PAIRS for split_name in pair)
METRICS = ("precision", "recall", "f1", "accuracy")
COUNT_FIELDS = ("tp", "fp", "fn", "tn")
COLORS = {
    "valid_m2_manual": "#1f77b4",
    "test_m2_manual": "#ff7f0e",
    "valid_m2_mfa": "#2ca02c",
    "test_m2_mfa": "#d62728",
    "valid_lmdb_mfa": "#9467bd",
    "test_lmdb_mfa": "#8c564b",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_predictions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"probability", "target", "valid"}
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
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def write_tsv(path, rows):
    fields = [
        "epoch",
        "split",
        "threshold_scheme",
        "threshold",
        *METRICS,
        *COUNT_FIELDS,
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_grid(rows, epochs, output_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for split in SPLITS:
            values = [
                row[metric]
                for row in rows
                if row["split"] == split
            ]
            axis.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[split],
                label=split,
            )
        axis.set_title(metric.upper())
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(epochs)
        axis.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # 总标题位于最上层，图例单独占下一层，子图顶部再留出安全间距。
    fig.suptitle(title, y=0.985, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_pair_metrics(rows, epochs, split_pair, output_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    for axis, metric in zip(axes, ("precision", "recall", "f1")):
        for split_name, linestyle in zip(split_pair, ("-", "--")):
            values = [
                row[metric]
                for row in rows
                if row["split"] == split_name
            ]
            axis.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                linestyle=linestyle,
                label=split_name,
            )
        axis.set_title(metric.upper())
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(epochs)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    # 标题和 valid/test 图例分层放置，避免顶部文字互相遮挡。
    fig.suptitle(title, y=0.985, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def find_best_epochs(rows):
    groups = {}
    for row in rows:
        key = row["split"]
        current = groups.get(key)
        if current is None or (row["f1"], row["recall"], -row["epoch"]) > (
            current["f1"],
            current["recall"],
            -current["epoch"],
        ):
            groups[key] = row
    return groups


def main():
    args = parse_args()
    epochs = sorted(set(args.epochs))
    if len(epochs) != len(args.epochs) or any(epoch <= 0 for epoch in epochs):
        raise ValueError("epochs 必须是互不重复的正整数")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold 必须在 [0, 1] 内")

    series_root = Path(args.series_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch in epochs:
        epoch_root = series_root / f"epoch_{epoch:03d}"
        for split in SPLITS:
            prediction_path = epoch_root / split / f"{split}_pause_predictions.tsv"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"找不到逐词预测：{prediction_path}")
            predictions = load_predictions(prediction_path)
            rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "threshold_scheme": "fixed",
                    "threshold": args.threshold,
                    **calculate_metrics(predictions, args.threshold),
                }
            )

    tsv_path = output_root / "pause_checkpoint_series_metrics.tsv"
    json_path = output_root / "pause_checkpoint_series_metrics.json"
    write_tsv(tsv_path, rows)
    threshold_name = str(args.threshold).replace(".", "p")
    plots = {
        "all_metrics": output_root / f"epoch_all_metrics_threshold_{threshold_name}.png",
        "valid_m2_manual__test_m2_manual": output_root / f"epoch_valid_m2_manual__test_m2_manual_threshold_{threshold_name}.png",
        "valid_m2_mfa__test_m2_mfa": output_root / f"epoch_valid_m2_mfa__test_m2_mfa_threshold_{threshold_name}.png",
        "valid_lmdb_mfa__test_lmdb_mfa": output_root / f"epoch_valid_lmdb_mfa__test_lmdb_mfa_threshold_{threshold_name}.png",
    }
    plot_metric_grid(
        rows,
        epochs,
        plots["all_metrics"],
        f"Checkpoint metrics at fixed threshold {args.threshold:.2f}",
    )
    plot_split_pair_metrics(
        rows,
        epochs,
        ("valid_m2_manual", "test_m2_manual"),
        plots["valid_m2_manual__test_m2_manual"],
        f"Kaoshiyuan manual labels (threshold={args.threshold:.2f})",
    )
    plot_split_pair_metrics(
        rows,
        epochs,
        ("valid_m2_mfa", "test_m2_mfa"),
        plots["valid_m2_mfa__test_m2_mfa"],
        f"Kaoshiyuan MFA labels (threshold={args.threshold:.2f})",
    )
    plot_split_pair_metrics(
        rows,
        epochs,
        ("valid_lmdb_mfa", "test_lmdb_mfa"),
        plots["valid_lmdb_mfa__test_lmdb_mfa"],
        f"Part00-04 MFA labels (threshold={args.threshold:.2f})",
    )
    output = {
        "series_root": str(series_root),
        "epochs": epochs,
        "threshold": args.threshold,
        "selection_note": "All epochs and all six splits use the same fixed threshold.",
        "best_epochs_by_f1": find_best_epochs(rows),
        "rows": rows,
        "plots": {name: str(path) for name, path in plots.items()},
    }
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "metrics_tsv": str(tsv_path), "metrics_json": str(json_path), **output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
