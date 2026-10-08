#!/usr/bin/env python3
"""统计 JSONL 中“至少含一个正停顿标签”的样本数量与比例。"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="统计 all_pause_data.jsonl 的样本级停顿占比，并复核标签级正停顿比例。"
    )
    parser.add_argument("--manifest", required=True, help="all_pause_data.jsonl 路径")
    parser.add_argument("--output", required=True, help="统计结果 TXT 路径")
    return parser.parse_args()


def normalize_binary_labels(raw_labels, utt_id, line_number):
    if not isinstance(raw_labels, list):
        raise ValueError(
            f"第 {line_number} 行 {utt_id}: pause_label 必须是 list，"
            f"实际为 {type(raw_labels).__name__}"
        )

    labels = []
    for index, value in enumerate(raw_labels):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int) or value not in (0, 1):
            raise ValueError(
                f"第 {line_number} 行 {utt_id}: pause_label[{index}]={value!r}，"
                "只允许整数 0 或 1"
            )
        labels.append(value)
    return labels


def summarize_manifest(manifest_path):
    total_samples = 0
    samples_with_pause = 0
    total_labels = 0
    positive_labels = 0
    seen_utt_ids = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON: {exc}") from exc

            utt_id = record.get("utt_id")
            if not isinstance(utt_id, str) or not utt_id:
                raise ValueError(f"第 {line_number} 行缺少有效 utt_id")
            if utt_id in seen_utt_ids:
                raise ValueError(f"第 {line_number} 行出现重复 utt_id: {utt_id}")
            seen_utt_ids.add(utt_id)

            labels = normalize_binary_labels(record.get("pause_label"), utt_id, line_number)
            positive_count = sum(labels)
            total_samples += 1
            samples_with_pause += int(positive_count > 0)
            total_labels += len(labels)
            positive_labels += positive_count

    samples_without_pause = total_samples - samples_with_pause
    sample_ratio = samples_with_pause / total_samples if total_samples else 0.0
    negative_labels = total_labels - positive_labels
    label_ratio = positive_labels / total_labels if total_labels else 0.0
    return {
        "total_samples": total_samples,
        "samples_with_pause": samples_with_pause,
        "samples_without_pause": samples_without_pause,
        "sample_ratio": sample_ratio,
        "total_labels": total_labels,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "label_ratio": label_ratio,
    }


def format_report(manifest_path, summary):
    return "\n".join(
        [
            "音频停顿训练数据：样本级停顿统计",
            f"输入 manifest: {manifest_path}",
            f"总样本数: {summary['total_samples']}",
            f"含至少一个停顿的样本数: {summary['samples_with_pause']}",
            f"不含停顿的样本数: {summary['samples_without_pause']}",
            f"含停顿样本比例: {summary['sample_ratio']:.8f}",
            f"含停顿样本百分比: {summary['sample_ratio'] * 100.0:.6f}%",
            "",
            "以下为标签级复核统计：",
            f"标签总数: {summary['total_labels']}",
            f"正停顿标签数: {summary['positive_labels']}",
            f"负停顿标签数: {summary['negative_labels']}",
            f"正停顿标签比例: {summary['label_ratio']:.8f}",
            f"正停顿标签百分比: {summary['label_ratio'] * 100.0:.6f}%",
            "",
            "公式：含停顿样本比例 = 含至少一个 pause_label=1 的样本数 / 总样本数",
        ]
    ) + "\n"


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到输入 manifest: {manifest_path}")

    summary = summarize_manifest(manifest_path)
    report = format_report(manifest_path, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"统计结果已写入: {output_path}")


if __name__ == "__main__":
    main()
