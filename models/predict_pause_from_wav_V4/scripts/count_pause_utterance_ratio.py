#!/usr/bin/env python3
"""Count and report the number and proportion of samples with at least one positive pause label."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report the utterance-level pause ratio in all_pause_data.jsonl and verify the positive-label ratio."
    )
    parser.add_argument("--manifest", required=True, help="Path to all_pause_data.jsonl")
    parser.add_argument("--output", required=True, help="Path for the statistics TXT file")
    return parser.parse_args()


def normalize_binary_labels(raw_labels, utt_id, line_number):
    if not isinstance(raw_labels, list):
        raise ValueError(
            f"Line {line_number}, {utt_id}: pause_label must be a list; "
            f"got {type(raw_labels).__name__}"
        )

    labels = []
    for index, value in enumerate(raw_labels):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int) or value not in (0, 1):
            raise ValueError(
                f"Line {line_number}, {utt_id}: pause_label[{index}]={value!r}; "
                "only integer 0 or 1 is allowed"
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
                raise ValueError(f"Line {line_number} is not valid JSON: {exc}") from exc

            utt_id = record.get("utt_id")
            if not isinstance(utt_id, str) or not utt_id:
                raise ValueError(f"Line {line_number} is missing a valid utt_id")
            if utt_id in seen_utt_ids:
                raise ValueError(f"Duplicate utt_id on line {line_number}: {utt_id}")
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
            "Audio pause training data: utterance-level pause statistics",
            f"Input manifest: {manifest_path}",
            f"Total utterances: {summary['total_samples']}",
            f"Utterances with at least one pause: {summary['samples_with_pause']}",
            f"Utterances without pauses: {summary['samples_without_pause']}",
            f"Utterance pause ratio: {summary['sample_ratio']:.8f}",
            f"Utterance pause percentage: {summary['sample_ratio'] * 100.0:.6f}%",
            "",
            "Label-level verification statistics:",
            f"Total labels: {summary['total_labels']}",
            f"Positive pause labels: {summary['positive_labels']}",
            f"Negative pause labels: {summary['negative_labels']}",
            f"Positive pause-label ratio: {summary['label_ratio']:.8f}",
            f"Positive pause-label percentage: {summary['label_ratio'] * 100.0:.6f}%",
            "",
            "Formula: utterance pause ratio = utterances with at least one pause_label=1 / total utterances",
        ]
    ) + "\n"


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")

    summary = summarize_manifest(manifest_path)
    report = format_report(manifest_path, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Statistics written to: {output_path}")


if __name__ == "__main__":
    main()
