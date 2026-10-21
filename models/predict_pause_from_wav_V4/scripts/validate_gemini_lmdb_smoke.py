#!/usr/bin/env python3
"""Validate smoke-test results for Gemini LMDB, Excel labels, and external WAV files."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--expected-wav-root", required=True)
    parser.add_argument("--expected-source-index", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    expected_wav_root = Path(args.expected_wav_root).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Smoke-test manifest not found: {manifest}")

    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            records.append(record)

    if len(records) != args.expected_count:
        raise ValueError(
            f"Unexpected number of successful samples: expected={args.expected_count}, actual={len(records)}"
        )

    seen_ids = set()
    positive_samples = 0
    positive_labels = 0
    total_labels = 0
    for record in records:
        utt_id = record.get("utt_id")
        if not utt_id or utt_id in seen_ids:
            raise ValueError(f"Missing or duplicate utt_id: {utt_id!r}")
        seen_ids.add(utt_id)
        if record.get("pause_label_source") != "xlsx_second_column":
            raise ValueError(f"{utt_id}: pause_label did not come from the second Excel column")
        if record.get("wav_source") != "external_wav_root":
            raise ValueError(f"{utt_id}: WAV did not come from external_wav_root")
        if int(record.get("source_index", -1)) != args.expected_source_index:
            raise ValueError(f"{utt_id}: incorrect source_index: {record.get('source_index')}")
        if int(record.get("sample_rate", 0)) != 24000:
            raise ValueError(f"{utt_id}: sample_rate is not 24000")

        wav_path = Path(record.get("wav_path", "")).resolve()
        try:
            wav_path.relative_to(expected_wav_root)
        except ValueError as exc:
            raise ValueError(f"{utt_id}: WAV is outside the external directory: {wav_path}") from exc
        if not wav_path.is_file():
            raise FileNotFoundError(f"{utt_id}: external WAV not found: {wav_path}")

        words = record.get("word_list")
        ends = record.get("word_end_time")
        labels = record.get("pause_label")
        if not isinstance(words, list) or not isinstance(ends, list) or not isinstance(labels, list):
            raise ValueError(f"{utt_id}: word_list/word_end_time/pause_label must be lists")
        if not (len(words) == len(ends) == len(labels)):
            raise ValueError(f"{utt_id}: word, time, and label lengths do not match")
        if any(not isinstance(value, int) or value not in (0, 1) for value in labels):
            raise ValueError(f"{utt_id}: pause_label is not a binary integer")
        numeric_ends = [float(value) for value in ends if value is not None]
        if not numeric_ends or any(not math.isfinite(value) for value in numeric_ends):
            raise ValueError(f"{utt_id}: word_end_time is missing a valid numeric value")
        leading = float(record.get("leading_silence_end_seconds", -1))
        duration = float(record.get("audio_duration_seconds", -1))
        alignment_duration = float(record.get("alignment_duration_seconds", -1))
        if not (0 <= leading <= alignment_duration <= duration + 0.05):
            raise ValueError(
                f"{utt_id}: invalid time range: leading={leading}, "
                f"alignment={alignment_duration}, wav={duration}"
            )
        count = sum(labels)
        positive_samples += int(count > 0)
        positive_labels += count
        total_labels += len(labels)

    print("Gemini LMDB smoke-test validation passed")
    print(f"Samples: {len(records)}")
    print(f"Samples containing pauses: {positive_samples}")
    print(f"Positive pause labels: {positive_labels}")
    print(f"Total labels: {total_labels}")
    print("Timestamp source: LMDB text/phonemes/fa")
    print("Pause-label source: second Excel column <PAUSE>")
    print(f"WAV source: {expected_wav_root}")


if __name__ == "__main__":
    main()
