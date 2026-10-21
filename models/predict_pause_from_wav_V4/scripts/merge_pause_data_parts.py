#!/usr/bin/env python3
"""Merge JSONL manifests and statistics from extraction shards without copying WAV files."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_jsonl(path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def sample_id_from_error(row):
    """Recover utt_id from an error record using the extraction script's rules; return None if it cannot be recovered."""
    source_index = row.get("source_index")
    key_line = row.get("key_line", "")
    if not isinstance(source_index, int) or "@" not in key_line:
        return None
    key = key_line.split("@", 1)[1].split(" ", 1)[0].strip()
    if not key:
        return None
    readable = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("._-")[:80] or "sample"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"all_{source_index:03d}_{readable}_{digest}"


def records_equivalent(left, right):
    """For the same sample across shard directories, only output paths or newly added audit fields may differ."""
    ignored = {"wav_path", "hash_marker_count_before_cleanup"}
    return (
        {key: value for key, value in left.items() if key not in ignored}
        == {key: value for key, value in right.items() if key not in ignored}
    )


def main():
    args = parse_args()
    input_roots = [Path(value).resolve() for value in args.input_root]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = {}
    duplicate_count = 0
    leading_boundary_errors = []
    for root in input_roots:
        manifest = root / "all_pause_data.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Shard manifest not found: {manifest}")
        for record in read_jsonl(manifest):
            utt_id = record.get("utt_id")
            if not utt_id:
                raise ValueError(f"Record is missing utt_id: {manifest}")
            leading = record.get("leading_silence_end_seconds")
            alignment_duration = record.get("alignment_duration_seconds")
            if leading is None:
                leading_boundary_errors.append(f"{utt_id}: missing leading_silence_end_seconds")
                continue
            try:
                leading = float(leading)
                alignment_duration = float(alignment_duration)
            except (TypeError, ValueError):
                leading_boundary_errors.append(f"{utt_id}: leading-silence or alignment time is not numeric")
                continue
            if (
                not math.isfinite(leading)
                or not math.isfinite(alignment_duration)
                or leading < 0
                or leading > alignment_duration
            ):
                leading_boundary_errors.append(
                    f"{utt_id}: invalid boundary leading={leading}, alignment={alignment_duration}"
                )
                continue
            if utt_id in records:
                if not records_equivalent(records[utt_id], record):
                    raise ValueError(f"Conflicting duplicate utt_id found: {utt_id}")
                duplicate_count += 1
                # Inputs are ordered with the old single-process directory before the new shard directories; prefer a valid WAV path from the latter.
            records[utt_id] = record

    if leading_boundary_errors:
        preview = "\n".join(leading_boundary_errors[:20])
        raise ValueError(
            "Merged inputs contain missing or invalid leading-silence boundaries; re-extract using the updated extraction script. "
            f"Error count={len(leading_boundary_errors)}\n{preview}"
        )

    merged_manifest = output_root / "all_pause_data.jsonl"
    with merged_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for utt_id in sorted(records):
            handle.write(json.dumps(records[utt_id], ensure_ascii=False) + "\n")

    error_rows = []
    seen_errors = set()
    resolved_error_count = 0
    for root in input_roots:
        for row in read_jsonl(root / "extraction_errors.jsonl"):
            if sample_id_from_error(row) in records:
                resolved_error_count += 1
                continue
            signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if signature not in seen_errors:
                seen_errors.add(signature)
                error_rows.append(row)
    with (output_root / "extraction_errors.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports = {}
    duplicate_source_report_count = 0
    for root in input_roots:
        for row in read_jsonl(root / "source_report.jsonl"):
            source_index = row.get("source_index")
            if source_index in reports:
                duplicate_source_report_count += 1
            # As with the manifest, the old directory comes first and the new shards later; keep the report from the latest complete run.
            reports[source_index] = row
    with (output_root / "source_report.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for source_index in sorted(reports, key=lambda value: (-1 if value is None else value)):
            handle.write(json.dumps(reports[source_index], ensure_ascii=False) + "\n")

    counters = Counter()
    speakers = set()
    sources = set()
    for record in records.values():
        words = record.get("word_list", [])
        labels = record.get("pause_label", [])
        if len(words) != len(labels):
            raise ValueError(
                f"{record.get('utt_id', '<unknown>')}: "
                f"word_list and pause_label lengths differ: {len(words)} != {len(labels)}"
            )
        normalized_labels = [int(value) for value in labels]
        if any(value not in (0, 1) for value in normalized_labels):
            raise ValueError(
                f"{record.get('utt_id', '<unknown>')}: pause_label may contain only 0 or 1"
            )
        positive_pause_count = sum(normalized_labels)
        counters["words"] += len(words)
        counters["positive_pauses"] += positive_pause_count
        counters["utterances_with_pause"] += int(positive_pause_count > 0)
        counters["audio_samples"] += int(record.get("audio_num_samples", 0))
        counters["hash_markers"] += int(record.get("hash_marker_count_before_cleanup", 0))
        counters[f"phoneme_layout_{record.get('phoneme_storage_layout', 'unknown')}"] += 1
        if record.get("wav_path") and not Path(record["wav_path"]).is_file():
            counters["missing_wav_files"] += 1
        if record.get("speaker_id"):
            speakers.add(record["speaker_id"])
        if record.get("source_index") is not None:
            sources.add(record["source_index"])

    negative_pauses = counters["words"] - counters["positive_pauses"]
    positive_ratio = counters["positive_pauses"] / counters["words"] if counters["words"] else 0.0
    utterances_without_pause = len(records) - counters["utterances_with_pause"]
    utterance_pause_ratio = (
        counters["utterances_with_pause"] / len(records) if records else 0.0
    )
    total_hours = sum(float(row.get("audio_duration_seconds", 0.0)) for row in records.values()) / 3600.0
    lines = [
        f"Samples containing at least one pause: {counters['utterances_with_pause']}",
        f"Samples without pauses: {utterances_without_pause}",
        f"Fraction of samples containing pauses: {utterance_pause_ratio:.8f}",
        f"Percentage of samples containing pauses: {utterance_pause_ratio * 100.0:.6f}%",
        "Audio pause training-data shard merge statistics",
        f"Input directories: {len(input_roots)}",
        f"Unique samples: {len(records)}",
        f"Deduplicated records: {duplicate_count}",
        f"LMDB sources: {len(sources)}",
        f"Speakers: {len(speakers)}",
        f"Total audio hours: {total_hours:.6f}",
        f"Total words: {counters['words']}",
        f"Positive pause labels: {counters['positive_pauses']}",
        f"Negative pause labels: {negative_pauses}",
        f"Positive pause ratio: {positive_ratio:.8f}",
        f"Total # markers before cleanup: {counters['hash_markers']}",
        f"Samples with phonemes int16_x3 layout: {counters['phoneme_layout_int16_x3']}",
        f"Samples with phonemes int16_x6 layout: {counters['phoneme_layout_int16_x6']}",
        f"Samples with unrecorded phonemes layout: {counters['phoneme_layout_unknown']}",
        f"Error records: {len(error_rows)}",
        f"Historical errors resolved by successful samples: {resolved_error_count}",
        f"Old source reports superseded by new shards: {duplicate_source_report_count}",
        f"Missing WAV files: {counters['missing_wav_files']}",
        "Missing leading_silence_end_seconds values: 0",
        "Invalid leading_silence_end_seconds values: 0",
        "Note: only manifests and statistics are merged; WAV files are not copied, and wav_path still points to the original shard directory.",
    ]
    stats = "\n".join(lines) + "\n"
    (output_root / "training_data_statistics.txt").write_text(stats, encoding="utf-8")
    print(stats, end="")
    print(f"Merged manifest: {merged_manifest}")


if __name__ == "__main__":
    main()
