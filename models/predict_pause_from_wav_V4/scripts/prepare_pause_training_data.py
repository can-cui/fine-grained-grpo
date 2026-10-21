#!/usr/bin/env python3
"""Convert merged pause JSONL/WAV data to PCM16 LMDBs for online wav2vec training."""

import argparse
import io
import json
import math
import random
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchaudio


SPLITS = ("train", "valid", "test")
PAUSE_SYMBOLS = {
    ",", "，", "、", ".", "。", "；", "...", "：",
    "~", "*", "#", "@", "&", ":", ";", "!", "！", "?", "？",
}


class NoValidPauseTargetError(ValueError):
    """Record samples with fewer than two words after removing pause markers, which have no valid training targets."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-mode", choices=("ratio", "count"), default="ratio")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--train-count", type=int, default=0)
    parser.add_argument("--valid-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument(
        "--test-positive-utterance-ratio",
        type=float,
        default=0.50,
        help=(
            "In count mode with train-count=0, the ratio of test sentences containing at least one valid positive pause; "
            "rounded to obtain an exact sentence count."
        ),
    )
    parser.add_argument(
        "--eval-source-index",
        type=int,
        default=None,
        help=(
            "Used only in count mode with train-count=0; valid/test are sampled only from this "
            "source_index, and all other sources plus its remaining samples go to train."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--source-sample-rate", type=int, default=24000)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path):
    records = []
    skipped = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped.append(
                    make_skipped_record(
                        None,
                        "load_manifest",
                        f"Invalid JSON: {exc}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(record, dict):
                skipped.append(
                    make_skipped_record(
                        None,
                        "load_manifest",
                        f"JSON top-level value must be an object, got {type(record).__name__}",
                        line_number=line_number,
                    )
                )
                continue
            utt_id = str(record.get("utt_id", "")).strip()
            if not utt_id:
                skipped.append(
                    make_skipped_record(
                        record,
                        "load_manifest",
                        "Missing utt_id",
                        line_number=line_number,
                    )
                )
                continue
            if utt_id in seen:
                skipped.append(
                    make_skipped_record(
                        record,
                        "load_manifest",
                        f"Duplicate utt_id: {utt_id}",
                        line_number=line_number,
                    )
                )
                continue
            seen.add(utt_id)
            records.append(record)
    if not records:
        raise ValueError(f"Merged manifest is empty: {path}")
    return sorted(records, key=lambda item: item["utt_id"]), skipped


def make_skipped_record(record, stage, reason, line_number=None, split=None):
    record = record if isinstance(record, dict) else {}
    row = {
        "stage": stage,
        "utt_id": str(record.get("utt_id", "")),
        "source_index": record.get("source_index"),
        "reason": str(reason),
    }
    if line_number is not None:
        row["line_number"] = int(line_number)
    if split is not None:
        row["split"] = split
    return row


def write_skipped_records(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ratio_counts(total, ratios):
    if any(value < 0 for value in ratios):
        raise ValueError("train/valid/test ratios cannot be negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Ratios must sum to 1, got {sum(ratios)}")
    raw = [total * value for value in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def record_source_index(record):
    try:
        return int(record.get("source_index", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{record['utt_id']}: source_index is not an integer: {record.get('source_index')!r}"
        ) from exc


def record_has_positive_pause(record):
    """Determine whether a sentence contains at least one positive pause target used for training."""
    _, _, _, labels, valid_mask = collapse_to_lexical(record)
    return any(label and valid for label, valid in zip(labels, valid_mask))


def allocate_strata(strata, count):
    total = sum(len(items) for items in strata.values())
    if count < 0 or count > total:
        raise ValueError(f"Invalid stratified sample count: count={count}, total={total}")
    if count == 0:
        return {key: 0 for key in strata}
    raw = {key: count * len(items) / total for key, items in strata.items()}
    quota = {
        key: min(len(strata[key]), int(math.floor(raw[key])))
        for key in strata
    }
    remaining = count - sum(quota.values())
    order = sorted(strata, key=lambda key: (-(raw[key] - quota[key]), str(key)))
    while remaining:
        moved = False
        for key in order:
            if quota[key] < len(strata[key]):
                quota[key] += 1
                remaining -= 1
                moved = True
            if remaining == 0:
                break
        if not moved:
            raise ValueError("Could not allocate stratified sample counts")
    return quota


def build_split_rows(records):
    rows = []
    for record in records:
        rows.append(
            {
                "record": record,
                "utt_id": str(record["utt_id"]),
                "source_index": record_source_index(record),
                "positive_utterance": record_has_positive_pause(record),
            }
        )
    return rows


def filter_splittable_records(records):
    """Check required fields for each split; exclude invalid samples and retain the audit reason."""
    eligible = []
    skipped = []
    for record in records:
        try:
            record_source_index(record)
            collapse_to_lexical(record)
        except Exception as exc:
            skipped.append(
                make_skipped_record(record, "pre_split_validation", exc)
            )
        else:
            eligible.append(record)
    if not eligible:
        raise ValueError("No records can produce valid word-final pause targets")
    return eligible, skipped


def take_stratified(rows, count, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[(row["source_index"], row["positive_utterance"])].append(row)
    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)
    quota = allocate_strata(strata, count)
    selected = []
    remaining = []
    for key, items in strata.items():
        selected.extend(items[:quota[key]])
        remaining.extend(items[quota[key]:])
    return selected, remaining


def take_test_with_positive_ratio(rows, count, positive_ratio, seed):
    if count == 0:
        return [], list(rows), 0
    positive_count = int(math.floor(count * positive_ratio + 0.5))
    negative_count = count - positive_count
    groups = {
        True: [row for row in rows if row["positive_utterance"]],
        False: [row for row in rows if not row["positive_utterance"]],
    }
    requested = {True: positive_count, False: negative_count}
    for status in (True, False):
        if requested[status] > len(groups[status]):
            name = "with positive pauses" if status else "without positive pauses"
            raise ValueError(
                f"test requires {requested[status]} samples {name}, "
                f"but only {len(groups[status])} are available"
            )

    selected = []
    selected_ids = set()
    for offset, status in enumerate((True, False)):
        by_source = defaultdict(list)
        for row in groups[status]:
            by_source[row["source_index"]].append(row)
        rng = random.Random(seed + offset)
        for items in by_source.values():
            rng.shuffle(items)
        quota = allocate_strata(by_source, requested[status])
        for source, items in by_source.items():
            chosen = items[:quota[source]]
            selected.extend(chosen)
            selected_ids.update(row["utt_id"] for row in chosen)
    return selected, [row for row in rows if row["utt_id"] not in selected_ids], positive_count


def ensure_binary_validation(valid_rows, train_rows):
    """When valid has at least two sentences, try to include both positive- and negative-pause sentences."""
    if len(valid_rows) < 2:
        return
    statuses = {row["positive_utterance"] for row in valid_rows}
    if len(statuses) == 2:
        return
    missing_status = not next(iter(statuses))
    donor = next(
        (row for row in train_rows if row["positive_utterance"] == missing_status),
        None,
    )
    replacement = next(
        (row for row in valid_rows if row["positive_utterance"] != missing_status),
        None,
    )
    if donor is not None and replacement is not None:
        valid_rows.remove(replacement)
        train_rows.remove(donor)
        valid_rows.append(donor)
        train_rows.append(replacement)


def split_records(records, args):
    shuffled = list(records)
    random.Random(args.split_seed).shuffle(shuffled)
    if args.split_mode == "ratio":
        if args.eval_source_index is not None:
            raise ValueError("eval-source-index is supported only in count mode with train-count=0")
        counts = ratio_counts(
            len(shuffled),
            [args.train_ratio, args.valid_ratio, args.test_ratio],
        )
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(f"train and valid must both be non-empty, got counts {counts}")
        result = {}
        cursor = 0
        for split, count in zip(SPLITS, counts):
            result[split] = shuffled[cursor:cursor + count]
            cursor += count
        return result, {
            "strategy": "Random split with a fixed seed (ratio mode)",
            "eval_source_index": None,
            "eval_source_utterance_count": None,
            "test_positive_utterance_ratio_requested": None,
            "test_positive_utterance_count_target": None,
        }

    counts = [args.train_count, args.valid_count, args.test_count]
    if any(value < 0 for value in counts):
        raise ValueError("train/valid/test counts cannot be negative")
    if not 0.0 <= args.test_positive_utterance_ratio <= 1.0:
        raise ValueError("test-positive-utterance-ratio must be between 0 and 1")

    if args.train_count == 0:
        if args.valid_count <= 0:
            raise ValueError("In count mode with train-count=0, valid-count must be greater than 0")
        if args.valid_count + args.test_count >= len(records):
            raise ValueError(
                "valid-count + test-count must be less than the total sample count so train remains non-empty"
            )
        rows = build_split_rows(records)
        if args.eval_source_index is None:
            eval_rows = rows
            always_train_rows = []
        else:
            eval_rows = [
                row for row in rows
                if row["source_index"] == args.eval_source_index
            ]
            always_train_rows = [
                row for row in rows
                if row["source_index"] != args.eval_source_index
            ]
            required_eval_count = args.valid_count + args.test_count
            if required_eval_count > len(eval_rows):
                raise ValueError(
                    f"source_index={args.eval_source_index} has {len(eval_rows)} records, "
                    f"fewer than valid+test={required_eval_count}; cannot create the requested split"
                )
        test_rows, remaining_rows, test_positive_count = take_test_with_positive_ratio(
            eval_rows,
            args.test_count,
            args.test_positive_utterance_ratio,
            args.split_seed + 1,
        )
        valid_rows, train_rows = take_stratified(
            remaining_rows,
            args.valid_count,
            args.split_seed + 2,
        )
        ensure_binary_validation(valid_rows, train_rows)
        train_rows = always_train_rows + train_rows
        result = {
            "train": [row["record"] for row in train_rows],
            "valid": [row["record"] for row in valid_rows],
            "test": [row["record"] for row in test_rows],
        }
        if args.eval_source_index is None:
            strategy = (
                "Sample test first using the requested ratio of sentences with positive pauses, stratified by source_index within positive/negative classes; "
                "then stratify valid by source_index and positive-pause status; train uses the remaining samples"
            )
        else:
            strategy = (
                f"Sample valid/test only from source_index={args.eval_source_index}; "
                "sample test first using the requested ratio of sentences with positive pauses, then stratify valid by positive-pause status; "
                "all other sources and remaining samples from this source go to train"
            )
        return result, {
            "strategy": strategy,
            "eval_source_index": args.eval_source_index,
            "eval_source_utterance_count": len(eval_rows),
            "test_positive_utterance_ratio_requested": args.test_positive_utterance_ratio,
            "test_positive_utterance_count_target": test_positive_count,
        }

    if args.eval_source_index is not None:
        raise ValueError("eval-source-index is supported only in count mode with train-count=0")
    if sum(counts) != len(shuffled):
        raise ValueError(
            f"Counts must sum to the total number of samples: counts={counts}, total={len(shuffled)}; "
            "set train-count to 0 to assign the remainder to train automatically"
        )
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"train and valid must both be non-empty, got counts {counts}")
    result = {}
    cursor = 0
    for split, count in zip(SPLITS, counts):
        result[split] = shuffled[cursor:cursor + count]
        cursor += count
    return result, {
        "strategy": "Random split with a fixed seed (count mode with train-count specified)",
        "eval_source_index": None,
        "eval_source_utterance_count": None,
        "test_positive_utterance_ratio_requested": None,
        "test_positive_utterance_count_target": None,
    }


def read_pcm16_wav(path, expected_sample_rate):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if channels != 1 or sample_width != 2:
        raise ValueError(
            f"{path}: only mono PCM16 WAV is supported, got channels={channels}, sample_width={sample_width}"
        )
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path}: expected sample rate {expected_sample_rate}, got {sample_rate}"
        )
    waveform = np.frombuffer(raw, dtype="<i2").copy()
    if len(waveform) == 0:
        raise ValueError(f"{path}: WAV is empty")
    return waveform


def resample_pcm16(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return waveform.astype(np.int16, copy=True)
    source = torch.from_numpy(waveform.astype(np.float32) / 32768.0).unsqueeze(0)
    with torch.no_grad():
        target = torchaudio.functional.resample(source, source_rate, target_rate)
    target = target.squeeze(0).clamp(-1.0, 32767.0 / 32768.0)
    return torch.round(target * 32768.0).to(torch.int16).cpu().numpy()


def collapse_to_lexical(record):
    words = list(record.get("word_list", []))
    ends = list(record.get("word_end_time", []))
    labels = list(record.get("pause_label", []))
    if not (len(words) == len(ends) == len(labels)):
        raise ValueError(
            f"{record['utt_id']}: word_list/word_end_time/pause_label lengths do not match"
        )

    lexical_words = []
    lexical_ends = []
    lexical_labels = []
    for word, end_time, label in zip(words, ends, labels):
        if end_time is None or not math.isfinite(float(end_time)):
            raise ValueError(f"{record['utt_id']}: invalid end time for {word!r}: {end_time}")
        end_time = float(end_time)
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"{record['utt_id']}: pause label is not 0/1: {label}")
        if word in PAUSE_SYMBOLS:
            if lexical_words:
                lexical_ends[-1] = end_time
                lexical_labels[-1] = max(lexical_labels[-1], label)
            continue
        lexical_words.append(str(word))
        lexical_ends.append(end_time)
        lexical_labels.append(label)

    if len(lexical_words) < 2:
        raise NoValidPauseTargetError(
            f"{record['utt_id']}: fewer than two words; cannot produce valid pause targets"
        )
    leading = record.get("leading_silence_end_seconds")
    if leading is None:
        raise ValueError(
            f"{record['utt_id']}: missing leading_silence_end_seconds; rerun LMDB extraction"
        )
    leading = float(leading)
    if not math.isfinite(leading) or leading < 0 or leading > lexical_ends[0]:
        raise ValueError(
            f"{record['utt_id']}: invalid leading-silence boundary: leading={leading}, first_end={lexical_ends[0]}"
        )
    if any(right <= left for left, right in zip([leading] + lexical_ends[:-1], lexical_ends)):
        raise ValueError(f"{record['utt_id']}: word-window times are not strictly increasing")

    starts = [leading] + lexical_ends[:-1]
    valid_mask = [1] * (len(lexical_words) - 1) + [0]
    return lexical_words, starts, lexical_ends, lexical_labels, valid_mask


def seconds_to_samples(values, sample_rate, waveform_samples, utt_id):
    result = np.rint(np.asarray(values, dtype=np.float64) * sample_rate).astype(np.int64)
    tolerance = int(round(0.05 * sample_rate))
    if np.any(result < 0) or np.any(result > waveform_samples + tolerance):
        raise ValueError(
            f"{utt_id}: time boundaries exceed WAV length, range={result.min()}..{result.max()}, wav={waveform_samples}"
        )
    return np.clip(result, 0, waveform_samples)


def encode_record(record, source_rate, target_rate):
    wav_path = Path(record["wav_path"])
    if not wav_path.is_file():
        raise FileNotFoundError(f"{record['utt_id']}: WAV not found: {wav_path}")
    source_waveform = read_pcm16_wav(wav_path, source_rate)
    waveform = resample_pcm16(source_waveform, source_rate, target_rate)
    words, starts_seconds, ends_seconds, labels, valid_mask = collapse_to_lexical(record)
    starts = seconds_to_samples(starts_seconds, target_rate, len(waveform), record["utt_id"])
    ends = seconds_to_samples(ends_seconds, target_rate, len(waveform), record["utt_id"])
    if np.any(ends <= starts):
        raise ValueError(f"{record['utt_id']}: empty word window after conversion to sample indices")

    arrays = {
        "waveform_pcm16": waveform.astype(np.int16, copy=False),
        "sample_rate": np.asarray(target_rate, dtype=np.int32),
        "pool_start_sample": starts,
        "pool_end_sample": ends,
        "pool_start_seconds": starts.astype(np.float64) / target_rate,
        "pool_end_seconds": ends.astype(np.float64) / target_rate,
        "pause_label": np.asarray(labels, dtype=np.uint8),
        "valid_mask": np.asarray(valid_mask, dtype=np.uint8),
        "word_list": np.asarray(words, dtype=np.str_),
        "utt_id": np.asarray(str(record["utt_id"]), dtype=np.str_),
        "speaker_id": np.asarray(str(record.get("speaker_id", "")), dtype=np.str_),
        "source_index": np.asarray(int(record.get("source_index", -1)), dtype=np.int64),
        "source_config_group": np.asarray(str(record.get("source_config_group", "")), dtype=np.str_),
        "source_lmdb_spec": np.asarray(str(record.get("source_lmdb_spec", "")), dtype=np.str_),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue(), arrays


def remove_existing_split(output_root, split):
    lmdb_path = output_root / split
    key_path = output_root / f"{split}.key"
    manifest_path = output_root / f"{split}.jsonl"
    if lmdb_path.is_dir():
        shutil.rmtree(str(lmdb_path))
    for path in (key_path, manifest_path):
        if path.exists():
            path.unlink()


def write_split(output_root, split, records, args, skipped_records, skipped_path):
    if not records:
        lmdb_path = output_root / split
        key_path = output_root / f"{split}.key"
        manifest_path = output_root / f"{split}.jsonl"
        existing = [path for path in (lmdb_path, key_path, manifest_path) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{split} count is 0, but old output still exists; use --overwrite to delete after confirmation: {existing}"
            )
        if args.overwrite:
            remove_existing_split(output_root, split)
        return None
    lmdb_path = output_root / split
    key_path = output_root / f"{split}.key"
    manifest_path = output_root / f"{split}.jsonl"
    existing = [path for path in (lmdb_path, key_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{split} output already exists; use --overwrite after confirmation: {existing}")
    if args.overwrite:
        remove_existing_split(output_root, split)

    env = lmdb.open(
        str(lmdb_path),
        map_size=args.map_size_gb * 1024 ** 3,
        subdir=True,
        create=True,
        lock=True,
    )
    counters = Counter()
    speakers = set()
    sources = defaultdict(Counter)
    key_rows = []
    manifest_rows = []
    txn = env.begin(write=True)
    success = False
    try:
        for index, record in enumerate(records, 1):
            try:
                raw, arrays = encode_record(
                    record, args.source_sample_rate, args.target_sample_rate
                )
            except Exception as exc:
                skipped = make_skipped_record(record, "encode_record", exc, split=split)
                skipped_records.append(skipped)
                with skipped_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                continue
            key = str(record["utt_id"])
            if not txn.put(key.encode("utf-8"), raw, overwrite=False):
                raise ValueError(f"{split}: duplicate LMDB key: {key}")
            word_count = len(arrays["pause_label"])
            valid = arrays["valid_mask"].astype(bool)
            labels = arrays["pause_label"].astype(bool)
            speaker = str(arrays["speaker_id"].item())
            source_index = int(arrays["source_index"].item())
            key_rows.append(f"{key}\t{len(arrays['waveform_pcm16'])}\t{word_count}\t{speaker}\n")
            manifest_rows.append(
                json.dumps(
                    {
                        "utt_id": key,
                        "speaker_id": speaker,
                        "audio_num_samples": len(arrays["waveform_pcm16"]),
                        "word_count": word_count,
                        "valid_target_count": int(valid.sum()),
                        "positive_target_count": int((labels & valid).sum()),
                        "source_index": source_index,
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
            counters["utterances"] += 1
            counters["audio_samples"] += len(arrays["waveform_pcm16"])
            counters["words"] += word_count
            counters["valid"] += int(valid.sum())
            counters["positive"] += int((labels & valid).sum())
            counters["masked"] += int((~valid).sum())
            counters["positive_utterances"] += int(bool((labels & valid).any()))
            speakers.add(speaker)
            sources[source_index]["utterances"] += 1
            if index % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        txn = None
        env.sync()
        success = True
    finally:
        if txn is not None:
            txn.abort()
        env.close()
        if not success and lmdb_path.is_dir():
            shutil.rmtree(str(lmdb_path), ignore_errors=True)
    key_path.write_text("".join(key_rows), encoding="utf-8")
    manifest_path.write_text("".join(manifest_rows), encoding="utf-8")
    counters["speakers"] = len(speakers)
    return counters, sources


def main():
    args = parse_args()
    manifest = Path(args.merged_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Merged manifest not found: {manifest}")
    if args.source_sample_rate <= 0 or args.target_sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer")
    if args.map_size_gb <= 0:
        raise ValueError("map-size-gb must be greater than 0")
    output_root.mkdir(parents=True, exist_ok=True)

    input_records, skipped_records = load_manifest(manifest)
    input_record_count = len(input_records) + len(skipped_records)
    records, pre_split_skipped = filter_splittable_records(input_records)
    skipped_records.extend(pre_split_skipped)
    skipped_path = output_root / "prepare_skipped_records.jsonl"
    write_skipped_records(skipped_path, skipped_records)
    split_map, split_strategy = split_records(records, args)
    split_reports = {}
    for split in SPLITS:
        result = write_split(
            output_root,
            split,
            split_map[split],
            args,
            skipped_records,
            skipped_path,
        )
        if result is not None:
            split_reports[split] = result

    lines = [
        "Audio Pause Prediction V4: Online wav2vec Training Data Statistics",
        "=" * 72,
        f"Input manifest: {manifest}",
        f"Output directory: {output_root}",
        f"Split mode: {args.split_mode}",
        f"Random seed: {args.split_seed}",
        f"Split strategy: {split_strategy['strategy']}",
        f"Sample-rate conversion: {args.source_sample_rate} -> {args.target_sample_rate}",
        f"Input samples: {input_record_count}",
        f"Splittable samples before partitioning: {len(records)}",
        f"Cumulative skipped samples: {len(skipped_records)}",
        f"Skipped-sample manifest: {skipped_path}",
    ]
    if split_strategy["eval_source_index"] is not None:
        lines.extend(
            [
                "Restricted source_index for valid/test: {}".format(
                    split_strategy["eval_source_index"]
                ),
                "Available sentences from restricted sources: {}".format(
                    split_strategy["eval_source_utterance_count"]
                ),
            ]
        )
    if split_strategy["test_positive_utterance_ratio_requested"] is not None:
        lines.extend(
            [
                "Target ratio of test sentences with positive pauses: {:.8f}".format(
                    split_strategy["test_positive_utterance_ratio_requested"]
                ),
                "Target count of test sentences with positive pauses: {}".format(
                    split_strategy["test_positive_utterance_count_target"]
                ),
            ]
        )
    for split in SPLITS:
        if split not in split_reports:
            lines.extend(["", f"[{split}] Not generated (count is 0)"])
            continue
        counters, sources = split_reports[split]
        negative = counters["valid"] - counters["positive"]
        lines.extend(
            [
                "",
                f"[{split}]",
                f"Sentences: {counters['utterances']}",
                f"Speakers: {counters['speakers']}",
                f"Audio hours: {counters['audio_samples'] / args.target_sample_rate / 3600.0:.6f}",
                f"Total words: {counters['words']}",
                f"Valid targets: {counters['valid']}",
                f"Masked targets: {counters['masked']}",
                f"Sentences with positive pauses: {counters['positive_utterances']}",
                "Ratio of sentences with positive pauses: {:.8f}".format(
                    counters["positive_utterances"] / counters["utterances"]
                ),
                f"Positive pauses: {counters['positive']}",
                f"Negative pauses: {negative}",
                f"Positive-pause ratio: {(counters['positive'] / counters['valid'] if counters['valid'] else 0):.8f}",
                "Sentences by source: " + ", ".join(
                    f"{source_index}={sources[source_index]['utterances']}"
                    for source_index in sorted(sources)
                ),
            ]
        )
    statistics = "\n".join(lines) + "\n"
    (output_root / "training_feed_statistics.txt").write_text(statistics, encoding="utf-8")
    (output_root / "split_config.json").write_text(
        json.dumps(
            {"arguments": vars(args), "split_strategy": split_strategy},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(statistics, end="")


if __name__ == "__main__":
    main()
