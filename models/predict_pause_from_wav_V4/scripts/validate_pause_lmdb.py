#!/usr/bin/env python3
"""Validate the pause-prediction LMDB schema without importing Fairseq."""

import argparse
import io
import json
from pathlib import Path

import lmdb
import numpy as np


REQUIRED_ARRAYS = (
    "waveform_pcm16",
    "sample_rate",
    "pool_start_sample",
    "pool_end_sample",
    "pool_start_seconds",
    "pool_end_seconds",
    "pause_label",
    "valid_mask",
    "word_list",
    "utt_id",
    "speaker_id",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--splits", default="train,valid,test")
    return parser.parse_args()


def load_keys(path):
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number}: expected four tab-separated fields")
            entries.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
    return entries


def validate_record(raw, key, expected_samples, expected_words, sample_rate):
    with np.load(io.BytesIO(raw), allow_pickle=False) as record:
        missing = sorted(set(REQUIRED_ARRAYS) - set(record.files))
        if missing:
            raise ValueError(f"{key}: missing arrays: {missing}")
        waveform = record["waveform_pcm16"]
        starts = record["pool_start_sample"]
        ends = record["pool_end_sample"]
        labels = record["pause_label"]
        mask = record["valid_mask"]
        if waveform.dtype != np.int16 or waveform.shape != (expected_samples,):
            raise ValueError(f"{key}: invalid waveform dtype/shape {waveform.dtype} {waveform.shape}")
        if int(record["sample_rate"].item()) != sample_rate:
            raise ValueError(f"{key}: sample_rate is not {sample_rate}")
        one_dimensional = (
            starts, ends, labels, mask, record["pool_start_seconds"],
            record["pool_end_seconds"], record["word_list"],
        )
        if any(array.ndim != 1 or len(array) != expected_words for array in one_dimensional):
            raise ValueError(f"{key}: word-level arrays do not match word count {expected_words}")
        if np.any(starts < 0) or np.any(starts >= ends) or np.any(ends > expected_samples):
            raise ValueError(f"{key}: pooling ranges are invalid")
        if not np.isin(labels, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{key}: labels and valid_mask must be binary")
        if int(mask.sum()) == 0:
            raise ValueError(f"{key}: no valid target")
        if int(mask[-1]) != 0 or np.any(mask[:-1] != 1):
            raise ValueError(f"{key}: only the final lexical word may be masked")
        if str(record["utt_id"].item()) != key:
            raise ValueError(f"{key}: embedded utt_id does not match LMDB key")
        expected_starts = np.rint(record["pool_start_seconds"] * sample_rate).astype(np.int64)
        expected_ends = np.rint(record["pool_end_seconds"] * sample_rate).astype(np.int64)
        if np.max(np.abs(expected_starts - starts)) > 1 or np.max(np.abs(expected_ends - ends)) > 1:
            raise ValueError(f"{key}: seconds and sample boundaries disagree")


def main():
    args = parse_args()
    root = Path(args.data_root).resolve()
    report = {"data_root": str(root), "splits": {}, "ok": False}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        lmdb_path, key_path = root / split, root / f"{split}.key"
        if not lmdb_path.exists() and not key_path.exists() and split == "test":
            report["splits"][split] = {"count": 0, "optional_missing": True}
            continue
        if not lmdb_path.is_dir() or not key_path.is_file():
            raise FileNotFoundError(f"missing LMDB/key pair for split {split}")
        entries = load_keys(key_path)
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, subdir=True)
        try:
            with env.begin() as txn:
                if txn.stat()["entries"] != len(entries):
                    raise ValueError(f"{split}: key count and LMDB entry count differ")
                for key, samples, words, _speaker in entries:
                    raw = txn.get(key.encode("utf-8"))
                    if raw is None:
                        raise KeyError(f"{split}: missing LMDB key {key}")
                    validate_record(raw, key, samples, words, args.sample_rate)
        finally:
            env.close()
        report["splits"][split] = {"count": len(entries)}
    report["ok"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
