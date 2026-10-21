#!/usr/bin/env python3
"""Validate counts, sources, pairing, leakage, and SHA256 fingerprints for V2-V5 shared splits."""

import argparse
import hashlib
import json
import math
from pathlib import Path


SPLITS = (
    "train",
    "valid_m2_manual",
    "valid_m2_mfa",
    "valid_lmdb_mfa",
    "test_m2_manual",
    "test_m2_mfa",
    "test_lmdb_mfa",
)
M2_PAIRS = (
    ("valid_m2_manual", "valid_m2_mfa"),
    ("test_m2_manual", "test_m2_mfa"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-valid-count", type=int, default=500)
    parser.add_argument("--expected-test-count", type=int, default=100)
    parser.add_argument("--m2-source-index", type=int, default=27)
    parser.add_argument("--m2-supplement-source-indices", default="25,26")
    parser.add_argument("--lmdb-source-index-min", type=int, default=0)
    parser.add_argument("--lmdb-source-index-max", type=int, default=24)
    parser.add_argument("--ratio-tolerance-utterances", type=int, default=1)
    parser.add_argument(
        "--verify-existing-fingerprint",
        action="store_true",
        help="Read-only comparison with the existing fingerprint; do not create or modify shared_dataset_fingerprint.json",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_keys(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number}: key row must have 4 columns")
            rows.append(parts[0])
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"Key file is empty or contains duplicate utt_id values: {path}")
    return rows


def read_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def main():
    args = parse_args()
    root = Path(args.data_root).resolve()
    split_config_path = root / "shared_split_config.json"
    if not split_config_path.is_file():
        raise FileNotFoundError(f"Missing shared split configuration: {split_config_path}")
    split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
    split_keys = {}
    split_manifests = {}
    hashes = {}
    metadata_files = (
        "shared_split_config.json",
        "shared_split_statistics.json",
        "pause_label_dataset_comparison.json",
        "prepare_skipped_records.jsonl",
        "train_ratio_removed_negative_records.jsonl",
    )
    for filename in metadata_files:
        metadata_path = root / filename
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing shared-data audit file: {metadata_path}")
        hashes[filename] = sha256_file(metadata_path)
    for split in SPLITS:
        lmdb_path = root / split
        key_path = root / f"{split}.key"
        manifest_path = root / f"{split}.jsonl"
        if not lmdb_path.is_dir() or not key_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"{split}: LMDB/key/jsonl is incomplete")
        split_keys[split] = read_keys(key_path)
        split_manifests[split] = read_manifest(manifest_path)
        if len(split_keys[split]) != len(split_manifests[split]):
            raise ValueError(f"{split}: key and manifest counts differ")
        hashes[f"{split}.key"] = sha256_file(key_path)
        hashes[f"{split}.jsonl"] = sha256_file(manifest_path)

    for left, right in M2_PAIRS:
        if split_keys[left] != split_keys[right]:
            raise ValueError(f"utt_id values or order differ between {left} and {right}")
    expected = {
        "valid_m2_manual": args.expected_valid_count,
        "valid_m2_mfa": args.expected_valid_count,
        "valid_lmdb_mfa": args.expected_valid_count,
        "test_m2_manual": args.expected_test_count,
        "test_m2_mfa": args.expected_test_count,
        "test_lmdb_mfa": args.expected_test_count,
    }
    for split, count in expected.items():
        if len(split_keys[split]) != count:
            raise ValueError(f"{split}: expected {count} records, got {len(split_keys[split])}")

    unique_eval_splits = (
        "valid_m2_manual", "test_m2_manual", "valid_lmdb_mfa", "test_lmdb_mfa"
    )
    train_ids = set(split_keys["train"])
    seen_eval = set()
    for split in unique_eval_splits:
        current = set(split_keys[split])
        if train_ids & current:
            raise ValueError(f"Data leakage between train and {split}")
        if seen_eval & current:
            raise ValueError(f"Data leakage between unique evaluation sets: {split}")
        seen_eval.update(current)
    expected_unique_eval = 2 * (args.expected_valid_count + args.expected_test_count)
    if len(seen_eval) != expected_unique_eval:
        raise ValueError(
            f"Expected {expected_unique_eval} unique evaluation utt_id values, got {len(seen_eval)}"
        )

    supplement_sources = {
        int(token.strip())
        for token in args.m2_supplement_source_indices.split(",")
        if token.strip()
    }
    allowed_m2_sources = {args.m2_source_index, *supplement_sources}
    bad_test = [
        row.get("utt_id") for row in split_manifests["test_m2_manual"]
        if int(row.get("source_index", -1)) != args.m2_source_index
    ]
    if bad_test:
        raise ValueError(f"test_m2_manual: contains non-examination-authority sources, e.g. {bad_test[:5]}")
    bad_valid = [
        row.get("utt_id") for row in split_manifests["valid_m2_manual"]
        if int(row.get("source_index", -1)) not in allowed_m2_sources
    ]
    if bad_valid:
        raise ValueError(f"valid_m2_manual: contains invalid sources, e.g. {bad_valid[:5]}")
    bad_supplements = [
        row.get("utt_id") for row in split_manifests["valid_m2_manual"]
        if int(row.get("source_index", -1)) in supplement_sources
        and int(row.get("positive_target_count", 0)) > 0
    ]
    if bad_supplements:
        raise ValueError(
            f"valid_m2_manual: Gemini supplement sources may only provide negative examples, e.g. {bad_supplements[:5]}"
        )
    for split in ("valid_lmdb_mfa", "test_lmdb_mfa"):
        bad = [row.get("utt_id") for row in split_manifests[split]
               if not args.lmdb_source_index_min <= int(row.get("source_index", -1)) <= args.lmdb_source_index_max]
        if bad:
            raise ValueError(f"{split}: contains sources outside part00-04, e.g. {bad[:5]}")

    removed_rows = read_manifest(root / "train_ratio_removed_negative_records.jsonl")
    bad_removed = [
        row.get("utt_id") for row in removed_rows
        if bool(row.get("positive_utterance"))
        or not args.lmdb_source_index_min <= int(row.get("source_index", -1)) <= args.lmdb_source_index_max
    ]
    if bad_removed:
        raise ValueError(f"Training-ratio removal list contains positive examples or samples outside part00-04, e.g. {bad_removed[:5]}")
    removed_ids = {str(row.get("utt_id", "")) for row in removed_rows}
    if removed_ids & (train_ids | seen_eval):
        raise ValueError("Training-ratio removal list overlaps with the final train/eval sets")

    train_rows = split_manifests["train"]
    train_positive = sum(int(row.get("positive_target_count", 0)) > 0 for row in train_rows)
    train_ratio = train_positive / len(train_rows)
    positive_counts = {
        split: sum(
            int(row.get("positive_target_count", 0)) > 0
            for row in split_manifests[split]
        )
        for split in ("valid_m2_manual", "test_m2_manual", "valid_lmdb_mfa", "test_lmdb_mfa")
    }
    pair_ratio_checks = {
        "valid": abs(positive_counts["valid_m2_manual"] - positive_counts["valid_lmdb_mfa"]),
        "test": abs(positive_counts["test_m2_manual"] - positive_counts["test_lmdb_mfa"]),
    }
    if any(delta > args.ratio_tolerance_utterances for delta in pair_ratio_checks.values()):
        raise ValueError(f"Positive sentence counts differ between corresponding valid/test evaluation pairs: {pair_ratio_checks}")
    valid_target_ratio = positive_counts["valid_m2_manual"] / args.expected_valid_count
    train_balance_mode = split_config.get("train_balance_mode", "")
    if train_balance_mode == "match_fallback_valid_ratio_by_removing_part00_04_negatives":
        train_delta = abs(train_positive - len(train_rows) * valid_target_ratio)
        if train_delta > args.ratio_tolerance_utterances:
            raise ValueError(
                f"In fallback mode, the train ratio differs from the valid_m2_manual target by {train_delta:.6f}, "
                f"exceeding the tolerance of {args.ratio_tolerance_utterances}"
            )
        nearest_integer_checks = {}
    elif train_balance_mode == "preserve_natural_ratio_and_use_nearest_integer_eval_counts":
        if removed_rows:
            raise ValueError("Normal fill mode should not remove any training negatives")
        expected_valid_positive = int(math.floor(args.expected_valid_count * train_ratio + 0.5))
        expected_test_positive = int(math.floor(args.expected_test_count * train_ratio + 0.5))
        nearest_integer_checks = {
            "valid_m2_manual": abs(
                positive_counts["valid_m2_manual"] - expected_valid_positive
            ),
            "test_m2_manual": abs(
                positive_counts["test_m2_manual"] - expected_test_positive
            ),
        }
        if any(
            delta > args.ratio_tolerance_utterances
            for delta in nearest_integer_checks.values()
        ):
            raise ValueError(
                f"In normal fill mode, evaluation positive counts are not the nearest integers to the train ratio: {nearest_integer_checks}"
            )
        train_delta = 0.0
    else:
        raise ValueError(f"Unknown train_balance_mode: {train_balance_mode!r}")
    ratio_checks = {
        "positive_counts": positive_counts,
        "paired_eval_positive_count_delta": pair_ratio_checks,
        "valid_target_ratio": valid_target_ratio,
        "train_positive_utterances": train_positive,
        "train_utterances": len(train_rows),
        "train_delta_from_valid_expected_count": train_delta,
        "train_balance_mode": train_balance_mode,
        "nearest_integer_eval_count_delta": nearest_integer_checks,
        "removed_part00_04_negative_utterances": len(removed_rows),
    }

    fingerprint_payload = {
        "format_version": 2,
        "splits": {split: len(split_keys[split]) for split in SPLITS},
        "hashes": hashes,
        "train_positive_utterance_ratio": train_ratio,
        "ratio_checks": ratio_checks,
        "unique_eval_utterances": len(seen_eval),
    }
    canonical = json.dumps(
        fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint_payload["dataset_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path = root / "shared_dataset_fingerprint.json"
    if args.verify_existing_fingerprint:
        if not output_path.is_file():
            raise FileNotFoundError(f"Read-only validation requires an existing fingerprint: {output_path}")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != fingerprint_payload:
            raise ValueError(
                "Current shared-data statistics/SHA256 do not match the existing fingerprint; V3-V5 must not refresh it. "
                "Return to V2 to investigate the data drift."
            )
    else:
        output_path.write_text(
            json.dumps(fingerprint_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"ok": True, "report": str(output_path), **fingerprint_payload}, ensure_ascii=False))


if __name__ == "__main__":
    main()
