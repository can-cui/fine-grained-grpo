#!/usr/bin/env python3
"""Create comparison LMDBs with manual labels and MFA 30 ms labels for the same valid/test samples."""

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-manifest", required=True)
    parser.add_argument("--source-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--extract-script", required=True)
    parser.add_argument("--protobuf-python-dir", required=True)
    parser.add_argument("--phone-vocab", required=True)
    parser.add_argument("--pause-threshold-seconds", type=float, default=0.03)
    parser.add_argument("--expected-source-indices", default="25,26,27")
    parser.add_argument("--expected-test-source-index", type=int, default=27)
    parser.add_argument("--expected-valid-count", type=int, default=500)
    parser.add_argument("--expected-test-count", type=int, default=100)
    parser.add_argument("--source-valid-split", default="valid_m2_manual")
    parser.add_argument("--source-test-split", default="test_m2_manual")
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_extraction_runtime(extract_script, protobuf_python_dir, phone_vocab):
    script_path = Path(extract_script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Original extraction script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("pause_batch_extract", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datum_class = module.load_datum_class(protobuf_python_dir)
    vocab = module.load_vocab(phone_vocab)
    return module, datum_class, vocab


def read_key_file(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number}: key row does not have 4 columns")
            rows.append(parts)
    if not rows:
        raise ValueError(f"Key file is empty: {path}")
    keys = [row[0] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Key file contains duplicate utt_id values: {path}")
    return rows


def load_selected_manifest(manifest_path, wanted_ids):
    selected = {}
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            utt_id = str(record.get("utt_id", ""))
            if utt_id not in wanted_ids:
                continue
            if utt_id in selected:
                raise ValueError(f"Duplicate utt_id in merged manifest: {utt_id}")
            selected[utt_id] = record
            if len(selected) == len(wanted_ids):
                break
    missing = sorted(wanted_ids - set(selected))
    if missing:
        raise KeyError(f"Merged manifest is missing {len(missing)} evaluation samples, e.g. {missing[:5]}")
    return selected


def collapse_mfa_labels(words, raw_labels, pause_symbols):
    lexical_words, lexical_labels = [], []
    for word, label in zip(words, raw_labels):
        if word in pause_symbols:
            if lexical_labels:
                lexical_labels[-1] = max(lexical_labels[-1], label)
            continue
        lexical_words.append(str(word))
        lexical_labels.append(int(label))
    return lexical_words, np.asarray(lexical_labels, dtype=np.uint8)


def load_mfa_labels(selected_manifest, extract_module, datum_class, vocab, threshold):
    """Read back only the 600 selected source Datum records and reuse the official extraction function to generate MFA labels from phonemes/fa."""
    environments = {}
    results = {}
    try:
        for utt_id, record in selected_manifest.items():
            lmdb_spec = str(record.get("source_lmdb_spec", ""))
            source_key = str(record.get("source_lmdb_key", ""))
            lmdb_path, _ = extract_module.split_lmdb_spec(lmdb_spec)
            if not lmdb_path or not source_key:
                raise ValueError(f"{utt_id}: missing source_lmdb_spec/source_lmdb_key")
            if lmdb_path not in environments:
                environments[lmdb_path] = lmdb.open(
                    lmdb_path, readonly=True, lock=False, readahead=False, subdir=True
                )
            raw = environments[lmdb_path].begin().get(source_key.encode("utf-8"))
            if raw is None:
                raise KeyError(f"{utt_id}: key={source_key} not found in source LMDB")
            datum = datum_class()
            datum.ParseFromString(raw)
            (
                _wav,
                words,
                raw_labels,
                _word_end_time,
                _pause_durations,
                _text,
                _alignment_duration,
                _hash_marker_count,
                _phoneme_storage_layout,
                _leading_silence_end_seconds,
            ) = extract_module.process_datum(
                datum, vocab, threshold, decode_wav=False
            )
            results[utt_id] = collapse_mfa_labels(
                words, raw_labels, set(extract_module.PAUSE_SYMBOLS)
            )
    finally:
        for environment in environments.values():
            environment.close()
    return results


def decode_npz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as record:
        return {name: record[name].copy() for name in record.files}


def encode_npz(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def remove_split(output_root, split):
    lmdb_path = output_root / split
    if lmdb_path.exists():
        if not lmdb_path.is_dir():
            raise ValueError(f"Expected LMDB to be a directory, got: {lmdb_path}")
        shutil.rmtree(str(lmdb_path))
    for suffix in (".key", ".jsonl"):
        path = output_root / f"{split}{suffix}"
        if path.exists():
            path.unlink()


def build_split(
    source_root,
    output_root,
    source_split,
    key_rows,
    mfa_labels_by_id,
    map_size_bytes,
    overwrite,
):
    if not source_split.endswith("_manual"):
        raise ValueError(f"Manual-label source split must end in _manual: {source_split}")
    output_split = source_split[:-len("_manual")] + "_mfa"
    output_lmdb = output_root / output_split
    output_key = output_root / f"{output_split}.key"
    output_manifest = output_root / f"{output_split}.jsonl"
    if any(path.exists() for path in (output_lmdb, output_key, output_manifest)):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_split}; use --overwrite after confirmation")
        remove_split(output_root, output_split)

    source_env = lmdb.open(
        str(source_root / source_split), readonly=True, lock=False, readahead=False, subdir=True
    )
    output_env = lmdb.open(
        str(output_lmdb), map_size=map_size_bytes, subdir=True, create=True, lock=True
    )
    counters = Counter()
    rows = []
    transaction = output_env.begin(write=True)
    try:
        source_txn = source_env.begin()
        for index, key_row in enumerate(key_rows, 1):
            utt_id = key_row[0]
            raw = source_txn.get(utt_id.encode("utf-8"))
            if raw is None:
                raise KeyError(f"{source_split} LMDB is missing key: {utt_id}")
            arrays = decode_npz(raw)
            manual_labels = arrays["pause_label"].astype(np.uint8, copy=True)
            valid_mask = arrays["valid_mask"].astype(bool)
            lexical_words, mfa_labels = mfa_labels_by_id[utt_id]
            lmdb_words = arrays["word_list"].astype(str).tolist()
            if lexical_words != lmdb_words:
                raise ValueError(f"{utt_id}: lexical word_list in the manifest does not match the existing LMDB")
            if len(mfa_labels) != len(manual_labels):
                raise ValueError(f"{utt_id}: MFA and manual label lengths differ")

            output_labels = mfa_labels
            arrays["pause_label"] = mfa_labels
            if not transaction.put(utt_id.encode("utf-8"), encode_npz(arrays), overwrite=False):
                raise ValueError(f"{output_split}: duplicate key: {utt_id}")

            manual_valid = manual_labels.astype(bool) & valid_mask
            mfa_valid = mfa_labels.astype(bool) & valid_mask
            counters["utterances"] += 1
            counters["valid_targets"] += int(valid_mask.sum())
            counters["manual_positive_targets"] += int(manual_valid.sum())
            counters["mfa_positive_targets"] += int(mfa_valid.sum())
            counters["both_positive_targets"] += int((manual_valid & mfa_valid).sum())
            counters["manual_only_targets"] += int((manual_valid & ~mfa_valid).sum())
            counters["mfa_only_targets"] += int((~manual_valid & mfa_valid).sum())
            counters["manual_positive_utterances"] += int(manual_valid.any())
            counters["mfa_positive_utterances"] += int(mfa_valid.any())
            rows.append(
                json.dumps(
                    {
                        "utt_id": utt_id,
                        "label_mode": "mfa",
                        "source_index": int(arrays["source_index"].item()),
                        "speaker_id": str(arrays["speaker_id"].item()),
                        "valid_target_count": int(valid_mask.sum()),
                        "positive_target_count": int((output_labels.astype(bool) & valid_mask).sum()),
                        "manual_positive_target_count": int(manual_valid.sum()),
                        "mfa_positive_target_count": int(mfa_valid.sum()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 200 == 0:
                transaction.commit()
                transaction = output_env.begin(write=True)
        transaction.commit()
        transaction = None
        output_env.sync()
    finally:
        if transaction is not None:
            transaction.abort()
        output_env.close()
        source_env.close()

    output_key.write_text(
        "".join("\t".join(row) + "\n" for row in key_rows), encoding="utf-8"
    )
    output_manifest.write_text("".join(rows), encoding="utf-8")
    return dict(counters)


def main():
    args = parse_args()
    if args.pause_threshold_seconds < 0:
        raise ValueError("pause-threshold-seconds cannot be negative")
    if args.map_size_gb <= 0:
        raise ValueError("map-size-gb must be greater than 0")
    manifest_path = Path(args.merged_manifest).resolve()
    source_root = Path(args.source_data_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Merged manifest not found: {manifest_path}")

    source_splits = (args.source_valid_split, args.source_test_split)
    if len(set(source_splits)) != 2:
        raise ValueError("source-valid-split and source-test-split must differ")
    split_rows = {}
    wanted_ids = set()
    for split in source_splits:
        if not (source_root / split).is_dir():
            raise FileNotFoundError(f"Original split LMDB not found: {source_root / split}")
        split_rows[split] = read_key_file(source_root / f"{split}.key")
        wanted_ids.update(row[0] for row in split_rows[split])
    expected_counts = {
        args.source_valid_split: args.expected_valid_count,
        args.source_test_split: args.expected_test_count,
    }
    for split, rows in split_rows.items():
        if expected_counts[split] >= 0 and len(rows) != expected_counts[split]:
            raise ValueError(
                f"Expected {expected_counts[split]} records in original {split}, got {len(rows)}; "
                "stopped to avoid comparing mismatched data"
            )
    selected_manifest = load_selected_manifest(manifest_path, wanted_ids)
    allowed_sources = {
        int(token.strip())
        for token in args.expected_source_indices.split(",")
        if token.strip()
    }
    if not allowed_sources:
        raise ValueError("expected-source-indices cannot be empty")
    wrong_sources = [
        utt_id
        for utt_id, record in selected_manifest.items()
        if int(record.get("source_index", -1)) not in allowed_sources
    ]
    non_manual = [
        utt_id
        for utt_id, record in selected_manifest.items()
        if record.get("pause_label_source") != "xlsx_second_column"
    ]
    if wrong_sources:
        raise ValueError(
            f"{len(wrong_sources)} evaluation samples are not from the allowed sources="
            f"{sorted(allowed_sources)}, e.g. {wrong_sources[:5]}"
        )
    test_ids = {row[0] for row in split_rows[args.source_test_split]}
    wrong_test_sources = [
        utt_id for utt_id in test_ids
        if int(selected_manifest[utt_id].get("source_index", -1))
        != args.expected_test_source_index
    ]
    if wrong_test_sources:
        raise ValueError(
            f"All test samples must come from source_index={args.expected_test_source_index}; "
            f"examples of violations: {wrong_test_sources[:5]}"
        )
    if non_manual:
        raise ValueError(
            f"{len(non_manual)} evaluation samples do not use manual Excel labels, e.g. {non_manual[:5]}"
        )
    extract_module, datum_class, vocab = load_extraction_runtime(
        args.extract_script, args.protobuf_python_dir, args.phone_vocab
    )
    mfa_labels_by_id = load_mfa_labels(
        selected_manifest,
        extract_module,
        datum_class,
        vocab,
        args.pause_threshold_seconds,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    pairing_checks = {}
    for source_split in source_splits:
        output_split = source_split[:-len("_manual")] + "_mfa"
        summaries[output_split] = build_split(
            source_root,
            output_root,
            source_split,
            split_rows[source_split],
            mfa_labels_by_id,
            args.map_size_gb * 1024 ** 3,
            args.overwrite,
        )
        paired_ids = [row[0] for row in split_rows[source_split]]
        pairing_checks[source_split] = {
            "manual_split": source_split,
            "mfa_split": output_split,
            "same_utt_id_order": True,
            "utterances": len(paired_ids),
            "utt_id_sha256": hashlib.sha256(
                ("\n".join(paired_ids) + "\n").encode("utf-8")
            ).hexdigest(),
        }

    comparison = {
        "source_data_root": str(source_root),
        "merged_manifest": str(manifest_path),
        "pause_threshold_seconds": args.pause_threshold_seconds,
        "expected_source_indices": sorted(allowed_sources),
        "expected_test_source_index": args.expected_test_source_index,
        "splits": summaries,
        "pairing_checks": pairing_checks,
    }
    summary_path = output_root / "pause_label_dataset_comparison.json"
    summary_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "summary": str(summary_path), **comparison}, ensure_ascii=False))


if __name__ == "__main__":
    main()
