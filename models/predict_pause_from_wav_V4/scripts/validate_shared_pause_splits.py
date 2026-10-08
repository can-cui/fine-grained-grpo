#!/usr/bin/env python3
"""校验 V2-V5 共享 split 的数量、来源、配对、泄漏与 SHA256 指纹。"""

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
        help="只读比对已有指纹；不创建或改写 shared_dataset_fingerprint.json",
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
                raise ValueError(f"{path}:{line_number}: key 行必须为 4 列")
            rows.append(parts[0])
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"key 文件为空或含重复 utt_id：{path}")
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
                raise ValueError(f"{path}:{line_number}: JSON 损坏") from exc
    return rows


def main():
    args = parse_args()
    root = Path(args.data_root).resolve()
    split_config_path = root / "shared_split_config.json"
    if not split_config_path.is_file():
        raise FileNotFoundError(f"缺少共享划分配置：{split_config_path}")
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
            raise FileNotFoundError(f"缺少共享数据审计文件：{metadata_path}")
        hashes[filename] = sha256_file(metadata_path)
    for split in SPLITS:
        lmdb_path = root / split
        key_path = root / f"{split}.key"
        manifest_path = root / f"{split}.jsonl"
        if not lmdb_path.is_dir() or not key_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"{split}: LMDB/key/jsonl 不完整")
        split_keys[split] = read_keys(key_path)
        split_manifests[split] = read_manifest(manifest_path)
        if len(split_keys[split]) != len(split_manifests[split]):
            raise ValueError(f"{split}: key 与 manifest 数量不同")
        hashes[f"{split}.key"] = sha256_file(key_path)
        hashes[f"{split}.jsonl"] = sha256_file(manifest_path)

    for left, right in M2_PAIRS:
        if split_keys[left] != split_keys[right]:
            raise ValueError(f"{left} 与 {right} 的 utt_id 或顺序不同")
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
            raise ValueError(f"{split}: 预期 {count} 条，实际 {len(split_keys[split])} 条")

    unique_eval_splits = (
        "valid_m2_manual", "test_m2_manual", "valid_lmdb_mfa", "test_lmdb_mfa"
    )
    train_ids = set(split_keys["train"])
    seen_eval = set()
    for split in unique_eval_splits:
        current = set(split_keys[split])
        if train_ids & current:
            raise ValueError(f"train 与 {split} 存在数据泄漏")
        if seen_eval & current:
            raise ValueError(f"唯一评测集合之间存在数据泄漏：{split}")
        seen_eval.update(current)
    expected_unique_eval = 2 * (args.expected_valid_count + args.expected_test_count)
    if len(seen_eval) != expected_unique_eval:
        raise ValueError(
            f"唯一评测 utt_id 应为 {expected_unique_eval}，实际 {len(seen_eval)}"
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
        raise ValueError(f"test_m2_manual: 存在非考试院来源，例如 {bad_test[:5]}")
    bad_valid = [
        row.get("utt_id") for row in split_manifests["valid_m2_manual"]
        if int(row.get("source_index", -1)) not in allowed_m2_sources
    ]
    if bad_valid:
        raise ValueError(f"valid_m2_manual: 存在非法来源，例如 {bad_valid[:5]}")
    bad_supplements = [
        row.get("utt_id") for row in split_manifests["valid_m2_manual"]
        if int(row.get("source_index", -1)) in supplement_sources
        and int(row.get("positive_target_count", 0)) > 0
    ]
    if bad_supplements:
        raise ValueError(
            f"valid_m2_manual: Gemini补充来源只能提供负例，例如 {bad_supplements[:5]}"
        )
    for split in ("valid_lmdb_mfa", "test_lmdb_mfa"):
        bad = [row.get("utt_id") for row in split_manifests[split]
               if not args.lmdb_source_index_min <= int(row.get("source_index", -1)) <= args.lmdb_source_index_max]
        if bad:
            raise ValueError(f"{split}: 存在 part00-04 之外来源，例如 {bad[:5]}")

    removed_rows = read_manifest(root / "train_ratio_removed_negative_records.jsonl")
    bad_removed = [
        row.get("utt_id") for row in removed_rows
        if bool(row.get("positive_utterance"))
        or not args.lmdb_source_index_min <= int(row.get("source_index", -1)) <= args.lmdb_source_index_max
    ]
    if bad_removed:
        raise ValueError(f"训练比例剔除清单包含正例或part00-04外样本，例如 {bad_removed[:5]}")
    removed_ids = {str(row.get("utt_id", "")) for row in removed_rows}
    if removed_ids & (train_ids | seen_eval):
        raise ValueError("训练比例剔除清单与最终 train/eval 存在重叠")

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
        raise ValueError(f"两批评测的对应 valid/test 正例句数不一致：{pair_ratio_checks}")
    valid_target_ratio = positive_counts["valid_m2_manual"] / args.expected_valid_count
    train_balance_mode = split_config.get("train_balance_mode", "")
    if train_balance_mode == "match_fallback_valid_ratio_by_removing_part00_04_negatives":
        train_delta = abs(train_positive - len(train_rows) * valid_target_ratio)
        if train_delta > args.ratio_tolerance_utterances:
            raise ValueError(
                f"回退模式下 train 与 valid_m2_manual 比例目标相差 {train_delta:.6f}，"
                f"超过容差 {args.ratio_tolerance_utterances}"
            )
        nearest_integer_checks = {}
    elif train_balance_mode == "preserve_natural_ratio_and_use_nearest_integer_eval_counts":
        if removed_rows:
            raise ValueError("正常补足模式不应删除任何训练负例")
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
                f"正常补足模式下评测正例数不是train比例的最邻近整数：{nearest_integer_checks}"
            )
        train_delta = 0.0
    else:
        raise ValueError(f"未知 train_balance_mode：{train_balance_mode!r}")
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
            raise FileNotFoundError(f"只读校验要求已有指纹：{output_path}")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != fingerprint_payload:
            raise ValueError(
                "共享数据当前统计/SHA256 与已有指纹不一致；V3-V5 禁止刷新指纹，"
                "请回到 V2 查明数据漂移"
            )
    else:
        output_path.write_text(
            json.dumps(fingerprint_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"ok": True, "report": str(output_path), **fingerprint_payload}, ensure_ascii=False))


if __name__ == "__main__":
    main()
