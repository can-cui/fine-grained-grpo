#!/usr/bin/env python3
"""为 V2-V5 构造共享 train、考试院评测和原 LMDB MFA 评测 split。"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import prepare_pause_training_data as base
from prepare_pause_shared_data_v2 import main


FINAL_BASE_SPLITS = (
    "train",
    "valid_m2_manual",
    "test_m2_manual",
    "valid_lmdb_mfa",
    "test_lmdb_mfa",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--m2-source-index", type=int, default=27)
    parser.add_argument("--lmdb-source-index-min", type=int, default=0)
    parser.add_argument("--lmdb-source-index-max", type=int, default=24)
    parser.add_argument("--valid-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--source-sample-rate", type=int, default=24000)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_half_up(value):
    return int(math.floor(float(value) + 0.5))


def stratified_take(rows, count, positive_count, seed, label):
    """按 source_index 分层，从正负句中分别抽取精确数量。"""
    if count < 0 or positive_count < 0 or positive_count > count:
        raise ValueError(f"{label}: 非法抽样数量 count={count}, positive={positive_count}")
    grouped = {True: defaultdict(list), False: defaultdict(list)}
    for row in rows:
        grouped[bool(row["positive_utterance"])][row["source_index"]].append(row)
    targets = {True: positive_count, False: count - positive_count}
    selected = []
    for offset, status in enumerate((True, False)):
        available = sum(len(values) for values in grouped[status].values())
        if targets[status] > available:
            raise ValueError(
                f"{label}: {'正' if status else '负'}停顿句需要 {targets[status]} 条，"
                f"实际只有 {available} 条"
            )
        allocation = base.allocate_strata(grouped[status], targets[status])
        for source_index in sorted(grouped[status]):
            candidates = list(grouped[status][source_index])
            random.Random(seed + offset * 100003 + source_index).shuffle(candidates)
            selected.extend(candidates[:allocation[source_index]])
    random.Random(seed + 999983).shuffle(selected)
    selected_ids = {row["utt_id"] for row in selected}
    if len(selected_ids) != count:
        raise AssertionError(f"{label}: 抽样后 utt_id 数量异常")
    return selected, [row for row in rows if row["utt_id"] not in selected_ids]


def select_eval_pool(rows, valid_count, test_count, positive_ratio, seed, label):
    test_positive = stable_half_up(test_count * positive_ratio)
    valid_positive = stable_half_up(valid_count * positive_ratio)
    test_rows, remaining = stratified_take(
        rows, test_count, test_positive, seed + 1, f"{label}/test"
    )
    valid_rows, remaining = stratified_take(
        remaining, valid_count, valid_positive, seed + 2, f"{label}/valid"
    )
    return valid_rows, test_rows, remaining, {
        "valid_positive_target": valid_positive,
        "test_positive_target": test_positive,
    }


def sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def legacy_main():
    args = parse_args()
    if args.valid_count <= 0 or args.test_count <= 0:
        raise ValueError("valid-count 和 test-count 必须大于 0")
    if args.lmdb_source_index_min > args.lmdb_source_index_max:
        raise ValueError("LMDB source_index 范围非法")
    manifest = Path(args.merged_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"找不到合并 manifest：{manifest}")
    output_root.mkdir(parents=True, exist_ok=True)

    input_records, skipped_records = base.load_manifest(manifest)
    records, pre_split_skipped = base.filter_splittable_records(input_records)
    skipped_records.extend(pre_split_skipped)
    if not records:
        raise ValueError("过滤后没有可划分记录")
    rows = base.build_split_rows(records)
    positive_count = sum(bool(row["positive_utterance"]) for row in rows)
    global_positive_ratio = positive_count / len(rows)

    m2_rows = [row for row in rows if row["source_index"] == args.m2_source_index]
    lmdb_rows = [
        row for row in rows
        if args.lmdb_source_index_min <= row["source_index"] <= args.lmdb_source_index_max
    ]
    required = args.valid_count + args.test_count
    if len(m2_rows) < required or len(lmdb_rows) < required:
        raise ValueError(
            f"评测候选不足：m2={len(m2_rows)}, lmdb={len(lmdb_rows)}, 每组需要={required}"
        )

    valid_m2_manual_rows, test_m2_manual_rows, _m2_remaining, m2_targets = select_eval_pool(
        m2_rows, args.valid_count, args.test_count, global_positive_ratio,
        args.split_seed + 1000, "m2_manual",
    )
    valid_lmdb, test_lmdb, _lmdb_remaining, lmdb_targets = select_eval_pool(
        lmdb_rows, args.valid_count, args.test_count, global_positive_ratio,
        args.split_seed + 2000, "lmdb_mfa",
    )
    eval_rows = [
        *valid_m2_manual_rows,
        *test_m2_manual_rows,
        *valid_lmdb,
        *test_lmdb,
    ]
    eval_ids = [row["utt_id"] for row in eval_rows]
    if len(eval_ids) != len(set(eval_ids)):
        raise AssertionError("考试院和 part00-04 评测集合出现 utt_id 重叠")
    eval_id_set = set(eval_ids)
    train_rows = [row for row in rows if row["utt_id"] not in eval_id_set]
    if not train_rows:
        raise ValueError("扣除评测集后 train 为空")

    split_map = {
        "train": [row["record"] for row in train_rows],
        "valid_m2_manual": [row["record"] for row in valid_m2_manual_rows],
        "test_m2_manual": [row["record"] for row in test_m2_manual_rows],
        "valid_lmdb_mfa": [row["record"] for row in valid_lmdb],
        "test_lmdb_mfa": [row["record"] for row in test_lmdb],
    }
    skipped_path = output_root / "prepare_skipped_records.jsonl"
    base.write_skipped_records(skipped_path, skipped_records)
    reports = {}
    for split in FINAL_BASE_SPLITS:
        result = base.write_split(
            output_root, split, split_map[split], args, skipped_records, skipped_path
        )
        if result is None:
            raise AssertionError(f"{split}: 未生成")
        counters, sources = result
        reports[split] = {
            "counters": dict(counters),
            "sources": {str(key): dict(value) for key, value in sources.items()},
        }

    expected_counts = {
        "valid_m2_manual": args.valid_count,
        "test_m2_manual": args.test_count,
        "valid_lmdb_mfa": args.valid_count,
        "test_lmdb_mfa": args.test_count,
    }
    wrong_counts = {
        split: reports[split]["counters"].get("utterances", 0)
        for split, expected in expected_counts.items()
        if reports[split]["counters"].get("utterances", 0) != expected
    }
    if wrong_counts:
        raise RuntimeError(
            f"评测 split 在音频编码阶段发生跳过，无法保持固定数量：{wrong_counts}；"
            "请检查 prepare_skipped_records.jsonl 后修复源记录"
        )

    train_positive_ratio = (
        reports["train"]["counters"]["positive_utterances"]
        / reports["train"]["counters"]["utterances"]
    )
    split_config = {
        "format_version": 1,
        "split_seed": args.split_seed,
        "valid_count": args.valid_count,
        "test_count": args.test_count,
        "m2_source_index": args.m2_source_index,
        "lmdb_source_index_range": [args.lmdb_source_index_min, args.lmdb_source_index_max],
        "sampling_positive_utterance_ratio": global_positive_ratio,
        "formal_splits": [
            "train",
            "valid_m2_manual",
            "valid_m2_mfa",
            "valid_lmdb_mfa",
            "test_m2_manual",
            "test_m2_mfa",
            "test_lmdb_mfa",
        ],
    }
    (output_root / "shared_split_config.json").write_text(
        json.dumps(split_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "format_version": 1,
        "arguments": vars(args),
        "merged_manifest": str(manifest),
        "output_root": str(output_root),
        "split_seed": args.split_seed,
        "global_positive_utterance_ratio": global_positive_ratio,
        "train_positive_utterance_ratio": train_positive_ratio,
        "m2_source_index": args.m2_source_index,
        "lmdb_source_index_range": [args.lmdb_source_index_min, args.lmdb_source_index_max],
        "m2_targets": m2_targets,
        "lmdb_targets": lmdb_targets,
        "unique_eval_utterances": len(eval_ids),
        "eval_utt_id_sha256": sha256_lines(sorted(eval_ids)),
        "reports": reports,
        "split_config": split_config,
        "skipped_records": len(skipped_records),
    }
    (output_root / "shared_split_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    from prepare_pause_shared_data_v2 import main as main_v2
    main_v2()
