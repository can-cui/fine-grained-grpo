#!/usr/bin/env python3
"""按500/100新规则构造 V2-V5 共享双标签评测数据。"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import prepare_pause_training_data as base


FINAL_BASE_SPLITS = (
    "train", "valid_m2_manual", "test_m2_manual",
    "valid_lmdb_mfa", "test_lmdb_mfa",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--m2-source-index", type=int, default=27)
    parser.add_argument("--m2-supplement-source-indices", default="25,26")
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


def parse_source_indices(value):
    values = [int(token.strip()) for token in str(value).split(",") if token.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("m2-supplement-source-indices 必须是互不重复的逗号分隔整数")
    return values


def stable_half_up(value):
    return int(math.floor(float(value) + 0.5))


def stratified_take(rows, count, positive_count, seed, label):
    """按 source_index 分层，从正负句中分别抽取精确数量。"""
    if count < 0 or positive_count < 0 or positive_count > count:
        raise ValueError(f"{label}: 非法抽样数量 count={count}, positive={positive_count}")
    grouped = {True: defaultdict(list), False: defaultdict(list)}
    for row in rows:
        grouped[bool(row["positive_utterance"])][row["source_index"]].append(row)
    selected = []
    for offset, status in enumerate((True, False)):
        target = positive_count if status else count - positive_count
        available = sum(len(values) for values in grouped[status].values())
        if target > available:
            raise ValueError(
                f"{label}: {'正' if status else '负'}停顿句需要 {target} 条，实际只有 {available} 条"
            )
        allocation = base.allocate_strata(grouped[status], target)
        for source_index in sorted(grouped[status]):
            candidates = list(grouped[status][source_index])
            random.Random(seed + offset * 100003 + source_index).shuffle(candidates)
            selected.extend(candidates[:allocation[source_index]])
    random.Random(seed + 999983).shuffle(selected)
    selected_ids = {row["utt_id"] for row in selected}
    if len(selected_ids) != count:
        raise AssertionError(f"{label}: 抽样后 utt_id 数量异常")
    return selected, [row for row in rows if row["utt_id"] not in selected_ids]


def shuffled_take(rows, count, seed, label):
    if count < 0 or count > len(rows):
        raise ValueError(f"{label}: 需要 {count} 条，实际只有 {len(rows)} 条")
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def select_m2_splits(rows, valid_count, test_count, positive_ratio, main_source,
                     supplement_sources, seed):
    """test只取考试院；valid负例按27、25、26顺序补足，不足时使用最接近比例。"""
    main_rows = [row for row in rows if row["source_index"] == main_source]
    test_positive_target = stable_half_up(test_count * positive_ratio)
    test_rows, main_remaining = stratified_take(
        main_rows, test_count, test_positive_target, seed + 1, "m2_manual/test"
    )

    desired_valid_positive = stable_half_up(valid_count * positive_ratio)
    desired_valid_negative = valid_count - desired_valid_positive
    negative_selected = []
    negative_by_source = {}
    remaining_negative = desired_valid_negative
    for offset, source_index in enumerate([main_source, *supplement_sources]):
        source_rows = main_remaining if source_index == main_source else rows
        candidates = [
            row for row in source_rows
            if row["source_index"] == source_index and not row["positive_utterance"]
        ]
        take_count = min(remaining_negative, len(candidates))
        chosen = shuffled_take(
            candidates, take_count, seed + 100 + offset,
            f"m2_manual/valid/source_{source_index}/negative",
        )
        negative_selected.extend(chosen)
        negative_by_source[str(source_index)] = len(chosen)
        remaining_negative -= len(chosen)
        if remaining_negative == 0:
            break

    fallback_used = remaining_negative > 0
    actual_valid_negative = len(negative_selected)
    actual_valid_positive = valid_count - actual_valid_negative
    positive_candidates = [row for row in main_remaining if row["positive_utterance"]]
    positive_selected = shuffled_take(
        positive_candidates, actual_valid_positive, seed + 500,
        "m2_manual/valid/Kaoshiyuan positive",
    )
    valid_rows = [*positive_selected, *negative_selected]
    random.Random(seed + 700).shuffle(valid_rows)
    if len({row["utt_id"] for row in [*valid_rows, *test_rows]}) != valid_count + test_count:
        raise AssertionError("m2_manual valid/test 出现重复 utt_id")
    return valid_rows, test_rows, {
        "reference_positive_ratio": positive_ratio,
        "test_positive_target": test_positive_target,
        "test_negative_target": test_count - test_positive_target,
        "test_positive_actual": sum(bool(row["positive_utterance"]) for row in test_rows),
        "valid_positive_target": desired_valid_positive,
        "valid_negative_target": desired_valid_negative,
        "valid_positive_actual": actual_valid_positive,
        "valid_negative_actual": actual_valid_negative,
        "valid_positive_ratio_actual": actual_valid_positive / valid_count,
        "valid_negative_sources": negative_by_source,
        "closest_feasible_fallback_used": fallback_used,
        "unfilled_negative_target_before_fallback": remaining_negative,
    }


def select_matching_lmdb_splits(rows, valid_count, test_count, valid_positive_count,
                                test_positive_count, seed):
    test_rows, remaining = stratified_take(
        rows, test_count, test_positive_count, seed + 1, "lmdb_mfa/test"
    )
    valid_rows, _ = stratified_take(
        remaining, valid_count, valid_positive_count, seed + 2, "lmdb_mfa/valid"
    )
    return valid_rows, test_rows, {
        "valid_positive_target": valid_positive_count,
        "test_positive_target": test_positive_count,
    }


def balance_train_by_removing_part00_04_negatives(rows, target_ratio, source_min,
                                                   source_max, seed):
    """仅删除 part00-04 负例，使 train 正例比例最接近 valid_m2_manual。"""
    positive_count = sum(bool(row["positive_utterance"]) for row in rows)
    current_ratio = positive_count / len(rows)
    current_target_delta = abs(positive_count - len(rows) * target_ratio)
    if current_target_delta <= 1.0:
        return rows, [], {
            "before_ratio": current_ratio, "after_ratio": current_ratio,
            "target_ratio": target_ratio, "removed_negative_utterances": 0,
            "removed_source_counts": {},
            "target_positive_count_delta": current_target_delta,
        }
    if current_ratio > target_ratio + 1e-12:
        raise ValueError(
            f"train 当前正例比例 {current_ratio:.8f} 已高于目标 {target_ratio:.8f}；"
            "只删除负例无法降低比例"
        )
    removable = [
        row for row in rows
        if not row["positive_utterance"] and source_min <= row["source_index"] <= source_max
    ]
    ideal_remove = len(rows) - positive_count / target_ratio
    candidates = {
        max(0, min(len(removable), int(math.floor(ideal_remove)))),
        max(0, min(len(removable), int(math.ceil(ideal_remove)))),
    }
    remove_count = min(
        candidates,
        key=lambda count: (abs(positive_count / (len(rows) - count) - target_ratio), count),
    )
    best_possible = positive_count / (len(rows) - remove_count)
    if remove_count == len(removable) and best_possible + 1e-12 < target_ratio:
        raise ValueError(
            f"part00-04 可删除负例只有 {len(removable)} 条，无法把 train 从 "
            f"{current_ratio:.8f} 提高到 {target_ratio:.8f}"
        )
    removed, _ = stratified_take(
        removable, remove_count, 0, seed, "train/remove_part00_04_negative"
    )
    removed_ids = {row["utt_id"] for row in removed}
    kept = [row for row in rows if row["utt_id"] not in removed_ids]
    after_ratio = positive_count / len(kept)
    return kept, removed, {
        "before_ratio": current_ratio,
        "after_ratio": after_ratio,
        "target_ratio": target_ratio,
        "removed_negative_utterances": len(removed),
        "target_positive_count_delta": abs(positive_count - len(kept) * target_ratio),
        "removed_source_counts": {
            str(key): value for key, value in Counter(row["source_index"] for row in removed).items()
        },
    }


def sha256_lines(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    args = parse_args()
    supplement_sources = parse_source_indices(args.m2_supplement_source_indices)
    if args.m2_source_index in supplement_sources:
        raise ValueError("考试院 source_index 不能同时出现在补充来源中")
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
    global_positive_count = sum(bool(row["positive_utterance"]) for row in rows)
    reference_positive_ratio = global_positive_count / len(rows)

    allowed_m2_sources = {args.m2_source_index, *supplement_sources}
    m2_candidate_count = sum(row["source_index"] in allowed_m2_sources for row in rows)
    lmdb_rows = [
        row for row in rows
        if args.lmdb_source_index_min <= row["source_index"] <= args.lmdb_source_index_max
    ]
    required = args.valid_count + args.test_count
    if m2_candidate_count < required or len(lmdb_rows) < required:
        raise ValueError(
            f"评测候选不足：m2_allowed={m2_candidate_count}, lmdb={len(lmdb_rows)}, 每组需要={required}"
        )

    valid_m2_manual_rows, test_m2_manual_rows, m2_targets = select_m2_splits(
        rows, args.valid_count, args.test_count, reference_positive_ratio,
        args.m2_source_index, supplement_sources, args.split_seed + 1000,
    )
    valid_positive_count = sum(bool(row["positive_utterance"]) for row in valid_m2_manual_rows)
    test_positive_count = sum(bool(row["positive_utterance"]) for row in test_m2_manual_rows)
    valid_lmdb, test_lmdb, lmdb_targets = select_matching_lmdb_splits(
        lmdb_rows, args.valid_count, args.test_count, valid_positive_count,
        test_positive_count, args.split_seed + 2000,
    )

    eval_rows = [*valid_m2_manual_rows, *test_m2_manual_rows, *valid_lmdb, *test_lmdb]
    eval_ids = [row["utt_id"] for row in eval_rows]
    if len(eval_ids) != len(set(eval_ids)):
        raise AssertionError("两组评测集合出现 utt_id 重叠")
    eval_id_set = set(eval_ids)
    train_candidates = [row for row in rows if row["utt_id"] not in eval_id_set]
    train_target_ratio = valid_positive_count / args.valid_count
    if m2_targets["closest_feasible_fallback_used"]:
        train_balance_mode = "match_fallback_valid_ratio_by_removing_part00_04_negatives"
        train_rows, train_removed_rows, train_balance = balance_train_by_removing_part00_04_negatives(
            train_candidates, train_target_ratio, args.lmdb_source_index_min,
            args.lmdb_source_index_max, args.split_seed + 3000,
        )
    else:
        train_balance_mode = "preserve_natural_ratio_and_use_nearest_integer_eval_counts"
        train_rows = train_candidates
        train_removed_rows = []
        natural_train_ratio = (
            sum(bool(row["positive_utterance"]) for row in train_rows) / len(train_rows)
        )
        train_balance = {
            "before_ratio": natural_train_ratio,
            "after_ratio": natural_train_ratio,
            "target_ratio": reference_positive_ratio,
            "removed_negative_utterances": 0,
            "removed_source_counts": {},
            "reason": "valid负例已补足；保留train自然比例，评测集使用最邻近整数正例数",
        }
    if not train_rows:
        raise ValueError("扣除评测集和比例平衡负例后 train 为空")

    removed_path = output_root / "train_ratio_removed_negative_records.jsonl"
    removed_path.write_text(
        "".join(
            json.dumps({
                "utt_id": row["utt_id"], "source_index": row["source_index"],
                "positive_utterance": bool(row["positive_utterance"]),
                "reason": "remove_part00_04_negative_to_match_valid_m2_manual_ratio",
            }, ensure_ascii=False) + "\n"
            for row in train_removed_rows
        ), encoding="utf-8",
    )
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
        "valid_m2_manual": args.valid_count, "test_m2_manual": args.test_count,
        "valid_lmdb_mfa": args.valid_count, "test_lmdb_mfa": args.test_count,
    }
    wrong_counts = {
        split: reports[split]["counters"].get("utterances", 0)
        for split, expected in expected_counts.items()
        if reports[split]["counters"].get("utterances", 0) != expected
    }
    if wrong_counts:
        raise RuntimeError(
            f"评测 split 在音频编码阶段发生跳过，无法保持固定数量：{wrong_counts}；"
            "请检查 prepare_skipped_records.jsonl"
        )

    train_positive_ratio = (
        reports["train"]["counters"]["positive_utterances"]
        / reports["train"]["counters"]["utterances"]
    )
    split_config = {
        "format_version": 2,
        "split_seed": args.split_seed,
        "valid_count": args.valid_count,
        "test_count": args.test_count,
        "m2_source_index": args.m2_source_index,
        "m2_supplement_source_indices": supplement_sources,
        "lmdb_source_index_range": [args.lmdb_source_index_min, args.lmdb_source_index_max],
        "reference_positive_utterance_ratio": reference_positive_ratio,
        "valid_actual_positive_utterance_ratio": train_target_ratio,
        "fallback_train_target_positive_utterance_ratio": (
            train_target_ratio if m2_targets["closest_feasible_fallback_used"] else None
        ),
        "train_balance_mode": train_balance_mode,
        "formal_splits": [
            "train", "valid_m2_manual", "valid_m2_mfa", "valid_lmdb_mfa",
            "test_m2_manual", "test_m2_mfa", "test_lmdb_mfa",
        ],
    }
    (output_root / "shared_split_config.json").write_text(
        json.dumps(split_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "format_version": 2,
        "arguments": vars(args),
        "merged_manifest": str(manifest),
        "output_root": str(output_root),
        "split_seed": args.split_seed,
        "global_positive_utterance_ratio": reference_positive_ratio,
        "train_positive_utterance_ratio": train_positive_ratio,
        "m2_source_index": args.m2_source_index,
        "m2_supplement_source_indices": supplement_sources,
        "lmdb_source_index_range": [args.lmdb_source_index_min, args.lmdb_source_index_max],
        "m2_targets": m2_targets,
        "lmdb_targets": lmdb_targets,
        "train_balance": train_balance,
        "train_balance_mode": train_balance_mode,
        "train_removed_utt_id_sha256": sha256_lines(
            sorted(row["utt_id"] for row in train_removed_rows)
        ),
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
    main()
