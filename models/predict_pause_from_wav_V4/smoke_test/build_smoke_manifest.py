#!/usr/bin/env python3
"""从正式提取 manifest 中确定性抽取平衡的极小端到端测试集。"""

import argparse
import json
import math
import random
import wave
from collections import Counter
from pathlib import Path


PAUSE_SYMBOLS = {
    ",", "，", "。", ".", "、", "；", "...", "：",
    "~", "*", "#", "@", "&", ":", ";", "!", "！", "?", "？",
}
SPLITS = ("train", "valid", "test")
POOL_LIMIT = 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-dir", required=True)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--valid-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    return parser.parse_args()


def collapse_labels(record):
    words = list(record.get("word_list", []))
    ends = list(record.get("word_end_time", []))
    labels = list(record.get("pause_label", []))
    if not (len(words) == len(ends) == len(labels)):
        raise ValueError("word_list/word_end_time/pause_label 长度不一致")
    lexical_words = []
    lexical_ends = []
    lexical_labels = []
    for word, end_time, label in zip(words, ends, labels):
        if end_time is None or not math.isfinite(float(end_time)):
            raise ValueError("词结束时间非法")
        label = int(label)
        if label not in (0, 1):
            raise ValueError("停顿标签不是 0/1")
        if word in PAUSE_SYMBOLS:
            if lexical_words:
                lexical_ends[-1] = float(end_time)
                lexical_labels[-1] = max(lexical_labels[-1], label)
            continue
        lexical_words.append(str(word))
        lexical_ends.append(float(end_time))
        lexical_labels.append(label)
    if len(lexical_words) < 2:
        raise ValueError("清理停顿符号后实际词数少于 2")
    leading = record.get("leading_silence_end_seconds")
    if leading is None:
        raise ValueError("缺少 leading_silence_end_seconds")
    leading = float(leading)
    if not math.isfinite(leading) or leading < 0 or leading > lexical_ends[0]:
        raise ValueError("leading_silence_end_seconds 非法")
    boundaries = [leading] + lexical_ends
    if any(right <= left for left, right in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("词窗口时间不是严格递增")
    valid_labels = lexical_labels[:-1]
    return {
        "word_count": len(lexical_words),
        "valid_target_count": len(valid_labels),
        "positive_target_count": sum(valid_labels),
    }


def inspect_wav(path, sample_rate, max_seconds):
    if not path.is_file():
        raise ValueError("WAV 不存在")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        compression = handle.getcomptype()
    if channels != 1 or width != 2 or compression != "NONE":
        raise ValueError("WAV 不是单声道 PCM16")
    if rate != sample_rate:
        raise ValueError("WAV 采样率不是 {}".format(sample_rate))
    duration = frames / float(rate)
    if frames <= 0:
        raise ValueError("WAV 为空")
    if duration > max_seconds:
        raise ValueError("WAV 超过最大时长")
    return frames, duration


def trim_pool(pool):
    if len(pool) > POOL_LIMIT * 2:
        pool.sort(key=lambda item: (item["duration_seconds"], item["utt_id"]))
        del pool[POOL_LIMIT:]


def expected_split(selected, counts, seed):
    shuffled = sorted(selected, key=lambda item: item["utt_id"])
    random.Random(seed).shuffle(shuffled)
    result = {}
    cursor = 0
    for split, count in zip(SPLITS, counts):
        result[split] = shuffled[cursor:cursor + count]
        cursor += count
    return result


def split_is_balanced(split_map):
    return all(
        any(item["is_positive"] for item in rows)
        and any(not item["is_positive"] for item in rows)
        for rows in split_map.values()
    )


def select_records(positive, negative, counts, seed):
    total = sum(counts)
    if len(positive) < len(SPLITS) or len(negative) < len(SPLITS):
        raise ValueError(
            "至少需要 3 条正停顿样本和 3 条负停顿样本；当前 positive={} negative={}".format(
                len(positive), len(negative)
            )
        )
    feasible = [
        value for value in range(len(SPLITS), total - len(SPLITS) + 1)
        if value <= len(positive) and total - value <= len(negative)
    ]
    feasible.sort(key=lambda value: (abs(value - total / 2.0), value))
    if not feasible:
        raise ValueError("正负候选总数不足以组成 {} 条平衡样本".format(total))
    rng = random.Random(seed)
    positive = sorted(positive, key=lambda item: (item["duration_seconds"], item["utt_id"]))
    negative = sorted(negative, key=lambda item: (item["duration_seconds"], item["utt_id"]))
    for trial in range(10000):
        positive_count = feasible[trial % len(feasible)]
        if trial == 0:
            chosen_positive = positive[:positive_count]
            chosen_negative = negative[:total - positive_count]
        else:
            chosen_positive = rng.sample(positive, positive_count)
            chosen_negative = rng.sample(negative, total - positive_count)
        selected = chosen_positive + chosen_negative
        split_map = expected_split(selected, counts, seed)
        if split_is_balanced(split_map):
            return selected, split_map, trial + 1
    raise ValueError("尝试 10000 次后仍无法让每个 split 同时包含正、负样本")


def main():
    args = parse_args()
    counts = [args.train_count, args.valid_count, args.test_count]
    if any(value < 2 for value in counts):
        raise ValueError("为保证正负平衡，train/valid/test 数量都必须至少为 2")
    if args.max_audio_seconds <= 0 or args.sample_rate <= 0:
        raise ValueError("最大时长和采样率必须大于 0")

    input_path = Path(args.input_manifest).resolve()
    output_path = Path(args.output_manifest).resolve()
    report_path = Path(args.output_report).resolve()
    expected_dir = Path(args.expected_dir).resolve()
    positive = []
    negative = []
    reasons = Counter()
    seen = set()
    scanned = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            scanned += 1
            try:
                record = json.loads(line)
                utt_id = str(record.get("utt_id", "")).strip()
                if not utt_id:
                    raise ValueError("缺少 utt_id")
                if utt_id in seen:
                    raise ValueError("重复 utt_id")
                seen.add(utt_id)
                stats = collapse_labels(record)
                frames, duration = inspect_wav(
                    Path(str(record.get("wav_path", ""))),
                    args.sample_rate,
                    args.max_audio_seconds,
                )
                candidate = {
                    "utt_id": utt_id,
                    "record": record,
                    "duration_seconds": duration,
                    "audio_frames": frames,
                    "is_positive": stats["positive_target_count"] > 0,
                    **stats,
                }
                pool = positive if candidate["is_positive"] else negative
                pool.append(candidate)
                trim_pool(pool)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                reasons[str(exc)] += 1

    positive.sort(key=lambda item: (item["duration_seconds"], item["utt_id"]))
    negative.sort(key=lambda item: (item["duration_seconds"], item["utt_id"]))
    positive = positive[:POOL_LIMIT]
    negative = negative[:POOL_LIMIT]
    selected, split_map, attempts = select_records(
        positive, negative, counts, args.split_seed
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(selected, key=lambda item: item["utt_id"])
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            handle.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    split_reports = {}
    for split, rows in split_map.items():
        ids = [item["utt_id"] for item in rows]
        (expected_dir / ("expected_{}_utt_ids.txt".format(split))).write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )
        split_reports[split] = {
            "count": len(rows),
            "utt_ids": ids,
            "positive_utterances": sum(item["is_positive"] for item in rows),
            "negative_utterances": sum(not item["is_positive"] for item in rows),
            "word_count": sum(item["word_count"] for item in rows),
            "valid_target_count": sum(item["valid_target_count"] for item in rows),
            "positive_target_count": sum(item["positive_target_count"] for item in rows),
        }
    report = {
        "ok": True,
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "scanned_records": scanned,
        "eligible_positive_pool": len(positive),
        "eligible_negative_pool": len(negative),
        "selection_attempts": attempts,
        "split_seed": args.split_seed,
        "max_audio_seconds": args.max_audio_seconds,
        "rejection_reasons": dict(reasons.most_common()),
        "splits": split_reports,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
