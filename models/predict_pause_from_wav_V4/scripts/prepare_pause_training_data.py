#!/usr/bin/env python3
"""把合并后的停顿 JSONL/WAV 转换为在线 wav2vec 训练使用的 PCM16 LMDB。"""

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
    """记录清除停顿符号后不足两个词，因而没有有效训练目标。"""


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
            "count 模式且 train-count=0 时，test 中含至少一个有效正停顿的句子比例；"
            "按四舍五入换算为精确句数。"
        ),
    )
    parser.add_argument(
        "--eval-source-index",
        type=int,
        default=None,
        help=(
            "仅在 count 模式且 train-count=0 时使用；valid/test 只从该 "
            "source_index 抽取，其余来源及该来源剩余样本全部进入 train。"
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
                        f"JSON 损坏：{exc}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(record, dict):
                skipped.append(
                    make_skipped_record(
                        None,
                        "load_manifest",
                        f"JSON 顶层类型必须为对象，实际为 {type(record).__name__}",
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
                        "缺少 utt_id",
                        line_number=line_number,
                    )
                )
                continue
            if utt_id in seen:
                skipped.append(
                    make_skipped_record(
                        record,
                        "load_manifest",
                        f"重复 utt_id：{utt_id}",
                        line_number=line_number,
                    )
                )
                continue
            seen.add(utt_id)
            records.append(record)
    if not records:
        raise ValueError(f"合并 manifest 为空：{path}")
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
        raise ValueError("train/valid/test ratio 不能为负数")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"ratio 之和必须为 1，实际为 {sum(ratios)}")
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
            f"{record['utt_id']}: source_index 不是整数：{record.get('source_index')!r}"
        ) from exc


def record_has_positive_pause(record):
    """判断句子是否包含至少一个参与训练的正停顿目标。"""
    _, _, _, labels, valid_mask = collapse_to_lexical(record)
    return any(label and valid for label, valid in zip(labels, valid_mask))


def allocate_strata(strata, count):
    total = sum(len(items) for items in strata.values())
    if count < 0 or count > total:
        raise ValueError(f"分层抽样数量非法：count={count}, total={total}")
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
            raise ValueError("无法完成分层数量分配")
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
    """逐条检查划分所需字段，任何样本错误都排除并保留审计原因。"""
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
        raise ValueError("所有记录都无法产生有效词末停顿目标")
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
            name = "含正停顿" if status else "无正停顿"
            raise ValueError(
                f"test 需要 {name} 样本 {requested[status]} 条，"
                f"但可用样本只有 {len(groups[status])} 条"
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
    """valid 至少两句时，尽量保证正负停顿句均存在。"""
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
            raise ValueError("eval-source-index 仅支持 count 模式且 train-count=0")
        counts = ratio_counts(
            len(shuffled),
            [args.train_ratio, args.valid_ratio, args.test_ratio],
        )
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(f"train 和 valid 都必须非空，实际数量为 {counts}")
        result = {}
        cursor = 0
        for split, count in zip(SPLITS, counts):
            result[split] = shuffled[cursor:cursor + count]
            cursor += count
        return result, {
            "strategy": "按固定随机种子随机划分（ratio 模式）",
            "eval_source_index": None,
            "eval_source_utterance_count": None,
            "test_positive_utterance_ratio_requested": None,
            "test_positive_utterance_count_target": None,
        }

    counts = [args.train_count, args.valid_count, args.test_count]
    if any(value < 0 for value in counts):
        raise ValueError("train/valid/test count 不能为负数")
    if not 0.0 <= args.test_positive_utterance_ratio <= 1.0:
        raise ValueError("test-positive-utterance-ratio 必须在 0 到 1 之间")

    if args.train_count == 0:
        if args.valid_count <= 0:
            raise ValueError("count 模式且 train-count=0 时，valid-count 必须大于 0")
        if args.valid_count + args.test_count >= len(records):
            raise ValueError(
                "valid-count + test-count 必须小于总样本数，保证 train 自动保留非空"
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
                    f"source_index={args.eval_source_index} 共有 {len(eval_rows)} 条，"
                    f"少于 valid+test={required_eval_count}，无法完成指定划分"
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
                "test 先按指定含正停顿句比例、在正负类别内按 source_index 分层抽样；"
                "valid 再按 source_index 和是否含正停顿分层抽样；train 为剩余样本"
            )
        else:
            strategy = (
                f"valid/test 仅从 source_index={args.eval_source_index} 抽取；"
                "test 先按指定含正停顿句比例抽样，valid 再按是否含正停顿分层抽样；"
                "其余来源及该来源剩余样本全部进入 train"
            )
        return result, {
            "strategy": strategy,
            "eval_source_index": args.eval_source_index,
            "eval_source_utterance_count": len(eval_rows),
            "test_positive_utterance_ratio_requested": args.test_positive_utterance_ratio,
            "test_positive_utterance_count_target": test_positive_count,
        }

    if args.eval_source_index is not None:
        raise ValueError("eval-source-index 仅支持 count 模式且 train-count=0")
    if sum(counts) != len(shuffled):
        raise ValueError(
            f"count 之和必须等于样本总数：counts={counts}, total={len(shuffled)}；"
            "若希望 train 自动取余，请将 train-count 设为 0"
        )
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"train 和 valid 都必须非空，实际数量为 {counts}")
    result = {}
    cursor = 0
    for split, count in zip(SPLITS, counts):
        result[split] = shuffled[cursor:cursor + count]
        cursor += count
    return result, {
        "strategy": "按固定随机种子随机划分（count 模式，指定了 train-count）",
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
            f"{path}: 仅支持单声道 PCM16 WAV，实际 channels={channels}, sample_width={sample_width}"
        )
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{path}: 采样率应为 {expected_sample_rate}，实际为 {sample_rate}"
        )
    waveform = np.frombuffer(raw, dtype="<i2").copy()
    if len(waveform) == 0:
        raise ValueError(f"{path}: WAV 为空")
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
            f"{record['utt_id']}: word_list/word_end_time/pause_label 长度不一致"
        )

    lexical_words = []
    lexical_ends = []
    lexical_labels = []
    for word, end_time, label in zip(words, ends, labels):
        if end_time is None or not math.isfinite(float(end_time)):
            raise ValueError(f"{record['utt_id']}: {word!r} 的结束时间非法：{end_time}")
        end_time = float(end_time)
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"{record['utt_id']}: 停顿标签不是 0/1：{label}")
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
            f"{record['utt_id']}: 实际词数少于 2，无法产生有效停顿目标"
        )
    leading = record.get("leading_silence_end_seconds")
    if leading is None:
        raise ValueError(
            f"{record['utt_id']}: 缺少 leading_silence_end_seconds，请重新执行 LMDB 提取"
        )
    leading = float(leading)
    if not math.isfinite(leading) or leading < 0 or leading > lexical_ends[0]:
        raise ValueError(
            f"{record['utt_id']}: 句首静音边界非法：leading={leading}, first_end={lexical_ends[0]}"
        )
    if any(right <= left for left, right in zip([leading] + lexical_ends[:-1], lexical_ends)):
        raise ValueError(f"{record['utt_id']}: 词窗口时间不是严格递增")

    starts = [leading] + lexical_ends[:-1]
    valid_mask = [1] * (len(lexical_words) - 1) + [0]
    return lexical_words, starts, lexical_ends, lexical_labels, valid_mask


def seconds_to_samples(values, sample_rate, waveform_samples, utt_id):
    result = np.rint(np.asarray(values, dtype=np.float64) * sample_rate).astype(np.int64)
    tolerance = int(round(0.05 * sample_rate))
    if np.any(result < 0) or np.any(result > waveform_samples + tolerance):
        raise ValueError(
            f"{utt_id}: 时间边界超出 WAV，范围={result.min()}..{result.max()}, wav={waveform_samples}"
        )
    return np.clip(result, 0, waveform_samples)


def encode_record(record, source_rate, target_rate):
    wav_path = Path(record["wav_path"])
    if not wav_path.is_file():
        raise FileNotFoundError(f"{record['utt_id']}: 找不到 WAV：{wav_path}")
    source_waveform = read_pcm16_wav(wav_path, source_rate)
    waveform = resample_pcm16(source_waveform, source_rate, target_rate)
    words, starts_seconds, ends_seconds, labels, valid_mask = collapse_to_lexical(record)
    starts = seconds_to_samples(starts_seconds, target_rate, len(waveform), record["utt_id"])
    ends = seconds_to_samples(ends_seconds, target_rate, len(waveform), record["utt_id"])
    if np.any(ends <= starts):
        raise ValueError(f"{record['utt_id']}: 转换到采样点后存在空词窗口")

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
                f"{split} 数量为 0，但旧输出仍存在；确认后使用 --overwrite 删除：{existing}"
            )
        if args.overwrite:
            remove_existing_split(output_root, split)
        return None
    lmdb_path = output_root / split
    key_path = output_root / f"{split}.key"
    manifest_path = output_root / f"{split}.jsonl"
    existing = [path for path in (lmdb_path, key_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{split} 输出已存在；确认后使用 --overwrite：{existing}")
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
                raise ValueError(f"{split}: LMDB 重复 key：{key}")
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
        raise FileNotFoundError(f"找不到合并 manifest：{manifest}")
    if args.source_sample_rate <= 0 or args.target_sample_rate <= 0:
        raise ValueError("采样率必须为正整数")
    if args.map_size_gb <= 0:
        raise ValueError("map-size-gb 必须大于 0")
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
        "音频停顿预测 V4：在线 wav2vec 训练喂入数据统计",
        "=" * 72,
        f"输入 manifest: {manifest}",
        f"输出目录: {output_root}",
        f"划分模式: {args.split_mode}",
        f"随机种子: {args.split_seed}",
        f"划分策略: {split_strategy['strategy']}",
        f"采样率转换: {args.source_sample_rate} -> {args.target_sample_rate}",
        f"输入样本数: {input_record_count}",
        f"划分前可划分样本数: {len(records)}",
        f"累计跳过样本数: {len(skipped_records)}",
        f"跳过样本清单: {skipped_path}",
    ]
    if split_strategy["eval_source_index"] is not None:
        lines.extend(
            [
                "valid/test 限定来源 source_index: {}".format(
                    split_strategy["eval_source_index"]
                ),
                "限定来源可用句子数: {}".format(
                    split_strategy["eval_source_utterance_count"]
                ),
            ]
        )
    if split_strategy["test_positive_utterance_ratio_requested"] is not None:
        lines.extend(
            [
                "test 含正停顿句目标比例: {:.8f}".format(
                    split_strategy["test_positive_utterance_ratio_requested"]
                ),
                "test 含正停顿句目标数量: {}".format(
                    split_strategy["test_positive_utterance_count_target"]
                ),
            ]
        )
    for split in SPLITS:
        if split not in split_reports:
            lines.extend(["", f"[{split}] 未生成（数量为 0）"])
            continue
        counters, sources = split_reports[split]
        negative = counters["valid"] - counters["positive"]
        lines.extend(
            [
                "",
                f"[{split}]",
                f"句子数: {counters['utterances']}",
                f"说话人数: {counters['speakers']}",
                f"音频小时数: {counters['audio_samples'] / args.target_sample_rate / 3600.0:.6f}",
                f"总词数: {counters['words']}",
                f"有效目标数: {counters['valid']}",
                f"mask 目标数: {counters['masked']}",
                f"含正停顿句数: {counters['positive_utterances']}",
                "含正停顿句比例: {:.8f}".format(
                    counters["positive_utterances"] / counters["utterances"]
                ),
                f"正停顿数: {counters['positive']}",
                f"负停顿数: {negative}",
                f"正停顿比例: {(counters['positive'] / counters['valid'] if counters['valid'] else 0):.8f}",
                "来源句数: " + ", ".join(
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
