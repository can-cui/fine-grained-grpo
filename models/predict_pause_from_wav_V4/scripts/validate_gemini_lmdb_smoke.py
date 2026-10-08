#!/usr/bin/env python3
"""校验 Gemini LMDB + Excel 标签 + 外部 WAV 试跑结果。"""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--expected-wav-root", required=True)
    parser.add_argument("--expected-source-index", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    expected_wav_root = Path(args.expected_wav_root).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"找不到试跑 manifest：{manifest}")

    records = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行 JSON 损坏：{exc}") from exc
            records.append(record)

    if len(records) != args.expected_count:
        raise ValueError(
            f"成功样本数不是预期值：expected={args.expected_count}, actual={len(records)}"
        )

    seen_ids = set()
    positive_samples = 0
    positive_labels = 0
    total_labels = 0
    for record in records:
        utt_id = record.get("utt_id")
        if not utt_id or utt_id in seen_ids:
            raise ValueError(f"utt_id 缺失或重复：{utt_id!r}")
        seen_ids.add(utt_id)
        if record.get("pause_label_source") != "xlsx_second_column":
            raise ValueError(f"{utt_id}: pause_label 不是来自 Excel 第二列")
        if record.get("wav_source") != "external_wav_root":
            raise ValueError(f"{utt_id}: WAV 不是来自 external_wav_root")
        if int(record.get("source_index", -1)) != args.expected_source_index:
            raise ValueError(f"{utt_id}: source_index 不正确：{record.get('source_index')}")
        if int(record.get("sample_rate", 0)) != 24000:
            raise ValueError(f"{utt_id}: sample_rate 不是 24000")

        wav_path = Path(record.get("wav_path", "")).resolve()
        try:
            wav_path.relative_to(expected_wav_root)
        except ValueError as exc:
            raise ValueError(f"{utt_id}: WAV 不在外部目录内：{wav_path}") from exc
        if not wav_path.is_file():
            raise FileNotFoundError(f"{utt_id}: 外部 WAV 不存在：{wav_path}")

        words = record.get("word_list")
        ends = record.get("word_end_time")
        labels = record.get("pause_label")
        if not isinstance(words, list) or not isinstance(ends, list) or not isinstance(labels, list):
            raise ValueError(f"{utt_id}: word_list/word_end_time/pause_label 必须是 list")
        if not (len(words) == len(ends) == len(labels)):
            raise ValueError(f"{utt_id}: 词、时间和标签长度不一致")
        if any(not isinstance(value, int) or value not in (0, 1) for value in labels):
            raise ValueError(f"{utt_id}: pause_label 不是二值整数")
        numeric_ends = [float(value) for value in ends if value is not None]
        if not numeric_ends or any(not math.isfinite(value) for value in numeric_ends):
            raise ValueError(f"{utt_id}: word_end_time 缺少有效数值")
        leading = float(record.get("leading_silence_end_seconds", -1))
        duration = float(record.get("audio_duration_seconds", -1))
        alignment_duration = float(record.get("alignment_duration_seconds", -1))
        if not (0 <= leading <= alignment_duration <= duration + 0.05):
            raise ValueError(
                f"{utt_id}: 时间范围非法：leading={leading}, "
                f"alignment={alignment_duration}, wav={duration}"
            )
        count = sum(labels)
        positive_samples += int(count > 0)
        positive_labels += count
        total_labels += len(labels)

    print("Gemini LMDB 试跑校验通过")
    print(f"样本数: {len(records)}")
    print(f"含停顿样本数: {positive_samples}")
    print(f"正停顿标签数: {positive_labels}")
    print(f"标签总数: {total_labels}")
    print("时间戳来源: LMDB text/phonemes/fa")
    print("停顿标签来源: Excel 第二列 <PAUSE>")
    print(f"WAV 来源: {expected_wav_root}")


if __name__ == "__main__":
    main()
