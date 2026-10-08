#!/usr/bin/env python3
"""合并分片提取结果的 JSONL 与统计信息；不复制 WAV 文件。"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_jsonl(path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 损坏：{path}:{line_number}: {exc}") from exc


def sample_id_from_error(row):
    """按提取脚本相同规则，从错误记录还原 utt_id；无法还原时返回 None。"""
    source_index = row.get("source_index")
    key_line = row.get("key_line", "")
    if not isinstance(source_index, int) or "@" not in key_line:
        return None
    key = key_line.split("@", 1)[1].split(" ", 1)[0].strip()
    if not key:
        return None
    readable = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("._-")[:80] or "sample"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"all_{source_index:03d}_{readable}_{digest}"


def records_equivalent(left, right):
    """同一条样本在不同分片目录中只有输出路径或新增审计字段可以不同。"""
    ignored = {"wav_path", "hash_marker_count_before_cleanup"}
    return (
        {key: value for key, value in left.items() if key not in ignored}
        == {key: value for key, value in right.items() if key not in ignored}
    )


def main():
    args = parse_args()
    input_roots = [Path(value).resolve() for value in args.input_root]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = {}
    duplicate_count = 0
    leading_boundary_errors = []
    for root in input_roots:
        manifest = root / "all_pause_data.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"找不到分片 manifest：{manifest}")
        for record in read_jsonl(manifest):
            utt_id = record.get("utt_id")
            if not utt_id:
                raise ValueError(f"记录缺少 utt_id：{manifest}")
            leading = record.get("leading_silence_end_seconds")
            alignment_duration = record.get("alignment_duration_seconds")
            if leading is None:
                leading_boundary_errors.append(f"{utt_id}: 缺少 leading_silence_end_seconds")
                continue
            try:
                leading = float(leading)
                alignment_duration = float(alignment_duration)
            except (TypeError, ValueError):
                leading_boundary_errors.append(f"{utt_id}: 句首或对齐时间不是数值")
                continue
            if (
                not math.isfinite(leading)
                or not math.isfinite(alignment_duration)
                or leading < 0
                or leading > alignment_duration
            ):
                leading_boundary_errors.append(
                    f"{utt_id}: 非法边界 leading={leading}, alignment={alignment_duration}"
                )
                continue
            if utt_id in records:
                if not records_equivalent(records[utt_id], record):
                    raise ValueError(f"发现内容冲突的重复 utt_id：{utt_id}")
                duplicate_count += 1
                # 输入顺序为旧单进程目录、再到新分片目录；优先保留后者的有效 WAV 路径。
            records[utt_id] = record

    if leading_boundary_errors:
        preview = "\n".join(leading_boundary_errors[:20])
        raise ValueError(
            "合并输入包含缺失或非法的句首静音边界；请使用更新后的提取脚本重新提取。"
            f"错误数={len(leading_boundary_errors)}\n{preview}"
        )

    merged_manifest = output_root / "all_pause_data.jsonl"
    with merged_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for utt_id in sorted(records):
            handle.write(json.dumps(records[utt_id], ensure_ascii=False) + "\n")

    error_rows = []
    seen_errors = set()
    resolved_error_count = 0
    for root in input_roots:
        for row in read_jsonl(root / "extraction_errors.jsonl"):
            if sample_id_from_error(row) in records:
                resolved_error_count += 1
                continue
            signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if signature not in seen_errors:
                seen_errors.add(signature)
                error_rows.append(row)
    with (output_root / "extraction_errors.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    reports = {}
    duplicate_source_report_count = 0
    for root in input_roots:
        for row in read_jsonl(root / "source_report.jsonl"):
            source_index = row.get("source_index")
            if source_index in reports:
                duplicate_source_report_count += 1
            # 与 manifest 一样，旧目录在前、新分片在后，保留后一次完整运行的报告。
            reports[source_index] = row
    with (output_root / "source_report.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for source_index in sorted(reports, key=lambda value: (-1 if value is None else value)):
            handle.write(json.dumps(reports[source_index], ensure_ascii=False) + "\n")

    counters = Counter()
    speakers = set()
    sources = set()
    for record in records.values():
        words = record.get("word_list", [])
        labels = record.get("pause_label", [])
        if len(words) != len(labels):
            raise ValueError(
                f"{record.get('utt_id', '<unknown>')}: "
                f"word_list 与 pause_label 长度不一致: {len(words)} != {len(labels)}"
            )
        normalized_labels = [int(value) for value in labels]
        if any(value not in (0, 1) for value in normalized_labels):
            raise ValueError(
                f"{record.get('utt_id', '<unknown>')}: pause_label 只能包含 0 或 1"
            )
        positive_pause_count = sum(normalized_labels)
        counters["words"] += len(words)
        counters["positive_pauses"] += positive_pause_count
        counters["utterances_with_pause"] += int(positive_pause_count > 0)
        counters["audio_samples"] += int(record.get("audio_num_samples", 0))
        counters["hash_markers"] += int(record.get("hash_marker_count_before_cleanup", 0))
        counters[f"phoneme_layout_{record.get('phoneme_storage_layout', 'unknown')}"] += 1
        if record.get("wav_path") and not Path(record["wav_path"]).is_file():
            counters["missing_wav_files"] += 1
        if record.get("speaker_id"):
            speakers.add(record["speaker_id"])
        if record.get("source_index") is not None:
            sources.add(record["source_index"])

    negative_pauses = counters["words"] - counters["positive_pauses"]
    positive_ratio = counters["positive_pauses"] / counters["words"] if counters["words"] else 0.0
    utterances_without_pause = len(records) - counters["utterances_with_pause"]
    utterance_pause_ratio = (
        counters["utterances_with_pause"] / len(records) if records else 0.0
    )
    total_hours = sum(float(row.get("audio_duration_seconds", 0.0)) for row in records.values()) / 3600.0
    lines = [
        f"含至少一个停顿的样本数: {counters['utterances_with_pause']}",
        f"不含停顿的样本数: {utterances_without_pause}",
        f"含停顿样本比例: {utterance_pause_ratio:.8f}",
        f"含停顿样本百分比: {utterance_pause_ratio * 100.0:.6f}%",
        "音频停顿训练数据分片合并统计",
        f"输入目录数: {len(input_roots)}",
        f"唯一样本数: {len(records)}",
        f"去重记录数: {duplicate_count}",
        f"LMDB 来源数: {len(sources)}",
        f"说话人数: {len(speakers)}",
        f"总音频小时数: {total_hours:.6f}",
        f"总词数: {counters['words']}",
        f"正停顿标签数: {counters['positive_pauses']}",
        f"负停顿标签数: {negative_pauses}",
        f"正停顿比例: {positive_ratio:.8f}",
        f"清理前含 # 标记总数: {counters['hash_markers']}",
        f"phonemes int16_x3 样本数: {counters['phoneme_layout_int16_x3']}",
        f"phonemes int16_x6 样本数: {counters['phoneme_layout_int16_x6']}",
        f"phonemes 未记录布局样本数: {counters['phoneme_layout_unknown']}",
        f"错误记录数: {len(error_rows)}",
        f"已由成功样本消解的历史错误数: {resolved_error_count}",
        f"被新分片覆盖的旧来源报告数: {duplicate_source_report_count}",
        f"缺失 WAV 文件数: {counters['missing_wav_files']}",
        "缺失 leading_silence_end_seconds 数: 0",
        "非法 leading_silence_end_seconds 数: 0",
        "说明: 合并过程只合并清单和统计，不复制 WAV；wav_path 仍指向原分片目录。",
    ]
    stats = "\n".join(lines) + "\n"
    (output_root / "training_data_statistics.txt").write_text(stats, encoding="utf-8")
    print(stats, end="")
    print(f"合并 manifest：{merged_manifest}")


if __name__ == "__main__":
    main()
