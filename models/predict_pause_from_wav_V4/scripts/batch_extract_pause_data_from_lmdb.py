#!/usr/bin/env python3
"""从配置 JSON 中列出的全部 LMDB 批量导出 V4 停顿训练数据。

本实现仅依据用户提供的 filter_fa_lmdb.py 截图所显示的字段和逻辑：
Datum_LLMTTS.text / phonemes / fa / wav、spkid.key、ENsil/ENL3、
word>phone+phone 文本格式，以及标点持续时间大于阈值的停顿判定。
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


CONFIG_GROUPS = ("train", "valid", "test")
PAUSE_SYMBOLS = {
    ",", "，", "、", ".", "。", "；", "...", "：",
    "~", "*", "#", "@", "&", ":", ";", "!", "！", "?", "？",
}
SCREENSHOT_SPECIAL_PAUSE = {"~", "*", "#", "@", "&"}
LEADING_SILENCE_PHONES = {"ENsil", "ENL3"}


class SampleError(ValueError):
    """单条 LMDB 数据无法安全转换。"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-json",
        required=True,
        help="包含原 LMDB 列表的 JSON；各列表只作为来源读取并合并，不作为数据划分",
    )
    parser.add_argument("--output-root", required=True, help="导出 WAV、JSONL 和统计文件的目录")
    parser.add_argument("--protobuf-python-dir", required=True, help="datum_llmtts_pb2.py 所在内网目录")
    parser.add_argument("--phone-vocab", required=True, help="截图中的 vocab_extended.txt")
    parser.add_argument("--sample-rate", type=int, default=24000, help="已确认的 datum.wav 原始采样率")
    parser.add_argument("--pause-threshold-seconds", type=float, default=0.03)
    parser.add_argument("--default-key-tag", default="100ms10khz")
    parser.add_argument("--skip-key-substring", default="lmdb_0_Long20_60s_HeadTail200ms")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples-per-source", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1, help="LMDB 来源分片总数")
    parser.add_argument("--part-index", type=int, default=0, help="当前分片编号，从 0 开始")
    parser.add_argument(
        "--source-index-offset",
        type=int,
        default=0,
        help="写入记录前给配置内 source_index 增加的偏移；追加新 LMDB 时用于避免来源编号冲突",
    )
    return parser.parse_args()


def load_vocab(vocab_file):
    """按截图逻辑读取每行第一个字段。"""
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab.append(parts[0])
    if not vocab:
        raise ValueError(f"音素词表为空：{vocab_file}")
    return vocab


def ids_to_string(id_list, vocab):
    """按截图逻辑把越界 ID 映射为 <UNK>。"""
    return " ".join(vocab[phone_id] if 0 <= phone_id < len(vocab) else "<UNK>" for phone_id in id_list)


def load_datum_class(protobuf_python_dir):
    """只从用户指定的公司内网目录加载截图所示 Datum_LLMTTS。"""
    protobuf_dir = str(Path(protobuf_python_dir).resolve())
    if protobuf_dir not in sys.path:
        sys.path.insert(0, protobuf_dir)
    module = importlib.import_module("datum_llmtts_pb2")
    try:
        return getattr(module, "Datum_LLMTTS")
    except AttributeError as exc:
        raise ImportError(
            "datum_llmtts_pb2 中找不到截图所示的 Datum_LLMTTS；"
            "请确认 PROTOBUF_PYTHON_DIR 指向生成该 LMDB 时使用的内网代码。"
        ) from exc


def split_lmdb_spec(lmdb_spec):
    """把 /path/lmdb:100ms10khz 解析成 LMDB 目录和 key tag。"""
    value = str(lmdb_spec).strip()
    if not value:
        raise ValueError("lmdb_path 为空")
    last_slash = max(value.rfind("/"), value.rfind("\\"))
    last_colon = value.rfind(":")
    if last_colon > last_slash:
        return value[:last_colon], value[last_colon + 1:]
    return value, ""


def resolve_lmdb_and_key(entry, default_key_tag):
    lmdb_path, key_tag = split_lmdb_spec(entry["lmdb_path"])
    lmdb_path = Path(lmdb_path)
    if entry.get("key_path"):
        key_path = Path(entry["key_path"])
    elif key_tag:
        key_path = Path(f"{lmdb_path}_{key_tag}.spkid.key")
    else:
        candidates = [
            Path(f"{lmdb_path}.spkid.key"),
            Path(f"{lmdb_path}_{default_key_tag}.spkid.key"),
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) == 1:
            key_path = existing[0]
        elif len(existing) > 1:
            raise ValueError(
                f"{lmdb_path}: 找到多个候选 key 文件，请在配置项中显式增加 key_path：{existing}"
            )
        else:
            key_path = candidates[-1]
    return lmdb_path, key_path, key_tag or default_key_tag


def parse_key_line(key_line):
    """严格沿用截图：取 @ 后、空格前的内容作为 LMDB key。"""
    if "@" not in key_line:
        raise SampleError("spkid.key 行中没有截图要求的 @ 分隔符")
    key = key_line.split("@", 1)[1].split(" ", 1)[0].strip()
    if not key:
        raise SampleError("spkid.key 行解析出的 key 为空")
    return key


def safe_sample_id(source_index, key):
    readable = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("._-")[:80] or "sample"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"all_{source_index:03d}_{readable}_{digest}"


def xlsx_column_index(cell_reference):
    """把 A/B/AA 等 Excel 列名转成从 0 开始的序号。"""
    match = re.match(r"([A-Za-z]+)", str(cell_reference))
    if not match:
        raise ValueError(f"无法解析 Excel 单元格地址：{cell_reference!r}")
    result = 0
    for character in match.group(1).upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def read_xlsx_first_two_columns(path):
    """只用标准库读取第一个工作表的 A/B 列，避免引入新的 Excel 依赖。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到停顿标签 Excel：{path}")
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(str(path), "r") as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{main_ns}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t"))
                )

        workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook_root.find(f".//{{{main_ns}}}sheet")
        if first_sheet is None:
            raise ValueError(f"Excel 中没有工作表：{path}")
        relationship_id = first_sheet.attrib.get(f"{{{rel_ns}}}id")
        rel_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in rel_root.findall(f"{{{pkg_rel_ns}}}Relationship"):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"无法定位 Excel 第一个工作表：{path}")
        sheet_member = target.lstrip("/")
        if not sheet_member.startswith("xl/"):
            sheet_member = f"xl/{sheet_member}"
        sheet_root = ElementTree.fromstring(archive.read(sheet_member))

        rows = []
        for row_node in sheet_root.findall(f".//{{{main_ns}}}row"):
            values = ["", ""]
            for cell in row_node.findall(f"{{{main_ns}}}c"):
                column = xlsx_column_index(cell.attrib.get("r", ""))
                if column > 1:
                    continue
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{main_ns}}}t")
                    )
                else:
                    value_node = cell.find(f"{{{main_ns}}}v")
                    value = "" if value_node is None else (value_node.text or "")
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                values[column] = value.strip()
            if values[0] or values[1]:
                rows.append(tuple(values))
    return rows


def load_xlsx_pause_labels(path, pause_token="<PAUSE>"):
    """读取 Excel 第一列 WAV ID 和第二列标签；重复表头会被忽略。"""
    labels_by_number = {}
    for row_number, (wav_id, tagged_text) in enumerate(
        read_xlsx_first_two_columns(path), 1
    ):
        if wav_id in {"文件名称", "文件名", "wavid", "wav_id"}:
            continue
        if not wav_id or not tagged_text:
            raise ValueError(f"{path}: 第 {row_number} 行第一列或第二列为空")
        stem = Path(wav_id).stem
        match = re.search(r"(\d+)$", stem)
        if not match:
            raise ValueError(f"{path}: 第 {row_number} 行 WAV ID 没有数字后缀：{wav_id}")
        numeric_id = int(match.group(1))
        if numeric_id in labels_by_number:
            raise ValueError(f"{path}: 重复数字 ID：{numeric_id}")
        if pause_token.lower() not in tagged_text.lower():
            pause_count = 0
        else:
            pause_count = len(re.findall(re.escape(pause_token), tagged_text, re.IGNORECASE))
        labels_by_number[numeric_id] = {
            "wav_id": wav_id,
            "tagged_text": tagged_text,
            "pause_count": pause_count,
        }
    if not labels_by_number:
        raise ValueError(f"停顿标签 Excel 没有有效数据：{path}")
    return labels_by_number


def normalize_alignment_word(word):
    return re.sub(r"[^0-9a-z]+", "", str(word).lower().replace("’", "'"))


def parse_tagged_pause_words(tagged_text, pause_token="<PAUSE>"):
    """把 Excel 第二列转成 lexical words 及“停顿归前词”的 0/1 标签。"""
    pieces = re.findall(
        rf"{re.escape(pause_token)}|[0-9A-Za-z]+(?:['’\-][0-9A-Za-z]+)*",
        tagged_text,
        flags=re.IGNORECASE,
    )
    words = []
    labels = []
    for piece in pieces:
        if piece.lower() == pause_token.lower():
            if not labels:
                raise SampleError("Excel 第二列在首词之前出现 <PAUSE>")
            labels[-1] = 1
            continue
        normalized = normalize_alignment_word(piece)
        if normalized:
            words.append(normalized)
            labels.append(0)
    if not words:
        raise SampleError("Excel 第二列没有可用英文单词")
    return words, labels


def normalized_lmdb_lexical_words(word_list):
    """取得 LMDB word_list 的标准化 lexical words，忽略静音和标点 token。"""
    ignored_words = {"ensil", "enl3", "sil", "silv", "sp", "head", "tail"}
    result = []
    for word in word_list:
        if word in PAUSE_SYMBOLS:
            continue
        normalized = normalize_alignment_word(word)
        if normalized and normalized not in ignored_words:
            result.append(normalized)
    return result


def build_xlsx_text_index(labels_by_number, pause_token="<PAUSE>"):
    """按去掉 <PAUSE>/标点后的完整句子建立 Excel 内容索引。"""
    index = defaultdict(list)
    invalid_rows = []
    for numeric_id, row in labels_by_number.items():
        try:
            words, _ = parse_tagged_pause_words(row["tagged_text"], pause_token)
        except SampleError as exc:
            invalid_rows.append(
                {
                    "numeric_id": numeric_id,
                    "wav_id": row["wav_id"],
                    "tagged_text": row["tagged_text"],
                    "error": str(exc),
                }
            )
            continue
        text_key = "".join(words)
        enriched = dict(row)
        enriched["numeric_id"] = numeric_id
        enriched["normalized_text_key"] = text_key
        index[text_key].append(enriched)
    if not index:
        raise ValueError("停顿标签 Excel 没有任何可用于句子匹配的英文记录")
    return index, invalid_rows


def lookup_xlsx_label_by_lmdb_words(text_index, word_list, lmdb_key):
    """优先按完整句子内容匹配；重复句子再用 key 数字后缀消歧。"""
    lmdb_words = normalized_lmdb_lexical_words(word_list)
    text_key = "".join(lmdb_words)
    candidates = text_index.get(text_key, [])
    if not candidates:
        raise SampleError(
            "Excel 中找不到与 LMDB 全句内容一致的记录："
            f"key={lmdb_key}, lmdb_words={lmdb_words}"
        )
    if len(candidates) == 1:
        return candidates[0]

    numeric_match = re.search(r"(\d+)$", str(lmdb_key))
    if numeric_match:
        numeric_id = int(numeric_match.group(1))
        narrowed = [row for row in candidates if row["numeric_id"] == numeric_id]
        if len(narrowed) == 1:
            return narrowed[0]
    raise SampleError(
        "Excel 中存在多条相同句子，且无法用 LMDB key 唯一消歧："
        f"key={lmdb_key}, candidates={[row['wav_id'] for row in candidates]}"
    )


def labels_from_xlsx_for_lmdb_words(word_list, tagged_text, pause_token="<PAUSE>"):
    """对齐 Excel 词序列与 LMDB 词序列，并把 Excel 标签投到 LMDB lexical word。"""
    xlsx_words, xlsx_labels = parse_tagged_pause_words(tagged_text, pause_token)
    lmdb_indices = []
    lmdb_words = []
    ignored_words = {"ensil", "enl3", "sil", "silv", "sp", "head", "tail"}
    for index, word in enumerate(word_list):
        if word in PAUSE_SYMBOLS:
            continue
        normalized = normalize_alignment_word(word)
        if normalized and normalized not in ignored_words:
            lmdb_indices.append(index)
            lmdb_words.append(normalized)
    if not lmdb_words:
        raise SampleError("LMDB word_list 没有 lexical word")

    output_labels = [0] * len(word_list)
    left_index = 0
    right_index = 0
    while left_index < len(xlsx_words) and right_index < len(lmdb_words):
        left_start = left_index
        right_start = right_index
        left_text = xlsx_words[left_index]
        right_text = lmdb_words[right_index]
        left_index += 1
        right_index += 1
        while left_text != right_text:
            if left_text.startswith(right_text) and right_index < len(lmdb_words):
                right_text += lmdb_words[right_index]
                right_index += 1
            elif right_text.startswith(left_text) and left_index < len(xlsx_words):
                left_text += xlsx_words[left_index]
                left_index += 1
            else:
                raise SampleError(
                    "Excel 与 LMDB 词序列不一致："
                    f"xlsx={xlsx_words[left_start:left_index]}, "
                    f"lmdb={lmdb_words[right_start:right_index]}"
                )
        positive_offsets = [
            offset
            for offset, label in enumerate(xlsx_labels[left_start:left_index])
            if label == 1
        ]
        if positive_offsets and positive_offsets != [left_index - left_start - 1]:
            raise SampleError(
                "Excel 的 <PAUSE> 位于无法安全映射的合并词内部："
                f"{xlsx_words[left_start:left_index]}"
            )
        if positive_offsets:
            output_labels[lmdb_indices[right_index - 1]] = 1

    if left_index != len(xlsx_words) or right_index != len(lmdb_words):
        raise SampleError(
            "Excel 与 LMDB 词数未完全对齐："
            f"xlsx={len(xlsx_words)}, lmdb={len(lmdb_words)}"
        )
    return output_labels


def lookup_xlsx_label(labels_by_number, lmdb_key):
    match = re.search(r"(\d+)$", str(lmdb_key))
    if not match:
        raise SampleError(f"LMDB key 没有数字后缀，无法匹配 Excel：{lmdb_key}")
    numeric_id = int(match.group(1))
    try:
        return labels_by_number[numeric_id]
    except KeyError as exc:
        raise SampleError(f"Excel 中找不到数字 ID={numeric_id}，LMDB key={lmdb_key}") from exc


def decode_datum_arrays(datum, vocab, decode_wav=True):
    try:
        text = datum.text.decode("utf-8")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise SampleError(f"datum.text 不是有效 UTF-8 bytes：{exc}") from exc

    phoneme_raw = np.frombuffer(datum.phonemes, dtype=np.int16)
    if phoneme_raw.size == 0:
        raise SampleError("datum.phonemes 为空")
    if len(datum.fa) == 0 or len(datum.fa) % np.dtype(np.float32).itemsize != 0:
        raise SampleError(f"datum.fa 无法按 float32 解析：字节数={len(datum.fa)}")

    # 已通过真实 LMDB 样本确认存在两种布局：
    # 1. 截图原格式 int16_x3，每个逻辑音素 3 个 int16；
    # 2. 新语料格式 int16_x6，每个逻辑音素 6 个 int16，第 0 列为实际音素 ID。
    # 两种格式都使用每个逻辑音素一个 float32 FA 时间，严格按数量判定，不猜测或截断。
    audio_fa = np.copy(np.frombuffer(datum.fa, dtype=np.float32).reshape(-1))
    if phoneme_raw.size % 3 == 0 and phoneme_raw.size // 3 == len(audio_fa):
        phoneme_array = np.copy(phoneme_raw.reshape(-1, 3))
        phoneme_storage_layout = "int16_x3"
    elif phoneme_raw.size % 6 == 0 and phoneme_raw.size // 6 == len(audio_fa):
        phoneme_array = np.copy(phoneme_raw.reshape(-1, 6))
        phoneme_storage_layout = "int16_x6"
    else:
        raise SampleError(
            "无法识别 phonemes 与 fa 的对应格式："
            f"phoneme_int16_count={phoneme_raw.size}, fa_count={len(audio_fa)}"
        )

    phone_names = ids_to_string(phoneme_array[:, 0].tolist(), vocab).split()
    if len(phone_names) < 2:
        raise SampleError("去掉句首 ENsil 前的音素序列长度不足")

    # audio_fa[0] 是被截图原脚本一并切掉的句首 ENsil 结束时间。
    # V4 需要保留该边界，确保第一个词的声学窗口不包含句首静音。
    leading_silence_end_seconds = float(audio_fa[0])
    if not math.isfinite(leading_silence_end_seconds) or leading_silence_end_seconds < 0:
        raise SampleError(f"句首静音结束时间非法：{leading_silence_end_seconds}")

    wav = None
    if decode_wav:
        wav = np.copy(np.frombuffer(datum.wav, dtype=np.int16).reshape(-1))
        if wav.size == 0:
            raise SampleError("datum.wav 为空")
    return (
        text,
        phone_names[1:],
        audio_fa[1:],
        wav,
        phoneme_storage_layout,
        leading_silence_end_seconds,
    )


def extract_word_pause_like_screenshot(text, phones, audio_fa, pause_threshold):
    """逐步复现截图中的 word_list、word_end_time、pause_label。"""
    tokens = text.split()
    if not tokens:
        raise SampleError("datum.text 分词后为空")

    word_list = []
    word_end_time = []
    phone_index = 0
    for token in tokens:
        if ">" not in token:
            word_list.append(token)
            if phone_index < len(phones) and phones[phone_index] in LEADING_SILENCE_PHONES:
                word_end_time.append(float(audio_fa[phone_index]))
                phone_index += 1
            else:
                word_end_time.append(None)
            continue

        # 截图原式为 token.split(">")，正常词只能包含一个 >。
        parts = token.split(">")
        if len(parts) != 2:
            raise SampleError(f"词 token 的 > 数量不是 1：{token!r}")
        word, phone_text = parts
        expected_phones = [re.sub(r"\d+$", "", value) for value in phone_text.split("+") if value]
        if not expected_phones:
            raise SampleError(f"词 token 没有音素：{token!r}")
        phone_index += len(expected_phones)
        if phone_index > len(audio_fa):
            raise SampleError(
                f"词 {word!r} 需要 {len(expected_phones)} 个音素，但 fa 已越界："
                f"phone_index={phone_index}, fa_len={len(audio_fa)}"
            )
        word_list.append(word)
        word_end_time.append(float(audio_fa[phone_index - 1]))

    pause_label = [0] * len(word_list)
    pause_durations = []
    for index, word in enumerate(word_list):
        # 截图原逻辑：第一个无持续时间，最后一个对应句末 ENsil，均跳过。
        if index == 0 or index == len(word_list) - 1:
            continue
        if word_end_time[index] is None or word_end_time[index - 1] is None:
            raise SampleError(f"停顿时长计算遇到 None：index={index}, word={word!r}")
        duration = word_end_time[index] - word_end_time[index - 1]
        if word in PAUSE_SYMBOLS and duration > pause_threshold:
            pause_label[index] = 1
            pause_durations.append(float(duration))

    # 原脚本在这里进入 pdb；批处理不能交互，改为明确记录该样本错误。
    # 截图原脚本在这里用 pdb 检查 #。批处理不能交互，因此记录数量后，
    # 继续执行截图紧随其后的 special_pause 清理逻辑；# 会被删除并归并到前一词。
    hash_marker_count = word_list.count("#")

    new_word_list = []
    new_pause_label = []
    new_word_end_time = []
    for word, end_time, label in zip(word_list, word_end_time, pause_label):
        if word in SCREENSHOT_SPECIAL_PAUSE:
            if new_pause_label:
                new_pause_label[-1] = max(new_pause_label[-1], label)
            if new_word_end_time:
                new_word_end_time[-1] = end_time
            continue
        new_word_list.append(word)
        new_pause_label.append(label)
        new_word_end_time.append(end_time)

    return new_word_list, new_pause_label, new_word_end_time, pause_durations, hash_marker_count


def normalize_wav_like_screenshot(wav):
    """逐式复现截图第 162--165 行；wav_f32 按截图计算但不作为输出。"""
    wav_f32 = wav.astype(np.float32) / 32768.0
    _ = wav_f32
    wav = wav / np.abs(wav).max() * 0.6
    return (wav * 32768).astype(np.int16)


def write_pcm16_wav(path, wav, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(np.asarray(wav, dtype="<i2").tobytes())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def inspect_external_pcm16_wav(path, expected_sample_rate, alignment_duration):
    """检查外部训练 WAV；只读取文件头，不重写、不归一化音频。"""
    path = Path(path)
    if not path.is_file():
        raise SampleError(f"找不到 Excel 第一列对应的外部 WAV：{path}")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise SampleError(f"外部 WAV 无法读取：{path}: {exc}") from exc
    if channels != 1 or sample_width != 2:
        raise SampleError(
            f"外部 WAV 必须为单声道 PCM16：{path}, "
            f"channels={channels}, sample_width={sample_width}"
        )
    if sample_rate != expected_sample_rate:
        raise SampleError(
            f"外部 WAV 采样率错误：{path}, expected={expected_sample_rate}, actual={sample_rate}"
        )
    if frame_count <= 0:
        raise SampleError(f"外部 WAV 没有采样点：{path}")
    duration = frame_count / sample_rate
    if alignment_duration > duration + 0.05:
        raise SampleError(
            f"LMDB 对齐时间超出外部 WAV：{path}, "
            f"alignment={alignment_duration:.6f}, wav={duration:.6f}"
        )
    return frame_count, duration


def is_storage_exhausted_error(exc):
    """磁盘空间或用户配额耗尽时必须停止，不能当作普通坏样本继续写错误日志。"""
    return isinstance(exc, OSError) and exc.errno in {28, 122}


def process_datum(datum, vocab, pause_threshold, decode_wav=True):
    (
        text,
        phones,
        audio_fa,
        wav,
        phoneme_storage_layout,
        leading_silence_end_seconds,
    ) = decode_datum_arrays(datum, vocab, decode_wav=decode_wav)
    alignment_duration = float(audio_fa[-1])
    if not math.isfinite(alignment_duration) or alignment_duration <= 0:
        raise SampleError(f"最后一个 fa 时间非法：{alignment_duration}")
    word_list, pause_label, word_end_time, pause_durations, hash_marker_count = extract_word_pause_like_screenshot(
        text, phones, audio_fa, pause_threshold
    )
    first_word_boundary = next(
        (
            float(end_time)
            for word, end_time in zip(word_list, word_end_time)
            if word not in PAUSE_SYMBOLS and end_time is not None and math.isfinite(float(end_time))
        ),
        None,
    )
    if first_word_boundary is None:
        raise SampleError("清理停顿符号后找不到第一个有效词边界")
    if leading_silence_end_seconds > first_word_boundary:
        raise SampleError(
            "句首静音结束时间晚于第一个有效词边界："
            f"leading={leading_silence_end_seconds}, first_word_end={first_word_boundary}"
        )
    return (
        normalize_wav_like_screenshot(wav) if wav is not None else None,
        word_list,
        pause_label,
        word_end_time,
        pause_durations,
        text,
        alignment_duration,
        hash_marker_count,
        phoneme_storage_layout,
        leading_silence_end_seconds,
    )


def load_config_entries(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("file_type") != "LMDB":
        raise ValueError(f"配置 file_type 必须为 LMDB，实际为 {config.get('file_type')!r}")
    entries = []
    source_index = 0
    for config_group in CONFIG_GROUPS:
        group_entries = config.get(config_group, [])
        if not isinstance(group_entries, list):
            raise ValueError(f"配置字段 {config_group} 必须是列表")
        for item in group_entries:
            if not isinstance(item, dict) or "lmdb_path" not in item:
                raise ValueError(f"{config_group} 中存在缺少 lmdb_path 的配置项")
            entries.append((source_index, config_group, item))
            source_index += 1
    if not entries:
        raise ValueError("配置中没有可汇总的 LMDB")
    return entries


def existing_ids(manifest_path):
    ids = set()
    if not manifest_path.is_file():
        return ids
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["utt_id"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"已有 manifest 第 {line_number} 行损坏：{exc}") from exc
    return ids


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def summarize_manifest(manifest_path, run_counters, source_reports):
    total = Counter()
    audio_durations = []
    alignment_durations = []
    words_per_utt = []
    pause_durations = []
    by_source = defaultdict(Counter)
    total_wav_bytes = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            word_list = item["word_list"]
            pause_label = item["pause_label"]
            positive_count = int(sum(pause_label))
            total["utterances"] += 1
            total["words"] += len(word_list)
            total["positive_words"] += positive_count
            total["utterances_with_pause"] += int(positive_count > 0)
            total["audio_samples"] += int(item["audio_num_samples"])
            total[f"phoneme_layout_{item.get('phoneme_storage_layout', 'unknown')}"] += 1
            audio_durations.append(float(item["audio_duration_seconds"]))
            alignment_durations.append(float(item["alignment_duration_seconds"]))
            words_per_utt.append(len(word_list))
            pause_durations.extend(float(value) for value in item["positive_pause_durations_seconds"])
            source_key = f"{item['source_index']:03d}:{item['source_lmdb_spec']}"
            by_source[source_key]["utterances"] += 1
            by_source[source_key]["words"] += len(word_list)
            by_source[source_key]["positive_words"] += positive_count
            wav_path = Path(item["wav_path"])
            if wav_path.is_file():
                total_wav_bytes += wav_path.stat().st_size

    lines = [
        "音频停顿预测 V4：LMDB 批量提取统计",
        "=" * 72,
        f"Manifest: {manifest_path}",
        f"统计生成时间戳由文件修改时间记录；脚本不随机抽样 ratio/times。",
        "",
        "一、总体运行统计",
        f"配置 LMDB 数: {len(source_reports)}",
        f"成功打开 LMDB 数: {sum(1 for x in source_reports if x['status'] == 'ok')}",
        f"失败 LMDB 数: {sum(1 for x in source_reports if x['status'] != 'ok')}",
        f"扫描 key 行数: {run_counters['key_lines']}",
        f"本次新导出样本数: {run_counters['exported']}",
        f"断点续跑跳过样本数: {run_counters['resume_skipped']}",
        f"skip-key-substring 跳过数: {run_counters['substring_skipped']}",
        f"LMDB 缺失 key 数: {run_counters['missing_lmdb_key']}",
        f"单样本解析失败数: {run_counters['sample_errors']}",
        f"phonemes int16_x3 样本数: {total['phoneme_layout_int16_x3']}",
        f"phonemes int16_x6 样本数: {total['phoneme_layout_int16_x6']}",
        f"phonemes 未记录布局样本数: {total['phoneme_layout_unknown']}",
        f"导出 WAV 总字节: {total_wav_bytes}",
        "",
        "二、合并后总数据统计（尚未划分 train/valid/test）",
        f"句子数: {total['utterances']}",
        f"按 24000 Hz PCM 采样点计算的音频总时长(小时): {sum(audio_durations) / 3600.0:.6f}",
        f"WAV 音频时长 秒 min/mean/p50/p90/p95/max: "
        f"{min(audio_durations, default=0):.4f}/"
        f"{(sum(audio_durations) / len(audio_durations) if audio_durations else 0):.4f}/"
        f"{percentile(audio_durations, 50):.4f}/{percentile(audio_durations, 90):.4f}/"
        f"{percentile(audio_durations, 95):.4f}/{max(audio_durations, default=0):.4f}",
        f"按 fa 末时间累计时长(小时): {sum(alignment_durations) / 3600.0:.6f}",
        f"fa 对齐时长 秒 min/mean/p50/p90/p95/max: "
        f"{min(alignment_durations, default=0):.4f}/"
        f"{(sum(alignment_durations) / len(alignment_durations) if alignment_durations else 0):.4f}/"
        f"{percentile(alignment_durations, 50):.4f}/{percentile(alignment_durations, 90):.4f}/"
        f"{percentile(alignment_durations, 95):.4f}/{max(alignment_durations, default=0):.4f}",
        f"原始 PCM 采样点总数: {total['audio_samples']}",
        f"词总数: {total['words']}",
        f"每句词数 min/mean/p50/p90/p95/max: "
        f"{min(words_per_utt, default=0)}/"
        f"{(sum(words_per_utt) / len(words_per_utt) if words_per_utt else 0):.4f}/"
        f"{percentile(words_per_utt, 50):.2f}/{percentile(words_per_utt, 90):.2f}/"
        f"{percentile(words_per_utt, 95):.2f}/{max(words_per_utt, default=0)}",
        f"正停顿词数: {total['positive_words']}",
        f"正停顿占最终 word_list 比例: "
        f"{(total['positive_words'] / total['words'] if total['words'] else 0):.6%}",
        f"含停顿句子数/比例: {total['utterances_with_pause']}/"
        f"{(total['utterances_with_pause'] / total['utterances'] if total['utterances'] else 0):.6%}",
        f"正停顿时长 秒 min/mean/p50/p90/p95/max: "
        f"{min(pause_durations, default=0):.4f}/"
        f"{(sum(pause_durations) / len(pause_durations) if pause_durations else 0):.4f}/"
        f"{percentile(pause_durations, 50):.4f}/{percentile(pause_durations, 90):.4f}/"
        f"{percentile(pause_durations, 95):.4f}/{max(pause_durations, default=0):.4f}",
        "",
        "三、按原 LMDB 统计",
    ]
    for source_key in sorted(by_source):
        counter = by_source[source_key]
        word_count = counter["words"]
        lines.extend(
            [
                f"[{source_key}]",
                f"  句子数: {counter['utterances']}",
                f"  最终 word_list 元素数: {word_count}",
                f"  正停顿词数/比例: {counter['positive_words']}/"
                f"{(counter['positive_words'] / word_count if word_count else 0):.6%}",
            ]
        )

    lines.extend(["", "四、LMDB 打开与扫描状态（config_group 仅记录原 JSON 来源，不是数据划分）"])
    for report in source_reports:
        lines.append(
            f"[{report['source_index']:03d}] config_group={report['config_group']} status={report['status']} "
            f"keys={report.get('key_lines', 0)} exported={report.get('exported', 0)} "
            f"ratio={report.get('ratio')} times={report.get('times')} "
            f"lmdb={report['lmdb_spec']} message={report.get('message', '')}"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.sample_rate != 24000 or args.pause_threshold_seconds < 0:
        raise ValueError("已确认 datum.wav 原始采样率为 24000；sample-rate 必须为 24000")
    if args.num_parts <= 0:
        raise ValueError("--num-parts 必须大于 0")
    if not 0 <= args.part_index < args.num_parts:
        raise ValueError("--part-index 必须满足 0 <= part-index < num-parts")
    if args.source_index_offset < 0:
        raise ValueError("--source-index-offset 不能为负数")

    try:
        import lmdb
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("运行需要安装截图中使用的 lmdb 和 tqdm") from exc

    config_path = Path(args.config_json).resolve()
    output_root = Path(args.output_root).resolve()
    wav_root = output_root / "wavs"
    manifest_path = output_root / "all_pause_data.jsonl"
    errors_path = output_root / "extraction_errors.jsonl"
    source_report_path = output_root / "source_report.jsonl"
    stats_path = output_root / "training_data_statistics.txt"

    output_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"输出 manifest 已存在：{manifest_path}。如需断点续跑，请增加 --resume。"
        )

    all_entries = load_config_entries(config_path)
    selected_entries = [
        item for item in all_entries if item[0] % args.num_parts == args.part_index
    ]
    entries = [
        (source_index + args.source_index_offset, config_group, entry)
        for source_index, config_group, entry in selected_entries
    ]
    if not entries:
        raise ValueError(
            f"当前分片没有 LMDB 来源：part-index={args.part_index}, "
            f"num-parts={args.num_parts}, source-count={len(all_entries)}"
        )
    vocab = load_vocab(args.phone_vocab)
    datum_class = load_datum_class(args.protobuf_python_dir)
    completed_ids = existing_ids(manifest_path) if args.resume else set()
    run_counters = Counter()
    source_reports = []

    manifest_mode = "a" if args.resume else "w"
    error_mode = "a" if args.resume else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8", newline="\n") as manifest_handle, \
            errors_path.open(error_mode, encoding="utf-8", newline="\n") as error_handle:
        for source_index, config_group, entry in entries:
            lmdb_spec = entry["lmdb_path"]
            report = {
                "source_index": source_index,
                "config_group": config_group,
                "lmdb_spec": lmdb_spec,
                "ratio": entry.get("ratio"),
                "times": entry.get("times"),
                "status": "failed",
                "key_lines": 0,
                "exported": 0,
                "resume_skipped": 0,
            }
            source_reports.append(report)
            env = None
            try:
                label_xlsx_path = entry.get("pause_label_xlsx")
                label_pause_token = entry.get("pause_token", "<PAUSE>")
                external_wav_root = entry.get("external_wav_root")
                xlsx_labels = None
                xlsx_text_index = None
                if label_xlsx_path:
                    xlsx_labels = load_xlsx_pause_labels(
                        label_xlsx_path, pause_token=label_pause_token
                    )
                    xlsx_text_index, invalid_xlsx_rows = build_xlsx_text_index(
                        xlsx_labels, pause_token=label_pause_token
                    )
                    report["pause_label_source"] = "xlsx_second_column"
                    report["excel_match_strategy"] = "normalized_full_sentence"
                    report["pause_label_xlsx"] = str(label_xlsx_path)
                    report["pause_label_rows"] = len(xlsx_labels)
                    report["pause_label_unique_sentence_count"] = len(xlsx_text_index)
                    report["pause_label_invalid_rows"] = invalid_xlsx_rows
                    report["pause_label_rows"] = len(xlsx_labels)
                else:
                    report["pause_label_source"] = "lmdb_alignment_legacy"
                if external_wav_root and xlsx_labels is None:
                    raise ValueError("配置 external_wav_root 时必须同时配置 pause_label_xlsx")
                if external_wav_root:
                    report["wav_source"] = "external_wav_root"
                    report["external_wav_root"] = str(external_wav_root)
                else:
                    report["wav_source"] = "datum.wav_legacy"
                lmdb_path, key_path, key_tag = resolve_lmdb_and_key(entry, args.default_key_tag)
                report["resolved_lmdb_path"] = str(lmdb_path)
                report["resolved_key_path"] = str(key_path)
                report["key_tag"] = key_tag
                if not lmdb_path.is_dir():
                    raise FileNotFoundError(f"LMDB 目录不存在：{lmdb_path}")
                if not key_path.is_file():
                    raise FileNotFoundError(f"spkid.key 不存在：{key_path}")
                env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
                with key_path.open("r", encoding="utf-8") as key_handle:
                    key_lines = key_handle.readlines()
                report["key_lines"] = len(key_lines)
                run_counters["key_lines"] += len(key_lines)

                exported_this_source = 0
                with env.begin() as txn:
                    for line_number, key_line in enumerate(
                        tqdm(key_lines, desc=f"all:{source_index:03d}", unit="sample"), 1
                    ):
                        if args.skip_key_substring and args.skip_key_substring in key_line:
                            run_counters["substring_skipped"] += 1
                            continue
                        try:
                            key = parse_key_line(key_line)
                            sample_id = safe_sample_id(source_index, key)
                            if sample_id in completed_ids:
                                run_counters["resume_skipped"] += 1
                                report["resume_skipped"] += 1
                                continue
                            raw = txn.get(key.encode("utf-8"))
                            if raw is None:
                                run_counters["missing_lmdb_key"] += 1
                                raise SampleError("spkid.key 中的 key 在 LMDB 中不存在")
                            datum = datum_class()
                            datum.ParseFromString(raw)
                            (
                                wav,
                                word_list,
                                pause_label,
                                word_end_time,
                                pause_durations,
                                text,
                                alignment_duration,
                                hash_marker_count,
                                phoneme_storage_layout,
                                leading_silence_end_seconds,
                            ) = process_datum(
                                datum,
                                vocab,
                                args.pause_threshold_seconds,
                                decode_wav=not bool(external_wav_root),
                            )
                            label_audit = None
                            if xlsx_labels is not None:
                                label_audit = lookup_xlsx_label_by_lmdb_words(
                                    xlsx_text_index,
                                    word_list,
                                    key,
                                )
                                pause_label = labels_from_xlsx_for_lmdb_words(
                                    word_list,
                                    label_audit["tagged_text"],
                                    pause_token=label_pause_token,
                                )
                                # 此字段只用于统计展示；训练标签只来自 Excel 第二列。
                                pause_durations = []
                            if external_wav_root:
                                wav_path = Path(external_wav_root) / label_audit["wav_id"]
                                audio_num_samples, audio_duration_seconds = (
                                    inspect_external_pcm16_wav(
                                        wav_path,
                                        args.sample_rate,
                                        alignment_duration,
                                    )
                                )
                            else:
                                wav_path = wav_root / f"{sample_id}.wav"
                                write_pcm16_wav(wav_path, wav, args.sample_rate)
                                audio_num_samples = int(len(wav))
                                audio_duration_seconds = float(len(wav) / args.sample_rate)
                            record = {
                                "utt_id": sample_id,
                                "speaker_id": key_line.split("@", 1)[0].strip(),
                                "wav_path": str(wav_path),
                                "word_list": word_list,
                                "word_end_time": word_end_time,
                                "pause_label": pause_label,
                                "positive_pause_durations_seconds": pause_durations,
                                "audio_num_samples": int(audio_num_samples),
                                "audio_duration_seconds": float(audio_duration_seconds),
                                "alignment_duration_seconds": float(alignment_duration),
                                "leading_silence_end_seconds": float(leading_silence_end_seconds),
                                "sample_rate": args.sample_rate,
                                "source_index": source_index,
                                "source_config_group": config_group,
                                "source_lmdb_spec": lmdb_spec,
                                "source_lmdb_key": key,
                                "source_ratio": entry.get("ratio"),
                                "source_times": entry.get("times"),
                                "source_py_ratio": entry.get("py_ratio"),
                                "source_text": text,
                                "hash_marker_count_before_cleanup": hash_marker_count,
                                "phoneme_storage_layout": phoneme_storage_layout,
                            }
                            if label_audit is not None:
                                record.update(
                                    {
                                        "pause_label_source": "xlsx_second_column",
                                        "pause_label_xlsx": str(label_xlsx_path),
                                        "pause_label_wav_id": label_audit["wav_id"],
                                        "pause_label_text": label_audit["tagged_text"],
                                        "pause_label_match_strategy": "normalized_full_sentence",
                                        "wav_source": "external_wav_root",
                                    }
                                )
                            manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            manifest_handle.flush()
                            completed_ids.add(sample_id)
                            run_counters["exported"] += 1
                            report["exported"] += 1
                            exported_this_source += 1
                            if (
                                args.max_samples_per_source > 0
                                and exported_this_source >= args.max_samples_per_source
                            ):
                                break
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            if is_storage_exhausted_error(exc):
                                raise
                            run_counters["sample_errors"] += 1
                            error_handle.write(
                                json.dumps(
                                    {
                                        "source_index": source_index,
                                        "config_group": config_group,
                                        "lmdb_spec": lmdb_spec,
                                        "line_number": line_number,
                                        "key_line": key_line.rstrip("\n"),
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            error_handle.flush()
                report["status"] = "ok"
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if is_storage_exhausted_error(exc):
                    raise RuntimeError(
                        f"输出存储空间或用户磁盘配额已耗尽（errno={exc.errno}）。"
                        "请把 --output-root 改到有足够容量且有写配额的文件系统后，"
                        "使用 --resume 续跑。"
                    ) from exc
                report["message"] = f"{type(exc).__name__}: {exc}"
            finally:
                if env is not None:
                    try:
                        env.close()
                    except OSError as exc:
                        if not is_storage_exhausted_error(exc):
                            raise
                        report["close_warning"] = f"{type(exc).__name__}: {exc}"

    with source_report_path.open("w", encoding="utf-8", newline="\n") as handle:
        for report in source_reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    stats_text = summarize_manifest(manifest_path, run_counters, source_reports)
    stats_path.write_text(stats_text, encoding="utf-8")

    print(stats_text)
    print(f"未划分总数据 JSONL：{manifest_path}")
    print(f"WAV 目录：{wav_root}")
    print(f"统计文件：{stats_path}")
    print(f"错误明细：{errors_path}")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        if is_storage_exhausted_error(exc):
            raise SystemExit(
                f"输出存储空间或用户磁盘配额已耗尽（errno={exc.errno}）。"
                "请更换有足够容量且有写配额的输出目录后续跑。"
            ) from None
        raise
