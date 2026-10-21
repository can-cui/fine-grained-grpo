#!/usr/bin/env python3
"""Batch-export V4 pause training data from all LMDBs listed in the configuration JSON.

This implementation follows only the fields and logic shown in the user-provided filter_fa_lmdb.py screenshots:
Datum_LLMTTS.text / phonemes / fa / wav、spkid.key、ENsil/ENL3、
the word>phone+phone text format and pause detection based on punctuation duration exceeding a threshold.
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
    """A single LMDB record cannot be converted safely."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-json",
        required=True,
        help="JSON containing lists of source LMDBs; the lists are read and merged as sources, not used as data splits",
    )
    parser.add_argument("--output-root", required=True, help="Directory for exported WAV, JSONL, and statistics files")
    parser.add_argument("--protobuf-python-dir", required=True, help="Internal directory containing datum_llmtts_pb2.py")
    parser.add_argument("--phone-vocab", required=True, help="vocab_extended.txt shown in the screenshots")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Confirmed original sample rate of datum.wav")
    parser.add_argument("--pause-threshold-seconds", type=float, default=0.03)
    parser.add_argument("--default-key-tag", default="100ms10khz")
    parser.add_argument("--skip-key-substring", default="lmdb_0_Long20_60s_HeadTail200ms")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples-per-source", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1, help="Total number of LMDB source shards")
    parser.add_argument("--part-index", type=int, default=0, help="Current shard index, starting at 0")
    parser.add_argument(
        "--source-index-offset",
        type=int,
        default=0,
        help="Offset added to configured source_index values before writing records; avoids source-index conflicts when adding LMDBs",
    )
    return parser.parse_args()


def load_vocab(vocab_file):
    """Read the first field of each line, following the screenshot logic."""
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                vocab.append(parts[0])
    if not vocab:
        raise ValueError(f"Phone vocabulary is empty: {vocab_file}")
    return vocab


def ids_to_string(id_list, vocab):
    """Map out-of-range IDs to <UNK>, following the screenshot logic."""
    return " ".join(vocab[phone_id] if 0 <= phone_id < len(vocab) else "<UNK>" for phone_id in id_list)


def load_datum_class(protobuf_python_dir):
    """Load the Datum_LLMTTS shown in the screenshots only from the user-specified company intranet directory."""
    protobuf_dir = str(Path(protobuf_python_dir).resolve())
    if protobuf_dir not in sys.path:
        sys.path.insert(0, protobuf_dir)
    module = importlib.import_module("datum_llmtts_pb2")
    try:
        return getattr(module, "Datum_LLMTTS")
    except AttributeError as exc:
        raise ImportError(
            "Could not find the Datum_LLMTTS shown in the screenshots in datum_llmtts_pb2; "
            "confirm that PROTOBUF_PYTHON_DIR points to the internal code used to generate this LMDB."
        ) from exc


def split_lmdb_spec(lmdb_spec):
    """Parse /path/lmdb:100ms10khz into an LMDB directory and key tag."""
    value = str(lmdb_spec).strip()
    if not value:
        raise ValueError("lmdb_path is empty")
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
                f"{lmdb_path}: multiple candidate key files found; explicitly add key_path to the configuration: {existing}"
            )
        else:
            key_path = candidates[-1]
    return lmdb_path, key_path, key_tag or default_key_tag


def parse_key_line(key_line):
    """Follow the screenshots exactly: use the text after @ and before the space as the LMDB key."""
    if "@" not in key_line:
        raise SampleError("spkid.key line is missing the @ delimiter required by the screenshots")
    key = key_line.split("@", 1)[1].split(" ", 1)[0].strip()
    if not key:
        raise SampleError("Parsed key from spkid.key line is empty")
    return key


def safe_sample_id(source_index, key):
    readable = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("._-")[:80] or "sample"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"all_{source_index:03d}_{readable}_{digest}"


def xlsx_column_index(cell_reference):
    """Convert Excel column names such as A/B/AA to zero-based indices."""
    match = re.match(r"([A-Za-z]+)", str(cell_reference))
    if not match:
        raise ValueError(f"Could not parse Excel cell reference: {cell_reference!r}")
    result = 0
    for character in match.group(1).upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def read_xlsx_first_two_columns(path):
    """Use only the standard library to read columns A/B of the first worksheet, avoiding a new Excel dependency."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pause-label Excel file not found: {path}")
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
            raise ValueError(f"Excel file contains no worksheets: {path}")
        relationship_id = first_sheet.attrib.get(f"{{{rel_ns}}}id")
        rel_root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in rel_root.findall(f"{{{pkg_rel_ns}}}Relationship"):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"Could not locate the first Excel worksheet: {path}")
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
    """Read WAV IDs from the first Excel column and labels from the second; ignore repeated header rows."""
    labels_by_number = {}
    for row_number, (wav_id, tagged_text) in enumerate(
        read_xlsx_first_two_columns(path), 1
    ):
        if wav_id in {"\u6587\u4ef6\u540d\u79f0", "\u6587\u4ef6\u540d", "wavid", "wav_id"}:
            continue
        if not wav_id or not tagged_text:
            raise ValueError(f"{path}: first or second column is empty on row {row_number}")
        stem = Path(wav_id).stem
        match = re.search(r"(\d+)$", stem)
        if not match:
            raise ValueError(f"{path}: WAV ID on row {row_number} has no numeric suffix: {wav_id}")
        numeric_id = int(match.group(1))
        if numeric_id in labels_by_number:
            raise ValueError(f"{path}: duplicate numeric ID: {numeric_id}")
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
        raise ValueError(f"Pause-label Excel file contains no valid data: {path}")
    return labels_by_number


def normalize_alignment_word(word):
    return re.sub(r"[^0-9a-z]+", "", str(word).lower().replace("’", "'"))


def parse_tagged_pause_words(tagged_text, pause_token="<PAUSE>"):
    """Convert the second Excel column into lexical words and binary labels assigning each pause to the preceding word."""
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
                raise SampleError("<PAUSE> appears before the first word in the second Excel column")
            labels[-1] = 1
            continue
        normalized = normalize_alignment_word(piece)
        if normalized:
            words.append(normalized)
            labels.append(0)
    if not words:
        raise SampleError("No usable English words in the second Excel column")
    return words, labels


def normalized_lmdb_lexical_words(word_list):
    """Get normalized lexical words from the LMDB word_list, ignoring silence and punctuation tokens."""
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
    """Build an Excel content index keyed by the full sentence after removing <PAUSE> and punctuation."""
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
        raise ValueError("Pause-label Excel file has no English records usable for sentence matching")
    return index, invalid_rows


def lookup_xlsx_label_by_lmdb_words(text_index, word_list, lmdb_key):
    """Match by full sentence content first; disambiguate duplicate sentences using the numeric key suffix."""
    lmdb_words = normalized_lmdb_lexical_words(word_list)
    text_key = "".join(lmdb_words)
    candidates = text_index.get(text_key, [])
    if not candidates:
        raise SampleError(
            "No Excel record matches the full sentence content in LMDB: "
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
        "Multiple identical sentences exist in Excel and cannot be uniquely disambiguated using the LMDB key: "
        f"key={lmdb_key}, candidates={[row['wav_id'] for row in candidates]}"
    )


def labels_from_xlsx_for_lmdb_words(word_list, tagged_text, pause_token="<PAUSE>"):
    """Align Excel and LMDB word sequences and map Excel labels to LMDB lexical words."""
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
        raise SampleError("LMDB word_list contains no lexical words")

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
                    "Excel and LMDB word sequences do not match: "
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
                "Excel <PAUSE> falls inside a merged word and cannot be mapped safely: "
                f"{xlsx_words[left_start:left_index]}"
            )
        if positive_offsets:
            output_labels[lmdb_indices[right_index - 1]] = 1

    if left_index != len(xlsx_words) or right_index != len(lmdb_words):
        raise SampleError(
            "Excel and LMDB word counts did not align completely: "
            f"xlsx={len(xlsx_words)}, lmdb={len(lmdb_words)}"
        )
    return output_labels


def lookup_xlsx_label(labels_by_number, lmdb_key):
    match = re.search(r"(\d+)$", str(lmdb_key))
    if not match:
        raise SampleError(f"LMDB key has no numeric suffix and cannot be matched to Excel: {lmdb_key}")
    numeric_id = int(match.group(1))
    try:
        return labels_by_number[numeric_id]
    except KeyError as exc:
        raise SampleError(f"Numeric ID={numeric_id} not found in Excel, LMDB key={lmdb_key}") from exc


def decode_datum_arrays(datum, vocab, decode_wav=True):
    try:
        text = datum.text.decode("utf-8")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise SampleError(f"datum.text is not valid UTF-8 bytes: {exc}") from exc

    phoneme_raw = np.frombuffer(datum.phonemes, dtype=np.int16)
    if phoneme_raw.size == 0:
        raise SampleError("datum.phonemes is empty")
    if len(datum.fa) == 0 or len(datum.fa) % np.dtype(np.float32).itemsize != 0:
        raise SampleError(f"Could not parse datum.fa as float32: byte count={len(datum.fa)}")

    # Real LMDB samples confirm two layouts:
    # 1. Original screenshot format int16_x3: 3 int16 values per logical phoneme.
    # 2. New corpus format int16_x6: 6 int16 values per logical phoneme; column 0 is the actual phoneme ID.
    # Both formats use one float32 FA timestamp per logical phoneme. Determine the format strictly by count; do not guess or truncate.
    audio_fa = np.copy(np.frombuffer(datum.fa, dtype=np.float32).reshape(-1))
    if phoneme_raw.size % 3 == 0 and phoneme_raw.size // 3 == len(audio_fa):
        phoneme_array = np.copy(phoneme_raw.reshape(-1, 3))
        phoneme_storage_layout = "int16_x3"
    elif phoneme_raw.size % 6 == 0 and phoneme_raw.size // 6 == len(audio_fa):
        phoneme_array = np.copy(phoneme_raw.reshape(-1, 6))
        phoneme_storage_layout = "int16_x6"
    else:
        raise SampleError(
            "Could not identify the correspondence format between phonemes and fa: "
            f"phoneme_int16_count={phoneme_raw.size}, fa_count={len(audio_fa)}"
        )

    phone_names = ids_to_string(phoneme_array[:, 0].tolist(), vocab).split()
    if len(phone_names) < 2:
        raise SampleError("Phone sequence is too short before removing leading ENsil")

    # audio_fa[0] is the end time of the initial ENsil, which the original script shown in the screenshot also removed.
    # V4 must retain this boundary so the first word's acoustic window excludes leading silence.
    leading_silence_end_seconds = float(audio_fa[0])
    if not math.isfinite(leading_silence_end_seconds) or leading_silence_end_seconds < 0:
        raise SampleError(f"Invalid leading-silence end time: {leading_silence_end_seconds}")

    wav = None
    if decode_wav:
        wav = np.copy(np.frombuffer(datum.wav, dtype=np.int16).reshape(-1))
        if wav.size == 0:
            raise SampleError("datum.wav is empty")
    return (
        text,
        phone_names[1:],
        audio_fa[1:],
        wav,
        phoneme_storage_layout,
        leading_silence_end_seconds,
    )


def extract_word_pause_like_screenshot(text, phones, audio_fa, pause_threshold):
    """Reproduce the screenshot's word_list, word_end_time, and pause_label step by step."""
    tokens = text.split()
    if not tokens:
        raise SampleError("datum.text is empty after tokenization")

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

        # The original expression shown in the screenshot was token.split(">"); a normal word can contain only one >.
        parts = token.split(">")
        if len(parts) != 2:
            raise SampleError(f"Word token must contain exactly one >: {token!r}")
        word, phone_text = parts
        expected_phones = [re.sub(r"\d+$", "", value) for value in phone_text.split("+") if value]
        if not expected_phones:
            raise SampleError(f"Word token contains no phones: {token!r}")
        phone_index += len(expected_phones)
        if phone_index > len(audio_fa):
            raise SampleError(
                f"Word {word!r} requires {len(expected_phones)} phones, but fa is out of range: "
                f"phone_index={phone_index}, fa_len={len(audio_fa)}"
            )
        word_list.append(word)
        word_end_time.append(float(audio_fa[phone_index - 1]))

    pause_label = [0] * len(word_list)
    pause_durations = []
    for index, word in enumerate(word_list):
        # Original logic shown in the screenshot: skip the first item (no duration) and the last item (sentence-final ENsil).
        if index == 0 or index == len(word_list) - 1:
            continue
        if word_end_time[index] is None or word_end_time[index - 1] is None:
            raise SampleError(f"Encountered None while calculating pause duration: index={index}, word={word!r}")
        duration = word_end_time[index] - word_end_time[index - 1]
        if word in PAUSE_SYMBOLS and duration > pause_threshold:
            pause_label[index] = 1
            pause_durations.append(float(duration))

    # The original script entered pdb here; batch processing is non-interactive, so record this sample's error explicitly.
    # The original script shown in the screenshot used pdb to inspect #. Since batch processing is non-interactive, record the count,
    # then continue with the following special_pause cleanup logic; # is removed and merged into the preceding word.
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
    """Reproduce screenshot lines 162--165 exactly; calculate wav_f32 as shown but do not output it."""
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
    """Check external training WAV files by reading only their headers; do not rewrite or normalize audio."""
    path = Path(path)
    if not path.is_file():
        raise SampleError(f"External WAV corresponding to the first Excel column was not found: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise SampleError(f"Could not read external WAV: {path}: {exc}") from exc
    if channels != 1 or sample_width != 2:
        raise SampleError(
            f"External WAV must be mono PCM16: {path}, "
            f"channels={channels}, sample_width={sample_width}"
        )
    if sample_rate != expected_sample_rate:
        raise SampleError(
            f"External WAV has the wrong sample rate: {path}, expected={expected_sample_rate}, actual={sample_rate}"
        )
    if frame_count <= 0:
        raise SampleError(f"External WAV contains no samples: {path}")
    duration = frame_count / sample_rate
    if alignment_duration > duration + 0.05:
        raise SampleError(
            f"LMDB alignment time exceeds external WAV duration: {path}, "
            f"alignment={alignment_duration:.6f}, wav={duration:.6f}"
        )
    return frame_count, duration


def is_storage_exhausted_error(exc):
    """Stop if disk space or the user quota is exhausted; do not treat it as a bad sample and continue writing an error log."""
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
        raise SampleError(f"Invalid final fa timestamp: {alignment_duration}")
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
        raise SampleError("Could not find the first valid word boundary after removing pause markers")
    if leading_silence_end_seconds > first_word_boundary:
        raise SampleError(
            "Leading-silence end time is later than the first valid word boundary: "
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
        raise ValueError(f"Configuration file_type must be LMDB, got {config.get('file_type')!r}")
    entries = []
    source_index = 0
    for config_group in CONFIG_GROUPS:
        group_entries = config.get(config_group, [])
        if not isinstance(group_entries, list):
            raise ValueError(f"Configuration field {config_group} must be a list")
        for item in group_entries:
            if not isinstance(item, dict) or "lmdb_path" not in item:
                raise ValueError(f"{config_group} contains an entry missing lmdb_path")
            entries.append((source_index, config_group, item))
            source_index += 1
    if not entries:
        raise ValueError("Configuration contains no LMDBs to summarize")
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
                raise ValueError(f"Invalid existing manifest at line {line_number}: {exc}") from exc
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
        "Audio Pause Prediction V4: LMDB Batch Extraction Statistics",
        "=" * 72,
        f"Manifest: {manifest_path}",
        f"Statistics generation time is recorded using file modification time; the script does not randomly sample ratio/times.",
        "",
        "1. Overall run statistics",
        f"Configured LMDBs: {len(source_reports)}",
        f"LMDBs opened successfully: {sum(1 for x in source_reports if x['status'] == 'ok')}",
        f"LMDBs failed to open: {sum(1 for x in source_reports if x['status'] != 'ok')}",
        f"Key lines scanned: {run_counters['key_lines']}",
        f"New samples exported in this run: {run_counters['exported']}",
        f"Samples skipped during resume: {run_counters['resume_skipped']}",
        f"Skipped by skip-key-substring: {run_counters['substring_skipped']}",
        f"Keys missing from LMDB: {run_counters['missing_lmdb_key']}",
        f"Single-sample parsing failures: {run_counters['sample_errors']}",
        f"Samples with phonemes int16_x3 layout: {total['phoneme_layout_int16_x3']}",
        f"Samples with phonemes int16_x6 layout: {total['phoneme_layout_int16_x6']}",
        f"Samples with unrecorded phonemes layout: {total['phoneme_layout_unknown']}",
        f"Total bytes of exported WAV files: {total_wav_bytes}",
        "",
        "2. Combined-data statistics (train/valid/test split not yet created)",
        f"Sentences: {total['utterances']}",
        f"Total audio duration in hours (from 24000 Hz PCM samples): {sum(audio_durations) / 3600.0:.6f}",
        f"WAV duration in seconds min/mean/p50/p90/p95/max: "
        f"{min(audio_durations, default=0):.4f}/"
        f"{(sum(audio_durations) / len(audio_durations) if audio_durations else 0):.4f}/"
        f"{percentile(audio_durations, 50):.4f}/{percentile(audio_durations, 90):.4f}/"
        f"{percentile(audio_durations, 95):.4f}/{max(audio_durations, default=0):.4f}",
        f"Total duration in hours (from final fa timestamps): {sum(alignment_durations) / 3600.0:.6f}",
        f"fa alignment duration in seconds min/mean/p50/p90/p95/max: "
        f"{min(alignment_durations, default=0):.4f}/"
        f"{(sum(alignment_durations) / len(alignment_durations) if alignment_durations else 0):.4f}/"
        f"{percentile(alignment_durations, 50):.4f}/{percentile(alignment_durations, 90):.4f}/"
        f"{percentile(alignment_durations, 95):.4f}/{max(alignment_durations, default=0):.4f}",
        f"Total original PCM samples: {total['audio_samples']}",
        f"Total words: {total['words']}",
        f"Words per sentence min/mean/p50/p90/p95/max: "
        f"{min(words_per_utt, default=0)}/"
        f"{(sum(words_per_utt) / len(words_per_utt) if words_per_utt else 0):.4f}/"
        f"{percentile(words_per_utt, 50):.2f}/{percentile(words_per_utt, 90):.2f}/"
        f"{percentile(words_per_utt, 95):.2f}/{max(words_per_utt, default=0)}",
        f"Words with positive pauses: {total['positive_words']}",
        f"Positive pauses as a fraction of final word_list: "
        f"{(total['positive_words'] / total['words'] if total['words'] else 0):.6%}",
        f"Sentences with pauses/count ratio: {total['utterances_with_pause']}/"
        f"{(total['utterances_with_pause'] / total['utterances'] if total['utterances'] else 0):.6%}",
        f"Positive pause duration in seconds min/mean/p50/p90/p95/max: "
        f"{min(pause_durations, default=0):.4f}/"
        f"{(sum(pause_durations) / len(pause_durations) if pause_durations else 0):.4f}/"
        f"{percentile(pause_durations, 50):.4f}/{percentile(pause_durations, 90):.4f}/"
        f"{percentile(pause_durations, 95):.4f}/{max(pause_durations, default=0):.4f}",
        "",
        "3. Statistics by source LMDB",
    ]
    for source_key in sorted(by_source):
        counter = by_source[source_key]
        word_count = counter["words"]
        lines.extend(
            [
                f"[{source_key}]",
                f"  Sentences: {counter['utterances']}",
                f"  Final word_list elements: {word_count}",
                f"  Words with positive pauses/count ratio: {counter['positive_words']}/"
                f"{(counter['positive_words'] / word_count if word_count else 0):.6%}",
            ]
        )

    lines.extend(["", "4. LMDB open and scan status (config_group records source JSON only; it is not a data split)"])
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
        raise ValueError("The original datum.wav sample rate is confirmed as 24000; sample-rate must be 24000")
    if args.num_parts <= 0:
        raise ValueError("--num-parts must be greater than 0")
    if not 0 <= args.part_index < args.num_parts:
        raise ValueError("--part-index must satisfy 0 <= part-index < num-parts")
    if args.source_index_offset < 0:
        raise ValueError("--source-index-offset cannot be negative")

    try:
        import lmdb
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError("Install lmdb and tqdm, which are used in the screenshots, to run this script") from exc

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
            f"Output manifest already exists: {manifest_path}. Add --resume to continue from the last checkpoint."
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
            f"No LMDB sources assigned to the current shard: part-index={args.part_index}, "
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
                    raise ValueError("pause_label_xlsx must also be configured when external_wav_root is set")
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
                    raise FileNotFoundError(f"LMDB directory does not exist: {lmdb_path}")
                if not key_path.is_file():
                    raise FileNotFoundError(f"spkid.key does not exist: {key_path}")
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
                                raise SampleError("Key from spkid.key does not exist in LMDB")
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
                                # This field is for reporting only; training labels come exclusively from the second Excel column.
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
                        f"Output storage space or user disk quota exhausted (errno={exc.errno}). "
                        "Set --output-root to a filesystem with sufficient capacity and write quota, "
                        "then resume with --resume."
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
    print(f"Unsplit full-data JSONL: {manifest_path}")
    print(f"WAV directory: {wav_root}")
    print(f"Statistics file: {stats_path}")
    print(f"Error details: {errors_path}")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        if is_storage_exhausted_error(exc):
            raise SystemExit(
                f"Output storage space or user disk quota exhausted (errno={exc.errno}). "
                "Choose an output directory with sufficient capacity and write quota, then resume."
            ) from None
        raise
