#!/usr/bin/env python3
"""评测指定 checkpoint 在六个正式 split 上的词级与句级停顿指标。"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from fairseq import checkpoint_utils, utils


SPLITS = (
    "valid_m2_manual",
    "valid_m2_mfa",
    "valid_lmdb_mfa",
    "test_m2_manual",
    "test_m2_mfa",
    "test_lmdb_mfa",
)
METRICS = ("precision", "recall", "f1", "sentence_accuracy", "phrase_pause_accuracy")
PHRASE_EVALUABLE_SPLITS = frozenset((
    "valid_m2_manual",
    "valid_m2_mfa",
    "test_m2_manual",
    "test_m2_mfa",
))
FORMULA_REMARK = (
    "Precision=TP/(TP+FP，分母为0取0)；Recall=TP/(TP+FN，分母为0取0)；"
    "F1=2*Precision*Recall/(Precision+Recall，分母为0取0)；"
    "sentence_accuracy=预测序列与标签在该句全部 valid_mask=1 词位置完全一致的句数/总句数；"
    "phrase_pause_accuracy=【意群】内部停顿序列完全一致的句数/有可评测意群内部停顿的句数，意群左右边界及外部停顿不参与；"
    "LMDB split（valid_lmdb_mfa/test_lmdb_mfa）没有对应的【】意群标注，"
    "该指标输出 N/A；单词意群没有内部词间停顿，单列统计且不计入该指标分母"
)
ID_PATTERN = re.compile(r"\d+")
SOURCE_MARKED_ID_PATTERN = re.compile(
    r"(?:kaoshiyuan|gemini_no_pause|gemini_pause)_(\d+)(?:_|$)", re.IGNORECASE
)
WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
COLORS = {
    "valid_m2_manual": "#1f77b4",
    "test_m2_manual": "#ff7f0e",
    "valid_m2_mfa": "#2ca02c",
    "test_m2_mfa": "#d62728",
    "valid_lmdb_mfa": "#9467bd",
    "test_lmdb_mfa": "#8c564b",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--phrase-marked-txt", required=True, type=Path)
    parser.add_argument("--wav2vec-jit-path", required=True)
    parser.add_argument("--wav2vec-meta-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def normalize_token(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def tokenize_text(text):
    return tuple(
        token for token in (normalize_token(match.group(0)) for match in WORD_TOKEN_PATTERN.finditer(text))
        if token
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phrase_catalog(path):
    by_id = {}
    by_sentence = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                marked_id, marked_sentence = line.split("\t", 1)
                before, remaining = marked_sentence.split("【", 1)
                phrase_text, after = remaining.split("】", 1)
            except ValueError as error:
                raise ValueError(
                    "{}:{} 应为 id<TAB>且恰有一个【意群】的句子".format(path, line_number)
                ) from error
            if not marked_id.isdigit():
                raise ValueError("{}:{} 标注 id 不是数字：{}".format(path, line_number, marked_id))
            full_tokens = tokenize_text(before + phrase_text + after)
            phrase_tokens = tokenize_text(phrase_text)
            phrase_start = len(tokenize_text(before))
            if full_tokens[phrase_start:phrase_start + len(phrase_tokens)] != phrase_tokens:
                raise AssertionError("{}:{} 意群词边界解析异常".format(path, line_number))
            record = {
                "marked_id": str(int(marked_id)),
                "full_tokens": full_tokens,
                "phrase_tokens": phrase_tokens,
                "phrase_start": phrase_start,
            }
            if record["marked_id"] in by_id:
                raise ValueError("{}:{} 出现重复标注 id".format(path, line_number))
            if full_tokens in by_sentence:
                raise ValueError("{}:{} 出现重复完整句子，无法唯一文本匹配".format(path, line_number))
            by_id[record["marked_id"]] = record
            by_sentence[full_tokens] = record
    if not by_id:
        raise ValueError("意群标注文件为空：{}".format(path))
    return {"by_id": by_id, "by_sentence": by_sentence}


def find_phrase_spans(sequence, phrase_tokens):
    """在 LMDB 逐词序列中定位意群，允许连字符词在 LMDB 中被拆分。"""
    phrase_text = "".join(phrase_tokens)
    spans = []
    for start in range(len(sequence)):
        joined = ""
        for end in range(start, len(sequence)):
            joined += sequence[end]
            if joined == phrase_text:
                spans.append((start, end + 1))
                break
            if not phrase_text.startswith(joined):
                break
    return spans


def phrase_pause_indices(metadata, catalog, cache):
    utt_id = str(metadata["utt_id"])
    words = tuple(str(word) for word in metadata["word_list"])
    cache_key = (utt_id, words)
    if cache_key in cache:
        return cache[cache_key]
    normalized_positions = [
        (index, normalize_token(word)) for index, word in enumerate(words)
        if normalize_token(word)
    ]
    normalized_words = tuple(token for _index, token in normalized_positions)
    record = catalog["by_sentence"].get(normalized_words)
    matched_by = "text"
    if record is None:
        raw_id_values = ID_PATTERN.findall(utt_id)
        source_marked_ids = [
            str(int(value)) for value in SOURCE_MARKED_ID_PATTERN.findall(utt_id)
            if str(int(value)) in catalog["by_id"]
        ]
        candidate_ids = [
            str(int(value)) for value in raw_id_values
            if str(int(value)) in catalog["by_id"]
        ]
        padded_candidate_ids = [
            str(int(value)) for value in raw_id_values
            if len(value) > 1 and value.startswith("0") and str(int(value)) in catalog["by_id"]
        ]
        if len(set(source_marked_ids)) == 1:
            selected_id = source_marked_ids[0]
        elif len(set(source_marked_ids)) > 1:
            raise ValueError(
                "{}: 来源名后出现多个意群标注 id：{}".format(utt_id, source_marked_ids)
            )
        elif len(set(padded_candidate_ids)) == 1:
            selected_id = padded_candidate_ids[0]
        elif len(set(candidate_ids)) == 1:
            selected_id = candidate_ids[0]
        else:
            raise ValueError(
                "{}: 无法通过完整文本或可判定数字 id 对齐意群；数字候选={}".format(
                    utt_id, candidate_ids
                )
            )
        record = catalog["by_id"][selected_id]
        matched_by = "id"
    spans = find_phrase_spans(normalized_words, record["phrase_tokens"])
    if len(spans) != 1:
        raise ValueError(
            "{}: 已按 {} 对齐到意群，但无法唯一定位意群词字符序列，位置={}".format(
                utt_id, matched_by, spans
            )
        )
    phrase_start, phrase_end = spans[0]
    if phrase_end > len(normalized_positions):
        raise AssertionError("{}: 意群范围超出逐词序列".format(utt_id))
    indices = tuple(index for index, _token in normalized_positions[phrase_start:phrase_end - 1])
    result = {"indices": indices, "matched_by": matched_by, "marked_id": record["marked_id"]}
    cache[cache_key] = result
    return result


def move_net_input(net_input, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in net_input.items()
    }


def evaluate_split(model, task, split, args, device, phrase_catalog, phrase_cache):
    task.load_dataset(split)
    dataset = task.dataset(split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collater,
    )
    tp = fp = fn = tn = 0
    correct_utterances = utterances = 0
    phrase_correct_utterances = phrase_utterances = 0
    phrase_no_internal_pause_utterances = 0
    phrase_alignment_text_matched = phrase_alignment_id_matched = 0
    phrase_metric_applicable = split in PHRASE_EVALUABLE_SPLITS
    with torch.no_grad():
        for batch in loader:
            output = model(**move_net_input(batch["net_input"], device))
            predictions = torch.sigmoid(output["pause_logits"]).cpu() >= args.threshold
            targets = batch["target"].bool()
            valid_mask = batch["valid_mask"].bool()
            valid_counts = valid_mask.long().sum(dim=1)
            if torch.any(valid_counts == 0):
                bad = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
                raise ValueError("{} 存在无有效句中词位置的样本：{}".format(split, bad))
            tp += int(((predictions & targets) & valid_mask).long().sum().item())
            fp += int(((predictions & ~targets) & valid_mask).long().sum().item())
            fn += int(((~predictions & targets) & valid_mask).long().sum().item())
            tn += int(((~predictions & ~targets) & valid_mask).long().sum().item())
            sentence_correct = ((predictions == targets) | ~valid_mask).all(dim=1)
            correct_utterances += int(sentence_correct.long().sum().item())
            utterances += int(sentence_correct.numel())
            if not phrase_metric_applicable:
                continue
            for batch_index, metadata in enumerate(batch["metadata"]):
                phrase_info = phrase_pause_indices(metadata, phrase_catalog, phrase_cache)
                indices = list(phrase_info["indices"])
                if phrase_info["matched_by"] == "text":
                    phrase_alignment_text_matched += 1
                else:
                    phrase_alignment_id_matched += 1
                if not indices:
                    phrase_no_internal_pause_utterances += 1
                    continue
                phrase_valid = valid_mask[batch_index, indices]
                if not bool(phrase_valid.all()):
                    raise ValueError(
                        "{}:{} 意群内部位置包含无效停顿位置：{}".format(
                            split, metadata["utt_id"], indices
                        )
                    )
                phrase_correct_utterances += int(
                    torch.equal(predictions[batch_index, indices], targets[batch_index, indices])
                )
                phrase_utterances += 1
    if utterances == 0:
        raise ValueError("{} 没有可评测句子".format(split))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "sentence_accuracy": safe_div(correct_utterances, utterances),
        "phrase_pause_accuracy": (
            safe_div(phrase_correct_utterances, phrase_utterances)
            if phrase_metric_applicable else None
        ),
        "phrase_metric_status": (
            "evaluated" if phrase_metric_applicable else "not_applicable_no_phrase_annotation"
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_utterances": correct_utterances,
        "utterances": utterances,
        "phrase_correct_utterances": phrase_correct_utterances,
        "phrase_utterances": phrase_utterances,
        "phrase_no_internal_pause_utterances": phrase_no_internal_pause_utterances,
        "phrase_alignment_text_matched": phrase_alignment_text_matched,
        "phrase_alignment_id_matched": phrase_alignment_id_matched,
    }


def load_cache(path, threshold, phrase_marked_sha256, overwrite_cache):
    if overwrite_cache or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("format_version") != 3
        or payload.get("threshold") != threshold
        or payload.get("splits") != list(SPLITS)
        or payload.get("phrase_marked_sha256") != phrase_marked_sha256
    ):
        raise ValueError("指标缓存与当前阈值、split 或意群标注文件不一致；请更换输出目录或使用 --overwrite-cache")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("指标缓存 rows 非法：{}".format(path))
    return rows


def write_cache(path, threshold, phrase_marked_path, phrase_marked_sha256, rows):
    rows.sort(key=lambda row: (row["epoch"], SPLITS.index(row["split"])))
    payload = {
        "format_version": 3,
        "threshold": threshold,
        "splits": list(SPLITS),
        "metric_formula_remark": FORMULA_REMARK,
        "phrase_marked_txt": str(phrase_marked_path),
        "phrase_marked_sha256": phrase_marked_sha256,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path, rows):
    fields = (
        "epoch", "checkpoint", "split", "threshold", *METRICS,
        "tp", "fp", "fn", "tn", "correct_utterances", "utterances",
        "phrase_metric_status", "phrase_correct_utterances", "phrase_utterances",
        "phrase_no_internal_pause_utterances",
        "phrase_alignment_text_matched",
        "phrase_alignment_id_matched", "metric_formula_remark",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, epochs, metric, output_path):
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for split in SPLITS:
        by_epoch = {row["epoch"]: row[metric] for row in rows if row["split"] == split}
        available = [(epoch, by_epoch[epoch]) for epoch in epochs if by_epoch[epoch] is not None]
        if not available:
            continue
        axis.plot(
            [epoch for epoch, _value in available],
            [value for _epoch, value in available],
            marker="o",
            linewidth=2,
            color=COLORS[split],
            label=split,
        )
    axis.set_title("{} by checkpoint epoch (threshold={:.2f})".format(metric, rows[0]["threshold"]))
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_selected_reports(output_root, epochs, threshold, phrase_marked_path, phrase_marked_sha256, cache_rows):
    selected = [row for row in cache_rows if row["epoch"] in epochs]
    required = {(epoch, split) for epoch in epochs for split in SPLITS}
    actual = {(row["epoch"], row["split"]) for row in selected}
    if actual != required:
        raise RuntimeError("指定 checkpoint 的指标不完整，缺少：{}".format(sorted(required - actual)))
    selected.sort(key=lambda row: (row["epoch"], SPLITS.index(row["split"])))
    tsv_path = output_root / "pause_checkpoint_metrics.tsv"
    json_path = output_root / "pause_checkpoint_metrics.json"
    plot_paths = {
        metric: output_root / "pause_checkpoint_{}_by_epoch.png".format(metric)
        for metric in METRICS
    }
    write_tsv(tsv_path, selected)
    for metric, path in plot_paths.items():
        plot_metric(selected, epochs, metric, path)
    payload = {
        "format_version": 3,
        "requested_epochs": epochs,
        "threshold": threshold,
        "splits": list(SPLITS),
        "metric_formula_remark": FORMULA_REMARK,
        "phrase_marked_txt": str(phrase_marked_path),
        "phrase_marked_sha256": phrase_marked_sha256,
        "rows": selected,
        "plots": {metric: str(path) for metric, path in plot_paths.items()},
        "cache_file": str(output_root / "pause_checkpoint_metrics_cache.json"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tsv_path, json_path, plot_paths


def main():
    args = parse_args()
    epochs = sorted(set(args.epochs))
    if len(epochs) != len(args.epochs) or any(epoch <= 0 for epoch in epochs):
        raise ValueError("epochs 必须是互不重复的正整数")
    if not 0.0 <= args.threshold <= 1.0 or args.batch_size <= 0:
        raise ValueError("threshold 必须在 [0,1] 内且 batch-size 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 wav2vec TorchScript 资产为 CUDA 版本，评测必须使用 GPU")
    checkpoint_root = args.checkpoint_root.resolve()
    data_root = args.data_root.resolve()
    phrase_marked_path = args.phrase_marked_txt.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError("checkpoint-root 不存在：{}".format(checkpoint_root))
    for split in SPLITS:
        if not (data_root / split).is_dir() or not (data_root / "{}.key".format(split)).is_file():
            raise FileNotFoundError("缺少正式 split：{}/{}".format(data_root, split))
    if not phrase_marked_path.is_file():
        raise FileNotFoundError("意群标注文件不存在：{}".format(phrase_marked_path))
    phrase_marked_sha256 = sha256_file(phrase_marked_path)
    phrase_catalog = load_phrase_catalog(phrase_marked_path)
    phrase_cache = {}
    checkpoints = [(epoch, checkpoint_root / "checkpoint{}.pt".format(epoch)) for epoch in epochs]
    missing_files = [str(path) for _epoch, path in checkpoints if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("指定 checkpoint 不存在：{}".format(missing_files))
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / "pause_checkpoint_metrics_cache.json"
    rows = load_cache(cache_path, args.threshold, phrase_marked_sha256, args.overwrite_cache)
    cached_keys = {(row["checkpoint"], row["split"]) for row in rows}

    utils.import_user_module(args)
    device = torch.device("cuda")
    for epoch, checkpoint in checkpoints:
        missing_splits = [split for split in SPLITS if (checkpoint.name, split) not in cached_keys]
        if not missing_splits:
            print("跳过已缓存 checkpoint：{}".format(checkpoint.name))
            continue
        print("评测 {}：{}".format(checkpoint.name, ",".join(missing_splits)))
        models, _checkpoint_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint)],
            arg_overrides={
                "data": str(data_root),
                "wav2vec_jit_path": args.wav2vec_jit_path,
                "wav2vec_meta_path": args.wav2vec_meta_path,
            },
        )
        if len(models) != 1:
            raise ValueError("{} 未加载到唯一模型".format(checkpoint))
        model = models[0].to(device).eval()
        for split in missing_splits:
            rows.append({
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "split": split,
                "threshold": args.threshold,
                "metric_formula_remark": FORMULA_REMARK,
                **evaluate_split(model, task, split, args, device, phrase_catalog, phrase_cache),
            })
            cached_keys.add((checkpoint.name, split))
        write_cache(cache_path, args.threshold, phrase_marked_path, phrase_marked_sha256, rows)
        del model, models, task
        torch.cuda.empty_cache()

    tsv_path, json_path, plot_paths = write_selected_reports(
        output_root, epochs, args.threshold, phrase_marked_path, phrase_marked_sha256, rows
    )
    print(json.dumps({
        "ok": True,
        "requested_epochs": epochs,
        "metrics_tsv": str(tsv_path),
        "metrics_json": str(json_path),
        "plots": {metric: str(path) for metric, path in plot_paths.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
