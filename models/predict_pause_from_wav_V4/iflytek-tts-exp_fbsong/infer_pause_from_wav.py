#!/usr/bin/env python3
"""Run pause inference from a Fairseq checkpoint and pause LMDB split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fairseq import checkpoint_utils, utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--wav2vec-jit-path", required=True)
    parser.add_argument("--wav2vec-meta-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def move_net_input(net_input, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in net_input.items()
    }


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # 推理入口不经过 train.py 的参数初始化，必须在加载 checkpoint 前显式
    # 导入 V4 user-dir，注册 pause task/model/criterion。
    utils.import_user_module(args)

    models, _checkpoint_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint],
        arg_overrides={
            "data": args.data_root,
            "wav2vec_jit_path": args.wav2vec_jit_path,
            "wav2vec_meta_path": args.wav2vec_meta_path,
        },
    )
    if len(models) != 1:
        raise ValueError("V1 inference expects exactly one checkpoint")
    model = models[0].to(device).eval()
    task.load_dataset(args.split)
    dataset = task.dataset(args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collater,
    )

    jsonl_path = output_dir / f"{args.split}_pause_predictions.jsonl"
    tsv_path = output_dir / f"{args.split}_pause_predictions.tsv"
    tp = fp = fn = tn = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, tsv_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as tsv_handle, torch.no_grad():
        tsv_handle.write(
            "utt_id\tword_index\tword\tprobability\tprediction\ttarget\tvalid\t"
            "pool_start_seconds\tpool_end_seconds\n"
        )
        for batch in loader:
            output = model(**move_net_input(batch["net_input"], device))
            probabilities = torch.sigmoid(output["pause_logits"]).cpu()
            predictions = probabilities >= args.threshold
            targets = batch["target"].bool()
            valid_mask = batch["valid_mask"].bool()
            tp += int(((predictions & targets) & valid_mask).long().sum())
            fp += int(((predictions & ~targets) & valid_mask).long().sum())
            fn += int(((~predictions & targets) & valid_mask).long().sum())
            tn += int(((~predictions & ~targets) & valid_mask).long().sum())

            for batch_index, metadata in enumerate(batch["metadata"]):
                words = []
                for word_index, word in enumerate(metadata["word_list"]):
                    row = {
                        "word_index": word_index,
                        "word": word,
                        "probability": float(probabilities[batch_index, word_index]),
                        "prediction": int(predictions[batch_index, word_index]),
                        "target": int(targets[batch_index, word_index]),
                        "valid": int(valid_mask[batch_index, word_index]),
                        "pool_start_seconds": float(metadata["pool_start_seconds"][word_index]),
                        "pool_end_seconds": float(metadata["pool_end_seconds"][word_index]),
                    }
                    words.append(row)
                    tsv_handle.write(
                        "\t".join(
                            str(value if value is not None else "")
                            for value in [metadata["utt_id"], *row.values()]
                        )
                        + "\n"
                    )
                jsonl_handle.write(
                    json.dumps(
                        {
                            "utt_id": metadata["utt_id"],
                            "speaker_id": metadata["speaker_id"],
                            "threshold": args.threshold,
                            "words": words,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    summary_path = output_dir / f"{args.split}_pause_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": [str(jsonl_path), str(tsv_path), str(summary_path)], **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
