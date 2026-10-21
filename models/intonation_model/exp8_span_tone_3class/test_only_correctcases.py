"""
test_only_correctcase.py — exp8_3class: Export utterances with all tones correct (correct cases)

Opposite of the bad-case report in test_only.py:
  - Evaluate tone only; ignore break labels.
  - For each utterance, consider all valid tone-labeled words (tone_target >= 0).
    If every tone_pred equals tone_target, mark the utterance as a "tone-correct utterance".
  - Collect the UIDs of these correct utterances.
  - Extract the corresponding original lines from the provided GT text file (tone_ori.txt).
  - Output each ID and its original text, including 【Rising/Falling/Flat】 labels.

Outputs:
  - <output-dir>/correct_ids.txt        : one correct utterance ID per line
  - <output-dir>/correctcase.txt        : id <TAB> original GT text (primary deliverable)
  - <output-dir>/correct_summary.txt    : correct-utterance statistics for each test set

Usage:
    python test_only_correctcase.py --ckpt /train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/checkpoints/best_exp8_span_tone_3class0625.pt --vocab /train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/vocab.json --stats /train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/stats.npz --gt-txt  /yrfs5/tts/mezhao/kaoshiyuan_project/data/Adrian/tone_ori.txt --output-dir /train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/correctcase_adrian
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import config as C
from data_utils import (
    load_json, WordTokenizer, IntonationDataset, collate_fn, load_stats,
)
from model import IntonationV4Model


TEST_PATHS = {
    "adrian":   "/train29/tts/permanent/jyxu24/vc_data/vc_dataset/dataset_final_output_Adrian.json",  # Fill in the path
}

TONE_NAMES = ["rising", "falling", "flat"]


def collect_correct_uids(model, loader, device):
    """Return the UIDs of utterances with all tones correct (ignore break labels).

    An utterance is correct when every valid word with tone_target >= 0 has a matching tone_pred.
    Skip utterances without tone labels because they cannot be evaluated.
    """
    model.eval()

    correct_uids = []
    n_sents_with_tone = 0

    with torch.no_grad():
        for batch in loader:
            fbank = batch["fbank"].to(device)
            fbank_mask = batch["fbank_mask"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            tone_target = batch["tone_target"].to(device)
            sil_mask = batch["sil_mask"].to(device)
            word_spans = batch["word_spans"].to(device)
            uids = batch["uid"]

            out = model(fbank, fbank_mask, text_tokens, text_mask, word_spans)
            tone_pred = out["tone_logits"].argmax(-1)

            valid = (text_mask > 0.5) & (sil_mask < 0.5)
            tone_valid = (tone_target >= 0) & valid
            tone_correct = (tone_pred == tone_target)

            B = fbank.size(0)
            for i in range(B):
                mask_i = tone_valid[i]
                n_tone = int(mask_i.sum().item())
                if n_tone == 0:
                    continue  # No tone annotation; skip
                n_sents_with_tone += 1
                if bool(tone_correct[i][mask_i].all().item()):
                    correct_uids.append(uids[i])

    return correct_uids, n_sents_with_tone


def _num_key(s):
    """Extract trailing consecutive digits as a matching key, e.g. 'EnUs_kaoshiyuan_male_21780917' -> '21780917'.
    Return the original string if it has no digits. Strip leading zeros so '007' matches '7'."""
    import re
    m = re.search(r"(\d+)$", str(s))
    if not m:
        return str(s)
    return m.group(1).lstrip("0") or "0"


def load_gt_lines(gt_txt_path):
    """Load the GT text file into {numeric_key: text}. Each line is ID<whitespace>text, including labels.
    Input JSON IDs are numeric, so match them using the trailing digits of each GT ID."""
    id2text = {}
    n_dup = 0
    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)  # Split on the first whitespace; preserve whitespace within the text
            full_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            key = _num_key(full_id)
            if key in id2text:
                n_dup += 1
            id2text[key] = text
    print(f"[gt] Loaded {len(id2text)} lines from {gt_txt_path}" + (f"  (overwrote {n_dup} duplicate numeric keys)" if n_dup else ""))
    return id2text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the trained best-model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--stats", required=True, help="Path to stats.npz")
    parser.add_argument("--gt-txt", required=True, help="Path to the original GT text file (e.g. tone_ori.txt)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--encoder-type", type=str, default=C.ENCODER_TYPE,
                        choices=["bilstm", "bilstm_residual", "transformer"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] vocab from {args.vocab}")
    tokenizer = WordTokenizer()
    tokenizer.load(args.vocab)
    print(f"  vocab_size = {tokenizer.vocab_size}")

    print(f"[load] stats from {args.stats}")
    stats = load_stats(args.stats)

    print(f"[load] model from {args.ckpt}")
    model = IntonationV4Model(tokenizer.vocab_size, encoder_type=args.encoder_type).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter count: {n_params / 1e6:.2f}M, encoder_type={args.encoder_type}")
    if "epoch" in ckpt:
        print(f"  ckpt epoch: {ckpt['epoch']}, best_score: {ckpt.get('best_score', 'N/A')}")

    # Original ground-truth text
    id2text = load_gt_lines(args.gt_txt)

    # Collect the UIDs of correct sentences from all test sets (deduplicate while preserving order)
    all_correct_uids = []
    seen = set()
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Correct Case Summary (exp8 span_tone 3-class, tone-only, break ignored)")
    summary_lines.append(f"  ckpt:   {args.ckpt}")
    summary_lines.append(f"  gt_txt: {args.gt_txt}")
    summary_lines.append("=" * 80)

    for name, path in TEST_PATHS.items():
        if not path or not Path(path).exists():
            print(f"\n[skip] {name}: File does not exist or path is empty: {path}")
            summary_lines.append(f"\n[{name}]  SKIPPED (file not found or empty path: {path})")
            continue

        print(f"\n[eval] {name} <- {path}")
        test_data = load_json(path)
        print(f"  Number of samples: {len(test_data)}")

        test_dataset = IntonationDataset(test_data, stats, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)

        correct_uids, n_with_tone = collect_correct_uids(model, test_loader, device)

        n_correct = len(correct_uids)
        ratio = n_correct / n_with_tone if n_with_tone > 0 else 0.0
        line = (f"\n[{name}]  Utterances with tone labels: {n_with_tone}, "
                f"utterances with all tones correct: {n_correct}  (sent_tone_acc={ratio:.4f})")
        print(line)
        summary_lines.append(line)

        for uid in correct_uids:
            if uid not in seen:
                seen.add(uid)
                all_correct_uids.append(uid)

    # Write the ID list
    ids_path = output_dir / "correct_ids.txt"
    with open(ids_path, "w", encoding="utf-8") as f:
        for uid in all_correct_uids:
            f.write(f"{uid}\n")
    print(f"\n[save] {ids_path}  ({len(all_correct_uids)} ids)")

    # Write correctcase.txt: id <TAB> original ground-truth text
    correctcase_path = output_dir / "correctcase.txt"
    n_found = 0
    n_missing = 0
    missing_ids = []
    with open(correctcase_path, "w", encoding="utf-8") as f:
        for uid in all_correct_uids:
            key = _num_key(uid)
            if key in id2text:
                f.write(f"{uid}\t{id2text[key]}\n")
                n_found += 1
            else:
                n_missing += 1
                missing_ids.append(uid)
    print(f"[save] {correctcase_path}  (GT matches: {n_found}, missing: {n_missing})")

    if missing_ids:
        miss_path = output_dir / "correct_ids_missing_in_gt.txt"
        with open(miss_path, "w", encoding="utf-8") as f:
            for uid in missing_ids:
                f.write(f"{uid}\n")
        print(f"[warn] {n_missing} IDs were not found in the GT text file -> {miss_path}")

    summary_lines.append("")
    summary_lines.append(f"Total correct utterances (deduplicated): {len(all_correct_uids)}")
    summary_lines.append(f"Matched in original GT text: {n_found}")
    summary_lines.append(f"Missing from GT: {n_missing}")

    summary_path = output_dir / "correct_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[save] {summary_path}")

    print(f"\n[done] Correct-case extraction complete")


if __name__ == "__main__":
    main()
