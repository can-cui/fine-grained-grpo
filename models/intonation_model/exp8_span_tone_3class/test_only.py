"""
test_only.py — exp8_3class: 独立测试脚本（三分类版本）

功能:
  加载训练好的 best model，在测试集上评估
  输出 rising/falling/flat 各自的 P/R/F1/Accuracy + 总体指标 + 混淆矩阵
  生成 result.json 和 summary.txt

测试集:
  - 路径需要在 TEST_PATHS 中配置

用法:
    python test_only.py \\
        --ckpt    <你填模型路径>             \\
        --vocab   <你填 vocab.json 路径>     \\
        --stats   <你填 stats.npz 路径>      \\
        --output-dir <你填输出目录>
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)

import config as C
from data_utils import (
    load_json, WordTokenizer, IntonationDataset, collate_fn, load_stats,
)
from model import IntonationV4Model


TEST_PATHS = {
    "Angel_vc":   "/train29/tts/permanent/jyxu24/vc_data/vc_dataset/dataset_final_output_Angel_F_CZ.json",  # 你自己填路径
    "Matt_vc":   "/train29/tts/permanent/jyxu24/vc_data/vc_dataset/dataset_final_output_Matt_M_TY.json",  # 你自己填路径
    "vc_all_train":   "/train29/tts/permanent/jyxu24/vc_data/split/train.json",  # 你自己填路径
    "vc_all_val": "/train29/tts/permanent/jyxu24/vc_data/split/test.json",
    "Felix": "/train29/tts/permanent/jyxu24/0605_new_golden_data_24k/Felix_data.json", 
    "Lyra": "/train29/tts/permanent/jyxu24/0605_new_golden_data_24k/Lyra_data.json" # 你自己填路径
}

TONE_NAMES = ["rising", "falling", "flat"]


def evaluate_with_badcases(model, loader, device):
    model.eval()

    break_preds, break_gts = [], []
    tone_preds, tone_gts = [], []

    sent_break_correct = 0
    sent_tone_correct = 0
    sent_joint_correct = 0
    n_sents = 0

    word_joint_correct = 0
    word_joint_total = 0
    word_all_correct = 0
    word_all_total = 0

    break_fp_cases = []
    break_fn_cases = []
    tone_wrong_cases = []

    with torch.no_grad():
        for batch in loader:
            fbank = batch["fbank"].to(device)
            fbank_mask = batch["fbank_mask"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            break_target = batch["break_target"].to(device)
            tone_target = batch["tone_target"].to(device)
            sil_mask = batch["sil_mask"].to(device)
            word_spans = batch["word_spans"].to(device)
            uids = batch["uid"]
            words_batch = batch["words"]

            out = model(fbank, fbank_mask, text_tokens, text_mask, word_spans)

            break_prob = out["break_logits"].sigmoid()
            break_pred = (break_prob > 0.5).float()
            tone_pred = out["tone_logits"].argmax(-1)

            valid = (text_mask > 0.5) & (sil_mask < 0.5)
            break_preds.extend(break_pred[valid].cpu().tolist())
            break_gts.extend(break_target[valid].cpu().tolist())

            valid_tone = (tone_target >= 0) & valid
            tone_preds.extend(tone_pred[valid_tone].cpu().tolist())
            tone_gts.extend(tone_target[valid_tone].cpu().tolist())

            break_correct_per_word = (break_pred == break_target)
            tone_correct_per_word = (tone_pred == tone_target)
            joint_correct_per_word = break_correct_per_word & tone_correct_per_word
            word_joint_correct += joint_correct_per_word[valid_tone].sum().item()
            word_joint_total += valid_tone.sum().item()

            no_tone_mask = (tone_target < 0) & valid
            all_correct_mask = (valid_tone & joint_correct_per_word) | (no_tone_mask & break_correct_per_word)
            word_all_correct += all_correct_mask.sum().item()
            word_all_total += valid.sum().item()

            B = fbank.size(0)
            for i in range(B):
                mask_i = valid[i]
                n_words = mask_i.sum().item()
                if n_words == 0:
                    continue

                bp = break_pred[i][mask_i]
                bt = break_target[i][mask_i]
                break_match = (bp == bt).all()
                if break_match:
                    sent_break_correct += 1

                tone_mask_i = (tone_target[i] >= 0) & mask_i
                tone_match = True
                if tone_mask_i.sum() > 0:
                    tone_match = (tone_pred[i][tone_mask_i] == tone_target[i][tone_mask_i]).all().item()
                    if tone_match:
                        sent_tone_correct += 1
                    if break_match and tone_match:
                        sent_joint_correct += 1

                n_sents += 1

                n_total = int(text_mask[i].sum().item())
                words_i = words_batch[i]
                for j in range(n_total):
                    if sil_mask[i][j] > 0.5:
                        continue
                    w = words_i[j] if j < len(words_i) else ""
                    bp_j = int(break_pred[i][j].item())
                    bt_j = int(break_target[i][j].item())
                    tp_j = int(tone_pred[i][j].item())
                    tt_j = int(tone_target[i][j].item())

                    if bp_j == 1 and bt_j == 0:
                        break_fp_cases.append({
                            "uid": uids[i], "word_idx": j, "word": w,
                            "context": words_i[:n_total],
                        })
                    elif bp_j == 0 and bt_j == 1:
                        break_fn_cases.append({
                            "uid": uids[i], "word_idx": j, "word": w,
                            "tone_gt": tt_j,
                            "context": words_i[:n_total],
                        })

                    if bt_j == 1 and tt_j >= 0 and tp_j != tt_j:
                        tone_wrong_cases.append({
                            "uid": uids[i], "word_idx": j, "word": w,
                            "tone_gt": tt_j, "tone_pred": tp_j,
                            "context": words_i[:n_total],
                        })

    break_f1 = f1_score(break_gts, break_preds, average="binary", zero_division=0)
    break_prec = precision_score(break_gts, break_preds, average="binary", zero_division=0)
    break_rec = recall_score(break_gts, break_preds, average="binary", zero_division=0)
    break_acc = accuracy_score(break_gts, break_preds)
    break_cm = confusion_matrix(break_gts, break_preds, labels=[0, 1]).tolist() if len(break_gts) > 0 else [[0, 0], [0, 0]]

    if len(tone_gts) > 0:
        tone_acc = accuracy_score(tone_gts, tone_preds)
        tone_prec_macro = precision_score(tone_gts, tone_preds, average="macro", zero_division=0)
        tone_rec_macro = recall_score(tone_gts, tone_preds, average="macro", zero_division=0)
        tone_f1_macro = f1_score(tone_gts, tone_preds, average="macro", zero_division=0)
        tone_prec_per = precision_score(tone_gts, tone_preds, average=None, labels=[0, 1, 2], zero_division=0).tolist()
        tone_rec_per = recall_score(tone_gts, tone_preds, average=None, labels=[0, 1, 2], zero_division=0).tolist()
        tone_f1_per = f1_score(tone_gts, tone_preds, average=None, labels=[0, 1, 2], zero_division=0).tolist()
        tone_cm = confusion_matrix(tone_gts, tone_preds, labels=[0, 1, 2]).tolist()
    else:
        tone_acc = 0.0
        tone_prec_macro = 0.0
        tone_rec_macro = 0.0
        tone_f1_macro = 0.0
        tone_prec_per = [0.0, 0.0, 0.0]
        tone_rec_per = [0.0, 0.0, 0.0]
        tone_f1_per = [0.0, 0.0, 0.0]
        tone_cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    sent_break_acc = sent_break_correct / n_sents if n_sents > 0 else 0.0
    sent_tone_acc = sent_tone_correct / n_sents if n_sents > 0 else 0.0
    sent_joint_acc = sent_joint_correct / n_sents if n_sents > 0 else 0.0
    word_joint_acc = word_joint_correct / word_joint_total if word_joint_total > 0 else 0.0
    word_all_acc = word_all_correct / word_all_total if word_all_total > 0 else 0.0

    metrics = {
        "break_acc": break_acc,
        "break_prec": break_prec,
        "break_rec": break_rec,
        "break_f1": break_f1,
        "break_confusion_matrix": break_cm,
        "tone_acc": tone_acc,
        "tone_prec_macro": tone_prec_macro,
        "tone_rec_macro": tone_rec_macro,
        "tone_f1_macro": tone_f1_macro,
        "tone_prec_rising": tone_prec_per[0],
        "tone_rec_rising": tone_rec_per[0],
        "tone_f1_rising": tone_f1_per[0],
        "tone_prec_falling": tone_prec_per[1],
        "tone_rec_falling": tone_rec_per[1],
        "tone_f1_falling": tone_f1_per[1],
        "tone_prec_flat": tone_prec_per[2],
        "tone_rec_flat": tone_rec_per[2],
        "tone_f1_flat": tone_f1_per[2],
        "tone_confusion_matrix": tone_cm,
        "sent_break_acc": sent_break_acc,
        "sent_tone_acc": sent_tone_acc,
        "sent_joint_acc": sent_joint_acc,
        "word_joint_acc": word_joint_acc,
        "word_all_acc": word_all_acc,
        "n_sents": n_sents,
        "n_tone_words": len(tone_gts),
        "n_break_words": len(break_gts),
    }

    bad_cases = {
        "break_fp": break_fp_cases,
        "break_fn": break_fn_cases,
        "tone_wrong": tone_wrong_cases,
    }

    return metrics, bad_cases


def format_metrics(tag, metrics, bad_cases=None):
    lines = []
    lines.append(f"\n[{tag}]  样本数: {metrics['n_sents']}, tone标注词数: {metrics['n_tone_words']}, break词数: {metrics['n_break_words']}")

    lines.append(f"  === Break (binary) ===")
    lines.append(f"    Accuracy  : {metrics['break_acc']:.4f}")
    lines.append(f"    Precision : {metrics['break_prec']:.4f}")
    lines.append(f"    Recall    : {metrics['break_rec']:.4f}")
    lines.append(f"    F1        : {metrics['break_f1']:.4f}")
    cm = metrics["break_confusion_matrix"]
    lines.append(f"    Confusion Matrix  (rows=gt, cols=pred):")
    lines.append(f"              pred=0    pred=1")
    lines.append(f"      gt=0    {cm[0][0]:>8d}  {cm[0][1]:>8d}")
    lines.append(f"      gt=1    {cm[1][0]:>8d}  {cm[1][1]:>8d}")

    lines.append(f"  === Tone Overall (3-class: rising vs falling vs flat) ===")
    lines.append(f"    Accuracy        : {metrics['tone_acc']:.4f}")
    lines.append(f"    Precision(macro): {metrics['tone_prec_macro']:.4f}")
    lines.append(f"    Recall(macro)   : {metrics['tone_rec_macro']:.4f}")
    lines.append(f"    F1(macro)       : {metrics['tone_f1_macro']:.4f}")

    lines.append(f"  === Tone Per-Class ===")
    lines.append(f"    Rising  -> Precision: {metrics['tone_prec_rising']:.4f}  Recall: {metrics['tone_rec_rising']:.4f}  F1: {metrics['tone_f1_rising']:.4f}")
    lines.append(f"    Falling -> Precision: {metrics['tone_prec_falling']:.4f}  Recall: {metrics['tone_rec_falling']:.4f}  F1: {metrics['tone_f1_falling']:.4f}")
    lines.append(f"    Flat    -> Precision: {metrics['tone_prec_flat']:.4f}  Recall: {metrics['tone_rec_flat']:.4f}  F1: {metrics['tone_f1_flat']:.4f}")

    tcm = metrics["tone_confusion_matrix"]
    lines.append(f"    Confusion Matrix  (rows=gt, cols=pred):")
    lines.append(f"              " + "  ".join([f"{n:>8}" for n in TONE_NAMES]))
    for i, row in enumerate(tcm):
        vals = "  ".join([f"{v:>8d}" for v in row])
        lines.append(f"      gt={TONE_NAMES[i]:<8} {vals}")

    lines.append(f"  === Joint ===")
    lines.append(f"    word_joint_acc : {metrics['word_joint_acc']:.4f}  (有tone标注的词, break+tone 都对)")
    lines.append(f"    word_all_acc   : {metrics['word_all_acc']:.4f}  (所有valid词)")
    lines.append(f"    sent_break_acc : {metrics['sent_break_acc']:.4f}")
    lines.append(f"    sent_tone_acc  : {metrics['sent_tone_acc']:.4f}")
    lines.append(f"    sent_joint_acc : {metrics['sent_joint_acc']:.4f}")

    if bad_cases is not None:
        lines.append(f"  === Bad cases ===")
        lines.append(f"    break FP={len(bad_cases['break_fp'])}, "
                     f"FN={len(bad_cases['break_fn'])}, "
                     f"tone_wrong={len(bad_cases['tone_wrong'])}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="训练好的 best model checkpoint 路径")
    parser.add_argument("--vocab", required=True, help="vocab.json 路径")
    parser.add_argument("--stats", required=True, help="stats.npz 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录（保存 result.json 和 summary.txt）")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--encoder-type", type=str, default=C.ENCODER_TYPE,
                        choices=["bilstm", "bilstm_residual", "transformer"])
    parser.add_argument("--save-bad-cases", action="store_true",
                        help="是否在 result.json 中保存 bad cases (默认不保存以减小文件)")
    parser.add_argument("--save-tone-badcases-txt", type=str, default="",
                        help="保存 tone bad cases 到单独的 txt 文件（留空则不保存）")
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
    print(f"  参数量: {n_params / 1e6:.2f}M, encoder_type={args.encoder_type}")
    if "epoch" in ckpt:
        print(f"  ckpt epoch: {ckpt['epoch']}, best_score: {ckpt.get('best_score', 'N/A')}")

    all_results = {}
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"Test Summary (exp8 span_tone 3-class)")
    summary_lines.append(f"  ckpt:  {args.ckpt}")
    summary_lines.append(f"  vocab: {args.vocab}")
    summary_lines.append(f"  stats: {args.stats}")
    summary_lines.append("=" * 80)

    for name, path in TEST_PATHS.items():
        if not path or not Path(path).exists():
            print(f"\n[skip] {name}: 文件不存在或路径为空 {path}")
            summary_lines.append(f"\n[{name}]  SKIPPED (file not found or empty path: {path})")
            continue

        print(f"\n[eval] {name} <- {path}")
        test_data = load_json(path)
        print(f"  样本数: {len(test_data)}")

        test_dataset = IntonationDataset(test_data, stats, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)

        metrics, bad_cases = evaluate_with_badcases(model, test_loader, device)

        result_entry = {"path": path, "metrics": metrics}
        if args.save_bad_cases:
            result_entry["bad_cases"] = bad_cases
        all_results[name] = result_entry

        formatted = format_metrics(name, metrics, bad_cases)
        print(formatted)
        summary_lines.append(formatted)

    result_json = output_dir / "result.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt": args.ckpt,
            "vocab": args.vocab,
            "stats": args.stats,
            "encoder_type": args.encoder_type,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {result_json}")

    summary_txt = output_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[save] {summary_txt}")

    # 保存 tone bad cases 到单独文件
    if args.save_tone_badcases_txt:
        tone_badcases_path = Path(args.save_tone_badcases_txt)
        tone_badcases_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tone_badcases_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Tone Bad Cases (exp8 span_tone 3-class)\n")
            f.write(f"ckpt: {args.ckpt}\n")
            f.write("=" * 80 + "\n\n")

            for test_name, result_entry in all_results.items():
                if "bad_cases" not in result_entry:
                    continue

                test_path = result_entry.get("path", "?")
                tone_wrong = result_entry["bad_cases"].get("tone_wrong", [])

                f.write("=" * 80 + "\n")
                f.write(f"Test set: {test_name}\n")
                f.write(f"Path:     {test_path}\n")
                f.write(f"Tone bad cases: {len(tone_wrong)}\n")
                f.write("=" * 80 + "\n\n")

                if not tone_wrong:
                    f.write("(no tone bad cases)\n\n")
                    continue

                for i, case in enumerate(tone_wrong, 1):
                    uid = case.get("uid", "?")
                    word_idx = case.get("word_idx", -1)
                    word = case.get("word", "")
                    tone_gt = case.get("tone_gt", -1)
                    tone_pred = case.get("tone_pred", -1)
                    context = case.get("context", [])

                    tone_gt_name = TONE_NAMES[tone_gt] if 0 <= tone_gt < len(TONE_NAMES) else str(tone_gt)
                    tone_pred_name = TONE_NAMES[tone_pred] if 0 <= tone_pred < len(TONE_NAMES) else str(tone_pred)

                    f.write(f"{i:4d}. [{test_name}] uid={uid}, word_idx={word_idx}, word=\"{word}\"\n")
                    f.write(f"      GT={tone_gt_name}, PRED={tone_pred_name}\n")
                    f.write(f"      Context: {' '.join(context)}\n")
                    f.write(f"      From:    {test_path}\n")
                    f.write("\n")

                f.write("\n")

        print(f"[save] tone bad cases -> {tone_badcases_path}")

    print(f"\n[done] 测试完成")


if __name__ == "__main__":
    main()
