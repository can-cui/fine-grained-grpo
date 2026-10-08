"""
test_only_correctcase.py — exp8_3class: 输出 tone 全对的句子 (correct case)

与 test_only.py 的 badcase 相反:
  - 只看 tone (break 完全无视)
  - 对每个句子, 取所有有 tone 标注的词 (tone_target >= 0 且 valid),
    若这些词的 tone_pred 全部 == tone_target, 则该句记为 "tone 正确句"
  - 收集这些正确句的 uid (id list)
  - 到给定的 GT txt (tone_ori.txt) 里, 把这些 id 对应的原文行抽出来
  - 输出 id + text (text 即 GT 原文, 已带 【Rising/Falling/Flat】标签)

输出:
  - <output-dir>/correct_ids.txt        : 每行一个正确句 id
  - <output-dir>/correctcase.txt        : id <TAB> GT原文text  (核心交付)
  - <output-dir>/correct_summary.txt    : 每个测试集的正确句统计

用法:
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
    "adrian":   "/train29/tts/permanent/jyxu24/vc_data/vc_dataset/dataset_final_output_Adrian.json",  # 你自己填路径
}

TONE_NAMES = ["rising", "falling", "flat"]


def collect_correct_uids(model, loader, device):
    """返回 tone 全对句子的 uid 列表 (break 无视)。

    判定: 句子内所有 tone_target>=0 的 valid 词, tone_pred 全部命中。
    无 tone 标注的句子跳过 (无可评估)。
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
                    continue  # 没有 tone 标注, 跳过
                n_sents_with_tone += 1
                if bool(tone_correct[i][mask_i].all().item()):
                    correct_uids.append(uids[i])

    return correct_uids, n_sents_with_tone


def _num_key(s):
    """提取末尾连续数字作为匹配键。如 'EnUs_kaoshiyuan_male_21780917' -> '21780917'。
    若无数字, 返回原串。前导零去掉以保证 '007' 与 '7' 能匹配。"""
    import re
    m = re.search(r"(\d+)$", str(s))
    if not m:
        return str(s)
    return m.group(1).lstrip("0") or "0"


def load_gt_lines(gt_txt_path):
    """加载 GT txt -> {数字key: text}。每行格式: ID<空白>text (text 含标签)。
    input json 的 id 只有数字, 所以用 GT id 末尾数字做匹配键。"""
    id2text = {}
    n_dup = 0
    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)  # 首段空白切分, text 内部空白保留
            full_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            key = _num_key(full_id)
            if key in id2text:
                n_dup += 1
            id2text[key] = text
    print(f"[gt] 加载 {len(id2text)} 行 from {gt_txt_path}" + (f"  (数字key重复覆盖 {n_dup} 条)" if n_dup else ""))
    return id2text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="训练好的 best model checkpoint 路径")
    parser.add_argument("--vocab", required=True, help="vocab.json 路径")
    parser.add_argument("--stats", required=True, help="stats.npz 路径")
    parser.add_argument("--gt-txt", required=True, help="GT 原文 txt 路径 (如 tone_ori.txt)")
    parser.add_argument("--output-dir", required=True, help="输出目录")
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
    print(f"  参数量: {n_params / 1e6:.2f}M, encoder_type={args.encoder_type}")
    if "epoch" in ckpt:
        print(f"  ckpt epoch: {ckpt['epoch']}, best_score: {ckpt.get('best_score', 'N/A')}")

    # GT 原文
    id2text = load_gt_lines(args.gt_txt)

    # 收集所有测试集的正确句 uid (去重, 保持顺序)
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
            print(f"\n[skip] {name}: 文件不存在或路径为空 {path}")
            summary_lines.append(f"\n[{name}]  SKIPPED (file not found or empty path: {path})")
            continue

        print(f"\n[eval] {name} <- {path}")
        test_data = load_json(path)
        print(f"  样本数: {len(test_data)}")

        test_dataset = IntonationDataset(test_data, stats, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=4)

        correct_uids, n_with_tone = collect_correct_uids(model, test_loader, device)

        n_correct = len(correct_uids)
        ratio = n_correct / n_with_tone if n_with_tone > 0 else 0.0
        line = (f"\n[{name}]  含tone标注句: {n_with_tone}, "
                f"tone全对句: {n_correct}  (sent_tone_acc={ratio:.4f})")
        print(line)
        summary_lines.append(line)

        for uid in correct_uids:
            if uid not in seen:
                seen.add(uid)
                all_correct_uids.append(uid)

    # 写 id list
    ids_path = output_dir / "correct_ids.txt"
    with open(ids_path, "w", encoding="utf-8") as f:
        for uid in all_correct_uids:
            f.write(f"{uid}\n")
    print(f"\n[save] {ids_path}  ({len(all_correct_uids)} ids)")

    # 写 correctcase.txt: id <TAB> GT原文
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
    print(f"[save] {correctcase_path}  (匹配GT: {n_found}, 缺失: {n_missing})")

    if missing_ids:
        miss_path = output_dir / "correct_ids_missing_in_gt.txt"
        with open(miss_path, "w", encoding="utf-8") as f:
            for uid in missing_ids:
                f.write(f"{uid}\n")
        print(f"[warn] {n_missing} 个 id 在 GT txt 中找不到 -> {miss_path}")

    summary_lines.append("")
    summary_lines.append(f"总正确句 (去重): {len(all_correct_uids)}")
    summary_lines.append(f"匹配到 GT 原文: {n_found}")
    summary_lines.append(f"GT 中缺失:      {n_missing}")

    summary_path = output_dir / "correct_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[save] {summary_path}")

    print(f"\n[done] correct case 抽取完成")


if __name__ == "__main__":
    main()
