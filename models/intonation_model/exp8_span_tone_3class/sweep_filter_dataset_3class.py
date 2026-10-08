"""
sweep_filter_dataset_3class.py — exp8_3class: 三阈值组合扫描 + 高精度筛选 dataset.json

在二分类 sweep_filter_dataset.py 基础上改为三分类版本:
  - rising/falling/flat 阈值各传一个列表, 做笛卡尔积组合
  - 每个组合都跑一遍筛选, 输出一份与输入同格式的 dataset.json
  - 输出只给文件夹路径, 文件名按阈值自动命名:
        dataset_r{rising}_f{falling}_fl{flat}.json   (小数点 -> p, 如 dataset_r0p90_f0p80_fl0p70.json)
  - 末尾打印一张对比表: 每个组合的留存率 + rising/falling/flat precision + 各类词数

判定逻辑 (三分类 softmax):
  对每个有 tone 标注的词 (tone_labels[i] ∈ {0=rising, 1=falling, 2=flat}):
    c = argmax(P_rising, P_falling, P_flat)
    若 P(c) >= threshold[c] 且 原标签 == c  -> 该词过关
    否则 (置信度不足 或 预测与标签矛盾)    -> 该词不过关
  整条样本: 所有 tone 词都过关 -> 保留; 任一词不过关 -> 整条丢弃。
  (默认) 没有任何 rising/falling/flat 词的样本不保留, 可用 --keep-no-tone 改变。

用法:
    python sweep_filter_dataset_3class.py \\
        --ckpt   <模型路径>            \\
        --vocab  <vocab.json 路径>     \\
        --stats  <stats.npz 路径>      \\
        --input  <输入 dataset.json>   \\
        --output-dir <输出文件夹>      \\
        --rising-thresholds  0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \\
        --falling-thresholds 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 \\
        --flat-thresholds    0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
"""
import argparse
import json
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import config as C
from data_utils import (
    load_json, WordTokenizer, IntonationDataset, collate_fn, load_stats,
)
from model import IntonationV4Model

RISING, FALLING, FLAT = 0, 1, 2
TONE_NAMES = ["rising", "falling", "flat"]


def filter_dataset(model, loader, device, data, thresholds, keep_no_tone=False):
    """返回 (kept_samples: list, stats: dict, csv_rows: list)。逐词判定, 整条去留。

    thresholds: (T_rising, T_falling, T_flat)
    """
    model.eval()
    T_r, T_f, T_fl = thresholds

    kept_samples = []
    csv_rows = []
    n_total = 0
    n_kept = 0
    n_drop_gray = 0          # 因置信度不足被丢的句子数
    n_drop_mismatch = 0      # 因预测与标签矛盾被丢的句子数
    n_drop_no_tone = 0       # 因无 tone 词被丢的句子数
    n_words_total = 0
    n_words_pass = 0

    # 全集各类词总数 (分母)
    n_total_rising = 0
    n_total_falling = 0
    n_total_flat = 0

    # 保留集各类词数
    n_kept_rising = 0
    n_kept_falling = 0
    n_kept_flat = 0

    # 含 >=1 个该类词的保留句子数
    n_kept_with_rising = 0
    n_kept_with_falling = 0
    n_kept_with_flat = 0

    # precision 统计 (纯按阈值预测 vs 真值, 不受整句去留影响):
    #   预测 c = argmax(P_r, P_f, P_fl)
    #   若 P(c) >= T[c], 预测为 c; 否则拒识(不计入)
    #   rising_precision  = #(预测rising 且 真值rising)  / #(预测rising)
    rising_pred = 0
    rising_pred_correct = 0
    falling_pred = 0
    falling_pred_correct = 0
    flat_pred = 0
    flat_pred_correct = 0

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            fbank = batch["fbank"].to(device)
            fbank_mask = batch["fbank_mask"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            tone_target = batch["tone_target"].to(device)  # (B, N), 0/1/2/-1
            sil_mask = batch["sil_mask"].to(device)
            word_spans = batch["word_spans"].to(device)
            uids = batch["uid"]

            out = model(fbank, fbank_mask, text_tokens, text_mask, word_spans)
            tone_prob = out["tone_logits"].softmax(-1)  # (B, N, 3)
            p_rising = tone_prob[..., RISING]
            p_falling = tone_prob[..., FALLING]
            p_flat = tone_prob[..., FLAT]

            valid = (text_mask > 0.5) & (sil_mask < 0.5)
            tone_valid = valid & (tone_target >= 0)   # tone_target ∈ {0,1,2}

            B = fbank.size(0)
            for i in range(B):
                n_total += 1
                original_sample = data[sample_idx]
                sample_idx += 1

                words = original_sample["words"]
                tone_labels = original_sample["tone_labels"]
                word_boundaries = original_sample.get("word_boundaries", [])
                sentence = " ".join(words)
                uid = uids[i]

                tone_idx = tone_valid[i].nonzero(as_tuple=True)[0]
                if tone_idx.numel() == 0:
                    if keep_no_tone:
                        kept_samples.append(original_sample)
                        n_kept += 1
                    else:
                        n_drop_no_tone += 1
                    continue

                # 全集各类词数统计 + precision 统计 (遍历所有 tone 词)
                for j in tone_idx.tolist():
                    label = int(tone_target[i, j].item())
                    pr = float(p_rising[i, j].item())
                    pf = float(p_falling[i, j].item())
                    pfl = float(p_flat[i, j].item())

                    if label == RISING:
                        n_total_rising += 1
                    elif label == FALLING:
                        n_total_falling += 1
                    elif label == FLAT:
                        n_total_flat += 1

                    # argmax 候选
                    cand = RISING if pr >= pf and pr >= pfl else (FALLING if pf >= pfl else FLAT)
                    if cand == RISING and pr >= T_r:
                        rising_pred += 1
                        if label == RISING:
                            rising_pred_correct += 1
                    elif cand == FALLING and pf >= T_f:
                        falling_pred += 1
                        if label == FALLING:
                            falling_pred_correct += 1
                    elif cand == FLAT and pfl >= T_fl:
                        flat_pred += 1
                        if label == FLAT:
                            flat_pred_correct += 1

                # 整句判定
                sentence_ok = True
                drop_reason = None
                sent_rising = 0
                sent_falling = 0
                sent_flat = 0

                for j in tone_idx.tolist():
                    label = int(tone_target[i, j].item())
                    pr = float(p_rising[i, j].item())
                    pf = float(p_falling[i, j].item())
                    pfl = float(p_flat[i, j].item())

                    cand = RISING if pr >= pf and pr >= pfl else (FALLING if pf >= pfl else FLAT)
                    p_cand = pr if cand == RISING else (pf if cand == FALLING else pfl)
                    T_cand = T_r if cand == RISING else (T_f if cand == FALLING else T_fl)

                    word_pass = (p_cand >= T_cand) and (label == cand)

                    n_words_total += 1
                    if word_pass:
                        n_words_pass += 1
                        if label == RISING:
                            sent_rising += 1
                        elif label == FALLING:
                            sent_falling += 1
                        else:
                            sent_flat += 1
                    else:
                        sentence_ok = False
                        drop_reason = "mismatch" if cand != label else "gray"
                        break

                if sentence_ok:
                    kept_samples.append(original_sample)
                    n_kept += 1
                    n_kept_rising += sent_rising
                    n_kept_falling += sent_falling
                    n_kept_flat += sent_flat
                    if sent_rising > 0:
                        n_kept_with_rising += 1
                    if sent_falling > 0:
                        n_kept_with_falling += 1
                    if sent_flat > 0:
                        n_kept_with_flat += 1

                    # 准备 csv 行 (所有保留的 tone 词)
                    for j in tone_idx.tolist():
                        label = int(tone_target[i, j].item())
                        # word_boundaries[0] 是整句 [0, total_frames], 从 [1] 开始是每个词
                        if word_boundaries and len(word_boundaries) > j + 1:
                            word_boundary = word_boundaries[j + 1]
                            start_frame, end_frame = word_boundary[0], word_boundary[1]
                            start_s = start_frame * 0.01
                            end_s = end_frame * 0.01
                        else:
                            start_s = 0.0
                            end_s = 0.0

                        csv_rows.append({
                            "id": uid,
                            "word_idx": j,
                            "word_start_s": f"{start_s:.2f}",
                            "word_end_s": f"{end_s:.2f}",
                            "word": words[j] if j < len(words) else "",
                            "tone_label": TONE_NAMES[label] if 0 <= label < 3 else str(label),
                            "manual_label": "",
                            "sentence": sentence,
                        })

                elif drop_reason == "mismatch":
                    n_drop_mismatch += 1
                else:
                    n_drop_gray += 1

    stats = {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_drop_gray": n_drop_gray,
        "n_drop_mismatch": n_drop_mismatch,
        "n_drop_no_tone": n_drop_no_tone,
        "keep_rate": n_kept / n_total if n_total > 0 else 0.0,
        "n_words_evaluated": n_words_total,
        "n_words_pass": n_words_pass,
        "n_total_rising": n_total_rising,
        "n_total_falling": n_total_falling,
        "n_total_flat": n_total_flat,
        "n_kept_rising": n_kept_rising,
        "n_kept_falling": n_kept_falling,
        "n_kept_flat": n_kept_flat,
        "rising_keep_rate": (n_kept_rising / n_total_rising) if n_total_rising > 0 else 0.0,
        "falling_keep_rate": (n_kept_falling / n_total_falling) if n_total_falling > 0 else 0.0,
        "flat_keep_rate": (n_kept_flat / n_total_flat) if n_total_flat > 0 else 0.0,
        "n_kept_with_rising": n_kept_with_rising,
        "n_kept_with_falling": n_kept_with_falling,
        "n_kept_with_flat": n_kept_with_flat,
        "rising_pred": rising_pred,
        "rising_pred_correct": rising_pred_correct,
        "rising_precision": (rising_pred_correct / rising_pred) if rising_pred > 0 else 0.0,
        "falling_pred": falling_pred,
        "falling_pred_correct": falling_pred_correct,
        "falling_precision": (falling_pred_correct / falling_pred) if falling_pred > 0 else 0.0,
        "flat_pred": flat_pred,
        "flat_pred_correct": flat_pred_correct,
        "flat_precision": (flat_pred_correct / flat_pred) if flat_pred > 0 else 0.0,
    }
    return kept_samples, stats, csv_rows


def _fmt_thr(x):
    """0.96 -> '0p96' 用于文件名"""
    return f"{x:.2f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="训练好的 best model checkpoint 路径")
    parser.add_argument("--vocab", required=True, help="vocab.json 路径")
    parser.add_argument("--stats", required=True, help="stats.npz 路径")
    parser.add_argument("--input", required=True, help="输入 dataset.json")
    parser.add_argument("--output-dir", required=True,
                        help="输出文件夹 (每个阈值组合一份 dataset_r{r}_f{f}_fl{fl}.json)")
    parser.add_argument("--rising-thresholds", type=str, default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                        help="rising 阈值列表, 逗号分隔。P(rising)>=T 且标签==rising 才过关")
    parser.add_argument("--falling-thresholds", type=str, default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                        help="falling 阈值列表, 逗号分隔。P(falling)>=T 且标签==falling 才过关")
    parser.add_argument("--flat-thresholds", type=str, default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                        help="flat 阈值列表, 逗号分隔。P(flat)>=T 且标签==flat 才过关")
    parser.add_argument("--keep-no-tone", action="store_true",
                        help="保留没有任何 rising/falling/flat 词的样本 (默认丢弃)")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--encoder-type", type=str, default=C.ENCODER_TYPE,
                        choices=["bilstm", "bilstm_residual", "transformer"])
    parser.add_argument("--save-csv", action="store_true",
                        help="保存每个组合的 csv 文件 (默认只保存 sweep_summary.csv)")
    args = parser.parse_args()

    rising_ts = [float(t.strip()) for t in args.rising_thresholds.split(",") if t.strip()]
    falling_ts = [float(t.strip()) for t in args.falling_thresholds.split(",") if t.strip()]
    flat_ts = [float(t.strip()) for t in args.flat_thresholds.split(",") if t.strip()]
    combos = [(r, f, fl) for r in rising_ts for f in falling_ts for fl in flat_ts]
    print(f"[sweep] rising_thresholds={rising_ts}")
    print(f"[sweep] falling_thresholds={falling_ts}")
    print(f"[sweep] flat_thresholds={flat_ts}")
    print(f"[sweep] 共 {len(combos)} 组组合, keep_no_tone={args.keep_no_tone}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={device}")

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

    print(f"[load] input dataset from {args.input}")
    data = load_json(args.input)
    dataset = IntonationDataset(data, stats, tokenizer)
    # 预加载一次 loader, 所有组合复用 (DataLoader 可重复迭代)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    sweep_rows = []   # (rising, falling, flat, stats, out_path, n_saved)
    for (r, f, fl) in combos:
        print("\n" + "#" * 60)
        print(f"# rising={r}  falling={f}  flat={fl}")
        print("#" * 60)
        kept_samples, fstats, csv_rows = filter_dataset(
            model, loader, device, data,
            thresholds=(r, f, fl),
            keep_no_tone=args.keep_no_tone,
        )

        out_path = output_dir / f"dataset_r{_fmt_thr(r)}_f{_fmt_thr(f)}_fl{_fmt_thr(fl)}.json"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(kept_samples, fp, ensure_ascii=False, indent=2)
        tmp_path.replace(out_path)

        # 可选: 保存 csv
        if args.save_csv:
            csv_path = output_dir / f"words_r{_fmt_thr(r)}_f{_fmt_thr(f)}_fl{_fmt_thr(fl)}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                fieldnames = ["id", "word_idx", "word_start_s", "word_end_s",
                             "word", "tone_label", "manual_label", "sentence"]
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        print(f"[result] 输入样本: {fstats['n_total']}")
        print(f"[result] 保留样本: {fstats['n_kept']}  (keep_rate={fstats['keep_rate']:.4f})")
        print(f"[result] 丢弃-灰区: {fstats['n_drop_gray']}, "
              f"预测矛盾: {fstats['n_drop_mismatch']}, 无tone词: {fstats['n_drop_no_tone']}")
        print(f"[result] rising  词: {fstats['n_kept_rising']}/{fstats['n_total_rising']} "
              f"(保有率={fstats['rising_keep_rate']:.4f}, precision={fstats['rising_precision']:.4f})")
        print(f"[result] falling 词: {fstats['n_kept_falling']}/{fstats['n_total_falling']} "
              f"(保有率={fstats['falling_keep_rate']:.4f}, precision={fstats['falling_precision']:.4f})")
        print(f"[result] flat    词: {fstats['n_kept_flat']}/{fstats['n_total_flat']} "
              f"(保有率={fstats['flat_keep_rate']:.4f}, precision={fstats['flat_precision']:.4f})")
        print(f"[save] {out_path}  ({len(kept_samples)} 条)")

        sweep_rows.append((r, f, fl, fstats, out_path, len(kept_samples)))

    # 对比表
    print("\n" + "=" * 140)
    print("Sweep comparison")
    print("=" * 140)
    header = (f"{'rising':>7}{'falling':>9}{'flat':>7}  "
              f"{'kept':>7}{'total':>7}{'keep%':>8}  "
              f"{'R-prec':>8}{'F-prec':>8}{'Fl-prec':>9}  "
              f"{'R-keep':>8}{'F-keep':>8}{'Fl-keep':>9}  "
              f"{'rising_w':>9}{'falling_w':>10}{'flat_w':>8}")
    print(header)
    for (r, f, fl, st, _path, _n) in sweep_rows:
        print(f"{r:>7.2f}{f:>9.2f}{fl:>7.2f}  "
              f"{st['n_kept']:>7d}{st['n_total']:>7d}{st['keep_rate']*100:>7.2f}%  "
              f"{st['rising_precision']:>8.4f}{st['falling_precision']:>8.4f}{st['flat_precision']:>9.4f}  "
              f"{st['rising_keep_rate']:>8.4f}{st['falling_keep_rate']:>8.4f}{st['flat_keep_rate']:>9.4f}  "
              f"{st['n_kept_rising']:>9d}{st['n_kept_falling']:>10d}{st['n_kept_flat']:>8d}")

    # 同时把对比表写一份 csv 到输出目录
    summary_csv = output_dir / "sweep_summary.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("rising_threshold,falling_threshold,flat_threshold,n_total,n_kept,keep_rate,"
                "rising_precision,falling_precision,flat_precision,"
                "rising_keep_rate,falling_keep_rate,flat_keep_rate,"
                "rising_pred,rising_pred_correct,falling_pred,falling_pred_correct,"
                "flat_pred,flat_pred_correct,"
                "n_drop_gray,n_drop_mismatch,n_drop_no_tone,"
                "n_total_rising,n_total_falling,n_total_flat,"
                "n_kept_rising,n_kept_falling,n_kept_flat,"
                "n_kept_with_rising,n_kept_with_falling,n_kept_with_flat,output_file\n")
        for (r, ft, fl, st, path, _n) in sweep_rows:
            f.write(f"{r},{ft},{fl},{st['n_total']},{st['n_kept']},{st['keep_rate']:.6f},"
                    f"{st['rising_precision']:.6f},{st['falling_precision']:.6f},{st['flat_precision']:.6f},"
                    f"{st['rising_keep_rate']:.6f},{st['falling_keep_rate']:.6f},{st['flat_keep_rate']:.6f},"
                    f"{st['rising_pred']},{st['rising_pred_correct']},"
                    f"{st['falling_pred']},{st['falling_pred_correct']},"
                    f"{st['flat_pred']},{st['flat_pred_correct']},"
                    f"{st['n_drop_gray']},{st['n_drop_mismatch']},{st['n_drop_no_tone']},"
                    f"{st['n_total_rising']},{st['n_total_falling']},{st['n_total_flat']},"
                    f"{st['n_kept_rising']},{st['n_kept_falling']},{st['n_kept_flat']},"
                    f"{st['n_kept_with_rising']},{st['n_kept_with_falling']},{st['n_kept_with_flat']},{path.name}\n")
    print(f"\n[save] sweep summary -> {summary_csv}")
    print("=" * 140)
    print(f"\n[done] 共输出 {len(sweep_rows)} 份筛选后的 dataset.json")


if __name__ == "__main__":
    main()
