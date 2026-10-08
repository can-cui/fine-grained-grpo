"""
infer_threshold_3class.py — exp8_3class: txt + wav -> pred (三阈值带拒识, 纯推理)

和二分类 infer_threshold.py 的差异:
  - tone 是 3 类 softmax (rising/falling/flat), P(rising)+P(falling)+P(flat)=1
  - tone 判决: 每类独立阈值三态判决 (和二分类 infer_threshold.py 完全一致):
        P(rising)  >= rising_threshold   -> rising
        P(falling) >= falling_threshold  -> falling
        P(flat)    >= flat_threshold     -> flat
        都不满足 (灰区)                  -> 拒识, 该词直接丢掉, 不出现在结果里
        多个同时满足时优先级 rising > falling > flat
  - 对齐: whisperX forced alignment 拿逐词时间
  - 纯推理, 没有 GT, 不做任何指标测试

阈值含义:
  阈值越高灰区越大, 拒识越多。三类阈值都设 0 时几乎不拒识。

用法 (一般用 run.sh 包一层):
  python infer_threshold_3class.py \\
      --text-path  <id\\tsentence 文本>     \\
      --wav-dir    <wav 目录>               \\
      --fbk-dir    <fbank .npy 缓存目录>    \\
      --ckpt  <模型> --vocab <vocab.json> --stats <stats.npz> \\
      --output-txt <输出 txt>               \\
      --rising-threshold 0.9 --falling-threshold 0.8 --flat-threshold 0.7
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config as C
from data_utils import WordTokenizer, collate_fn, load_stats, _parse_word_boundaries
from model import IntonationV4Model

TONE_NAMES = ["rising", "falling", "flat"]
RISING, FALLING, FLAT, REJECT = 0, 1, 2, 3


# =========================================================================
# 读取 id\tsentence 文本
# =========================================================================

def load_text(text_path):
    """返回 [(id, sentence), ...]。每行 id<空格>sentence, 跳过空行。"""
    pairs = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"  [skip] 无法解析(缺 sentence): {line[:60]}")
                continue
            uid, sentence = parts[0].strip(), parts[1].strip()
            pairs.append((uid, sentence))
    print(f"[text] 读到 {len(pairs)} 条 (id sentence) from {text_path}")
    return pairs


# =========================================================================
# whisperX forced alignment: (wav, transcript) -> 逐词 (start_s, end_s, word)
# =========================================================================

def build_aligner(device):
    """加载 whisperX 对齐模型一次, 返回 (whisperx, align_model, metadata)。"""
    import whisperx
    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=str(device)
    )
    return whisperx, align_model, metadata


def align_one(whisperx, align_model, metadata, wav_path, sentence, device):
    """对单条 wav 做 forced alignment, 返回 [(start_s, end_s, word), ...]。"""
    audio = whisperx.load_audio(str(wav_path))
    dur = len(audio) / 16000.0  # whisperx.load_audio 重采样到 16k
    segments = [{"start": 0.0, "end": dur, "text": sentence}]
    result = whisperx.align(
        segments, align_model, metadata, audio, str(device),
        return_char_alignments=False,
    )
    intervals = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue  # whisperX 对某些词 (标点/未对上) 不给时间, 跳过
            txt = str(w.get("word", "")).strip()
            if not txt:
                continue
            intervals.append((float(w["start"]), float(w["end"]), txt))
    return intervals


# =========================================================================
# fbank 自动提取 (和 infer.py 一致)
# =========================================================================

def ensure_fbank_batch(wav_dir, fbk_dir, extract_script, n_thread=40):
    """批量提取整个 wav 目录的 fbank 到 fbk_dir。
    提取脚本是目录级: --input_wav_dir / --output_fbk_dir / --n_thread,
    命名规则 <id>.wav -> <id>.npy, 已存在的 .npy 会被脚本自动跳过。"""
    Path(fbk_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, extract_script,
        "--input_wav_dir", str(wav_dir),
        "--output_fbk_dir", str(fbk_dir),
        "--n_thread", str(n_thread),
    ]
    print(f"[fbk] 批量提取: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [extract failed] {wav_dir}: {e}")
        return False


# =========================================================================
# 构造样本 (whisperX intervals -> sample dict)
# =========================================================================

def build_sample(uid, fbk_path, intervals):
    words = [t for (_, _, t) in intervals]
    total_sec = intervals[-1][1] if intervals else 0.0
    total_frames = int(round(total_sec * 100))  # 10ms/frame
    wb = [[0, total_frames]]
    for (s, e, _) in intervals:
        wb.append([int(round(s * 100)), int(round(e * 100))])
    return {
        "ID": uid,
        "fbk": str(fbk_path),
        "words": words,
        "break_labels": [0] * len(words),
        "tone_labels": [-1] * len(words),
        "word_boundaries": wb,
    }


# =========================================================================
# Dataset (推理用)
# =========================================================================

class InferDataset(Dataset):
    def __init__(self, samples, stats, tokenizer,
                 max_frames=C.MAX_FRAMES, max_words=C.MAX_WORDS):
        self.samples = samples
        self.stats = stats
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_words = max_words

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fbank = np.load(s["fbk"]).astype(np.float32)
        T = fbank.shape[0]
        drop = 0
        if T > self.max_frames:
            drop = T - self.max_frames
            fbank = fbank[drop:]
            T = self.max_frames
        fbank = (fbank - self.stats["fbank_mean"]) / self.stats["fbank_std"]

        words = s["words"][:self.max_words]
        text_tokens = self.tokenizer.encode(words)

        raw_spans = _parse_word_boundaries(s.get("word_boundaries", []), len(words), T + drop)
        word_spans = []
        for (sf, ef) in raw_spans:
            sf = max(0, sf - drop)
            ef = max(0, ef - drop)
            sf_d = sf // C.CONV_STRIDE
            ef_d = max(sf_d + 1, ef // C.CONV_STRIDE)
            word_spans.append((sf_d, ef_d))

        return {
            "fbank": torch.from_numpy(fbank),
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "break_target": torch.zeros(len(words), dtype=torch.float32),
            "tone_target": torch.full((len(words),), -1, dtype=torch.long),
            "sil_mask": torch.zeros(len(words), dtype=torch.float32),
            "word_spans": torch.tensor(word_spans, dtype=torch.long),
            "words": words,
            "fbank_len": T,
            "text_len": len(text_tokens),
            "uid": s["ID"],
        }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", required=True, help="id\\tsentence 文本文件")
    parser.add_argument("--wav-dir", required=True, help="wav 目录, id -> <wav-dir>/<id>.wav")
    parser.add_argument("--fbk-dir", required=True, help="fbank .npy 缓存目录 (没有就生成)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--extract-script", default=C.EXTRACT_SCRIPT)
    parser.add_argument("--output-txt", required=True, help="输出: id\\tword\\ttone_pred\\tP(pred)")
    parser.add_argument("--output-json", default="", help="可选: 详细 json, 留空则不写")
    parser.add_argument("--wav-suffix", default=".wav")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--encoder-type", type=str, default=C.ENCODER_TYPE,
                        choices=["bilstm", "bilstm_residual", "transformer"])
    parser.add_argument("--rising-threshold", type=float, default=0.0,
                        help="P(rising)>=此值判 rising")
    parser.add_argument("--falling-threshold", type=float, default=0.0,
                        help="P(falling)>=此值判 falling")
    parser.add_argument("--flat-threshold", type=float, default=0.0,
                        help="P(flat)>=此值判 flat; 三者都不满足则拒识丢弃")
    args = parser.parse_args()

    print(f"[infer] rising_threshold={args.rising_threshold}, "
          f"falling_threshold={args.falling_threshold}, "
          f"flat_threshold={args.flat_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}")

    text_pairs = load_text(args.text_path)
    wav_dir = Path(args.wav_dir)
    fbk_dir = Path(args.fbk_dir)
    fbk_dir.mkdir(parents=True, exist_ok=True)

    # 1. 先批量提取整个 wav 目录的 fbank (目录级脚本, 已存在的 .npy 自动跳过)
    ensure_fbank_batch(wav_dir, fbk_dir, args.extract_script)

    # 2. whisperX 对齐, 构造样本
    print("[align] 加载 whisperX 对齐模型...")
    whisperx, align_model, metadata = build_aligner(device)

    samples = []
    skipped = []
    for uid, sentence in text_pairs:
        wav_path = wav_dir / f"{uid}{args.wav_suffix}"
        if not wav_path.exists():
            skipped.append((uid, "wav_missing"))
            continue
        fbk_path = fbk_dir / f"{uid}.npy"
        if not fbk_path.exists():
            skipped.append((uid, "fbk_missing"))
            continue
        try:
            intervals = align_one(whisperx, align_model, metadata, wav_path, sentence, device)
        except Exception as e:
            skipped.append((uid, f"align_failed:{str(e)[:60]}"))
            continue
        if not intervals:
            skipped.append((uid, "align_empty"))
            continue
        samples.append(build_sample(uid, fbk_path, intervals))

    print(f"  ready: {len(samples)}, skipped: {len(skipped)}")
    for uid, reason in skipped[:10]:
        print(f"  [skip] {uid}: {reason}")

    if not samples:
        print("[infer] no samples to run")
        return

    # 3. 加载模型
    print(f"[load] vocab from {args.vocab}")
    tokenizer = WordTokenizer()
    tokenizer.load(args.vocab)

    print(f"[load] stats from {args.stats}")
    stats = load_stats(args.stats)

    print(f"[load] model from {args.ckpt}")
    model = IntonationV4Model(tokenizer.vocab_size, encoder_type=args.encoder_type).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if "epoch" in ckpt:
        print(f"  ckpt epoch={ckpt['epoch']}, best_score={ckpt.get('best_score', 'N/A')}")

    # 4. 推理 (argmax 候选 + 每类阈值带拒识)
    dataset = InferDataset(samples, stats, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    Tr = args.rising_threshold
    Tf = args.falling_threshold
    Tfl = args.flat_threshold

    results = []
    n_words_total = 0
    n_words_kept = 0
    with torch.no_grad():
        for batch in loader:
            fbank = batch["fbank"].to(device)
            fbank_mask = batch["fbank_mask"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)
            word_spans = batch["word_spans"].to(device)
            uids = batch["uid"]
            words_batch = batch["words"]

            out = model(fbank, fbank_mask, text_tokens, text_mask, word_spans)
            tone_prob = F.softmax(out["tone_logits"], dim=-1)  # (B, N, 3)
            p_rising = tone_prob[..., RISING]
            p_falling = tone_prob[..., FALLING]
            p_flat = tone_prob[..., FLAT]
            # 每类独立阈值三态判决 (和二分类一致, 重叠区优先 rising > falling > flat)
            pred_rising = p_rising >= Tr
            pred_falling = p_falling >= Tf
            pred_flat = p_flat >= Tfl
            tone_pred = torch.full(p_rising.shape, REJECT, dtype=torch.long, device=device)
            tone_pred[pred_flat] = FLAT
            tone_pred[pred_falling] = FALLING
            tone_pred[pred_rising] = RISING

            for i, uid in enumerate(uids):
                words = words_batch[i]
                kept = []
                for j, w in enumerate(words):
                    n_words_total += 1
                    tp = int(tone_pred[i, j].item())
                    if tp == REJECT:        # 置信度不足 -> 直接丢掉
                        continue
                    n_words_kept += 1
                    kept.append({
                        "word_idx": j,
                        "word": w,
                        "tone_pred": TONE_NAMES[tp],
                        "p_rising": round(float(tone_prob[i, j, RISING]), 4),
                        "p_falling": round(float(tone_prob[i, j, FALLING]), 4),
                        "p_flat": round(float(tone_prob[i, j, FLAT]), 4),
                    })
                results.append({"id": uid, "words": words, "kept_words": kept})

    # 5. 输出
    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        for r in results:
            for kw in r["kept_words"]:
                p = kw[f"p_{kw['tone_pred']}"]
                f.write(f"{r['id']}\t{kw['word']}\t{kw['tone_pred']}\t{p:.4f}\n")
    print(f"[save] {args.output_txt}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "ckpt": args.ckpt,
                "rising_threshold": args.rising_threshold,
                "falling_threshold": args.falling_threshold,
                "flat_threshold": args.flat_threshold,
                "n_total": len(text_pairs),
                "n_predicted": len(results),
                "n_skipped": len(skipped),
                "skipped": [{"id": u, "reason": rr} for u, rr in skipped],
                "predictions": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[save] {args.output_json}")

    print(f"[done] {len(results)} 句, 保留词 {n_words_kept}/{n_words_total} "
          f"(拒识 {n_words_total - n_words_kept})")


if __name__ == "__main__":
    main()
