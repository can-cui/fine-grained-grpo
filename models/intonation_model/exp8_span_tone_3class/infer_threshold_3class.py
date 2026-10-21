"""
infer_threshold_3class.py — exp8_3class: txt + wav -> predictions (three thresholds with rejection; inference only)

Differences from the binary-classification infer_threshold.py:
  - Tone uses a 3-class softmax (rising/falling/flat), with P(rising)+P(falling)+P(flat)=1
  - Tone decision: apply an independent three-way threshold decision to each class (same as in binary infer_threshold.py):
        P(rising)  >= rising_threshold   -> rising
        P(falling) >= falling_threshold  -> falling
        P(flat)    >= flat_threshold     -> flat
        None satisfied (gray zone)       -> reject; discard the word from the results
        If multiple thresholds are met, priority is rising > falling > flat
  - Alignment: use whisperX forced alignment to obtain word-level timestamps
  - Inference only: no ground truth and no metric evaluation

Thresholds:
  Higher thresholds create a wider gray zone and more rejections. Setting all three thresholds to 0 results in almost no rejections.

Usage (usually invoked through a wrapper such as run.sh):
  python infer_threshold_3class.py \\
      --text-path  <id\\tsentence text file> \\
      --wav-dir    <wav directory>            \\
      --fbk-dir    <fbank .npy cache directory> \\
      --ckpt  <model> --vocab <vocab.json> --stats <stats.npz> \\
      --output-txt <output txt file>          \\
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
# Read id\tsentence text
# =========================================================================

def load_text(text_path):
    """Return [(id, sentence), ...]. Each line is id<whitespace>sentence; blank lines are skipped."""
    pairs = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"  [skip] Could not parse line (missing sentence): {line[:60]}")
                continue
            uid, sentence = parts[0].strip(), parts[1].strip()
            pairs.append((uid, sentence))
    print(f"[text] Read {len(pairs)} (id, sentence) pairs from {text_path}")
    return pairs


# =========================================================================
# whisperX forced alignment: (wav, transcript) -> word-level (start_s, end_s, word)
# =========================================================================

def build_aligner(device):
    """Load the whisperX alignment model once and return (whisperx, align_model, metadata)."""
    import whisperx
    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=str(device)
    )
    return whisperx, align_model, metadata


def align_one(whisperx, align_model, metadata, wav_path, sentence, device):
    """Run forced alignment on one WAV file and return [(start_s, end_s, word), ...]."""
    audio = whisperx.load_audio(str(wav_path))
    dur = len(audio) / 16000.0  # whisperx.load_audio resamples to 16 kHz
    segments = [{"start": 0.0, "end": dur, "text": sentence}]
    result = whisperx.align(
        segments, align_model, metadata, audio, str(device),
        return_char_alignments=False,
    )
    intervals = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue  # whisperX provides no timestamps for some words (punctuation or failed alignment); skip them
            txt = str(w.get("word", "")).strip()
            if not txt:
                continue
            intervals.append((float(w["start"]), float(w["end"]), txt))
    return intervals


# =========================================================================
# Automatic fbank extraction (same as infer.py)
# =========================================================================

def ensure_fbank_batch(wav_dir, fbk_dir, extract_script, n_thread=40):
    """Batch-extract fbank features for the entire WAV directory into fbk_dir.
    The extractor operates on directories via --input_wav_dir / --output_fbk_dir / --n_thread.
    Files follow the naming rule <id>.wav -> <id>.npy; existing .npy files are skipped automatically."""
    Path(fbk_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, extract_script,
        "--input_wav_dir", str(wav_dir),
        "--output_fbk_dir", str(fbk_dir),
        "--n_thread", str(n_thread),
    ]
    print(f"[fbk] Batch extraction: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [extract failed] {wav_dir}: {e}")
        return False


# =========================================================================
# Build samples (whisperX intervals -> sample dict)
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
# Dataset (for inference)
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
    parser.add_argument("--text-path", required=True, help="Text file with id\\tsentence rows")
    parser.add_argument("--wav-dir", required=True, help="WAV directory; id maps to <wav-dir>/<id>.wav")
    parser.add_argument("--fbk-dir", required=True, help="fbank .npy cache directory (created if missing)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--extract-script", default=C.EXTRACT_SCRIPT)
    parser.add_argument("--output-txt", required=True, help="Output columns: id\\tword\\ttone_pred\\tP(pred)")
    parser.add_argument("--output-json", default="", help="Optional detailed JSON output; leave empty to disable")
    parser.add_argument("--wav-suffix", default=".wav")
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--encoder-type", type=str, default=C.ENCODER_TYPE,
                        choices=["bilstm", "bilstm_residual", "transformer"])
    parser.add_argument("--rising-threshold", type=float, default=0.0,
                        help="Predict rising when P(rising) is at least this value")
    parser.add_argument("--falling-threshold", type=float, default=0.0,
                        help="Predict falling when P(falling) is at least this value")
    parser.add_argument("--flat-threshold", type=float, default=0.0,
                        help="Predict flat when P(flat) is at least this value; reject and discard if none pass")
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

    # 1. Extract fbank features for the entire wav directory (directory-level script; existing .npy files are skipped)
    ensure_fbank_batch(wav_dir, fbk_dir, args.extract_script)

    # 2. Align with whisperX and build samples
    print("[align] Loading the whisperX alignment model...")
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

    # 3. Load the model
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

    # 4. Run inference (argmax candidate + per-class thresholds with rejection)
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
            # Apply a three-way decision with an independent threshold per class (same as binary classification; in overlaps, prioritize rising > falling > flat)
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
                    if tp == REJECT:        # Insufficient confidence -> discard
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

    # 5. Write outputs
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

    print(f"[done] {len(results)} utterances, retained words: {n_words_kept}/{n_words_total} "
          f"(rejected: {n_words_total - n_words_kept})")


if __name__ == "__main__":
    main()
