"""
sample_demo.py — Sample examples by tone from correctcase.txt

Rules:
  - A sentence is eligible for a tone only if all its labels have that tone (rising/falling/flat).
    Sentences with multiple tones are excluded so the three classes do not overlap.
  - Randomly sample 10 sentences per class.
  - Create one directory per class containing:
      <tone>/<tone>_samples.txt   : id <TAB> labeled text
      <tone>/wav/<matched wav>    : wav copied from the source directory

ID matching:
  - IDs in correctcase.txt are numeric (e.g. 10010301).
  - WAV names such as EnUs_kaoshiyuan_male_50000332.wav are matched by their trailing digits.

Usage:
    python sample_demo.py \\
        --correctcase /train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/correctcase_adrian/correctcase.txt \\
        --wav-dir     /yrfs5/tts/mezhao/kaoshiyuan_project/data/Adrian/24k_rename_wav \\
        --out-dir     . \\
        --n 10 --seed 42
"""
import argparse
import random
import re
import shutil
from pathlib import Path

TONES = ["rising", "falling", "flat"]
TAG_RE = re.compile(r"【\s*(Rising|Falling|Flat)\s*】", re.IGNORECASE)


def num_key(s):
    """Extract trailing consecutive digits and remove leading zeros: 'EnUs_..._50000332' -> '50000332'."""
    m = re.search(r"(\d+)\s*$", str(s))
    if not m:
        return str(s)
    return m.group(1).lstrip("0") or "0"


def parse_correctcase(path):
    """Read correctcase.txt into a list of (id, text, tone_set) tuples.
    Each line contains id<whitespace>text (including tone labels)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            uid = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            tags = {t.lower() for t in TAG_RE.findall(text)}
            rows.append((uid, text, tags))
    print(f"[correctcase] Read {len(rows)} sentences from {path}")
    return rows


def build_wav_index(wav_dir):
    """Scan the WAV directory and return {numeric_key: Path}."""
    wav_dir = Path(wav_dir)
    index = {}
    n_dup = 0
    for p in wav_dir.glob("*.wav"):
        key = num_key(p.stem)
        if key in index:
            n_dup += 1
        index[key] = p
    print(f"[wav] Indexed {len(index)} WAV files from {wav_dir}" + (f"  (duplicate trailing digits: {n_dup})" if n_dup else ""))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctcase", required=True, help="Path to correctcase.txt")
    parser.add_argument("--wav-dir", required=True, help="Source WAV directory")
    parser.add_argument("--out-dir", default=".", help="Output root directory (defaults to the current directory)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = parse_correctcase(args.correctcase)
    wav_index = build_wav_index(args.wav_dir)

    # Classify by sentence only
    pure = {t: [] for t in TONES}
    for uid, text, tags in rows:
        if len(tags) == 1:
            t = next(iter(tags))
            pure[t].append((uid, text))

    out_root = Path(args.out_dir)

    for tone in TONES:
        pool = pure[tone]
        print(f"\n[{tone}] Pure-sentence pool: {len(pool)} sentences")

        k = min(args.n, len(pool))
        if k < args.n:
            print(f"  [warn] Only {len(pool)} pure {tone} sentences are available, fewer than {args.n}; using all")
        picked = random.sample(pool, k) if k > 0 else []

        tone_dir = out_root / tone
        wav_out = tone_dir / "wav"
        wav_out.mkdir(parents=True, exist_ok=True)

        # Write the text file
        txt_path = tone_dir / f"{tone}_samples.txt"
        n_wav_ok = 0
        n_wav_miss = 0
        with open(txt_path, "w", encoding="utf-8") as f:
            for uid, text in picked:
                f.write(f"{uid}\t{text}\n")
                key = num_key(uid)
                src = wav_index.get(key)
                if src is not None and src.exists():
                    shutil.copy2(src, wav_out / src.name)
                    n_wav_ok += 1
                else:
                    n_wav_miss += 1
                    print(f"  [warn] No matching WAV found for id={uid} (key={key})")
        print(f"  Sampled {len(picked)} sentences -> {txt_path}")
        print(f"  WAV copies: {n_wav_ok} succeeded, {n_wav_miss} missing -> {wav_out}")

    print(f"\n[done] Demo sampling complete; output is in {out_root.resolve()}")


if __name__ == "__main__":
    main()
