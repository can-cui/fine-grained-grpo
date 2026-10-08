"""
sample_demo.py — 从 correctcase.txt 按 tone 抽样做 demo

规则:
  - 纯句: 整句只含同一种 tone 标签 (rising/falling/flat) 才参与该类抽样
    (同时含多种 tone 的句子被排除, 保证三类互不重叠)
  - 每类随机抽 10 句
  - 每类一个文件夹, 里面存:
      <tone>/<tone>_samples.txt   : id <TAB> 带标签text
      <tone>/wav/<匹配到的wav>     : 从 wav 源目录拷贝过来的 wav

id 匹配:
  - correctcase.txt 的 id 是纯数字 (如 10010301)
  - wav 文件名形如 EnUs_kaoshiyuan_male_50000332.wav, 用末尾数字匹配

用法:
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
    """提取末尾连续数字, 去前导零。'EnUs_..._50000332' -> '50000332'。"""
    m = re.search(r"(\d+)\s*$", str(s))
    if not m:
        return str(s)
    return m.group(1).lstrip("0") or "0"


def parse_correctcase(path):
    """读 correctcase.txt -> list of (id, text, tone_set)。
    每行: id<空白>text(含标签)。"""
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
    print(f"[correctcase] 读取 {len(rows)} 句 from {path}")
    return rows


def build_wav_index(wav_dir):
    """扫描 wav 目录 -> {数字key: Path}。"""
    wav_dir = Path(wav_dir)
    index = {}
    n_dup = 0
    for p in wav_dir.glob("*.wav"):
        key = num_key(p.stem)
        if key in index:
            n_dup += 1
        index[key] = p
    print(f"[wav] 索引 {len(index)} 个 wav from {wav_dir}" + (f"  (末尾数字重复 {n_dup})" if n_dup else ""))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctcase", required=True, help="correctcase.txt 路径")
    parser.add_argument("--wav-dir", required=True, help="wav 源目录")
    parser.add_argument("--out-dir", default=".", help="输出根目录 (默认当前路径)")
    parser.add_argument("--n", type=int, default=10, help="每类抽样数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = parse_correctcase(args.correctcase)
    wav_index = build_wav_index(args.wav_dir)

    # 按纯句分类
    pure = {t: [] for t in TONES}
    for uid, text, tags in rows:
        if len(tags) == 1:
            t = next(iter(tags))
            pure[t].append((uid, text))

    out_root = Path(args.out_dir)

    for tone in TONES:
        pool = pure[tone]
        print(f"\n[{tone}] 纯句池: {len(pool)} 句")

        k = min(args.n, len(pool))
        if k < args.n:
            print(f"  [warn] 纯 {tone} 句只有 {len(pool)} 句, 不足 {args.n}, 全取")
        picked = random.sample(pool, k) if k > 0 else []

        tone_dir = out_root / tone
        wav_out = tone_dir / "wav"
        wav_out.mkdir(parents=True, exist_ok=True)

        # 写 txt
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
                    print(f"  [warn] id={uid} (key={key}) 找不到对应 wav")
        print(f"  抽样 {len(picked)} 句 -> {txt_path}")
        print(f"  wav 拷贝: {n_wav_ok} 成功, {n_wav_miss} 缺失 -> {wav_out}")

    print(f"\n[done] demo 抽样完成, 输出在 {out_root.resolve()}")


if __name__ == "__main__":
    main()
