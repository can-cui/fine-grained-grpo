"""
data_utils.py — exp8: 加载 word_boundaries 用于 span pooling

相比 exp5:
  - Dataset 加载 word_boundaries 字段
  - 转换为下采样后的帧索引 (÷ CONV_STRIDE)
  - collate_fn 新增 word_spans padding (B, N, 2)
  - 删除 F0 相关逻辑
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import random

import config as C


class WordTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    @staticmethod
    def _clean(word):
        if word is None:
            return ""
        w = str(word).split("|", 1)[0]
        return w.lower()

    def build_vocab(self, all_samples):
        for s in all_samples:
            for word in s["words"]:
                w = self._clean(word)
                if w not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
        print(f"[vocab] 词表大小: {len(self.word2idx)}")

    def encode(self, words):
        return [self.word2idx.get(self._clean(w), 1) for w in words]

    def decode(self, ids):
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in ids]

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def save(self, path):
        from pathlib import Path as _P
        p = _P(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}


def compute_and_save_stats(samples, stats_path=C.STATS_NPZ):
    fbank_sum = np.zeros(C.FBANK_DIM, dtype=np.float64)
    fbank_sq = np.zeros(C.FBANK_DIM, dtype=np.float64)
    n_frames = 0

    for i, s in enumerate(samples):
        if i % 500 == 0:
            print(f"  [stats] {i}/{len(samples)}")
        try:
            fb = np.load(s["fbk"]).astype(np.float64)
            fbank_sum += fb.sum(axis=0)
            fbank_sq += (fb ** 2).sum(axis=0)
            n_frames += fb.shape[0]
        except Exception as e:
            print(f"  WARN: {s.get('ID', '?')}: {e}")
            continue

    mean = (fbank_sum / n_frames).astype(np.float32)
    var = fbank_sq / n_frames - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    std = np.maximum(std, 1e-5)

    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    actual_tmp_path = Path(str(tmp_path) + ".npz")
    np.savez(str(tmp_path), fbank_mean=mean, fbank_std=std)
    actual_tmp_path.replace(stats_path)
    print(f"[stats] 保存 -> {stats_path} ({n_frames} 帧)")
    return {"fbank_mean": mean, "fbank_std": std}


def load_stats(stats_path=C.STATS_NPZ):
    d = np.load(str(stats_path), allow_pickle=False)
    return {"fbank_mean": d["fbank_mean"], "fbank_std": d["fbank_std"]}


def _parse_word_boundaries(word_boundaries, n_words, total_frames):
    """将 dataset.json 中的 word_boundaries 转为每个词的 [start, end) 帧索引。

    word_boundaries 格式:
      - 第 0 项: [0, total_ms] 整段时长
      - 第 1~N 项: 每个词的 [start_ms, end_ms]（对应 words[0] ~ words[N-1]）

    但 words[0] 可能是 silence marker（如 "3"），此时 word_boundaries[1] 对应 words[0]。
    所以 word_boundaries[i+1] 对应 words[i]（如果 len(wb) == n_words + 1）。

    返回: list of (start_frame, end_frame)，长度 = n_words
    """
    if not word_boundaries or len(word_boundaries) < 2:
        # fallback: 均匀分配
        chunk = max(1, total_frames // max(n_words, 1))
        return [(i * chunk, min((i + 1) * chunk, total_frames)) for i in range(n_words)]

    # 判断 offset
    if len(word_boundaries) == n_words + 1:
        offset = 1
    elif len(word_boundaries) == n_words:
        offset = 0
    else:
        offset = 1 if len(word_boundaries) > n_words else 0

    spans = []
    for i in range(n_words):
        wb_idx = i + offset
        if wb_idx < len(word_boundaries):
            s_ms, e_ms = word_boundaries[wb_idx]
            s_frame = int(s_ms)
            e_frame = int(e_ms)
        else:
            s_frame = total_frames - 1
            e_frame = total_frames
        s_frame = min(s_frame, total_frames)
        e_frame = min(e_frame, total_frames)
        if s_frame >= e_frame:
            e_frame = min(s_frame + 1, total_frames)
        spans.append((s_frame, e_frame))
    return spans


class IntonationDataset(Dataset):
    def __init__(self, samples, stats, tokenizer, max_frames=C.MAX_FRAMES,
                 max_words=C.MAX_WORDS):
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

        # 如果超长，从前面截掉
        drop = 0
        if T > self.max_frames:
            drop = T - self.max_frames
            fbank = fbank[drop:]
            T = self.max_frames

        fbank = (fbank - self.stats["fbank_mean"]) / self.stats["fbank_std"]

        words = s["words"][:self.max_words]
        text_tokens = self.tokenizer.encode(words)

        break_labels = s["break_labels"][:self.max_words]
        tone_labels = s["tone_labels"][:self.max_words]

        # word_boundaries → 帧级 span（考虑 drop 和下采样）
        raw_wb = s.get("word_boundaries", [])
        raw_spans = _parse_word_boundaries(raw_wb, len(words), T + drop)
        # 调整 drop offset 并转为下采样后的索引
        word_spans = []
        for (sf, ef) in raw_spans:
            sf = max(0, sf - drop)
            ef = max(0, ef - drop)
            sf_down = sf // C.CONV_STRIDE
            ef_down = max(sf_down + 1, ef // C.CONV_STRIDE)
            word_spans.append((sf_down, ef_down))

        # sil_mask
        sil_mask = [0] * len(words)
        if len(break_labels) > 0 and break_labels[0] == 1 and tone_labels[0] == -1:
            sil_mask[0] = 1

        return {
            "fbank": torch.from_numpy(fbank),
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "break_target": torch.tensor(break_labels, dtype=torch.float32),
            "tone_target": torch.tensor(tone_labels, dtype=torch.long),
            "sil_mask": torch.tensor(sil_mask, dtype=torch.float32),
            "word_spans": torch.tensor(word_spans, dtype=torch.long),
            "words": words,
            "fbank_len": T,
            "text_len": len(text_tokens),
            "uid": s["ID"],
        }


def collate_fn(batch):
    max_T = max(b["fbank_len"] for b in batch)
    max_N = max(b["text_len"] for b in batch)
    B = len(batch)

    fbank      = torch.zeros(B, max_T, C.FBANK_DIM)
    fbank_mask = torch.zeros(B, max_T)
    text_tokens = torch.zeros(B, max_N, dtype=torch.long)
    text_mask   = torch.zeros(B, max_N)
    break_target = torch.zeros(B, max_N)
    tone_target  = torch.full((B, max_N), -1, dtype=torch.long)
    sil_mask     = torch.zeros(B, max_N)
    word_spans   = torch.zeros(B, max_N, 2, dtype=torch.long)
    uids = []
    words_batch = []

    for i, b in enumerate(batch):
        T = b["fbank_len"]
        N = b["text_len"]
        fbank[i, :T]        = b["fbank"]
        fbank_mask[i, :T]   = 1.0
        text_tokens[i, :N]  = b["text_tokens"]
        text_mask[i, :N]    = 1.0
        break_target[i, :N] = b["break_target"]
        tone_target[i, :N]  = b["tone_target"]
        sil_mask[i, :N]     = b["sil_mask"]
        word_spans[i, :N]   = b["word_spans"]
        uids.append(b["uid"])
        words_batch.append(b["words"])

    return {
        "fbank": fbank,
        "fbank_mask": fbank_mask,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "break_target": break_target,
        "tone_target": tone_target,
        "sil_mask": sil_mask,
        "word_spans": word_spans,
        "uid": uids,
        "words": words_batch,
    }


class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        seed = self.base_seed + self.epoch * 1000
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        chunk = self.batch_size * 50
        batches = []
        for start in range(0, len(order), chunk):
            block = order[start:start + chunk]
            if self.shuffle:
                random.Random(seed + start).shuffle(block)
            for i in range(0, len(block), self.batch_size):
                batches.append(block[i:i + self.batch_size])
        if self.shuffle:
            random.Random(seed).shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_sample_lengths(samples):
    lengths = []
    for s in samples:
        try:
            fb = np.load(s["fbk"], mmap_mode="r")
            lengths.append(min(fb.shape[0], C.MAX_FRAMES))
        except Exception:
            lengths.append(300)
    return lengths


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[data] 加载 {len(data)} 条样本 from {json_path}")
    return data
