"""Fairseq dataset for PCM16 pause-prediction LMDB records."""

import io
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch

from fairseq.data import FairseqDataset


class PauseLMDBDataset(FairseqDataset):
    def __init__(self, data_root, split, shuffle=True):
        self.data_root = Path(data_root)
        self.split = split
        self.lmdb_path = self.data_root / split
        self.key_path = self.data_root / f"{split}.key"
        if not self.lmdb_path.is_dir() or not self.key_path.is_file():
            raise FileNotFoundError(f"missing pause LMDB/key pair for {split} under {data_root}")
        self.keys, audio_sizes, word_sizes = [], [], []
        with self.key_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    raise ValueError(f"{self.key_path}:{line_number}: invalid key row")
                self.keys.append(parts[0])
                audio_sizes.append(int(parts[1]))
                word_sizes.append(int(parts[2]))
        self.audio_sizes = np.asarray(audio_sizes, dtype=np.int64)
        self.word_sizes = np.asarray(word_sizes, dtype=np.int64)
        self.sizes = self.audio_sizes
        self.shuffle = shuffle
        self._open_lmdb()

    def _open_lmdb(self):
        self.lmdb_env = lmdb.open(
            str(self.lmdb_path), readonly=True, lock=False, readahead=False, subdir=True
        )
        self.txn = self.lmdb_env.begin()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("lmdb_env", None)
        state.pop("txn", None)
        state.pop("_read_record", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_lmdb()

    def __del__(self):
        if getattr(self, "lmdb_env", None) is not None:
            self.lmdb_env.close()

    def __len__(self):
        return len(self.keys)

    @lru_cache(maxsize=8)
    def _read_record(self, index):
        raw = self.txn.get(self.keys[index].encode("utf-8"))
        if raw is None:
            raise KeyError(f"missing LMDB key {self.keys[index]}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as record:
            return {name: record[name].copy() for name in record.files}

    def __getitem__(self, index):
        record = self._read_record(index)
        waveform = torch.from_numpy(record["waveform_pcm16"].astype(np.float32)) / 32768.0
        return {
            "id": index,
            "utt_id": str(record["utt_id"].item()),
            "speaker_id": str(record["speaker_id"].item()),
            "waveform": waveform,
            "pool_start_sample": torch.from_numpy(record["pool_start_sample"]).long(),
            "pool_end_sample": torch.from_numpy(record["pool_end_sample"]).long(),
            "target": torch.from_numpy(record["pause_label"].astype(np.float32)),
            "valid_mask": torch.from_numpy(record["valid_mask"].astype(np.bool_)),
            "word_list": record["word_list"].astype(str).tolist(),
            "pool_start_seconds": record["pool_start_seconds"],
            "pool_end_seconds": record["pool_end_seconds"],
            "sample_rate": int(record["sample_rate"].item()),
        }

    def size(self, index):
        return int(self.audio_sizes[index])

    def num_tokens(self, index):
        return int(self.audio_sizes[index])

    def ordered_indices(self):
        indices = np.random.permutation(len(self)) if self.shuffle else np.arange(len(self))
        return indices[np.argsort(self.audio_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return False

    def collater(self, samples):
        if not samples:
            return {}
        batch_size = len(samples)
        max_audio = max(sample["waveform"].numel() for sample in samples)
        max_words = max(sample["target"].size(0) for sample in samples)
        waveforms = torch.zeros(batch_size, max_audio, dtype=torch.float32)
        starts = torch.zeros(batch_size, max_words, dtype=torch.long)
        ends = torch.ones(batch_size, max_words, dtype=torch.long)
        targets = torch.zeros(batch_size, max_words, dtype=torch.float32)
        valid_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
        waveform_lengths = torch.zeros(batch_size, dtype=torch.long)
        word_lengths = torch.zeros(batch_size, dtype=torch.long)
        for batch_index, sample in enumerate(samples):
            audio_count, word_count = sample["waveform"].numel(), sample["target"].size(0)
            waveforms[batch_index, :audio_count] = sample["waveform"]
            starts[batch_index, :word_count] = sample["pool_start_sample"]
            ends[batch_index, :word_count] = sample["pool_end_sample"]
            targets[batch_index, :word_count] = sample["target"]
            valid_mask[batch_index, :word_count] = sample["valid_mask"]
            waveform_lengths[batch_index], word_lengths[batch_index] = audio_count, word_count
        return {
            "id": torch.tensor([sample["id"] for sample in samples], dtype=torch.long),
            "utt_id": [sample["utt_id"] for sample in samples],
            "speaker_id": [sample["speaker_id"] for sample in samples],
            "nsentences": batch_size,
            "ntokens": int(valid_mask.long().sum().item()),
            "net_input": {
                "waveform": waveforms,
                "waveform_lengths": waveform_lengths,
                "pool_start_sample": starts,
                "pool_end_sample": ends,
                "word_lengths": word_lengths,
            },
            "target": targets,
            "valid_mask": valid_mask,
            "metadata": samples,
        }
