import sys, torch
from . import register_reward, register_reward_target_type, register_reward_norm_type
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import torchaudio
import soundfile as sf
import re

# Path to the intonation prediction model (exp8)
CKPT_PATH = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/checkpoints/best_exp8_span_tone_3class0625.pt"
VOCAB_PATH = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/vocab.json"
STATS_PATH = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/stats.npz"

# Path to the WhisperX model
WHISPERX_DIR = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/whisperX-main"
WHISPERX_MAIN_DIR = "Fine-Grained-GRPO/models/intonation_model/exp8_span_tone_3class/whisperX-main"

# # Path to calc_fbk (used to extract filter-bank features from 24 kHz audio)
# CALC_FBK_DIR = "/train34/tts/permanent/cancui11/RL/original/src_format_code_silu_rms_split_rope_expand_lang_v3_code_en_edu_add_emo_tag_sep_ratio"
# sys.path.insert(0, CALC_FBK_DIR)
# from inference import calc_fbk
def safe_log10(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log10(torch.clip(x, min=clip_val))

mel_spec_func = torchaudio.transforms.MelSpectrogram(
                    sample_rate=24000,
                    n_mels=80,
                    f_min=0,
                    f_max=12000,
                    n_fft=1200,
                    hop_length=240,
                    win_length=1200,
                    center=True,
                    pad_mode='constant',
                    power=1.0,
                    norm='slaney',
                    mel_scale='slaney',
                )

def calc_fbk(audio_norm,preemphasis=0.97):
    # Pre-emphasis
    audio = torch.cat((audio_norm[...,0:1],audio_norm[...,1:] - preemphasis* audio_norm[...,:-1]),dim=-1) 
    # Mel spectrogram
    mel = mel_spec_func(audio)
    fbk = safe_log10(mel)
    return fbk

# Model configuration
FBANK_DIM = 80
MAX_FRAMES = 600
MAX_WORDS = 80
TONE_NAMES = ["rising", "falling"]
SAMPLE_RATE = 16000

# Add the paths
sys.path.insert(0, WHISPERX_DIR)
sys.path.insert(0, WHISPERX_MAIN_DIR)
import whisperx


# Import the exp8 configuration and model
sys.path.insert(0, "/train34/tts/permanent/cancui11/RL/models/exp8_span_tone_3class")
import config3c as C
from data_utils import WordTokenizer, load_stats
from reward_model3c import IntonationV4Model, span_pool


class WordTokenizerForReward(WordTokenizer):
    """A standalone tokenizer that does not depend on the config."""
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def load(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    @staticmethod
    def _clean(word):
        if word is None:
            return ""
        w = str(word).split("|", 1)[0]
        return w.lower()

    def encode(self, words):
        return [self.word2idx.get(self._clean(w), 1) for w in words]

    @property
    def vocab_size(self):
        return len(self.word2idx)


def _normalize_word(word):
    """Normalize a word (remove punctuation and lowercase it) for fuzzy matching."""
    if word == "…":
        return "…"
    return re.sub(r'[^/w]', '', word.lower())


def _merge_whisperx_to_user_words(user_words, whisperx_segments):
    """Map WhisperX fine-grained tokens to the user-provided coarse-grained tokens.

    Args:
        user_words: Sequence of user-provided words
        whisperx_segments: Output from WhisperX

    Returns:
        word_boundaries: List of [start_cs, end_cs] pairs, with length len(user_words)
    """
    user_norm = [_normalize_word(w) for w in user_words]
    whisperx_norm = [_normalize_word(seg["word"]) for seg in whisperx_segments]

    word_boundaries = []
    wx_idx = 0

    for i, user_w_norm in enumerate(user_norm):
        if not user_w_norm:
            return None

        matched_segs = []
        accumulated = ""

        while wx_idx < len(whisperx_norm):
            wx_word_norm = whisperx_norm[wx_idx]

            if wx_word_norm:
                accumulated += wx_word_norm

            matched_segs.append(whisperx_segments[wx_idx])
            wx_idx += 1

            if user_w_norm in accumulated:
                break

            if len(accumulated) >= len(user_w_norm) and user_w_norm in accumulated:
                break

            if len(accumulated) > len(user_w_norm) * 2:
                return None

        if not matched_segs:
            return None

        start_s = min(seg["start"] for seg in matched_segs)
        end_s = max(seg["end"] for seg in matched_segs)
        start_cs = int(start_s * 100)
        end_cs = int(end_s * 100)
        word_boundaries.append([start_cs, end_cs])

    if len(word_boundaries) != len(user_words):
        return None

    return word_boundaries


@register_reward('intonationAsrProb3c')
class IntonationASR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Load the tokenizer and statistics
        self.tokenizer = WordTokenizerForReward()
        self.tokenizer.load(VOCAB_PATH)
        self.stats = load_stats(STATS_PATH)

        # Load the exp8 intonation model
        self.model = IntonationV4Model(
            self.tokenizer.vocab_size,
            encoder_type=C.ENCODER_TYPE
        ).cuda()
        ckpt = torch.load(CKPT_PATH, map_location="cuda")
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Load the WhisperX transcription model
        # self.whisperx_model = whisperx.load_model("large-v3", "cuda", compute_type="int8_float16")
        # Load the WhisperX alignment model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            "en", "cuda", model_dir='/train34/tts/permanent/cancui11/RL/models/whisperX-main/checkpoints'
        )

        self.max_frames = MAX_FRAMES
        self.max_words = MAX_WORDS

    def whisperx_align_words(self, audio_16k, words):
        """Align with WhisperX and obtain word boundaries in centiseconds."""
        # try:
        duration_s = len(audio_16k) / SAMPLE_RATE
        text_str = " ".join(words)
        transcript = [{"start": 0.0, "end": duration_s, "text": text_str}]

        # Align the transcript
        result = whisperx.align(transcript, self.align_model, self.align_metadata,
                                audio_16k, "cuda", return_char_alignments=False)

        if "word_segments" not in result or not result["word_segments"]:
            return None

        whisperx_segments = result["word_segments"]
        word_boundaries = _merge_whisperx_to_user_words(words, whisperx_segments)
        try:
            for i in range(len(word_boundaries) - 1):
                word_boundaries[i][1] = word_boundaries[i + 1][0]
        except:
            import pdb;pdb.set_trace()
        return word_boundaries

        # except Exception as e:
        #     print(f"[warn] whisperx align failed: {e}")
        #     return None

    def cal_reward(self, pred_tone, words, intos):
        """
        Reward computation interface

        Args:
            pred_tone: Predicted intonation tensor
            words: List of text words
            intos: Ground-truth intonation labels

        Returns:
            reward: float
        """
        if not torch.is_tensor(pred_tone):
            pred_tone = torch.tensor(pred_tone)

        if not torch.is_tensor(intos):
            intos = torch.tensor(intos, device=pred_tone.device)

        assert len(pred_tone) == len(intos), /
            f"Length mismatch: pred={len(pred_tone)}, gt={len(intos)}"

        valid_mask = (intos != -1)
        mismatch_count = (pred_tone[valid_mask] != intos[valid_mask]).sum().item()
        reward = -float(mismatch_count)
        # import pdb;pdb.set_trace()
        return reward

    @torch.no_grad()
    def forward(self, wav_batch, labels, wav_masks=None, num_thread=1, infer_bsz=1):
        if num_thread <= 1:
            return self._forward(wav_batch, labels, wav_masks, infer_bsz)

        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        rewards = []
        with ThreadPoolExecutor(max_workers=num_thread) as pool:
            for i in range(total_batch):
                rewards.append(pool.submit(
                    self.reward_func,
                    wav_batch[i * infer_bsz:(i + 1) * infer_bsz],
                    labels[i * infer_bsz:(i + 1) * infer_bsz],
                    wav_masks[i * infer_bsz:(i + 1) * infer_bsz] if wav_masks is not None else None
                ))
        rewards = [f.result() for f in rewards]
        rewards = torch.cat(rewards, dim=0).unsqueeze(-1).to(wav_batch.device)
        return rewards

    def _forward(self, wav_batch, labels, wav_masks=None, infer_bsz=1):
        rewards = []
        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        for i in range(total_batch):
            wav = wav_batch[i * infer_bsz:(i + 1) * infer_bsz]
            wav_mask = wav_masks[i * infer_bsz:(i + 1) * infer_bsz] if wav_masks is not None else None
            label = labels[i]
            reward = self.process_batch(wav, wav_mask, label)
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1).to(wav_batch.device)
        return rewards

    def process_batch(self, wav_batch, wav_masks, labels):
        """
        Process a batch of samples and return rewards
        wav_batch: (batch, samples) waveform at 24 kHz
        labels: Tuple (text, intos) or a single text string
        """
        B = wav_batch.size(0)

        # Process labels
        text, intos = labels
        intos_list = intos


        # Get the valid audio (24 kHz)
        valid_wav = wav_batch[wav_masks.bool()] if wav_masks is not None else wav_batch
        audio_24k = valid_wav.detach().float().cpu()

        # Resample to 16 kHz for WhisperX
        audio_16k = torchaudio.functional.resample(
            audio_24k.unsqueeze(0),
            orig_freq=24000, new_freq=16000
        ).squeeze(0).numpy()

        # Extract filter-bank features from 24 kHz audio (10 ms per frame) using calc_fbk
        audio_24k_for_fbk = audio_24k.unsqueeze(0) if audio_24k.dim() == 1 else audio_24k
        fbank = calc_fbk(audio_24k_for_fbk).squeeze(0)  # (80, frames) from calc_fbk
        fbank = fbank.float().transpose(0, 1).unsqueeze(0).cuda()  # (1, frames, 80)
        # import pdb;pdb.set_trace()
        # Normalize filter-bank features
        fbank = (fbank - torch.from_numpy(self.stats["fbank_mean"]).to(fbank.device).float()) / /
                (torch.from_numpy(self.stats["fbank_std"]).to(fbank.device).float() + 1e-8)

        T = fbank.size(1)
        if T > self.max_frames:
            fbank = fbank[:, T - self.max_frames:, :]
            T = self.max_frames

        fbank_mask = torch.ones(1, T, device=fbank.device)
        # Use WhisperX alignment to obtain word boundaries in centiseconds
        words = text.split()[:self.max_words]
        word_boundaries = self.whisperx_align_words(audio_16k, words)
        
        # Compute word_spans: convert centiseconds to downsampled frame indices
        # At 24 kHz, 10 ms per frame corresponds to 100 frames per second
        # The convolutional front end downsamples by 4x, yielding 25 frames per second; each frame represents 4 centiseconds
        # Frame index = centiseconds // 4
        if word_boundaries is not None:
            word_spans = []
            for start_cs, end_cs in word_boundaries:
                start_frame = start_cs // 4
                end_frame = end_cs // 4
                word_spans.append([start_frame, end_frame])
            word_spans = torch.tensor([word_spans], dtype=torch.long, device=fbank.device)
        else:
            # If alignment fails, use uniformly distributed word spans
            n_words = len(words)
            word_spans = torch.zeros(1, n_words, 2, dtype=torch.long, device=fbank.device)
            frame_per_word = T // max(n_words, 1)
            for i in range(n_words):
                word_spans[0, i, 0] = i * frame_per_word
                word_spans[0, i, 1] = (i + 1) * frame_per_word
        # Process text tokens
        text_tokens = self.tokenizer.encode(words)
        text_tokens = torch.tensor([text_tokens], dtype=torch.long, device=fbank.device)
        text_mask = torch.ones(1, text_tokens.size(1), device=fbank.device)

        # Run model inference (exp8 requires word_spans)
        out = self.model(fbank, fbank_mask, text_tokens, text_mask, word_spans)
        tone_prob = F.softmax(out["tone_logits"], dim=-1)
        tone_pred = tone_prob.argmax(-1).squeeze(0)
        # Apply a threshold: set positions with predicted probabilities below 0.8 to -1
        max_prob = tone_prob.max(dim=-1)[0].squeeze(0)
        tone_pred = tone_pred.where(max_prob >= 0.9, torch.tensor(-1, device=tone_pred.device))
        # Compute the reward
        reward = self.cal_reward(tone_pred, words, intos_list)
        # import pdb;pdb.set_trace()
        return reward

    def reward_func(self, wav_batch, labels, wav_masks=None):
        """Process a single batch for multithreaded execution."""
        return self.process_batch(wav_batch, wav_masks, labels)


register_reward_target_type('intonationAsrProb3c', 'text_with_into')
register_reward_norm_type('intonationAsrProb3c', 'max_division')