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
import json

# WhisperX模型路径
WHISPERX_DIR = "/train34/tts/permanent/cancui11/RL/models/whisperX-main/whisperX-main"
WHISPERX_MAIN_DIR = "/train34/tts/permanent/cancui11/RL/models/whisperX-main"

# 停顿预测模型路径
PAUSE_PACKAGE_ROOT = "/train34/tts/permanent/cancui11/RL/models/predict_pause_from_wav_V4"
PAUSE_CHECKPOINT = "/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2/checkpoints/checkpoint_last.pt"
WAV2VEC_JIT_PATH = "/train34/tts/permanent/cancui11/RL/models/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH = "/train34/tts/permanent/cancui11/RL/models/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.meta.json"
PAUSE_USER_DIR = "/train34/tts/permanent/cancui11/RL/models/predict_pause_from_wav_V4/pause_user_dir"

# 模型配置
MAX_FRAMES = 600
MAX_WORDS = 80
SAMPLE_RATE = 16000
PAUSE_THRESHOLD = 0.5

# 添加路径
sys.path.insert(0, WHISPERX_DIR)
sys.path.insert(0, WHISPERX_MAIN_DIR)
import whisperx


def _normalize_word(word):
    """标准化词（去标点、小写），用于模糊匹配。"""
    if word == "…":
        return "…"
    return re.sub(r'[^\w]', '', word.lower())


def _merge_whisperx_to_user_words(user_words, whisperx_segments):
    """将 WhisperX 的细粒度分词映射到用户的粗粒度分词。

    Args:
        user_words: 用户的词序列
        whisperx_segments: WhisperX 输出

    Returns:
        word_boundaries: list of [start_cs, end_cs]，长度 = len(user_words)
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


def normalize_text(text):
    return "".join(character.lower() for character in text if character.isalnum())


def text_match_ratio(text, words):
    reference = normalize_text(text)
    hypothesis = normalize_text("".join(item["word"] for item in words))
    if not reference and not hypothesis:
        return 1.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, reference, hypothesis).ratio()


def load_audio(path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("WAV 为空：{}".format(path))
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    waveform = waveform.squeeze(0).contiguous()
    if not torch.isfinite(waveform).all():
        raise ValueError("WAV 包含非有限采样值：{}".format(path))
    return waveform


def build_sample(row, whisper_record, waveform, sample_rate):
    """构建停顿模型的输入样本

    Args:
        row: 包含id和text的字典
        whisper_record: 包含words列表的字典，每个word有word/start/end
        waveform: audio waveform tensor (1D)
        sample_rate: 采样率
    """
    words = whisper_record["words"]
    duration = waveform.numel() / float(sample_rate)
    if words[-1]["end"] > duration + 0.25:
        raise ValueError(
            "{}: Whisper 最后结束时间 {:.3f}s 超出 WAV 时长 {:.3f}s".format(
                row["id"], words[-1]["end"], duration
            )
        )
    # pool_starts: 第一个词的start + 前面每个词的end
    pool_starts_seconds = [words[0]["start"]] + [item["end"] for item in words[:-1]]
    pool_ends_seconds = [item["end"] for item in words]

    # 转换为采样点
    starts = torch.tensor(
        [round(min(value, duration) * sample_rate) for value in pool_starts_seconds],
        dtype=torch.long,
    )
    ends = torch.tensor(
        [round(min(value, duration) * sample_rate) for value in pool_ends_seconds],
        dtype=torch.long,
    )
    if torch.any(ends <= starts):
        raise ValueError("{}: 转成采样点后出现空词窗口".format(row["id"]))

    return {
        "id": row["id"],
        "text": row["text"],
        "waveform": waveform,
        "words": words,
        "pool_start_seconds": pool_starts_seconds,
        "pool_end_seconds": pool_ends_seconds,
        "pool_start_sample": starts,
        "pool_end_sample": ends,
        "text_match_ratio": text_match_ratio(row["text"], words),
    }


def collate(samples):
    """将样本collate成batch"""
    batch_size = len(samples)
    max_audio = max(sample["waveform"].numel() for sample in samples)
    max_words = max(len(sample["words"]) for sample in samples)
    waveforms = torch.zeros(batch_size, max_audio, dtype=torch.float32)
    starts = torch.zeros(batch_size, max_words, dtype=torch.long)
    ends = torch.ones(batch_size, max_words, dtype=torch.long)
    waveform_lengths = torch.zeros(batch_size, dtype=torch.long)
    word_lengths = torch.zeros(batch_size, dtype=torch.long)
    for index, sample in enumerate(samples):
        audio_count = sample["waveform"].numel()
        word_count = len(sample["words"])
        waveforms[index, :audio_count] = sample["waveform"]
        starts[index, :word_count] = sample["pool_start_sample"]
        ends[index, :word_count] = sample["pool_end_sample"]
        waveform_lengths[index] = audio_count
        word_lengths[index] = word_count
    return {
        "waveform": waveforms,
        "waveform_lengths": waveform_lengths,
        "pool_start_sample": starts,
        "pool_end_sample": ends,
        "word_lengths": word_lengths,
    }


@register_reward('pausePredWhisperToken')
class PausePredWhisperToken(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 加载WhisperX对齐模型
        self.align_model, self.align_metadata = whisperx.load_align_model(
            "en", "cuda", model_dir='/train34/tts/permanent/cancui11/RL/models/whisperX-main/checkpoints'
        )

        # 加载停顿预测模型
        from fairseq import checkpoint_utils, utils
        utils.import_user_module(type('obj', (object,), {'user_dir': PAUSE_USER_DIR})())

        models, _checkpoint_args, _task = checkpoint_utils.load_model_ensemble_and_task(
            [PAUSE_CHECKPOINT],
            arg_overrides={
                "wav2vec_jit_path": WAV2VEC_JIT_PATH,
                "wav2vec_meta_path": WAV2VEC_META_PATH,
            },
        )
        if len(models) != 1:
            raise ValueError("只能加载一个 checkpoint")
        self.model = models[0].cuda().eval()
        self.model.eval()

        self.max_frames = MAX_FRAMES
        self.max_words = MAX_WORDS
        self.sample_rate = SAMPLE_RATE
        self.threshold = PAUSE_THRESHOLD

    def whisperx_align_words(self, audio_16k, words):
        """用WhisperX对齐，获取词边界（厘秒）"""
        try:
            duration_s = len(audio_16k) / SAMPLE_RATE
            text_str = " ".join(words)
            transcript = [{"start": 0.0, "end": duration_s, "text": text_str}]

            result = whisperx.align(transcript, self.align_model, self.align_metadata,
                                    audio_16k, "cuda", return_char_alignments=False)

            if "word_segments" not in result or not result["word_segments"]:
                return None

            whisperx_segments = result["word_segments"]
            word_boundaries = _merge_whisperx_to_user_words(words, whisperx_segments)

            if word_boundaries is not None:
                for i in range(len(word_boundaries) - 1):
                    word_boundaries[i][1] = word_boundaries[i + 1][0]
            return word_boundaries
        except Exception as e:
            print(f"[warn] whisperx align failed: {e}")
            return None

    def cal_word_pause_predictions(self, waveform, words, word_boundaries):
        """用停顿模型预测每个词后面的停顿

        Args:
            waveform: audio waveform tensor (1D) at 16kHz
            words: 词列表
            word_boundaries: 词边界 [[start_cs, end_cs], ...]

        Returns:
            pause_predictions: list of int (0或1)，每个词后面是否有停顿
        """
        # 构建sample
        row = {"id": "0", "text": " ".join(words)}
        whisper_record = {"words": [{"word": w, "start": b[0]/100.0, "end": b[1]/100.0} for w, b in zip(words, word_boundaries)]}

        sample = build_sample(row, whisper_record, waveform, self.sample_rate)
        net_input = collate([sample])
        net_input = {name: value.cuda() for name, value in net_input.items()}

        with torch.no_grad():
            logits = self.model(**net_input)["pause_logits"]
            probs = torch.sigmoid(logits).cpu().squeeze(0)

        # 获取预测结果 (转为int 0/1)
        n_words = len(words)
        predictions = (probs[:n_words] >= self.threshold).long().tolist()

        # 最后一个词后面强制无停顿
        if n_words > 0:
            predictions[-1] = 0

        return predictions

    def cal_word_rewards(self, pred_pause, intos):
        """计算每个词级别的reward

        Args:
            pred_pause: 预测的停顿序列 list of int
            intos: 真实的语调标签

        Returns:
            word_rewards: list of float, 每个词一个reward
            sentence_reward: float, 句子级别reward
        """
        n_words = len(intos)
        word_rewards = [0.0] * n_words
        mismatch_count = 0

        for i, flag in enumerate(intos):
            if flag == -1:
                continue
            pred_i = pred_pause[i] if i < len(pred_pause) else 0
            gt_i = intos[i].item() if torch.is_tensor(intos[i]) else intos[i]
            if pred_i != gt_i:
                word_rewards[i] = -1.0
                mismatch_count += 1
            else:
                word_rewards[i] = 0.0

        sentence_reward = -float(mismatch_count)
        return word_rewards, sentence_reward

    def cal_reward(self, pred_pause, intos):
        """计算sentence-level reward（保持向后兼容）

        Args:
            pred_pause: 预测的停顿序列
            intos: 真实的语调标签

        Returns:
            reward: float
        """
        if not torch.is_tensor(pred_pause):
            pred_pause = torch.tensor(pred_pause)

        if not torch.is_tensor(intos):
            intos = torch.tensor(intos, device=pred_pause.device)

        valid_mask = (intos != -1)
        mismatch_count = (pred_pause[valid_mask] != intos[valid_mask]).sum().item()
        reward = -float(mismatch_count)
        return reward


    @torch.no_grad()
    def forward(self, wav_batch, labels, wav_masks=None, num_thread=1, infer_bsz=1):
        """批量推理

        Returns:
            rewards: tensor of shape [batch_size, 1], sentence-level reward
            word_rewards_list: list of list of float, word-level rewards per sample
            word_boundaries_list: list of list of [start_cs, end_cs], word boundaries per sample
        """
        if num_thread <= 1:
            return self._forward(wav_batch, labels, wav_masks, infer_bsz)

        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        results = []
        with ThreadPoolExecutor(max_workers=num_thread) as pool:
            for i in range(total_batch):
                results.append(pool.submit(
                    self.reward_func,
                    wav_batch[i * infer_bsz:(i + 1) * infer_bsz],
                    labels[i * infer_bsz:(i + 1) * infer_bsz],
                    wav_masks[i * infer_bsz:(i + 1) * infer_bsz] if wav_masks is not None else None
                ))
        results = [f.result() for f in results]

        rewards = []
        word_rewards_list = []
        word_boundaries_list = []
        for r in results:
            rewards.append(r["reward"])
            word_rewards_list.append(r["word_rewards"])
            word_boundaries_list.append(r["word_boundaries"])

        rewards_tensor = torch.tensor(rewards).unsqueeze(-1).to(wav_batch.device)
        return rewards_tensor, word_rewards_list, word_boundaries_list

    def _forward(self, wav_batch, labels, wav_masks=None, infer_bsz=1):
        """单线程批量推理"""
        results = []
        bsz = wav_batch.size(0)
        total_batch = bsz // infer_bsz
        if bsz % infer_bsz != 0:
            total_batch += 1

        for i in range(total_batch):
            wav = wav_batch[i * infer_bsz:(i + 1) * infer_bsz]
            wav_mask = wav_masks[i * infer_bsz:(i + 1) * infer_bsz] if wav_masks is not None else None
            label = labels[i]
            result = self.process_batch(wav, wav_mask, label)
            results.append(result)

        rewards = [r["reward"] for r in results]
        word_rewards_list = [r["word_rewards"] for r in results]
        word_boundaries_list = [r["word_boundaries"] for r in results]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1).to(wav_batch.device)
        return rewards_tensor, word_rewards_list, word_boundaries_list

    def process_batch(self, wav_batch, wav_masks, labels):
        """处理一个batch的样本，返回reward和word-level信息

        Returns:
            dict with keys: reward, word_rewards, word_boundaries
        """
        text, intos = labels

        # 获取有效音频 (24kHz)
        valid_wav = wav_batch[wav_masks.bool()] if wav_masks is not None else wav_batch
        audio_24k = valid_wav.detach().float().cpu()

        # 转换为16kHz用于WhisperX和停顿模型
        audio_16k_tensor = torchaudio.functional.resample(
            audio_24k.unsqueeze(0),
            orig_freq=24000, new_freq=16000
        ).squeeze(0)

        # WhisperX对齐获取词边界
        audio_16k_np = audio_16k_tensor.numpy()
        text=text.replace('-','')
        words = text.split()[:self.max_words]
        word_boundaries = self.whisperx_align_words(audio_16k_np, words)

        if word_boundaries is None:
            # 如果对齐失败，使用均匀分布的word_boundaries
            word_boundaries = [[i * 100, (i + 1) * 100] for i in range(len(words))]

        # 直接用tensor调用停顿模型预测每个词后面的停顿
        pause_predictions = self.cal_word_pause_predictions(audio_16k_tensor, words, word_boundaries)

        # 计算word-level rewards
        word_rewards, sentence_reward = self.cal_word_rewards(pause_predictions, intos)
        # if -1 in word_rewards:
        # import pdb;pdb.set_trace()
        return {
            "reward": sentence_reward,
            "word_rewards": word_rewards,
            "word_boundaries": word_boundaries,
        }

    def reward_func(self, wav_batch, labels, wav_masks=None):
        """单batch处理，用于多线程调用"""
        return self.process_batch(wav_batch, wav_masks, labels)


register_reward_target_type('pausePredWhisperToken', 'text_with_into')
register_reward_norm_type('pausePredWhisperToken', 'max_division')