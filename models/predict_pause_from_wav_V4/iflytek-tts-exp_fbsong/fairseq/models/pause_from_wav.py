"""Online wav2vec with only its last two Transformer blocks trainable."""

import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from fairseq.models import BaseFairseqModel, register_model, register_model_architecture


@register_model("pause_from_wav")
class PauseFromWavModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = args.wav2vec_feature_dim
        self.input_sample_rate = args.input_sample_rate
        metadata_path = Path(args.wav2vec_meta_path)
        model_path = Path(args.wav2vec_jit_path)
        if not metadata_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                "wav2vec TorchScript asset is missing: {} / {}".format(model_path, metadata_path)
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if int(metadata["sample_rate"]) != self.input_sample_rate:
            raise ValueError("wav2vec metadata sample rate does not match model input")
        if int(metadata["feature_dim"]) != self.feature_dim:
            raise ValueError("wav2vec metadata feature dimension does not match model")
        if metadata.get("format_version") != 3 or metadata.get("trace_device") != "cuda":
            raise ValueError(
                "V4 requires format-v3 CUDA-traced wav2vec TorchScript asset; "
                "run the provided training or smoke-test shell to re-export it"
            )
        kernels = [int(value) for value in metadata["conv_kernels"]]
        strides = [int(value) for value in metadata["conv_strides"]]
        if not kernels or len(kernels) != len(strides):
            raise ValueError("invalid wav2vec convolution metadata")
        self.conv_kernels = kernels
        self.conv_strides = strides
        # Do not load to CPU and move later: device constants embedded by
        # torch.jit.trace would keep the attention mask on CPU.  The exported
        # V4 asset is deliberately CUDA-only and must be loaded on CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError("V4 online wav2vec requires a CUDA device")
        self.wav2vec = torch.jit.load(str(model_path), map_location="cuda")
        self.wav2vec.eval()
        for parameter in self.wav2vec.parameters():
            parameter.requires_grad = False
        self.pause_head = nn.Linear(self.feature_dim, 1)
        self.unfrozen_wav2vec_layers = self.configure_unfrozen_wav2vec_layers(
            args.unfreeze_last_n
        )
        for parameter in self.pause_head.parameters():
            parameter._pause_lr_group = "head"

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wav2vec-feature-dim", type=int, default=768)
        parser.add_argument("--input-sample-rate", type=int, default=16000)
        parser.add_argument("--wav2vec-jit-path", required=True)
        parser.add_argument("--wav2vec-meta-path", required=True)
        parser.add_argument("--unfreeze-last-n", type=int, default=2)

    @classmethod
    def build_model(cls, args, task):
        pause_from_wav_w2v_unfreeze_last2(args)
        return cls(args)

    def configure_unfrozen_wav2vec_layers(self, last_n):
        if last_n <= 0:
            raise ValueError("unfreeze-last-n 必须大于 0")
        pattern = re.compile(r"(?:^|\.)encoder\.transformer\.layers\.(\d+)\.")
        indexed = []
        for name, parameter in self.wav2vec.named_parameters():
            match = pattern.search(name)
            if match:
                indexed.append((int(match.group(1)), name, parameter))
        layer_indices = sorted({index for index, _name, _parameter in indexed})
        if len(layer_indices) < last_n:
            examples = [name for name, _parameter in list(self.wav2vec.named_parameters())[:20]]
            raise RuntimeError(
                "TorchScript 中无法定位足够的 encoder.transformer.layers；"
                "检测到层={}，参数名前20项={}".format(layer_indices, examples)
            )
        selected = layer_indices[-last_n:]
        trainable_names = []
        for index, name, parameter in indexed:
            if index in selected:
                parameter.requires_grad = True
                parameter._pause_lr_group = "wav2vec"
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError("未找到需要解冻的 wav2vec 参数")
        print(json.dumps({
            "wav2vec_unfrozen_layer_indices": selected,
            "wav2vec_unfrozen_parameter_tensors": len(trainable_names),
            "wav2vec_unfrozen_parameters": sum(
                parameter.numel() for parameter in self.wav2vec.parameters()
                if parameter.requires_grad
            ),
            "wav2vec_unfrozen_parameter_names": trainable_names,
        }, ensure_ascii=False))
        return selected

    def train(self, mode=True):
        super().train(mode)
        self.wav2vec.eval()
        return self

    def samples_to_frames(self, sample_lengths):
        frame_lengths = sample_lengths.long()
        for kernel, stride in zip(self.conv_kernels, self.conv_strides):
            frame_lengths = torch.div(frame_lengths - kernel, stride, rounding_mode="floor") + 1
            frame_lengths = frame_lengths.clamp_min(0)
        return frame_lengths

    @staticmethod
    def mean_pool_words(features, pool_start_frame, pool_end_frame):
        if features.dim() != 3:
            raise ValueError("features must have shape [batch, frames, channels]")
        if pool_start_frame.shape != pool_end_frame.shape:
            raise ValueError("pool start/end tensors must have the same shape")
        if pool_start_frame.dim() != 2:
            raise ValueError("pool ranges must have shape [batch, words]")
        batch, frames, channels = features.shape
        if pool_start_frame.size(0) != batch:
            raise ValueError("pool range batch size does not match features")
        if torch.any(pool_start_frame < 0) or torch.any(pool_end_frame > frames):
            raise ValueError("pool range is outside the padded feature tensor")
        if torch.any(pool_end_frame <= pool_start_frame):
            raise ValueError("every pool range must contain at least one frame")

        prefix = torch.cat(
            [features.new_zeros(batch, 1, channels), features.cumsum(dim=1)], dim=1
        )
        gather_shape = (-1, -1, channels)
        starts = pool_start_frame.unsqueeze(-1).expand(*gather_shape)
        ends = pool_end_frame.unsqueeze(-1).expand(*gather_shape)
        sums = torch.gather(prefix, 1, ends) - torch.gather(prefix, 1, starts)
        lengths = (pool_end_frame - pool_start_frame).unsqueeze(-1).to(features.dtype)
        return sums / lengths

    @staticmethod
    def normalize_word_frame_windows(
        starts, ends, word_lengths, feature_lengths, maximum_frame
    ):
        """把词窗口限制在逐句有效帧内，并为极短真实词保留一个声学帧。"""
        if maximum_frame <= 0:
            raise ValueError("wav2vec returned no feature frames")
        if starts.shape != ends.shape or starts.dim() != 2:
            raise ValueError("word frame windows must have matching [batch, words] shapes")
        if word_lengths.dim() != 1 or feature_lengths.dim() != 1:
            raise ValueError("word_lengths and feature_lengths must be one-dimensional")
        if starts.size(0) != word_lengths.numel() or starts.size(0) != feature_lengths.numel():
            raise ValueError("word frame windows and length tensors have different batch sizes")

        word_positions = torch.arange(starts.size(1), device=starts.device).unsqueeze(0)
        real_words = word_positions < word_lengths.unsqueeze(1)
        per_sample_maximum = feature_lengths.long().clamp(min=1, max=maximum_frame).unsqueeze(1)

        starts = torch.minimum(starts.clamp_min(0), per_sample_maximum - 1)
        ends = torch.minimum(ends.clamp_min(1), per_sample_maximum)
        starts = torch.where(real_words, starts, torch.zeros_like(starts))
        ends = torch.where(real_words, ends, torch.ones_like(ends))

        # 小于 wav2vec 帧移的合法词窗口可能在卷积降采样后映射为同一边界。
        # 这种窗口使用其起点所在的一个有效帧，避免整批训练因单条极短词中断。
        empty_real_windows = real_words & (ends <= starts)
        ends = torch.where(empty_real_windows, starts + 1, ends)
        return starts, ends

    def forward(
        self,
        waveform,
        waveform_lengths,
        pool_start_sample,
        pool_end_sample,
        word_lengths,
        **unused
    ):
        self.wav2vec.eval()
        # 该资产在 CUDA 上 trace，TorchScript 内部的卷积长度计算及 attention
        # mask 都与声学特征位于同一 CUDA 设备。因此 Fairseq 移动后的 waveform
        # 与 waveform_lengths 均直接输入；不得单独将 lengths 移回 CPU。
        features, feature_lengths = self.wav2vec(waveform, waveform_lengths)
        if features.size(-1) != self.feature_dim:
            raise ValueError("wav2vec output feature dimension changed")
        expected_lengths = self.samples_to_frames(waveform_lengths)
        if not torch.equal(expected_lengths, feature_lengths.long()):
            raise ValueError("wav2vec output lengths disagree with convolution metadata")

        starts = self.samples_to_frames(pool_start_sample)
        ends = self.samples_to_frames(pool_end_sample)
        starts, ends = self.normalize_word_frame_windows(
            starts,
            ends,
            word_lengths,
            feature_lengths,
            features.size(1),
        )

        word_features = self.mean_pool_words(features, starts, ends)
        return {
            "pause_logits": self.pause_head(word_features).squeeze(-1),
            "word_features": word_features,
            "feature_lengths": feature_lengths,
            "pool_start_frame": starts,
            "pool_end_frame": ends,
        }


@register_model_architecture("pause_from_wav", "pause_from_wav_w2v_unfreeze_last2")
def pause_from_wav_w2v_unfreeze_last2(args):
    args.wav2vec_feature_dim = getattr(args, "wav2vec_feature_dim", 768)
    args.input_sample_rate = getattr(args, "input_sample_rate", 16000)
    args.unfreeze_last_n = getattr(args, "unfreeze_last_n", 2)


@register_model_architecture("pause_from_wav", "pause_from_wav_linear")
def pause_from_wav_linear(args):
    """保留旧 arch 名称，仅用于读取早期配置；数据接口仍为在线 waveform。"""
    pause_from_wav_w2v_unfreeze_last2(args)
