#!/usr/bin/env python3
"""导出或验证 V4 在线特征编码器使用的 wav2vec TorchScript 资产。"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--fairseq-project-root",
        default="",
        help="原 wav2vec 工程根目录；其中必须包含 fairseq 包",
    )
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--max-error", type=float, default=1e-4)
    parser.add_argument("--trace-device", choices=("cuda",), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def extract_conv_config(model):
    kernels = []
    strides = []
    for index, block in enumerate(model.feature_extractor.conv_layers):
        conv = getattr(block, "conv", None)
        if conv is None:
            raise RuntimeError("feature_extractor.conv_layers[{}] 没有 conv".format(index))
        kernel = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        kernels.append(int(kernel))
        strides.append(int(stride))
    if not kernels or len(kernels) != len(strides):
        raise RuntimeError("无法读取 wav2vec 卷积 kernel/stride")
    return kernels, strides


def validate_asset(model_path, meta_path, expected_sample_rate, expected_feature_dim):
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("TorchScript 或元数据不存在：{} / {}".format(model_path, meta_path))
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "sample_rate", "feature_dim", "conv_kernels", "conv_strides",
        "source_checkpoint_sha256", "torchscript_sha256", "trace_device",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError("wav2vec 元数据缺少字段：{}".format(missing))
    if int(metadata.get("format_version", 0)) != 3:
        raise ValueError("wav2vec metadata format_version must be 3")
    if int(metadata["sample_rate"]) != expected_sample_rate:
        raise ValueError("wav2vec sample_rate 不匹配")
    if int(metadata["feature_dim"]) != expected_feature_dim:
        raise ValueError("wav2vec feature_dim 不匹配")
    if len(metadata["conv_kernels"]) != len(metadata["conv_strides"]):
        raise ValueError("wav2vec 卷积元数据长度不一致")
    actual_hash = sha256_file(model_path)
    if actual_hash != metadata["torchscript_sha256"]:
        raise ValueError("TorchScript SHA256 与元数据不一致")

    trace_device = str(metadata["trace_device"])
    if trace_device != "cuda":
        raise ValueError("V1 TorchScript must be traced for CUDA runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to validate the V4 TorchScript asset")
    device = torch.device(trace_device)
    # TorchScript trace captures a few device-bearing constants used while
    # constructing wav2vec's length mask and Transformer attention mask.
    # Loading on CPU and subsequently calling ``.to(cuda)`` leaves those
    # constants on CPU, which fails inside attention.  Load directly on the
    # trace device so every tensor in the scripted graph is CUDA-resident.
    model = torch.jit.load(str(model_path), map_location=device).eval()
    previous_length = -1
    with torch.no_grad():
        for sample_count in (8000, 16000, 32000):
            waveform = torch.randn(1, sample_count, device=device)
            lengths = torch.tensor([sample_count], dtype=torch.long, device=device)
            features, output_lengths = model(waveform, lengths)
            if features.dim() != 3 or features.size(0) != 1 or features.size(2) != expected_feature_dim:
                raise RuntimeError("TorchScript 输出 shape 非法：{}".format(tuple(features.shape)))
            current_length = int(output_lengths[0].item())
            if current_length <= previous_length or current_length > features.size(1):
                raise RuntimeError("TorchScript 变长输出非法：{}".format(output_lengths.tolist()))
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("TorchScript 输出包含 NaN/Inf")
            previous_length = current_length
    print(json.dumps({"ok": True, "model": str(model_path), "metadata": metadata}, ensure_ascii=False))


class LastLayerWav2Vec(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform, waveform_lengths):
        features, output_lengths = self.model.extract_features(waveform, waveform_lengths)
        return features[-1], output_lengths


def export_asset(args):
    if not args.checkpoint:
        raise ValueError("导出模式必须提供 --checkpoint")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError("找不到 wav2vec checkpoint：{}".format(checkpoint))

    # 本脚本位于 V1/scripts，从绝对路径启动时，Python 不会自动把 shell 的
    # 当前工作目录放到 sys.path 首位，所以必须显式加入原 Fairseq 工程。
    fairseq_project_root = (
        Path(args.fairseq_project_root)
        if args.fairseq_project_root
        else Path.cwd()
    ).resolve()
    if not (fairseq_project_root / "fairseq").is_dir():
        raise FileNotFoundError(
            "原 Fairseq 工程目录中找不到 fairseq 包：{}；"
            "请传入 --fairseq-project-root".format(fairseq_project_root)
        )
    project_root_string = str(fairseq_project_root)
    if project_root_string not in sys.path:
        sys.path.insert(0, project_root_string)

    import fairseq
    import torchaudio
    from torchaudio.models.wav2vec2.utils import import_fairseq_model

    fairseq_file = Path(fairseq.__file__).resolve()
    try:
        fairseq_file.relative_to(fairseq_project_root)
    except ValueError:
        raise RuntimeError(
            "加载到了错误的 Fairseq：{}；预期位于 {}".format(
                fairseq_file, fairseq_project_root
            )
        )

    torch.manual_seed(20260807)
    models, cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
    if len(models) != 1:
        raise RuntimeError("checkpoint 中的模型数量不是 1：{}".format(len(models)))
    trace_device = torch.device(args.trace_device)
    if trace_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export the V4 TorchScript asset")
    converted = import_fairseq_model(models[0].eval()).to(trace_device).eval()
    kernels, strides = extract_conv_config(converted)

    check_waveform = torch.randn(1, args.sample_rate, device=trace_device)
    check_lengths = torch.tensor(
        [args.sample_rate], dtype=torch.long, device=trace_device
    )
    with torch.no_grad():
        before_layers, before_lengths = converted.extract_features(check_waveform, check_lengths)
        before = before_layers[-1].clone()

    removed = []
    for name, module in converted.named_modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
            removed.append(name)
        except ValueError:
            pass
    remaining = [
        name
        for name, module in converted.named_modules()
        for hook in module._forward_pre_hooks.values()
        if type(hook).__name__ == "WeightNorm"
    ]
    if remaining:
        raise RuntimeError("仍存在 WeightNorm hook：{}".format(remaining))
    with torch.no_grad():
        after_layers, after_lengths = converted.extract_features(check_waveform, check_lengths)
        after = after_layers[-1]
    error = float((before - after).abs().max().item())
    if error > args.max_error or not torch.equal(before_lengths, after_lengths):
        raise RuntimeError("移除 weight_norm 后输出不等价：error={}".format(error))
    if after.dim() != 3 or after.size(2) != args.feature_dim:
        raise RuntimeError("wav2vec 特征维度非法：{}".format(tuple(after.shape)))

    wrapper = LastLayerWav2Vec(converted).to(trace_device).eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (check_waveform, check_lengths),
            strict=False,
            check_trace=False,
        )
    output_model = Path(args.output_model)
    output_meta = Path(args.output_meta)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output_model.with_suffix(output_model.suffix + ".tmp")
    traced.save(str(temporary_model))
    os.replace(str(temporary_model), str(output_model))

    metadata = {
        "format_version": 3,
        "trace_device": args.trace_device,
        "sample_rate": args.sample_rate,
        "feature_dim": args.feature_dim,
        "conv_kernels": kernels,
        "conv_strides": strides,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "torchscript_sha256": sha256_file(output_model),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "fairseq_path": fairseq.__file__,
        "checkpoint_arch": getattr(getattr(cfg, "model", None), "_name", None),
        "removed_weight_norm_modules": removed,
        "weight_norm_max_abs_error": error,
    }
    temporary_meta = output_meta.with_suffix(output_meta.suffix + ".tmp")
    with temporary_meta.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary_meta), str(output_meta))
    validate_asset(output_model, output_meta, args.sample_rate, args.feature_dim)


def main():
    args = parse_args()
    if args.validate_only:
        validate_asset(args.output_model, args.output_meta, args.sample_rate, args.feature_dim)
    else:
        export_asset(args)


if __name__ == "__main__":
    main()
