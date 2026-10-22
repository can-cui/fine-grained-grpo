from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor


TARGET_SAMPLE_RATE = 16000


def configure_offline_env(root: Path) -> None:
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hf_cache" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf_cache" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch_cache"))
    os.environ.setdefault("TMPDIR", str(root / "temp"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for path in [
        root / "hf_cache",
        root / "hf_cache" / "hub",
        root / "hf_cache" / "transformers",
        root / "torch_cache",
        root / "temp",
        root / "outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_audio_16k_mono(path: Path):
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sample_rate


def decode_manual_labels(model, predicted_ids) -> str:
    ids = predicted_ids[0].detach().cpu().tolist()
    blank_id = getattr(model.config, "pad_token_id", None)
    id2label = getattr(model.config, "id2label", {})
    labels = []
    previous = None
    for item in ids:
        if blank_id is not None and item == blank_id:
            previous = item
            continue
        if item == previous:
            continue
        label = id2label.get(item, str(item))
        if label not in {"<pad>", "[PAD]", "<s>", "</s>", "<unk>"}:
            labels.append(label)
        previous = item
    return " ".join(labels).strip()


def infer_one(processor, model, device, wav_path: Path) -> str:
    audio, sample_rate = load_audio_16k_mono(wav_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    return decode_manual_labels(model, predicted_ids)


def iter_wavs(path: Path):
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--input", required=True, help="A .wav file or a directory containing .wav files.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    configure_offline_env(root)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else root / "model" / "huper_recognizer"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "outputs" / "huper_recognizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()

    wavs = list(iter_wavs(Path(args.input).resolve()))
    if not wavs:
        raise SystemExit(f"No .wav files found: {args.input}")
    print(f"model_dir={model_dir}")
    print(f"device={device}")
    print(f"wav_count={len(wavs)}")
    for wav_path in wavs:
        text = infer_one(processor, model, device, wav_path)
        out_path = output_dir / f"{wav_path.stem}.txt"
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"{wav_path.name}\t{text}")


if __name__ == "__main__":
    main()
