from __future__ import annotations

import argparse
from pathlib import Path

import torch
import sys

# sys.path.insert(
#     0,
#     "/home4/intern/yhchen70/miniconda3/envs/phoneme_huper/lib/python3.10/site-packages"
# )
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

from transformers import AutoModelForCTC, AutoProcessor

from infer_huper import configure_offline_env, infer_one


def read_text_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines]


def clean_cell(value: str) -> str:
    return " ".join(value.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def clean_phone_output(value: str) -> str:
    # Remove the CTC word boundary marker used by this model's label inventory.
    return clean_cell(value.replace("|", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--wav-dir", required=True, help="Directory containing .wav files.")
    parser.add_argument("--text-txt", required=True, help="UTF-8 text file; one line per wav after filename sort.")
    parser.add_argument("--output-txt", required=True, help="Output TSV text file: id<TAB>text<TAB>phoneme_output.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--header", action="store_true", help="Write a header row.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    configure_offline_env(root)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else root / "model" / "huper_recognizer"
    wav_dir = Path(args.wav_dir).resolve()
    text_txt = Path(args.text_txt).resolve()
    output_txt = Path(args.output_txt).resolve()
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files found: {wav_dir}")
    texts = read_text_lines(text_txt)
    if len(texts) != len(wavs):
        raise SystemExit(
            f"Line count mismatch: {text_txt} has {len(texts)} lines, but {wav_dir} has {len(wavs)} wav files. "
            "The text file must contain exactly one line per wav after sorting wav filenames."
        )

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()

    with output_txt.open("w", encoding="utf-8", newline="") as handle:
        if args.header:
            handle.write("id\ttext\tasr_phoneme_output\n")
        for wav_path, text in zip(wavs, texts):
            wav_id = wav_path.stem
            native_output = clean_phone_output(infer_one(processor, model, device, wav_path))
            handle.write(f"{clean_cell(wav_id)}\t{clean_cell(text)}\t{native_output}\n")
            print(f"{wav_id}\t{native_output}")

    print(f"wrote: {output_txt}")


if __name__ == "__main__":
    main()
