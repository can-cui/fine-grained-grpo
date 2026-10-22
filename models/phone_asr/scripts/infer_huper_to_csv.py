from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from infer_huper import configure_offline_env, infer_one
import torch
from transformers import AutoModelForCTC, AutoProcessor


def load_texts(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "id" not in reader.fieldnames:
            raise ValueError(f"metadata csv must contain an id column: {path}")
        text_key = "text" if "text" in reader.fieldnames else None
        if text_key is None:
            return {}
        return {row["id"].strip(): row[text_key].strip() for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metadata-csv", default=None, help="Optional CSV with id,text columns.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    configure_offline_env(root)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else root / "model" / "huper_recognizer"
    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(args.metadata_csv).resolve() if args.metadata_csv else root / "metadata" / "huper_label0_50_texts.csv"
    texts = load_texts(metadata_csv)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)
    model.eval()

    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files found: {input_dir}")

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "native_output"])
        writer.writeheader()
        for wav_path in wavs:
            wav_id = wav_path.stem
            native_output = infer_one(processor, model, device, wav_path)
            writer.writerow(
                {
                    "id": wav_id,
                    "text": texts.get(wav_id, ""),
                    "native_output": native_output,
                }
            )
            print(f"{wav_id}\t{native_output}")

    print(f"wrote: {output_csv}")


if __name__ == "__main__":
    main()
