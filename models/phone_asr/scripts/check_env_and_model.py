from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCTC, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "huper_recognizer"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

print("root:", ROOT)
print("model_dir:", MODEL_DIR)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("gpu 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForCTC.from_pretrained(str(MODEL_DIR), local_files_only=True)
print("model loaded: ok")
print("vocab size:", len(processor.tokenizer.get_vocab()))
print("id2label size:", len(model.config.id2label))
