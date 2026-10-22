#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$ROOT/hf_cache/transformers"
export TORCH_HOME="$ROOT/torch_cache"
export TMPDIR="$ROOT/temp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

INPUT_DIR="${1:-$ROOT/input_wavs}"
OUTPUT_DIR="${2:-$ROOT/outputs/huper_recognizer}"
DEVICE="${3:-cuda}"

python "$ROOT/scripts/infer_huper.py" --root "$ROOT" --input "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --device "$DEVICE"
