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

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_single.sh /path/to/audio.wav [cuda|cpu]" >&2
  exit 2
fi

DEVICE="${2:-cuda}"
python "$ROOT/scripts/infer_huper.py" --root "$ROOT" --input "$1" --device "$DEVICE"
