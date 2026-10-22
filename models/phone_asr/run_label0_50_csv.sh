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
export LD_LIBRARY_PATH=/opt/lib/gcc-11.4.0/lib64:$LD_LIBRARY_PATH
if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_label0_50_csv.sh <wav_dir> <text_txt> <output_txt> [cuda|cpu]

Example:
  bash run_label0_50_csv.sh \
    input_wavs/huper_label0_50_renumbered \
    input_wavs/huper_label0_50_renumbered.txt \
    reports/huper_label0_50_native_outputs_simple.txt \
    cuda

Output columns are tab-separated:
  id<TAB>text<TAB>asr_phoneme_output

The text txt must be UTF-8 and contain exactly one line per wav after sorting wav filenames.
EOF
  exit 2
fi

WAV_DIR="$1"
TEXT_TXT="$2"
OUTPUT_TXT="$3"
DEVICE="${4:-cuda}"

python "$ROOT/scripts/infer_huper_to_tsv.py" \
  --root "$ROOT" \
  --wav-dir "$WAV_DIR" \
  --text-txt "$TEXT_TXT" \
  --output-txt "$OUTPUT_TXT" \
  --device "$DEVICE"
