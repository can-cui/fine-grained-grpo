#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 修改区 =====
wav_dir="/train34/tts/permanent/cancui11/RL/models/phoneme_asr/input_wavs/test50"
text_txt="/train34/tts/permanent/cancui11/RL/models/phoneme_asr/input_txt/test50.txt"
output_txt="/train34/tts/permanent/cancui11/RL/models/phoneme_asr/test50_asr_result.txt"
device="cuda"
# ======================================================================
if [[ $# -eq 0 ]]; then
  exec bash "$ROOT/run_label0_50_csv.sh" \
    "$wav_dir" \
    "$text_txt" \
    "$output_txt" \
    "$device"
fi

exec bash "$ROOT/run_label0_50_csv.sh" "$@"
