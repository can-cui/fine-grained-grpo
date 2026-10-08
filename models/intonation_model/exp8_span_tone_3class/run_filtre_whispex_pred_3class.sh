#!/bin/bash
# run_3class.sh — txt + wav -> pred (三阈值带拒识, 纯推理)
#
# 用法:
# source /train29/tts/permanent/jyxu24/envs/whisperx_venv/bin/activate
#   sh run_3class.sh <text_path> <wav_dir>
#   sh run_3class.sh <text_path> <wav_dir> <output_txt>
#   sh run_3class.sh <text_path> <wav_dir> <output_txt> <rising_threshold> <falling_threshold> <flat_threshold>
#
# text_path: 每行 "id<TAB>sentence"
# wav_dir:   id 对应 <wav_dir>/<id>.wav
# whisperX 在推理时对 wav 做 forced alignment 拿逐词时间。
# 灰区(三个阈值都不过)的词直接丢掉, 不写中间文件。
set -e

TEXT_PATH="$1"
#/train29/tts/permanent/jyxu24/golden_test2/golden_test2.txt
WAV_DIR="$2"
#/yrfs5/tts/sjliu18/03_data_tools/tone_annotation/PyToBi/PyToBI-master/praatScripts/data_ksy/female
OUTPUT_TXT="${3:-pred.txt}"
#output.txt
RISING_THRESHOLD="${4:-0.98}"
FALLING_THRESHOLD="${5:-0.98}"
FLAT_THRESHOLD="${6:-0.97}"

if [ -z "$TEXT_PATH" ] || [ -z "$WAV_DIR" ]; then
  echo "用法: sh run_3class.sh <text_path> <wav_dir> [output_txt] [rising_threshold] [falling_threshold] [flat_threshold]"
  exit 1
fi

# ---- 训练产物路径 (按需修改) ----
CKPT="/train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/checkpoints/best_exp8_span_tone_3class0625.pt"
VOCAB="/train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/vocab.json"
STATS="/train29/tts/permanent/jyxu24/intonation_v4_0520_exp8_span_tone_3class_0622/stats.npz"

# fbank 缓存目录 (临时, 存到当前运行环境路径下的 fbk_tmp/, 同一 wav 第二次跑可复用)
FBK_DIR="$(pwd)/fbk_tmp"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/infer_threshold_3class.py" \
  --text-path          "$TEXT_PATH" \
  --wav-dir            "$WAV_DIR" \
  --fbk-dir            "$FBK_DIR" \
  --ckpt               "$CKPT" \
  --vocab              "$VOCAB" \
  --stats              "$STATS" \
  --output-txt         "$OUTPUT_TXT" \
  --rising-threshold   "$RISING_THRESHOLD" \
  --falling-threshold  "$FALLING_THRESHOLD" \
  --flat-threshold     "$FLAT_THRESHOLD"

echo "[run_3class.sh] done -> $OUTPUT_TXT"
