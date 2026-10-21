#!/bin/bash
set -euo pipefail

# ==================== Configuration ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
OUTPUT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2/checkpoint_sentence_accuracy"
THRESHOLD="0.50"
BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"  # Set to an available GPU; do not share a GPU used by local training.
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ===============================================

set +u
source "$CONDA_ACTIVATE" "$CONDA_ENV"
set -u

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
EVALUATE_SCRIPT="$PACKAGE_ROOT/scripts/evaluate_pause_sentence_accuracy_checkpoints.py"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"

[[ -d "$CHECKPOINT_ROOT" ]] || { echo "Error: CHECKPOINT_ROOT not found: $CHECKPOINT_ROOT" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "Error: DATA_ROOT not found: $DATA_ROOT" >&2; exit 2; }
[[ -f "$EVALUATE_SCRIPT" ]] || { echo "Error: evaluation script not found: $EVALUATE_SCRIPT" >&2; exit 2; }
[[ -s "$WAV2VEC_JIT_PATH" && -s "$WAV2VEC_META_PATH" ]] || { echo "Error: wav2vec assets are missing" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EVALUATE_SCRIPT" \
    --user-dir "$USER_DIR" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --data-root "$DATA_ROOT" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"

echo "Done: $OUTPUT_ROOT"
