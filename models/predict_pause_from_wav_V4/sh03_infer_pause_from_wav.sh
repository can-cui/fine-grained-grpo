#!/bin/bash
# ============================================================================
# Audio pause prediction V4: run unified inference on three valid/test groups using checkpoint_best.
# Run directly: bash sh03_infer_pause_from_wav.sh
# Important: the threshold is fixed at 0.5; this script does not select a threshold based on test data.
# ============================================================================

# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
CHECKPOINT_NAME="checkpoint_best.pt"
THRESHOLD="0.50"
BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail
source "$CONDA_ACTIVATE" "$CONDA_ENV"

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"
CHECKPOINT="$EXPERIMENT_ROOT/checkpoints/$CHECKPOINT_NAME"
OUTPUT_ROOT="$EXPERIMENT_ROOT/inference/${CHECKPOINT_NAME%.pt}_threshold_${THRESHOLD}"
SPLITS=(
    valid_m2_manual
    valid_m2_mfa
    valid_lmdb_mfa
    test_m2_manual
    test_m2_mfa
    test_lmdb_mfa
)

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    echo "Error: wav2vec TorchScript assets from this experiment are missing. Run sh02 first." >&2
    exit 1
fi
for split in "${SPLITS[@]}"; do
    if [[ ! -d "$DATA_ROOT/$split" || ! -f "$DATA_ROOT/$split.key" ]]; then
        echo "Error: official evaluation split is missing: $DATA_ROOT/$split" >&2
        exit 1
    fi
done

# Step 1: Output probabilities and fixed-threshold metrics for each of the six fixed splits.
echo "[1/1] Run inference on the six official splits using $CHECKPOINT_NAME"
mkdir -p "$OUTPUT_ROOT"
cd "$CODE_ROOT"
for split in "${SPLITS[@]}"; do
    split_output="$OUTPUT_ROOT/$split"
    mkdir -p "$split_output"
    echo "Running inference on split=$split"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
        python "$INFER_SCRIPT" \
        --user-dir "$USER_DIR" \
        --checkpoint "$CHECKPOINT" \
        --data-root "$DATA_ROOT" \
        --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
        --wav2vec-meta-path "$WAV2VEC_META_PATH" \
        --split "$split" \
        --output-dir "$split_output" \
        --threshold "$THRESHOLD" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"
done

echo "Done. Inference results for all six splits: $OUTPUT_ROOT"
echo "Next: run bash sh06_scan_pause_thresholds.sh."
