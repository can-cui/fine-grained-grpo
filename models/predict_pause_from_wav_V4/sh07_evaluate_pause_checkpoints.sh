#!/bin/bash
# ============================================================================
# Audio pause prediction V4: evaluate trends across the three valid/test groups for epochs 2-30.
#
# Main steps:
#   1. Load checkpoints for even-numbered epochs from checkpoint2 through checkpoint30.
#   2. Evaluate all six splits (three valid/test groups) for each checkpoint.
#   3. Summarize Precision, Recall, F1, and Accuracy using a fixed threshold.
#   4. Output JSON, TSV, and epoch trend plots.
#
# Run directly:
#   bash sh07_evaluate_pause_checkpoints.sh
#
# Important: use exactly the same data and threshold for every epoch; do not tune per epoch.
# Avoid running this at the same time as a training job on the same GPU.
# ============================================================================

# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
EPOCHS="2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"
THRESHOLD="0.50"
INFERENCE_BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail

source "$CONDA_ACTIVATE" "$CONDA_ENV"

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"
SUMMARY_SCRIPT="$PACKAGE_ROOT/scripts/summarize_pause_checkpoint_series.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
OUTPUT_ROOT="$EXPERIMENT_ROOT/checkpoint_series_epoch_2_to_30_even"
INFERENCE_ROOT="$OUTPUT_ROOT/inference"
REPORT_ROOT="$OUTPUT_ROOT/reports"
SPLITS=(valid_m2_manual valid_m2_mfa valid_lmdb_mfa test_m2_manual test_m2_mfa test_lmdb_mfa)
read -r -a EPOCH_ARRAY <<< "$EPOCHS"

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -f "$INFER_SCRIPT" || ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "Error: multi-checkpoint inference or summary script is missing" >&2
    exit 1
fi
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    echo "Error: wav2vec TorchScript assets are missing" >&2
    exit 1
fi
for split in "${SPLITS[@]}"; do
    if [[ ! -d "$DATA_ROOT/$split" || ! -f "$DATA_ROOT/$split.key" ]]; then
        echo "Error: comparison split not found: $DATA_ROOT/$split" >&2
        echo "Run V2's bash sh01_prepare_pause_training_data.sh first." >&2
        exit 1
    fi
done
for epoch in "${EPOCH_ARRAY[@]}"; do
    CHECKPOINT="$CHECKPOINT_ROOT/checkpoint${epoch}.pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Error: checkpoint not found: $CHECKPOINT" >&2
        exit 1
    fi
done

# Step 1: Save probabilities and baseline metrics at 0.5 for each checkpoint and split.
echo "[1/2] Start evaluating checkpoints: $EPOCHS"
mkdir -p "$INFERENCE_ROOT" "$REPORT_ROOT"
cd "$CODE_ROOT"
for epoch in "${EPOCH_ARRAY[@]}"; do
    CHECKPOINT="$CHECKPOINT_ROOT/checkpoint${epoch}.pt"
    EPOCH_ROOT="$INFERENCE_ROOT/epoch_$(printf '%03d' "$epoch")"
    echo "Evaluating epoch=$epoch"
    for split in "${SPLITS[@]}"; do
        SPLIT_OUTPUT="$EPOCH_ROOT/$split"
        mkdir -p "$SPLIT_OUTPUT"
        echo "  split=$split"
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
            python "$INFER_SCRIPT" \
            --user-dir "$USER_DIR" \
            --checkpoint "$CHECKPOINT" \
            --data-root "$DATA_ROOT" \
            --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
            --wav2vec-meta-path "$WAV2VEC_META_PATH" \
            --split "$split" \
            --output-dir "$SPLIT_OUTPUT" \
            --threshold "$THRESHOLD" \
            --batch-size "$INFERENCE_BATCH_SIZE" \
            --num-workers "$NUM_WORKERS"
    done
done

# Step 2: Recalculate metrics for all epochs using the fixed threshold and generate trend plots.
echo "[2/2] Aggregate metrics across checkpoints and generate line plots"
python "$SUMMARY_SCRIPT" \
    --series-root "$INFERENCE_ROOT" \
    --output-root "$REPORT_ROOT" \
    --epochs "${EPOCH_ARRAY[@]}" \
    --threshold "$THRESHOLD"

echo "Done. Per-checkpoint inference results: $INFERENCE_ROOT"
echo "Metrics TSV: $REPORT_ROOT/pause_checkpoint_series_metrics.tsv"
echo "Full JSON: $REPORT_ROOT/pause_checkpoint_series_metrics.json"
echo "Line plot directory: $REPORT_ROOT"
