#!/bin/bash
# ============================================================================
# Audio pause prediction V4: run wav2vec online with its weights frozen; train only a single-layer Linear pause classification head.
#
# Main steps:
#   1. Check the formal train/valid LMDBs and training entry point.
#   2. Export the permanent wav2vec TorchScript asset if it is missing.
#   3. Validate the model asset and run 30 epochs of formal training on one GPU.
#   4. Save and validate each epoch, then select the best checkpoint by valid pause_recall.
#
# Run directly:
#   bash sh02_train_pause_from_wav.sh
#
# Important: do not use offline wav2vec .npy files; keep wav2vec frozen and update only Linear(768,1).
# ============================================================================

# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
OLD_CONDA_ACTIVATE="/home/tts/sjliu18/yfw/anaconda3/bin/activate"
OLD_CONDA_ENV="python36"
NEW_CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
NEW_CONDA_ENV="torch20_fair_fromyjdong4"
ORIGINAL_WAV2VEC_PROJECT="/yrfs5/tts/sjliu18/02_front_end_tools/multiMFA_preL2_process/Pre_L123/iflytek-tts-exp"
WAV2VEC_CHECKPOINT="/yrfs5/tts/ycxu5/Pre_L123/wav2vec_small/wav2vec_small.pt"
WAV2VEC_JIT_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.meta.json"
CUDA_VISIBLE_DEVICES_VALUE="0"
LEARNING_RATE="0.001"
WAV2VEC_LEARNING_RATE="0.00001"
POSITIVE_WEIGHT="1.0"
MAX_EPOCH="30"
MAX_SENTENCES="8"
UPDATE_FREQ="4"
UNFREEZE_LAST_N="2"
SAVE_INTERVAL="1"
VALIDATE_INTERVAL="1"
TRAIN_SEED="20260807"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail

# The Conda activation script may read undefined variables under nounset; temporarily disable nounset during activation.
set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
TRAIN_SCRIPT="$CODE_ROOT/train.py"
EXPORT_SCRIPT="$PACKAGE_ROOT/scripts/export_wav2vec_torchscript.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints"
LOG_DIR="$EXPERIMENT_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/valid_m2_manual" || ! -d "$DATA_ROOT/valid_m2_mfa" || ! -d "$DATA_ROOT/valid_lmdb_mfa" ]]; then
    echo "Error: DATA_ROOT is missing train or one of the three official validation LMDBs: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$EXPORT_SCRIPT" ]]; then
    echo "Error: wav2vec export/validation script not found: $EXPORT_SCRIPT" >&2
    exit 1
fi

# Step 1: If the permanent model asset is missing, export it automatically in the original Python 3.6 environment.
WAV2VEC_NEEDS_EXPORT="0"
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    WAV2VEC_NEEDS_EXPORT="1"
elif ! python - "$WAV2VEC_META_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)

valid = (
    metadata.get("format_version") == 3
    and metadata.get("trace_device") == "cuda"
)
raise SystemExit(0 if valid else 1)
PY
then
    WAV2VEC_NEEDS_EXPORT="1"
fi

if [[ "$WAV2VEC_NEEDS_EXPORT" == "1" ]]; then
    if [[ ! -d "$ORIGINAL_WAV2VEC_PROJECT" || ! -f "$WAV2VEC_CHECKPOINT" ]]; then
        echo "Error: permanent assets are missing, and the original wav2vec project or checkpoint could not be found." >&2
        exit 1
    fi
    echo "[1/3] Export frozen wav2vec TorchScript assets"
    set +u
    source "$OLD_CONDA_ACTIVATE" "$OLD_CONDA_ENV"
    set -u
    cd "$ORIGINAL_WAV2VEC_PROJECT"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
        --checkpoint "$WAV2VEC_CHECKPOINT" \
        --fairseq-project-root "$ORIGINAL_WAV2VEC_PROJECT" \
        --output-model "$WAV2VEC_JIT_PATH" \
        --output-meta "$WAV2VEC_META_PATH" \
        --sample-rate 16000 \
        --feature-dim 768 \
        --trace-device cuda
fi

# Step 2: Return to the training environment and check cross-version loading, SHA256, and variable-length outputs.
set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u
echo "[2/3] Validate wav2vec TorchScript assets"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
    --output-model "$WAV2VEC_JIT_PATH" \
    --output-meta "$WAV2VEC_META_PATH" \
    --sample-rate 16000 \
    --feature-dim 768 \
    --validate-only

# Step 3: Unfreeze only the last two wav2vec layers and train with two learning rates.
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "[3/3] Start V4 partial-unfreezing experiment for audio pause prediction"
echo "Training parameters: GPU=1, batch_size=$MAX_SENTENCES, update_freq=$UPDATE_FREQ, epochs=$MAX_EPOCH"
echo "Learning rates: classifier=$LEARNING_RATE, wav2vec=$WAV2VEC_LEARNING_RATE, last layers to unfreeze=$UNFREEZE_LAST_N"
echo "Save/validation intervals: $SAVE_INTERVAL/$VALIDATE_INTERVAL epochs; random seed: $TRAIN_SEED"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python -u "$TRAIN_SCRIPT" "$DATA_ROOT" \
    --user-dir "$USER_DIR" \
    --save-dir "$CHECKPOINT_DIR" \
    --task pause_prediction \
    --valid-subset valid_m2_manual,valid_m2_mfa,valid_lmdb_mfa \
    --arch pause_from_wav_w2v_unfreeze_last2 \
    --unfreeze-last-n "$UNFREEZE_LAST_N" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --input-sample-rate 16000 \
    --criterion pause_bce_loss \
    --optimizer pause_adam \
    --adam-betas '(0.9,0.997)' \
    --wav2vec-lr "$WAV2VEC_LEARNING_RATE" \
    --lr-scheduler fixed \
    --lr "$LEARNING_RATE" \
    --positive-weight "$POSITIVE_WEIGHT" \
    --pause-threshold 0.5 \
    --best-checkpoint-metric pause_recall \
    --maximize-best-checkpoint-metric \
    --max-epoch "$MAX_EPOCH" \
    --max-sentences "$MAX_SENTENCES" \
    --update-freq "$UPDATE_FREQ" \
    --save-interval "$SAVE_INTERVAL" \
    --validate-interval "$VALIDATE_INTERVAL" \
    --seed "$TRAIN_SEED" \
    --num-workers 1 \
    --log-format simple \
    --log-interval 10 \
    --skip-invalid-size-inputs-valid-test \
    2>&1 | tee "$TRAIN_LOG"

echo "Done. Checkpoints: $CHECKPOINT_DIR"
echo "Training log: $TRAIN_LOG"
echo "Next: run bash sh03_infer_pause_from_wav.sh"
