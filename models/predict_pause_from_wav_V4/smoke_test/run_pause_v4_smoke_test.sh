#!/bin/bash
# End-to-end smoke test for audio pause prediction V4 using a tiny dataset.
# Stages: check the internal V4 package -> sample -> LMDB -> online wav2vec training -> inference -> validation.
# Run directly: bash run_pause_v4_smoke_test.sh
# Important: only clean fixed subdirectories under the pause_v4_smoke_test root; production data and experiments are untouched.

# ==================== Configuration ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
MERGED_MANIFEST="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/all_pause_data.jsonl"
SMOKE_OUTPUT_ROOT="/ng-mix02/tts/permanent/yhchen70/data/pause_v4_smoke_test"
OLD_CONDA_ACTIVATE="/home/tts/sjliu18/yfw/anaconda3/bin/activate"
OLD_CONDA_ENV="python36"
NEW_CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
NEW_CONDA_ENV="torch20_fair_fromyjdong4"
ORIGINAL_WAV2VEC_PROJECT="/yrfs5/tts/sjliu18/02_front_end_tools/multiMFA_preL2_process/Pre_L123/iflytek-tts-exp"
WAV2VEC_CHECKPOINT="/yrfs5/tts/ycxu5/Pre_L123/wav2vec_small/wav2vec_small.pt"
WAV2VEC_JIT_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.meta.json"
TRAIN_COUNT="8"
VALID_COUNT="4"
TEST_COUNT="4"
SPLIT_SEED="20260809"
MAX_AUDIO_SECONDS="10"
MAX_EPOCH="1"
MAX_SENTENCES="2"
CUDA_VISIBLE_DEVICES_VALUE="0"
RESET_SMOKE_OUTPUT="1"
# ========================================================

# ================== Usually no changes needed below ==================
if [[ ! -f "$NEW_CONDA_ACTIVATE" ]]; then
    echo "Error: training environment activation script not found: $NEW_CONDA_ACTIVATE" >&2
    exit 1
fi
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -euo pipefail

SMOKE_PACKAGE_ROOT="$PACKAGE_ROOT/smoke_test"
SELECTED_DIR="$SMOKE_OUTPUT_ROOT/selected"
SMOKE_MANIFEST="$SELECTED_DIR/smoke_manifest.jsonl"
SELECTION_REPORT="$SELECTED_DIR/smoke_selection_report.json"
PACKAGE_REPORT="$SELECTED_DIR/v1_package_check.json"
DATA_ROOT="$SMOKE_OUTPUT_ROOT/data"
EXPERIMENT_ROOT="$SMOKE_OUTPUT_ROOT/experiments"
CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints"
LOG_DIR="$SMOKE_OUTPUT_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"
INFERENCE_DIR="$SMOKE_OUTPUT_ROOT/inference"
FINAL_REPORT_JSON="$SMOKE_OUTPUT_ROOT/smoke_test_report.json"
FINAL_REPORT_TXT="$SMOKE_OUTPUT_ROOT/smoke_test_report.txt"

CHECK_PACKAGE_SCRIPT="$SMOKE_PACKAGE_ROOT/check_v1_package.py"
BUILD_MANIFEST_SCRIPT="$SMOKE_PACKAGE_ROOT/build_smoke_manifest.py"
VALIDATE_OUTPUT_SCRIPT="$SMOKE_PACKAGE_ROOT/validate_smoke_outputs.py"
PREPARE_SCRIPT="$PACKAGE_ROOT/scripts/prepare_pause_training_data.py"
VALIDATE_LMDB_SCRIPT="$PACKAGE_ROOT/scripts/validate_pause_lmdb.py"
EXPORT_SCRIPT="$PACKAGE_ROOT/scripts/export_wav2vec_torchscript.py"
CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
TRAIN_SCRIPT="$CODE_ROOT/train.py"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"

# Stage 0: strictly constrain which smoke-test directory may be cleaned.
if [[ "$(basename "$SMOKE_OUTPUT_ROOT")" != "pause_v4_smoke_test" ]]; then
    echo "Error: SMOKE_OUTPUT_ROOT directory name must be pause_v4_smoke_test: $SMOKE_OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$SMOKE_OUTPUT_ROOT" == "/" || "$SMOKE_OUTPUT_ROOT" == "$PACKAGE_ROOT" ]]; then
    echo "Error: refusing unsafe smoke-test output path: $SMOKE_OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$RESET_SMOKE_OUTPUT" != "0" && "$RESET_SMOKE_OUTPUT" != "1" ]]; then
    echo "Error: RESET_SMOKE_OUTPUT must be 0 or 1" >&2
    exit 1
fi
if [[ "$RESET_SMOKE_OUTPUT" == "1" ]]; then
    echo "[0/8] Clean isolated smoke-test output"
    for child in selected data experiments logs inference; do
        rm -rf -- "$SMOKE_OUTPUT_ROOT/$child"
    done
    rm -f -- "$FINAL_REPORT_JSON" "$FINAL_REPORT_TXT"
fi
mkdir -p "$SELECTED_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$INFERENCE_DIR"

# Stage 1: verify the internal V4 runtime tree, production manifest, and scripts.
echo "[1/8] Check completeness of the internal predict_pause_from_wav_V4 package"
python "$CHECK_PACKAGE_SCRIPT" \
    --package-root "$PACKAGE_ROOT" \
    --expected-arch "pause_from_wav_w2v_unfreeze_last2" \
    --manifest "$MERGED_MANIFEST" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --check-fairseq-registration \
    --report "$PACKAGE_REPORT"

# Stage 2: deterministically select short audio and preview the count split used by the production converter.
echo "[2/8] Sample a tiny balanced dataset from production extraction results"
python "$BUILD_MANIFEST_SCRIPT" \
    --input-manifest "$MERGED_MANIFEST" \
    --output-manifest "$SMOKE_MANIFEST" \
    --output-report "$SELECTION_REPORT" \
    --expected-dir "$SELECTED_DIR" \
    --train-count "$TRAIN_COUNT" \
    --valid-count "$VALID_COUNT" \
    --test-count "$TEST_COUNT" \
    --split-seed "$SPLIT_SEED" \
    --max-audio-seconds "$MAX_AUDIO_SECONDS" \
    --sample-rate 24000

# Stage 3: reuse the production converter to create 8/4/4 16 kHz PCM16 LMDBs.
echo "[3/8] Create smoke-test train/valid/test LMDBs"
python "$PREPARE_SCRIPT" \
    --merged-manifest "$SMOKE_MANIFEST" \
    --output-root "$DATA_ROOT" \
    --split-mode count \
    --train-count "$TRAIN_COUNT" \
    --valid-count "$VALID_COUNT" \
    --test-count "$TEST_COUNT" \
    --split-seed "$SPLIT_SEED" \
    --source-sample-rate 24000 \
    --target-sample-rate 16000 \
    --map-size-gb 8

# Stage 4: read each LMDB record and check PCM, word windows, labels, and masks.
echo "[4/8] Validate smoke-test LMDBs"
python "$VALIDATE_LMDB_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --sample-rate 16000 \
    --splits train,valid,test

# Stage 5: if permanent assets are missing, export them in the legacy environment, then switch back for cross-version validation.
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
    if [[ ! -f "$OLD_CONDA_ACTIVATE" || ! -d "$ORIGINAL_WAV2VEC_PROJECT" || ! -s "$WAV2VEC_CHECKPOINT" ]]; then
        echo "Error: wav2vec permanent assets are missing, and the legacy environment/original project/checkpoint is incomplete" >&2
        exit 1
    fi
    echo "[5/8] Export wav2vec TorchScript in the original Python 3.6 environment"
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
else
    echo "[5/8] Reuse existing permanent wav2vec TorchScript assets"
fi

set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
    --output-model "$WAV2VEC_JIT_PATH" \
    --output-meta "$WAV2VEC_META_PATH" \
    --sample-rate 16000 \
    --feature-dim 768 \
    --validate-only

# Stage 6: train for one epoch; write checkpoints and logs to an isolated directory.
echo "[6/8] Run a tiny online wav2vec + Linear training job"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python -u "$TRAIN_SCRIPT" "$DATA_ROOT" \
    --user-dir "$USER_DIR" \
    --save-dir "$CHECKPOINT_DIR" \
    --task pause_prediction \
    --arch pause_from_wav_w2v_unfreeze_last2 \
    --unfreeze-last-n 2 \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --input-sample-rate 16000 \
    --criterion pause_bce_loss \
    --optimizer pause_adam \
    --wav2vec-lr 0.00001 \
    --adam-betas '(0.9,0.997)' \
    --lr-scheduler fixed \
    --lr 0.001 \
    --positive-weight 1.0 \
    --pause-threshold 0.5 \
    --best-checkpoint-metric pause_recall \
    --maximize-best-checkpoint-metric \
    --max-epoch "$MAX_EPOCH" \
    --max-sentences "$MAX_SENTENCES" \
    --num-workers 0 \
    --log-format simple \
    --log-interval 1 \
    --save-interval 1 \
    --validate-interval 1 \
    --skip-invalid-size-inputs-valid-test \
    2>&1 | tee "$TRAIN_LOG"

# Stage 7: run inference on four test samples using the best checkpoint.
echo "[7/8] Run tiny test inference"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python "$INFER_SCRIPT" \
    --user-dir "$USER_DIR" \
    --checkpoint "$CHECKPOINT_DIR/checkpoint_best.pt" \
    --data-root "$DATA_ROOT" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --split test \
    --output-dir "$INFERENCE_DIR" \
    --threshold 0.5 \
    --batch-size "$MAX_SENTENCES" \
    --num-workers 0

# Stage 8: check splits, checkpoints, logs, word-level results, and finite metrics.
echo "[8/8] Summarize and validate end-to-end outputs"
python "$VALIDATE_OUTPUT_SCRIPT" \
    --selection-report "$SELECTION_REPORT" \
    --data-root "$DATA_ROOT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --train-log "$TRAIN_LOG" \
    --inference-dir "$INFERENCE_DIR" \
    --output-json "$FINAL_REPORT_JSON" \
    --output-txt "$FINAL_REPORT_TXT"

echo "SMOKE_TEST_PASS"
echo "Package integrity report: $PACKAGE_REPORT"
echo "Final JSON report: $FINAL_REPORT_JSON"
echo "Final TXT report: $FINAL_REPORT_TXT"
