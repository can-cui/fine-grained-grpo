#!/bin/bash
# ============================================================================
# Audio pause prediction V4: select thresholds on three valid sets and apply them to their corresponding test sets.
#
# Main steps:
#   1. Scan the m2_manual, m2_mfa, and lmdb_mfa valid sets separately.
#   2. Select a threshold with the highest F1 for each group and apply it unchanged to the corresponding test set.
#
# Run directly:
#   bash sh06_scan_pause_thresholds.sh
#
# Important: read only the word-level probabilities saved by sh03; do not rerun inference or use test data to select thresholds.
# ============================================================================

# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
CHECKPOINT_NAME="checkpoint_best.pt"
SOURCE_THRESHOLD="0.50"
THRESHOLD_START="0.01"
THRESHOLD_END="0.99"
THRESHOLD_STEP="0.01"
BASELINE_THRESHOLD="0.50"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail

source "$CONDA_ACTIVATE" "$CONDA_ENV"

SCAN_SCRIPT="$PACKAGE_ROOT/scripts/scan_pause_thresholds.py"
INFERENCE_ROOT="$EXPERIMENT_ROOT/inference/${CHECKPOINT_NAME%.pt}_threshold_${SOURCE_THRESHOLD}"
OUTPUT_ROOT="$INFERENCE_ROOT/threshold_scan"

if [[ ! -f "$SCAN_SCRIPT" ]]; then
    echo "Error: threshold-scan script not found: $SCAN_SCRIPT" >&2
    exit 1
fi
for split in valid_m2_manual valid_m2_mfa valid_lmdb_mfa test_m2_manual test_m2_mfa test_lmdb_mfa; do
    PREDICTION_TSV="$INFERENCE_ROOT/$split/${split}_pause_predictions.tsv"
    if [[ ! -f "$PREDICTION_TSV" ]]; then
        echo "Error: word-level predictions not found: $PREDICTION_TSV" >&2
        echo "Run bash sh03_infer_pause_from_wav.sh first." >&2
        exit 1
    fi
done

# Step 1: Read the existing probabilities and scan thresholds independently on the three valid sets.
echo "[1/2] Scan validation thresholds: $THRESHOLD_START to $THRESHOLD_END, step $THRESHOLD_STEP"
python "$SCAN_SCRIPT" \
    --inference-root "$INFERENCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --threshold-start "$THRESHOLD_START" \
    --threshold-end "$THRESHOLD_END" \
    --threshold-step "$THRESHOLD_STEP" \
    --baseline-threshold "$BASELINE_THRESHOLD"

# Step 2: Report the summary path; use only the thresholds selected on valid for test.
echo "[2/2] Threshold selection and test-set recomputation complete"
echo "Summary table: $OUTPUT_ROOT/pause_threshold_scan_summary.tsv"
echo "Full results: $OUTPUT_ROOT/pause_threshold_scan_summary.json"
echo "Exam-authority manual scan: $OUTPUT_ROOT/valid_m2_manual_threshold_scan.tsv"
echo "Exam-authority MFA scan: $OUTPUT_ROOT/valid_m2_mfa_threshold_scan.tsv"
echo "part00-04 MFA scan: $OUTPUT_ROOT/valid_lmdb_mfa_threshold_scan.tsv"
