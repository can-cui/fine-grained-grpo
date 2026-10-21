#!/bin/bash
# ============================================================================
# Audio pause prediction V4: read-only validation and reuse of the shared training/evaluation LMDBs created by V2.
# Run directly: bash sh01_prepare_pause_training_data.sh
# Important: this script never creates or repartitions data. If the fingerprint is missing, run V2's sh01 first.
# ============================================================================

# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
M2_SOURCE_INDEX="27"
M2_SUPPLEMENT_SOURCE_INDICES="25,26"
EXPECTED_VALID_COUNT="500"
EXPECTED_TEST_COUNT="100"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail
source "$CONDA_ACTIVATE" "$CONDA_ENV"

FINGERPRINT="$DATA_ROOT/shared_dataset_fingerprint.json"
VALIDATE_SCRIPT="$PACKAGE_ROOT/scripts/validate_shared_pause_splits.py"
if [[ ! -f "$FINGERPRINT" ]]; then
    echo "Error: shared data has not been generated or failed fingerprint validation: $FINGERPRINT" >&2
    echo "Run predict_pause_from_wav_V2/sh01_prepare_pause_training_data.sh first." >&2
    exit 1
fi
if [[ ! -f "$VALIDATE_SCRIPT" ]]; then
    echo "Error: shared-data validation script not found: $VALIDATE_SCRIPT" >&2
    exit 1
fi

echo "[1/1] Read-only validation of the data split shared by V2-V5"
python "$VALIDATE_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --expected-valid-count "$EXPECTED_VALID_COUNT" \
    --expected-test-count "$EXPECTED_TEST_COUNT" \
    --m2-source-index "$M2_SOURCE_INDEX" \
    --m2-supplement-source-indices "$M2_SUPPLEMENT_SOURCE_INDICES" \
    --verify-existing-fingerprint

echo "Done. V4 will reuse the shared data: $DATA_ROOT"
echo "Next: run bash sh02_train_pause_from_wav.sh or submit a KY training job."
