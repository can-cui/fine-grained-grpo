#!/bin/bash
# Audio pause prediction V4: smoke-test runner for a tiny dataset on the KY cluster.
# This script runs as a single KY GPU job after submission by run_pause_from_wav_smoke_test_ky.sh.
# ==================== Configuration ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
SMOKE_SCRIPT="$PACKAGE_ROOT/smoke_test/run_pause_v4_smoke_test.sh"
# ========================================================

# ================== Usually no changes needed below ==================
set -euo pipefail

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
    echo "Error: Conda activation script not found: $CONDA_ACTIVATE" >&2
    exit 1
fi
if [[ ! -f "$SMOKE_SCRIPT" ]]; then
    echo "Error: V4 smoke-test script not found: $SMOKE_SCRIPT" >&2
    exit 1
fi

source "$CONDA_ACTIVATE" "$CONDA_ENV"
echo "[KY smoke worker] Python：$(python --version 2>&1)"
echo "[KY smoke worker] Working directory: $PACKAGE_ROOT"
echo "[KY smoke worker] Starting the tiny-dataset smoke test"
cd "$PACKAGE_ROOT/smoke_test"
bash "$SMOKE_SCRIPT"
