#!/bin/bash
# Audio pause prediction V4: training worker script for the KY cluster.
# This script runs as a single KY GPU job after being submitted by run_pause_from_wav_train_ky.sh.
# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
TRAIN_SCRIPT="$PACKAGE_ROOT/sh02_train_pause_from_wav.sh"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
    echo "Error: Conda activation script not found: $CONDA_ACTIVATE" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: V4 training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

source "$CONDA_ACTIVATE" "$CONDA_ENV"
echo "[KY worker] Python：$(python --version 2>&1)"
echo "[KY worker] Working directory: $PACKAGE_ROOT"
echo "[KY worker] Starting the training script"
cd "$PACKAGE_ROOT"
bash "$TRAIN_SCRIPT"
