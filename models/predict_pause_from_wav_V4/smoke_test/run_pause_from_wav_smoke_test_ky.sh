#!/bin/bash
# Audio pause prediction V4: submit the tiny-dataset smoke test to the KY cluster.
# Run directly: bash run_pause_from_wav_smoke_test_ky.sh
# ==================== Configuration ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
KY_USERNAME="yhchen70"
KY_PROJECT_ID="2227"
KY_GPU_TYPE="TeslaV100-PCIE-32GB"
KY_RESERVED="dlp3-tts-prompt-reserved"
KY_IMAGE="reg.deeplearning.cn/ky/gpu:centos7-20260105-beta-glibc_2.40_mlnx_sndfile"
KY_GPU_COUNT="1"
KY_JOB_NAME="train-pause-v4"
KY_JOB_DESCRIPTION="train-pause-v4"
KY_LOG_DIR="/ng-mix02/tts/permanent/yhchen70/data/pause_v4_ky_logs/smoke"
KY_WORKER_SCRIPT="$PACKAGE_ROOT/smoke_test/ky_pause_from_wav_smoke_test.sh"
# ========================================================

# ================== Usually no changes needed below ==================
set -euo pipefail

if ! command -v ky >/dev/null 2>&1; then
    echo "Error: ky command is unavailable in this terminal; run this on an internal login node with the KY CLI." >&2
    exit 1
fi
if [[ ! -f "$KY_WORKER_SCRIPT" ]]; then
    echo "Error: KY smoke worker script not found: $KY_WORKER_SCRIPT" >&2
    exit 1
fi
if [[ "$KY_GPU_COUNT" -le 0 ]]; then
    echo "Error: KY_GPU_COUNT must be greater than 0" >&2
    exit 1
fi

mkdir -p "$KY_LOG_DIR"
TIMESTAMP="$(date '+%Y%m%d%H%M%S')"
STDOUT_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.log"
STDERR_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.error.log"

echo "Submitting KY smoke-test job: $KY_JOB_NAME"
echo "Standard output log: $STDOUT_LOG"
echo "Standard error log: $STDERR_LOG"
ky exp submit \
    -a "$KY_USERNAME" \
    -n "$KY_JOB_NAME" \
    -d "$KY_JOB_DESCRIPTION" \
    -i "$KY_IMAGE" \
    -e "$KY_WORKER_SCRIPT" \
    --useGpu \
    -g "$KY_GPU_COUNT" \
    -k "$KY_GPU_TYPE" \
    -t PtJob \
    -l "$STDOUT_LOG" \
    -o "$STDERR_LOG" \
    --proID "$KY_PROJECT_ID" \
    --modelPath "$KY_LOG_DIR" \
    --modelName "$TIMESTAMP" \
    -r "$KY_RESERVED"
