#!/bin/bash
# ============================================================================
# Audio pause prediction V4: submit the formal training job to the KY cluster.
#
# Main steps:
#   1. Check the KY CLI, worker script, and GPU resource configuration.
#   2. Create KY stdout/stderr log directories under the formal experiment root.
#   3. Submit the formal training job on one Tesla V100 PCIe 32 GB GPU.
#
# Run directly:
#   bash run_pause_from_wav_train_ky.sh
#
# Important: KY logs and model registration paths are both under the formal experiment directory on /ng-mix02.
# ============================================================================
# ==================== Edit these settings ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
KY_USERNAME="yhchen70"
KY_PROJECT_ID="2227"
KY_GPU_TYPE="TeslaV100-PCIE-32GB"
KY_RESERVED="dlp3-tts-prompt-reserved"
KY_IMAGE="reg.deeplearning.cn/ky/gpu:centos7-20260105-beta-glibc_2.40_mlnx_sndfile"
KY_GPU_COUNT="1"
KY_JOB_NAME="train-pause-v4"
KY_JOB_DESCRIPTION="train-pause-v4"
KY_LOG_DIR="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2/ky_logs/train"
KY_MODEL_PATH="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
KY_WORKER_SCRIPT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/ky_pause_from_wav_train.sh"
# ========================================================

# ================== The settings below usually do not need changes ==================
set -euo pipefail

if ! command -v ky >/dev/null 2>&1; then
    echo "Error: ky command is unavailable in this terminal; run this on an internal login node with the KY CLI." >&2
    exit 1
fi
if [[ ! -f "$KY_WORKER_SCRIPT" ]]; then
    echo "Error: KY worker script not found: $KY_WORKER_SCRIPT" >&2
    exit 1
fi
if [[ "$KY_GPU_COUNT" -le 0 ]]; then
    echo "Error: KY_GPU_COUNT must be greater than 0" >&2
    exit 1
fi

# Step 1: Create the KY scheduler log and model registration directories.
mkdir -p "$KY_LOG_DIR" "$KY_MODEL_PATH"
TIMESTAMP="$(date '+%Y%m%d%H%M%S')"
STDOUT_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.log"
STDERR_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.error.log"

echo "Submitting KY training job: $KY_JOB_NAME"
echo "Standard output log: $STDOUT_LOG"
echo "Standard error log: $STDERR_LOG"
echo "KY model registry directory: $KY_MODEL_PATH"

# Step 2: Submit the formal training job.
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
    --modelPath "$KY_MODEL_PATH" \
    --modelName "$TIMESTAMP" \
    -r "$KY_RESERVED"
