#!/bin/bash
# 音频停顿预测 V4：提交极小数据烟雾测试到 KY 集群。
# 直接运行：bash run_pause_from_wav_smoke_test_ky.sh
# ==================== 需要修改的地方 ====================
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

# ================== 下面一般不用修改 ==================
set -euo pipefail

if ! command -v ky >/dev/null 2>&1; then
    echo "错误：当前终端找不到 ky 命令；请在具备 KY CLI 的内网登录节点执行。" >&2
    exit 1
fi
if [[ ! -f "$KY_WORKER_SCRIPT" ]]; then
    echo "错误：找不到 KY smoke worker 脚本：$KY_WORKER_SCRIPT" >&2
    exit 1
fi
if [[ "$KY_GPU_COUNT" -le 0 ]]; then
    echo "错误：KY_GPU_COUNT 必须大于 0" >&2
    exit 1
fi

mkdir -p "$KY_LOG_DIR"
TIMESTAMP="$(date '+%Y%m%d%H%M%S')"
STDOUT_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.log"
STDERR_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.error.log"

echo "提交 KY 烟雾测试任务：$KY_JOB_NAME"
echo "标准输出日志：$STDOUT_LOG"
echo "错误输出日志：$STDERR_LOG"
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
