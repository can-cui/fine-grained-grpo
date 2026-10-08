#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：提交正式训练到 KY 集群。
#
# 主要阶段：
#   1. 检查 KY CLI、worker 脚本及 GPU 资源配置；
#   2. 在正式实验根目录下创建 KY stdout/stderr 日志目录；
#   3. 提交单张 Tesla V100 PCIe 32GB 正式训练任务。
#
# 直接运行：
#   bash run_pause_from_wav_train_ky.sh
#
# 重要约束：KY 日志和模型登记路径均位于 /ng-mix02 的正式实验目录。
# ============================================================================
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
KY_LOG_DIR="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2/ky_logs/train"
KY_MODEL_PATH="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
KY_WORKER_SCRIPT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/ky_pause_from_wav_train.sh"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail

if ! command -v ky >/dev/null 2>&1; then
    echo "错误：当前终端找不到 ky 命令；请在具备 KY CLI 的内网登录节点执行。" >&2
    exit 1
fi
if [[ ! -f "$KY_WORKER_SCRIPT" ]]; then
    echo "错误：找不到 KY worker 脚本：$KY_WORKER_SCRIPT" >&2
    exit 1
fi
if [[ "$KY_GPU_COUNT" -le 0 ]]; then
    echo "错误：KY_GPU_COUNT 必须大于 0" >&2
    exit 1
fi

# 第 1 阶段：创建 KY 调度日志和模型登记目录。
mkdir -p "$KY_LOG_DIR" "$KY_MODEL_PATH"
TIMESTAMP="$(date '+%Y%m%d%H%M%S')"
STDOUT_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.log"
STDERR_LOG="$KY_LOG_DIR/${KY_JOB_NAME}_${TIMESTAMP}.error.log"

echo "提交 KY 正式训练任务：$KY_JOB_NAME"
echo "标准输出日志：$STDOUT_LOG"
echo "错误输出日志：$STDERR_LOG"
echo "KY 模型登记目录：$KY_MODEL_PATH"

# 第 2 阶段：提交正式训练任务。
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
