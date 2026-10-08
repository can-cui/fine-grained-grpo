#!/bin/bash
# 音频停顿预测 V4：KY 集群极小数据烟雾测试执行脚本。
# 本脚本由 run_pause_from_wav_smoke_test_ky.sh 提交后在单个 KY GPU 任务中执行。
# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
SMOKE_SCRIPT="$PACKAGE_ROOT/smoke_test/run_pause_v4_smoke_test.sh"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail

if [[ ! -f "$CONDA_ACTIVATE" ]]; then
    echo "错误：找不到 Conda activate 脚本：$CONDA_ACTIVATE" >&2
    exit 1
fi
if [[ ! -f "$SMOKE_SCRIPT" ]]; then
    echo "错误：找不到 V4 烟雾测试脚本：$SMOKE_SCRIPT" >&2
    exit 1
fi

source "$CONDA_ACTIVATE" "$CONDA_ENV"
echo "[KY smoke worker] Python：$(python --version 2>&1)"
echo "[KY smoke worker] 工作目录：$PACKAGE_ROOT"
echo "[KY smoke worker] 开始调用极小数据烟雾测试"
cd "$PACKAGE_ROOT/smoke_test"
bash "$SMOKE_SCRIPT"
