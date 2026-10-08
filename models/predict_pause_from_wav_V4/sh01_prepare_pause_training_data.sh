#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：只读校验并复用 V2 创建的共享训练/评测 LMDB。
# 直接运行：bash sh01_prepare_pause_training_data.sh
# 重要约束：本脚本绝不创建或重划分数据；缺少指纹时请先运行 V2 的 sh01。
# ============================================================================

# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
M2_SOURCE_INDEX="27"
M2_SUPPLEMENT_SOURCE_INDICES="25,26"
EXPECTED_VALID_COUNT="500"
EXPECTED_TEST_COUNT="100"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail
source "$CONDA_ACTIVATE" "$CONDA_ENV"

FINGERPRINT="$DATA_ROOT/shared_dataset_fingerprint.json"
VALIDATE_SCRIPT="$PACKAGE_ROOT/scripts/validate_shared_pause_splits.py"
if [[ ! -f "$FINGERPRINT" ]]; then
    echo "错误：共享数据尚未生成或未通过指纹校验：$FINGERPRINT" >&2
    echo "请先运行 predict_pause_from_wav_V2/sh01_prepare_pause_training_data.sh" >&2
    exit 1
fi
if [[ ! -f "$VALIDATE_SCRIPT" ]]; then
    echo "错误：缺少共享数据校验脚本：$VALIDATE_SCRIPT" >&2
    exit 1
fi

echo "[1/1] 只读校验 V2-V5 共用的数据划分"
python "$VALIDATE_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --expected-valid-count "$EXPECTED_VALID_COUNT" \
    --expected-test-count "$EXPECTED_TEST_COUNT" \
    --m2-source-index "$M2_SOURCE_INDEX" \
    --m2-supplement-source-indices "$M2_SUPPLEMENT_SOURCE_INDICES" \
    --verify-existing-fingerprint

echo "完成。V4 将复用共享数据：$DATA_ROOT"
echo "下一步：运行 bash sh02_train_pause_from_wav.sh 或提交 KY 训练。"
