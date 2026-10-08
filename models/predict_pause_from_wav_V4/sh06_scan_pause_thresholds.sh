#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：在三组 valid 上选阈值，并应用到各自对应 test。
#
# 主要阶段：
#   1. 分别扫描 m2_manual、m2_mfa、lmdb_mfa 三组 valid；
#   2. 每组按最高 F1 选择阈值并原样应用到对应 test；
#
# 直接运行：
#   bash sh06_scan_pause_thresholds.sh
#
# 重要约束：只读取 sh03 已保存的逐词概率，不重新推理、不使用 test 选择阈值。
# ============================================================================

# ==================== 需要修改的地方 ====================
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

# ================== 下面一般不用修改 ==================
set -euo pipefail

source "$CONDA_ACTIVATE" "$CONDA_ENV"

SCAN_SCRIPT="$PACKAGE_ROOT/scripts/scan_pause_thresholds.py"
INFERENCE_ROOT="$EXPERIMENT_ROOT/inference/${CHECKPOINT_NAME%.pt}_threshold_${SOURCE_THRESHOLD}"
OUTPUT_ROOT="$INFERENCE_ROOT/threshold_scan"

if [[ ! -f "$SCAN_SCRIPT" ]]; then
    echo "错误：找不到阈值扫描脚本：$SCAN_SCRIPT" >&2
    exit 1
fi
for split in valid_m2_manual valid_m2_mfa valid_lmdb_mfa test_m2_manual test_m2_mfa test_lmdb_mfa; do
    PREDICTION_TSV="$INFERENCE_ROOT/$split/${split}_pause_predictions.tsv"
    if [[ ! -f "$PREDICTION_TSV" ]]; then
        echo "错误：找不到逐词预测：$PREDICTION_TSV" >&2
        echo "请先运行 bash sh03_infer_pause_from_wav.sh" >&2
        exit 1
    fi
done

# 第 1 阶段：读取已有概率，在三组 valid 上独立扫描阈值。
echo "[1/2] 扫描 valid 阈值：$THRESHOLD_START 到 $THRESHOLD_END，步长 $THRESHOLD_STEP"
python "$SCAN_SCRIPT" \
    --inference-root "$INFERENCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --threshold-start "$THRESHOLD_START" \
    --threshold-end "$THRESHOLD_END" \
    --threshold-step "$THRESHOLD_STEP" \
    --baseline-threshold "$BASELINE_THRESHOLD"

# 第 2 阶段：报告汇总路径；test 只使用 valid 选出的阈值。
echo "[2/2] 完成阈值选择与 test 复算"
echo "汇总表：$OUTPUT_ROOT/pause_threshold_scan_summary.tsv"
echo "完整结果：$OUTPUT_ROOT/pause_threshold_scan_summary.json"
echo "考试院人工扫描表：$OUTPUT_ROOT/valid_m2_manual_threshold_scan.tsv"
echo "考试院 MFA 扫描表：$OUTPUT_ROOT/valid_m2_mfa_threshold_scan.tsv"
echo "part00-04 MFA 扫描表：$OUTPUT_ROOT/valid_lmdb_mfa_threshold_scan.tsv"
