#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：评测 epoch 2-30 的三组 valid/test 变化规律。
#
# 主要阶段：
#   1. 依次加载偶数 epoch 的 checkpoint2 至 checkpoint30；
#   2. 每个 checkpoint 评测三组 valid/test 共六个 split；
#   3. 用固定阈值汇总 Precision、Recall、F1、Accuracy；
#   4. 输出 JSON、TSV 和 epoch 折线统计图。
#
# 直接运行：
#   bash sh07_evaluate_pause_checkpoints.sh
#
# 重要约束：不同 epoch 使用完全相同的数据与阈值，不逐 epoch 调参。
# 建议不要与同一张 GPU 上的训练任务同时运行。
# ============================================================================

# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
EPOCHS="2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"
THRESHOLD="0.50"
INFERENCE_BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail

source "$CONDA_ACTIVATE" "$CONDA_ENV"

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"
SUMMARY_SCRIPT="$PACKAGE_ROOT/scripts/summarize_pause_checkpoint_series.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
OUTPUT_ROOT="$EXPERIMENT_ROOT/checkpoint_series_epoch_2_to_30_even"
INFERENCE_ROOT="$OUTPUT_ROOT/inference"
REPORT_ROOT="$OUTPUT_ROOT/reports"
SPLITS=(valid_m2_manual valid_m2_mfa valid_lmdb_mfa test_m2_manual test_m2_mfa test_lmdb_mfa)
read -r -a EPOCH_ARRAY <<< "$EPOCHS"

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -f "$INFER_SCRIPT" || ! -f "$SUMMARY_SCRIPT" ]]; then
    echo "错误：缺少多 checkpoint 推理或汇总脚本" >&2
    exit 1
fi
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    echo "错误：缺少 wav2vec TorchScript 资产" >&2
    exit 1
fi
for split in "${SPLITS[@]}"; do
    if [[ ! -d "$DATA_ROOT/$split" || ! -f "$DATA_ROOT/$split.key" ]]; then
        echo "错误：找不到对照 split：$DATA_ROOT/$split" >&2
        echo "请先运行 V2 的 bash sh01_prepare_pause_training_data.sh" >&2
        exit 1
    fi
done
for epoch in "${EPOCH_ARRAY[@]}"; do
    CHECKPOINT="$CHECKPOINT_ROOT/checkpoint${epoch}.pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "错误：找不到 checkpoint：$CHECKPOINT" >&2
        exit 1
    fi
done

# 第 1 阶段：逐 checkpoint、逐 split 保存概率与 0.5 基础指标。
echo "[1/2] 开始评测 checkpoint：$EPOCHS"
mkdir -p "$INFERENCE_ROOT" "$REPORT_ROOT"
cd "$CODE_ROOT"
for epoch in "${EPOCH_ARRAY[@]}"; do
    CHECKPOINT="$CHECKPOINT_ROOT/checkpoint${epoch}.pt"
    EPOCH_ROOT="$INFERENCE_ROOT/epoch_$(printf '%03d' "$epoch")"
    echo "评测 epoch=$epoch"
    for split in "${SPLITS[@]}"; do
        SPLIT_OUTPUT="$EPOCH_ROOT/$split"
        mkdir -p "$SPLIT_OUTPUT"
        echo "  split=$split"
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
            python "$INFER_SCRIPT" \
            --user-dir "$USER_DIR" \
            --checkpoint "$CHECKPOINT" \
            --data-root "$DATA_ROOT" \
            --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
            --wav2vec-meta-path "$WAV2VEC_META_PATH" \
            --split "$split" \
            --output-dir "$SPLIT_OUTPUT" \
            --threshold "$THRESHOLD" \
            --batch-size "$INFERENCE_BATCH_SIZE" \
            --num-workers "$NUM_WORKERS"
    done
done

# 第 2 阶段：对所有 epoch 使用固定阈值重算指标并生成折线图。
echo "[2/2] 汇总多 checkpoint 指标并生成折线图"
python "$SUMMARY_SCRIPT" \
    --series-root "$INFERENCE_ROOT" \
    --output-root "$REPORT_ROOT" \
    --epochs "${EPOCH_ARRAY[@]}" \
    --threshold "$THRESHOLD"

echo "完成。逐 checkpoint 推理结果：$INFERENCE_ROOT"
echo "指标 TSV：$REPORT_ROOT/pause_checkpoint_series_metrics.tsv"
echo "完整 JSON：$REPORT_ROOT/pause_checkpoint_series_metrics.json"
echo "折线图目录：$REPORT_ROOT"
