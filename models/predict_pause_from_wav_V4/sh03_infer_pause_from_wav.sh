#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：使用 checkpoint_best 对三组 valid/test 统一推理。
# 直接运行：bash sh03_infer_pause_from_wav.sh
# 重要约束：阈值固定为 0.5；本脚本不根据 test 选择阈值。
# ============================================================================

# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
CHECKPOINT_NAME="checkpoint_best.pt"
THRESHOLD="0.50"
BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail
source "$CONDA_ACTIVATE" "$CONDA_ENV"

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"
CHECKPOINT="$EXPERIMENT_ROOT/checkpoints/$CHECKPOINT_NAME"
OUTPUT_ROOT="$EXPERIMENT_ROOT/inference/${CHECKPOINT_NAME%.pt}_threshold_${THRESHOLD}"
SPLITS=(
    valid_m2_manual
    valid_m2_mfa
    valid_lmdb_mfa
    test_m2_manual
    test_m2_mfa
    test_lmdb_mfa
)

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "错误：找不到 checkpoint：$CHECKPOINT" >&2
    exit 1
fi
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    echo "错误：缺少本实验训练所用 wav2vec TorchScript 资产。请先运行 sh02。" >&2
    exit 1
fi
for split in "${SPLITS[@]}"; do
    if [[ ! -d "$DATA_ROOT/$split" || ! -f "$DATA_ROOT/$split.key" ]]; then
        echo "错误：缺少正式评测 split：$DATA_ROOT/$split" >&2
        exit 1
    fi
done

# 第 1 阶段：对六个固定 split 逐一输出概率和固定阈值指标。
echo "[1/1] 使用 $CHECKPOINT_NAME 推理六个正式 split"
mkdir -p "$OUTPUT_ROOT"
cd "$CODE_ROOT"
for split in "${SPLITS[@]}"; do
    split_output="$OUTPUT_ROOT/$split"
    mkdir -p "$split_output"
    echo "推理 split=$split"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
        python "$INFER_SCRIPT" \
        --user-dir "$USER_DIR" \
        --checkpoint "$CHECKPOINT" \
        --data-root "$DATA_ROOT" \
        --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
        --wav2vec-meta-path "$WAV2VEC_META_PATH" \
        --split "$split" \
        --output-dir "$split_output" \
        --threshold "$THRESHOLD" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"
done

echo "完成。六 split 推理结果：$OUTPUT_ROOT"
echo "下一步：运行 bash sh06_scan_pause_thresholds.sh。"
