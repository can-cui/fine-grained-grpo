#!/bin/bash
# ============================================================================
# 音频停顿预测 V4：在线运行冻结 wav2vec，只训练单层 Linear 停顿分类头。
#
# 主要阶段：
#   1. 检查正式 train/valid LMDB 和训练入口；
#   2. 缺少永久 wav2vec TorchScript 资产时自动导出；
#   3. 校验模型资产并在单张 GPU 上执行 30 epoch 正式训练；
#   4. 每个 epoch 保存、验证，并按 valid pause_recall 选择最佳 checkpoint。
#
# 直接运行：
#   bash sh02_train_pause_from_wav.sh
#
# 重要约束：不使用离线 wav2vec .npy；wav2vec 始终冻结，仅更新 Linear(768,1)。
# ============================================================================

# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
OLD_CONDA_ACTIVATE="/home/tts/sjliu18/yfw/anaconda3/bin/activate"
OLD_CONDA_ENV="python36"
NEW_CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
NEW_CONDA_ENV="torch20_fair_fromyjdong4"
ORIGINAL_WAV2VEC_PROJECT="/yrfs5/tts/sjliu18/02_front_end_tools/multiMFA_preL2_process/Pre_L123/iflytek-tts-exp"
WAV2VEC_CHECKPOINT="/yrfs5/tts/ycxu5/Pre_L123/wav2vec_small/wav2vec_small.pt"
WAV2VEC_JIT_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.meta.json"
CUDA_VISIBLE_DEVICES_VALUE="0"
LEARNING_RATE="0.001"
WAV2VEC_LEARNING_RATE="0.00001"
POSITIVE_WEIGHT="1.0"
MAX_EPOCH="30"
MAX_SENTENCES="8"
UPDATE_FREQ="4"
UNFREEZE_LAST_N="2"
SAVE_INTERVAL="1"
VALIDATE_INTERVAL="1"
TRAIN_SEED="20260807"
# ========================================================

# ================== 下面一般不用修改 ==================
set -euo pipefail

# Conda 激活脚本在 nounset 下可能读取未定义变量，激活期间临时关闭 nounset。
set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
TRAIN_SCRIPT="$CODE_ROOT/train.py"
EXPORT_SCRIPT="$PACKAGE_ROOT/scripts/export_wav2vec_torchscript.py"
RUNTIME_CHECK_SCRIPT="$PACKAGE_ROOT/scripts/validate_runtime_files.py"
CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints"
LOG_DIR="$EXPERIMENT_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"

python "$RUNTIME_CHECK_SCRIPT" --package-root "$PACKAGE_ROOT"
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/valid_m2_manual" || ! -d "$DATA_ROOT/valid_m2_mfa" || ! -d "$DATA_ROOT/valid_lmdb_mfa" ]]; then
    echo "错误：DATA_ROOT 中缺少 train 或三组正式 valid LMDB：$DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$EXPORT_SCRIPT" ]]; then
    echo "错误：找不到 wav2vec 导出/校验脚本：$EXPORT_SCRIPT" >&2
    exit 1
fi

# 第 1 阶段：缺少永久模型资产时，在原 Python 3.6 环境中自动导出。
WAV2VEC_NEEDS_EXPORT="0"
if [[ ! -s "$WAV2VEC_JIT_PATH" || ! -s "$WAV2VEC_META_PATH" ]]; then
    WAV2VEC_NEEDS_EXPORT="1"
elif ! python - "$WAV2VEC_META_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)

valid = (
    metadata.get("format_version") == 3
    and metadata.get("trace_device") == "cuda"
)
raise SystemExit(0 if valid else 1)
PY
then
    WAV2VEC_NEEDS_EXPORT="1"
fi

if [[ "$WAV2VEC_NEEDS_EXPORT" == "1" ]]; then
    if [[ ! -d "$ORIGINAL_WAV2VEC_PROJECT" || ! -f "$WAV2VEC_CHECKPOINT" ]]; then
        echo "错误：缺少永久资产，且找不到原 wav2vec 工程或 checkpoint。" >&2
        exit 1
    fi
    echo "[1/3] 导出冻结 wav2vec TorchScript 模型资产"
    set +u
    source "$OLD_CONDA_ACTIVATE" "$OLD_CONDA_ENV"
    set -u
    cd "$ORIGINAL_WAV2VEC_PROJECT"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
        --checkpoint "$WAV2VEC_CHECKPOINT" \
        --fairseq-project-root "$ORIGINAL_WAV2VEC_PROJECT" \
        --output-model "$WAV2VEC_JIT_PATH" \
        --output-meta "$WAV2VEC_META_PATH" \
        --sample-rate 16000 \
        --feature-dim 768 \
        --trace-device cuda
fi

# 第 2 阶段：回到训练环境，检查跨版本加载、SHA256 和变长输出。
set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u
echo "[2/3] 校验 wav2vec TorchScript 模型资产"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
    --output-model "$WAV2VEC_JIT_PATH" \
    --output-meta "$WAV2VEC_META_PATH" \
    --sample-rate 16000 \
    --feature-dim 768 \
    --validate-only

# 第 3 阶段：只解冻 wav2vec 最后两层并使用双学习率训练。
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "[3/3] 开始训练音频停顿预测 V4 部分解冻实验"
echo "训练参数：GPU=1，batch_size=$MAX_SENTENCES，update_freq=$UPDATE_FREQ，epoch=$MAX_EPOCH"
echo "学习率：分类头=$LEARNING_RATE，wav2vec=$WAV2VEC_LEARNING_RATE，解冻最后=$UNFREEZE_LAST_N 层"
echo "保存/验证间隔：$SAVE_INTERVAL/$VALIDATE_INTERVAL epoch，随机种子：$TRAIN_SEED"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python -u "$TRAIN_SCRIPT" "$DATA_ROOT" \
    --user-dir "$USER_DIR" \
    --save-dir "$CHECKPOINT_DIR" \
    --task pause_prediction \
    --valid-subset valid_m2_manual,valid_m2_mfa,valid_lmdb_mfa \
    --arch pause_from_wav_w2v_unfreeze_last2 \
    --unfreeze-last-n "$UNFREEZE_LAST_N" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --input-sample-rate 16000 \
    --criterion pause_bce_loss \
    --optimizer pause_adam \
    --adam-betas '(0.9,0.997)' \
    --wav2vec-lr "$WAV2VEC_LEARNING_RATE" \
    --lr-scheduler fixed \
    --lr "$LEARNING_RATE" \
    --positive-weight "$POSITIVE_WEIGHT" \
    --pause-threshold 0.5 \
    --best-checkpoint-metric pause_recall \
    --maximize-best-checkpoint-metric \
    --max-epoch "$MAX_EPOCH" \
    --max-sentences "$MAX_SENTENCES" \
    --update-freq "$UPDATE_FREQ" \
    --save-interval "$SAVE_INTERVAL" \
    --validate-interval "$VALIDATE_INTERVAL" \
    --seed "$TRAIN_SEED" \
    --num-workers 1 \
    --log-format simple \
    --log-interval 10 \
    --skip-invalid-size-inputs-valid-test \
    2>&1 | tee "$TRAIN_LOG"

echo "完成。checkpoint：$CHECKPOINT_DIR"
echo "训练日志：$TRAIN_LOG"
echo "下一步：运行 bash sh03_infer_pause_from_wav.sh"
