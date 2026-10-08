#!/bin/bash
# 音频停顿预测 V4 极小数据端到端烟雾测试。
# 处理阶段：检查内网 V4 包 -> 抽样 -> LMDB -> 在线 wav2vec 训练 -> 推理 -> 验收。
# 直接运行：bash run_pause_v4_smoke_test.sh
# 重要约束：只清理 pause_v4_smoke_test 根目录下的固定子目录，不修改正式数据和正式实验。

# ==================== 需要修改的地方 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
MERGED_MANIFEST="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/all_pause_data.jsonl"
SMOKE_OUTPUT_ROOT="/ng-mix02/tts/permanent/yhchen70/data/pause_v4_smoke_test"
OLD_CONDA_ACTIVATE="/home/tts/sjliu18/yfw/anaconda3/bin/activate"
OLD_CONDA_ENV="python36"
NEW_CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
NEW_CONDA_ENV="torch20_fair_fromyjdong4"
ORIGINAL_WAV2VEC_PROJECT="/yrfs5/tts/sjliu18/02_front_end_tools/multiMFA_preL2_process/Pre_L123/iflytek-tts-exp"
WAV2VEC_CHECKPOINT="/yrfs5/tts/ycxu5/Pre_L123/wav2vec_small/wav2vec_small.pt"
WAV2VEC_JIT_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4/assets/wav2vec_small_last_layer_jit.meta.json"
TRAIN_COUNT="8"
VALID_COUNT="4"
TEST_COUNT="4"
SPLIT_SEED="20260809"
MAX_AUDIO_SECONDS="10"
MAX_EPOCH="1"
MAX_SENTENCES="2"
CUDA_VISIBLE_DEVICES_VALUE="0"
RESET_SMOKE_OUTPUT="1"
# ========================================================

# ================== 下面一般不用修改 ==================
if [[ ! -f "$NEW_CONDA_ACTIVATE" ]]; then
    echo "错误：找不到训练环境 activate：$NEW_CONDA_ACTIVATE" >&2
    exit 1
fi
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -euo pipefail

SMOKE_PACKAGE_ROOT="$PACKAGE_ROOT/smoke_test"
SELECTED_DIR="$SMOKE_OUTPUT_ROOT/selected"
SMOKE_MANIFEST="$SELECTED_DIR/smoke_manifest.jsonl"
SELECTION_REPORT="$SELECTED_DIR/smoke_selection_report.json"
PACKAGE_REPORT="$SELECTED_DIR/v1_package_check.json"
DATA_ROOT="$SMOKE_OUTPUT_ROOT/data"
EXPERIMENT_ROOT="$SMOKE_OUTPUT_ROOT/experiments"
CHECKPOINT_DIR="$EXPERIMENT_ROOT/checkpoints"
LOG_DIR="$SMOKE_OUTPUT_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"
INFERENCE_DIR="$SMOKE_OUTPUT_ROOT/inference"
FINAL_REPORT_JSON="$SMOKE_OUTPUT_ROOT/smoke_test_report.json"
FINAL_REPORT_TXT="$SMOKE_OUTPUT_ROOT/smoke_test_report.txt"

CHECK_PACKAGE_SCRIPT="$SMOKE_PACKAGE_ROOT/check_v1_package.py"
BUILD_MANIFEST_SCRIPT="$SMOKE_PACKAGE_ROOT/build_smoke_manifest.py"
VALIDATE_OUTPUT_SCRIPT="$SMOKE_PACKAGE_ROOT/validate_smoke_outputs.py"
PREPARE_SCRIPT="$PACKAGE_ROOT/scripts/prepare_pause_training_data.py"
VALIDATE_LMDB_SCRIPT="$PACKAGE_ROOT/scripts/validate_pause_lmdb.py"
EXPORT_SCRIPT="$PACKAGE_ROOT/scripts/export_wav2vec_torchscript.py"
CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
TRAIN_SCRIPT="$CODE_ROOT/train.py"
INFER_SCRIPT="$CODE_ROOT/infer_pause_from_wav.py"

# 第 0 阶段：严格限制可清理的烟雾测试目录。
if [[ "$(basename "$SMOKE_OUTPUT_ROOT")" != "pause_v4_smoke_test" ]]; then
    echo "错误：SMOKE_OUTPUT_ROOT 的目录名必须为 pause_v4_smoke_test：$SMOKE_OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$SMOKE_OUTPUT_ROOT" == "/" || "$SMOKE_OUTPUT_ROOT" == "$PACKAGE_ROOT" ]]; then
    echo "错误：拒绝使用危险的烟雾测试输出路径：$SMOKE_OUTPUT_ROOT" >&2
    exit 1
fi
if [[ "$RESET_SMOKE_OUTPUT" != "0" && "$RESET_SMOKE_OUTPUT" != "1" ]]; then
    echo "错误：RESET_SMOKE_OUTPUT 只能为 0 或 1" >&2
    exit 1
fi
if [[ "$RESET_SMOKE_OUTPUT" == "1" ]]; then
    echo "[0/8] 清理独立烟雾测试输出"
    for child in selected data experiments logs inference; do
        rm -rf -- "$SMOKE_OUTPUT_ROOT/$child"
    done
    rm -f -- "$FINAL_REPORT_JSON" "$FINAL_REPORT_TXT"
fi
mkdir -p "$SELECTED_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$INFERENCE_DIR"

# 第 1 阶段：先确认内网 V4 运行树、正式 manifest 和脚本均完整。
echo "[1/8] 检查内网 predict_pause_from_wav_V4 实验包完整性"
python "$CHECK_PACKAGE_SCRIPT" \
    --package-root "$PACKAGE_ROOT" \
    --expected-arch "pause_from_wav_w2v_unfreeze_last2" \
    --manifest "$MERGED_MANIFEST" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --check-fairseq-registration \
    --report "$PACKAGE_REPORT"

# 第 2 阶段：确定性选择短音频，并预演与正式转换器完全相同的 count 划分。
echo "[2/8] 从正式提取结果中抽取极小平衡数据"
python "$BUILD_MANIFEST_SCRIPT" \
    --input-manifest "$MERGED_MANIFEST" \
    --output-manifest "$SMOKE_MANIFEST" \
    --output-report "$SELECTION_REPORT" \
    --expected-dir "$SELECTED_DIR" \
    --train-count "$TRAIN_COUNT" \
    --valid-count "$VALID_COUNT" \
    --test-count "$TEST_COUNT" \
    --split-seed "$SPLIT_SEED" \
    --max-audio-seconds "$MAX_AUDIO_SECONDS" \
    --sample-rate 24000

# 第 3 阶段：复用正式转换器生成 8/4/4 的 16 kHz PCM16 LMDB。
echo "[3/8] 生成烟雾测试 train/valid/test LMDB"
python "$PREPARE_SCRIPT" \
    --merged-manifest "$SMOKE_MANIFEST" \
    --output-root "$DATA_ROOT" \
    --split-mode count \
    --train-count "$TRAIN_COUNT" \
    --valid-count "$VALID_COUNT" \
    --test-count "$TEST_COUNT" \
    --split-seed "$SPLIT_SEED" \
    --source-sample-rate 24000 \
    --target-sample-rate 16000 \
    --map-size-gb 8

# 第 4 阶段：逐条回读三个 LMDB，检查 PCM、词窗口、标签和 mask。
echo "[4/8] 校验烟雾测试 LMDB"
python "$VALIDATE_LMDB_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --sample-rate 16000 \
    --splits train,valid,test

# 第 5 阶段：缺少永久资产时在旧环境导出，再切回训练环境做跨版本校验。
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
    if [[ ! -f "$OLD_CONDA_ACTIVATE" || ! -d "$ORIGINAL_WAV2VEC_PROJECT" || ! -s "$WAV2VEC_CHECKPOINT" ]]; then
        echo "错误：wav2vec 永久资产缺失，且旧环境/原工程/checkpoint 不完整" >&2
        exit 1
    fi
    echo "[5/8] 在原 Python 3.6 环境导出 wav2vec TorchScript"
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
else
    echo "[5/8] 复用已有 wav2vec TorchScript 永久资产"
fi

set +u
source "$NEW_CONDA_ACTIVATE" "$NEW_CONDA_ENV"
set -u
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EXPORT_SCRIPT" \
    --output-model "$WAV2VEC_JIT_PATH" \
    --output-meta "$WAV2VEC_META_PATH" \
    --sample-rate 16000 \
    --feature-dim 768 \
    --validate-only

# 第 6 阶段：只训练 1 个 epoch，checkpoint 和日志均写入独立目录。
echo "[6/8] 运行在线 wav2vec + Linear 极小训练"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python -u "$TRAIN_SCRIPT" "$DATA_ROOT" \
    --user-dir "$USER_DIR" \
    --save-dir "$CHECKPOINT_DIR" \
    --task pause_prediction \
    --arch pause_from_wav_w2v_unfreeze_last2 \
    --unfreeze-last-n 2 \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --input-sample-rate 16000 \
    --criterion pause_bce_loss \
    --optimizer pause_adam \
    --wav2vec-lr 0.00001 \
    --adam-betas '(0.9,0.997)' \
    --lr-scheduler fixed \
    --lr 0.001 \
    --positive-weight 1.0 \
    --pause-threshold 0.5 \
    --best-checkpoint-metric pause_recall \
    --maximize-best-checkpoint-metric \
    --max-epoch "$MAX_EPOCH" \
    --max-sentences "$MAX_SENTENCES" \
    --num-workers 0 \
    --log-format simple \
    --log-interval 1 \
    --save-interval 1 \
    --validate-interval 1 \
    --skip-invalid-size-inputs-valid-test \
    2>&1 | tee "$TRAIN_LOG"

# 第 7 阶段：使用最佳 checkpoint 对 4 条 test 样本推理。
echo "[7/8] 运行极小 test 推理"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    python "$INFER_SCRIPT" \
    --user-dir "$USER_DIR" \
    --checkpoint "$CHECKPOINT_DIR/checkpoint_best.pt" \
    --data-root "$DATA_ROOT" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --split test \
    --output-dir "$INFERENCE_DIR" \
    --threshold 0.5 \
    --batch-size "$MAX_SENTENCES" \
    --num-workers 0

# 第 8 阶段：检查划分、checkpoint、日志、逐词结果和有限指标。
echo "[8/8] 汇总并校验端到端输出"
python "$VALIDATE_OUTPUT_SCRIPT" \
    --selection-report "$SELECTION_REPORT" \
    --data-root "$DATA_ROOT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --train-log "$TRAIN_LOG" \
    --inference-dir "$INFERENCE_DIR" \
    --output-json "$FINAL_REPORT_JSON" \
    --output-txt "$FINAL_REPORT_TXT"

echo "SMOKE_TEST_PASS"
echo "完整性报告：$PACKAGE_REPORT"
echo "最终 JSON 报告：$FINAL_REPORT_JSON"
echo "最终 TXT 报告：$FINAL_REPORT_TXT"
