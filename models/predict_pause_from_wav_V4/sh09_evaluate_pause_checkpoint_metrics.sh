#!/bin/bash
set -euo pipefail

# ==================== 修改区 ====================
PACKAGE_ROOT="/train29/tts/permanent/yhchen70/yrfs5/projects/En_phrase/En-L2L3predict/pre_l123_from_wav-master/variation/predict_pause_from_wav_V4"
DATA_ROOT="/ng-mix02/tts/permanent/yhchen70/data/extracted_pause_training_data/pause_lmdb_v2_shared_dual_eval"
PHRASE_MARKED_TXT="/ng-mix02/tts/permanent/yhchen70/data/gemini_fake_yewu_6000/2000sentences_marked.txt"
EXPERIMENT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2"
OUTPUT_ROOT="/ng-mix02/tts/permanent/yhchen70/model/pause_v4_experiments/pause_from_wav_w2v_unfreeze_last2/checkpoint_metrics"
CHECKPOINTS="5 10 15 20"  # 后续例如改为：5 10 15 20 23
OVERWRITE_CACHE="1"  # 仅在更换阈值或需要全部重算时改为 1
THRESHOLD="0.50"
BATCH_SIZE="8"
NUM_WORKERS="0"
CUDA_VISIBLE_DEVICES_VALUE="0"  # 改为当前空闲 GPU
CONDA_ACTIVATE="/home/tts/cancui11/miniconda3/bin/activate"
CONDA_ENV="torch20_fair_fromyjdong4"
# ===============================================

set +u
source "$CONDA_ACTIVATE" "$CONDA_ENV"
set -u

CODE_ROOT="$PACKAGE_ROOT/iflytek-tts-exp_fbsong"
USER_DIR="$PACKAGE_ROOT/pause_user_dir"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
EVALUATE_SCRIPT="$PACKAGE_ROOT/scripts/evaluate_pause_checkpoint_metrics.py"
WAV2VEC_JIT_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.pt"
WAV2VEC_META_PATH="$PACKAGE_ROOT/assets/wav2vec_small_last_layer_jit.meta.json"
read -r -a CHECKPOINT_ARRAY <<< "$CHECKPOINTS"
EXTRA_ARGS=()
if [[ "$OVERWRITE_CACHE" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite-cache)
fi

[[ -d "$CHECKPOINT_ROOT" ]] || { echo "错误：找不到 CHECKPOINT_ROOT：$CHECKPOINT_ROOT" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "错误：找不到 DATA_ROOT：$DATA_ROOT" >&2; exit 2; }
[[ -f "$PHRASE_MARKED_TXT" ]] || { echo "错误：找不到 PHRASE_MARKED_TXT：$PHRASE_MARKED_TXT" >&2; exit 2; }
[[ -f "$EVALUATE_SCRIPT" ]] || { echo "错误：找不到评测脚本：$EVALUATE_SCRIPT" >&2; exit 2; }
[[ -s "$WAV2VEC_JIT_PATH" && -s "$WAV2VEC_META_PATH" ]] || { echo "错误：缺少 wav2vec 资产" >&2; exit 2; }
[[ ${#CHECKPOINT_ARRAY[@]} -gt 0 ]] || { echo "错误：请填写 CHECKPOINTS" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$EVALUATE_SCRIPT" \
    --user-dir "$USER_DIR" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --data-root "$DATA_ROOT" \
    --phrase-marked-txt "$PHRASE_MARKED_TXT" \
    --wav2vec-jit-path "$WAV2VEC_JIT_PATH" \
    --wav2vec-meta-path "$WAV2VEC_META_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --epochs "${CHECKPOINT_ARRAY[@]}" \
    --threshold "$THRESHOLD" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"

echo "完成：$OUTPUT_ROOT"
