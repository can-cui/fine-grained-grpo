#!/usr/bin/env bash
set -euo pipefail

# KY cluster submit wrapper for huper29/huper_recognizer.
# Usage:
#   bash submit_huper_ky.sh
#
# Edit the default paths below before submitting a new test set.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 手动修改区：默认直接 bash submit_huper_ky.sh 会使用这几个路径 =====
default_wav_dir="input_wavs/huper_label0_50_renumbered"
default_text_txt="input_wavs/huper_label0_50_renumbered.txt"
default_output_txt="reports/huper_label0_50_native_outputs_simple.txt"
default_device="cuda"
# ======================================================================

username="${KY_USERNAME:-yhchen70}"
proID="${KY_PRO_ID:-2227}"
gpuType="${KY_GPU_TYPE:-TeslaV100-PCIE-16GB}"
reserved="${KY_RESERVED:-dlp3-tts-prompt-reserved}"
image="${KY_IMAGE:-reg.deeplearning.cn/ky/ubuntu-kaldi-image:latest}"
conda_env="${KY_CONDA_ENV:-phoneme_huper}"

if [[ $# -eq 0 ]]; then
  wav_dir="$default_wav_dir"
  text_txt="$default_text_txt"
  output_txt="$default_output_txt"
  device="$default_device"
elif [[ $# -ge 3 ]]; then
  wav_dir="$1"
  text_txt="$2"
  output_txt="$3"
  device="${4:-$default_device}"
else
  cat >&2 <<'EOF'
Usage:
  bash submit_huper_ky.sh

Optional override:
  bash submit_huper_ky.sh <wav_dir> <text_txt> <output_txt> [cuda|cpu]

Required:
  wav_dir     Directory containing wav files.
  text_txt    UTF-8 txt file; one line per wav after filename sort.
  output_txt  Output tab-separated txt file.

Optional:
  device      cuda or cpu. Default: cuda.

Overridable KY settings:
  KY_USERNAME   default: yhchen70
  KY_PRO_ID     default: 2227
  KY_GPU_TYPE   default: TeslaV100-PCIE-16GB
  KY_RESERVED   default: dlp3-tts-prompt-reserved
  KY_IMAGE      default: reg.deeplearning.cn/ky/ubuntu-kaldi-image:latest
  KY_CONDA_ENV  default: phoneme_huper
EOF
  exit 2
fi

if [[ "$device" != "cuda" && "$device" != "cpu" ]]; then
  echo "device must be cuda or cpu: $device" >&2
  exit 2
fi

timestamp="$(date "+%Y%m%d%H%M%S")"
job_name="huper-asr-${timestamp}"
log_dir="${ROOT}/logs/ky_huper/${timestamp}"
submit_dir="${ROOT}/submit/ky_huper/${timestamp}"
job_script="${submit_dir}/run_huper_job.sh"

if [[ "$output_txt" = /* ]]; then
  output_parent="$(dirname "$output_txt")"
else
  output_parent="$(dirname "${ROOT}/${output_txt}")"
fi

mkdir -p "$log_dir" "$submit_dir" "$output_parent"

cat > "$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$ROOT"

export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$ROOT/hf_cache/transformers"
export TORCH_HOME="$ROOT/torch_cache"
export TMPDIR="$ROOT/temp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "\$HF_HOME" "\$HF_HUB_CACHE" "\$TRANSFORMERS_CACHE" "\$TORCH_HOME" "\$TMPDIR" "$(dirname "$output_txt")"

echo "project root: $ROOT"
echo "wav dir: $wav_dir"
echo "text txt: $text_txt"
echo "output txt: $output_txt"
echo "device: $device"
echo "conda env: $conda_env"
date

if [[ -n "$conda_env" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
  elif [[ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found. Set KY_CONDA_ENV empty if python is already active in the KY image." >&2
    exit 1
  fi
  conda activate "$conda_env"
fi

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

bash "$ROOT/run_huper_tsv.sh" "$wav_dir" "$text_txt" "$output_txt" "$device"

date
echo "done: $output_txt"
EOF

chmod +x "$job_script"

echo "KY job script: $job_script"
echo "KY log dir: $log_dir"

ky exp submit \
  -a "$username" \
  -n "$job_name" \
  -d "huper-asr" \
  -i "$image" \
  -e "$job_script" \
  --useGpu \
  -g 1 \
  -k "$gpuType" \
  -t PtJob \
  -l "$log_dir/${job_name}.log" \
  -o "$log_dir/${job_name}_error.log" \
  --proID "$proID" \
  --modelPath "$log_dir" \
  --modelName "$timestamp" \
  -r "$reserved"
