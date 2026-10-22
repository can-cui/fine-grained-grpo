# Huper 音素识别

本目录提供离线 CTC 音素识别：输入 WAV，输出模型标签对应的音素序列。推理会把音频转成单声道 16 kHz，加载本地 Hugging Face `AutoProcessor` 和 `AutoModelForCTC`，执行贪心解码；无需联网下载模型。

## 脚本用途

- `scripts/infer_huper.py`：单个 WAV 或 WAV 目录推理；每个音频输出一个同名 `.txt`。
- `scripts/infer_huper_to_tsv.py`：批量推理并输出 `id / text / asr_phoneme_output` 三列 TSV；音频按文件名排序，文本文件须逐行对应。
- `scripts/infer_huper_to_csv.py`：批量输出 CSV，可用含 `id,text` 列的 metadata CSV 填入文本。
- `run_single.sh`、`run_batch.sh`：`infer_huper.py` 的快捷包装。
- `run_label0_50_csv.sh`：TSV 推理包装；`run_huper_tsv.sh` 是旧实验快捷入口，默认参数仍含旧服务器绝对路径，建议直接使用通用包装。
- `run_ky.sh`、`submit_huper_ky.sh`：KY 集群提交脚本，依赖集群命令和原服务器配置，不适用于普通本机终端。
- `check_offline.sh`、`scripts/check_env_and_model.py`：检查离线环境/模型文件。
- `text_to_huper_phones.py`：使用 CMU 发音词典将英文文本转为参考音素序列（G2P），不运行声学模型；输入输出路径需在脚本常量中配置。

## 准备模型与环境

默认模型目录为 `models/phone_asr/model/huper_recognizer/`，但当前仓库实际提供的模型目录名为 `model/best_by_target_metrics_0623/`。后者包含 `model.safetensors`、配置、processor 和词表；运行时请通过 `--model-dir` 显式指定该目录。需安装兼容版本的 `torch`、`transformers`、`numpy`、`soundfile` 和 `scipy`（可参考 `environment_gpu.yml` 或 `requirements_min.txt`）。GPU 不可用时传入 `cpu`。

## 直接推理

从仓库根目录直接运行单个文件或目录。请显式传入本仓库收录的模型目录（快捷脚本默认使用另一个目录名）：

```bash
python models/phone_asr/scripts/infer_huper.py \
  --root models/phone_asr \
  --model-dir models/phone_asr/model/best_by_target_metrics_0623 \
  --input /path/to/audio.wav \
  --output-dir models/phone_asr/outputs \
  --device cpu
```

批量 TSV 推理（WAV 文件名排序后与文本逐行配对，行数必须一致）：

```bash
python models/phone_asr/scripts/infer_huper_to_tsv.py \
  --root models/phone_asr \
  --model-dir models/phone_asr/model/best_by_target_metrics_0623 \
  --wav-dir /path/to/wavs --text-txt /path/to/sentences.txt \
  --output-txt /path/to/predictions.tsv --device cpu
```

CSV 入口参数可用 `python models/phone_asr/scripts/infer_huper_to_csv.py --help` 查看。输出音素使用模型原生标签；TSV 入口会将 CTC 词边界符 `|` 替换为空格。
