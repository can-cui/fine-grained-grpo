# 音调模型（3 类）

本目录里的可推理实验位于 `exp8_span_tone_3class/`。模型根据音频和对应文本，对齐到词级后预测 `rising`、`falling`、`flat` 三类音调。阈值以下的词会被拒绝并从结果中省略。

## 文件用途

- `exp8_span_tone_3class/infer_threshold_3class.py`：推理入口；输入 `id + 句子` 文本和同 ID 的 WAV，调用 WhisperX 强制对齐，再提取 fbank 并运行分类器。
- `run_filtre_whispex_pred_3class.sh`：上述推理脚本的便捷包装；可传文本、WAV 目录、输出文件和三类阈值。但脚本内模型路径仍是旧服务器绝对路径，需先改为本机路径，或直接调用 Python 入口。
- `reward_model3c.py`：音调模型网络结构定义，供训练/推理代码加载。
- `config3c.py`、`data_utils.py`：模型配置、数据预处理、分词和特征处理。
- `test_only.py`、`test_only_correctcases.py`：离线测试集评估及正确样例分析，不是通用单批推理入口；测试数据路径在代码内配置。
- `sweep_filter_dataset_3class.py`、`sweep_filter_dataset_3class_with_errors.py`：对已有预测/标注数据扫描阈值并生成筛选结果或错误分析。
- `intonation_v4_0520_exp8_span_tone_3class_0622/`：模型产物，目前包含 checkpoint、`vocab.json` 和 `stats.npz`。
- `whisperX-main/`：WhisperX 对齐源码；推理需要安装其依赖及模型资源。

## 直接推理

在仓库根目录安装 PyTorch、NumPy、WhisperX 及该实验需要的依赖。输入文本每行格式为 `音频ID<TAB>英文句子`，音频文件名需对应为 `<音频ID>.wav`。

当前脚本有一个会阻止直接运行的源码缺口：`infer_threshold_3class.py` 导入 `config` 和 `model`，而此目录中对应文件是 `config3c.py` 和 `reward_model3c.py`。因此需要先在代码中统一这些导入名（或提供兼容模块）；否则即使权重路径正确也会在导入阶段失败。推理脚本默认 fbank 提取器路径也指向旧服务器，需将 `--extract-script` 指向可用的 fbank 提取脚本。以下命令展示完成这些源码/路径准备后的调用方式：

```bash
python models/intonation_model/exp8_span_tone_3class/infer_threshold_3class.py \
  --text-path /path/to/ids_and_sentences.txt \
  --wav-dir /path/to/wavs \
  --fbk-dir /path/to/fbank_cache \
  --ckpt models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/checkpoints/best_exp8_span_tone_3class0625.pt \
  --vocab models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/vocab.json \
  --stats models/intonation_model/exp8_span_tone_3class/intonation_v4_0520_exp8_span_tone_3class_0622/stats.npz \
  --extract-script /path/to/get_filterbank_feature_24k.py \
  --output-txt /path/to/tone_predictions.tsv \
  --rising-threshold 0.98 --falling-threshold 0.98 --flat-threshold 0.97
```

阈值可按任务调整。输出 TSV 的列为 `id`、`word`、`tone_pred`、`P(pred)`；如需详细 JSON，可增加 `--output-json /path/to/predictions.json`。权重和数据路径均可替换为实际部署位置。
