# Pause from WAV V4

本目录实现基于 wav2vec 表征的词级停顿二分类。Fairseq 模型输出每个词的停顿概率，再用阈值（默认 `0.5`）转为类别。当前正式推理入口以已准备好的 pause LMDB split 为输入，而不是直接接收任意 WAV 文件夹；LMDB 中包含音频特征、词边界及标签/元数据。

## 主要脚本

- `iflytek-tts-exp_fbsong/infer_pause_from_wav.py`：单个指定 split 的底层推理入口；输出 JSONL、TSV 和指标 JSON。
- `sh03_infer_pause_from_wav.sh`：正式推理包装脚本；按默认配置遍历六个 valid/test split，固定阈值 0.50。脚本开头的 `PACKAGE_ROOT`、`DATA_ROOT`、`EXPERIMENT_ROOT`、conda 环境和 GPU 设置目前均为旧服务器路径，运行前需要替换为实际路径。
- `sh01_prepare_pause_training_data.sh`：验证共享的训练/评估数据集，不生成新数据。
- `sh02_train_pause_from_wav.sh`、`run_pause_from_wav_train_ky.sh`、`ky_pause_from_wav_train.sh`：训练及集群提交脚本。
- `sh06_scan_pause_thresholds.sh`：在评估集上扫描阈值；`sh07`、`sh08`、`sh09`：检查点系列评估、句子准确率评估和指标汇总。
- `scripts/prepare_*.py`、`batch_extract_pause_data_from_lmdb.py`、`merge_pause_data_parts.py`：数据准备、提取与合并。
- `scripts/validate_*.py`、`smoke_test/`：数据、运行文件及推理结果校验/冒烟检查。
- `scripts/export_wav2vec_torchscript.py`：导出 wav2vec TorchScript 推理资源；`count_pause_utterance_ratio.py`：统计数据比例。
- `iflytek-tts-exp_fbsong/fairseq/`、`pause_user_dir/`：模型需要的 Fairseq 代码和任务/模型注册模块。

## 直接推理

### 使用六个官方 split

确认 pause LMDB、checkpoint、wav2vec TorchScript 文件、metadata、Fairseq 运行依赖和 conda 环境都已准备好；然后编辑 `sh03_infer_pause_from_wav.sh` 顶部配置并运行：

```bash
bash models/predict_pause_from_wav_V4/sh03_infer_pause_from_wav.sh
```

结果写入实验目录下的 `inference/checkpoint_best_threshold_0.50/`，每个 split 包含：

- `<split>_pause_predictions.jsonl`：按句保存词级预测。
- `<split>_pause_predictions.tsv`：逐词概率、预测、目标、有效标记及时间范围。
- `<split>_pause_metrics.json`：基于有效标记计算的分类指标。

### 对单个 LMDB split 推理

可直接调用 Python 入口（请按部署位置替换参数）：

```bash
cd models/predict_pause_from_wav_V4/iflytek-tts-exp_fbsong
python infer_pause_from_wav.py \
  --user-dir ../pause_user_dir \
  --checkpoint /path/to/checkpoint.pt \
  --data-root /path/to/pause_lmdb_root \
  --wav2vec-jit-path ../assets/wav2vec_small_last_layer_jit.pt \
  --wav2vec-meta-path ../assets/wav2vec_small_last_layer_jit.meta.json \
  --split test_m2_manual \
  --output-dir /path/to/output \
  --threshold 0.5 --batch-size 8 --num-workers 0
```

加 `--cpu` 可强制使用 CPU。当前目录的 `checkpoints/` 中虽有 `checkpoint_last.pt`，包装脚本默认寻找的是外部实验目录下的 `checkpoint_best.pt`；请按实际模型选择检查点。由于该底层脚本的数据接口是 Fairseq pause task/LMDB，任意 WAV 文件不能直接作为 `--data-root` 使用；需要先按本项目数据格式构建 LMDB split。
