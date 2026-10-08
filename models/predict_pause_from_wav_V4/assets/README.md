# wav2vec 运行资产

本目录不预放模型二进制。首次运行 `sh02_train_pause_from_wav.sh` 或
`sh03_infer_pause_from_wav.sh` 时，会从原 `wav2vec_small.pt` 自动生成：

```text
wav2vec_small_last_layer_jit.pt
wav2vec_small_last_layer_jit.meta.json
```

TorchScript 只返回 wav2vec 最后一层 `[B,T,768]` 特征和有效帧数。元数据保存
原 checkpoint 与 TorchScript 的 SHA256、采样率、卷积 kernel/stride 和导出环境，
训练、恢复及推理前必须校验通过。不要把该资产理解为离线逐 WAV 特征文件。
