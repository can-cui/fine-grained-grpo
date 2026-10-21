# wav2vec Runtime Assets

This directory does not include model binaries. The first run of `sh02_train_pause_from_wav.sh` or
`sh03_infer_pause_from_wav.sh` automatically generates the following files from the original `wav2vec_small.pt`:

```text
wav2vec_small_last_layer_jit.pt
wav2vec_small_last_layer_jit.meta.json
```

The TorchScript model returns only the final wav2vec layer features `[B,T,768]` and the number of valid frames. Its metadata records the SHA256 hashes of the original checkpoint and TorchScript model, the sample rate, convolution kernel/stride, and export environment. Validation must pass before training, resuming, or inference. These assets are not offline per-WAV feature files.
