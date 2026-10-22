# Pause from WAV V4

This directory implements word-level pause detection using wav2vec representations and a binary classifier. The Fairseq model produces a pause probability for each word, which is converted to a class using a threshold (default `0.5`). The formal inference entry point expects an already prepared pause LMDB split, not an arbitrary WAV directory. The LMDB contains audio features, word boundaries, labels, and metadata.

## Main Scripts

- `iflytek-tts-exp_fbsong/infer_pause_from_wav.py`: low-level inference entry point for one named split; writes JSONL predictions, TSV word-level results, and a metrics JSON file.
- `sh03_infer_pause_from_wav.sh`: formal inference wrapper; runs six configured validation/test splits with a fixed threshold of 0.50. Its `PACKAGE_ROOT`, `DATA_ROOT`, `EXPERIMENT_ROOT`, conda environment, and GPU settings currently point to an old server and must be updated before use.
- `sh01_prepare_pause_training_data.sh`: validates the shared training/evaluation dataset; it does not create data.
- `sh02_train_pause_from_wav.sh`, `run_pause_from_wav_train_ky.sh`, and `ky_pause_from_wav_train.sh`: training and cluster submission scripts.
- `sh06_scan_pause_thresholds.sh`: scans thresholds on evaluation data. `sh07`, `sh08`, and `sh09` evaluate checkpoint series, sentence accuracy, and summarize metrics.
- `scripts/prepare_*.py`, `batch_extract_pause_data_from_lmdb.py`, and `merge_pause_data_parts.py`: prepare, extract, and merge data.
- `scripts/validate_*.py` and `smoke_test/`: validate data, runtime files, and inference outputs.
- `scripts/export_wav2vec_torchscript.py`: export wav2vec TorchScript inference assets. `count_pause_utterance_ratio.py` summarizes data ratios.
- `iflytek-tts-exp_fbsong/fairseq/` and `pause_user_dir/`: Fairseq code and task/model registration modules required by the model.

## Direct Inference

### Run the six configured splits

Make sure the pause LMDB, checkpoint, wav2vec TorchScript files, metadata, Fairseq runtime dependencies, and conda environment are available. Update the configuration at the top of `sh03_infer_pause_from_wav.sh`, then run:

```bash
bash models/predict_pause_from_wav_V4/sh03_infer_pause_from_wav.sh
```

Results are written under `inference/checkpoint_best_threshold_0.50/` in the experiment directory. Each split produces:

- `<split>_pause_predictions.jsonl`: word-level predictions grouped by utterance.
- `<split>_pause_predictions.tsv`: word-level probability, prediction, target, validity flag, and time range.
- `<split>_pause_metrics.json`: classification metrics computed over valid labels.

### Run inference on one LMDB split

Call the Python entry point directly, replacing paths with the deployment paths:

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

Add `--cpu` to force CPU inference. The local `checkpoints/` directory contains `checkpoint_last.pt`, while the wrapper defaults to an external `checkpoint_best.pt`; select the checkpoint that matches your setup. The Python entry point uses a Fairseq pause task and LMDB dataset, so an arbitrary WAV directory cannot be passed as `--data-root`; prepare an LMDB split in the expected format first.
