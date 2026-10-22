# Huper Phoneme Recognition

This directory provides offline CTC phoneme recognition. It accepts WAV audio and returns the phoneme sequence represented by the model labels. Inference converts audio to mono 16 kHz, loads a local Hugging Face `AutoProcessor` and `AutoModelForCTC`, and uses greedy decoding. It does not download model files from the network.

## Script Overview

- `scripts/infer_huper.py`: infer on one WAV file or a WAV directory; writes a `.txt` file for each audio file.
- `scripts/infer_huper_to_tsv.py`: batch inference to a three-column TSV (`id`, `text`, `asr_phoneme_output`). WAV files are sorted by filename and paired line-by-line with the text file.
- `scripts/infer_huper_to_csv.py`: batch inference to CSV; an optional metadata CSV with `id` and `text` columns supplies transcript text.
- `run_single.sh` and `run_batch.sh`: convenience wrappers around `infer_huper.py`.
- `run_label0_50_csv.sh`: TSV inference wrapper. `run_huper_tsv.sh` is a legacy experiment wrapper whose defaults still contain old server paths; use the generic entry point instead.
- `run_ky.sh` and `submit_huper_ky.sh`: KY cluster submission scripts; they depend on cluster commands and original server configuration and are not intended for a regular local terminal.
- `check_offline.sh` and `scripts/check_env_and_model.py`: validate the offline environment and model files.
- `text_to_huper_phones.py`: convert English text to reference phoneme sequences (G2P) with the CMU pronunciation dictionary; it does not run the acoustic model. Configure its input and output paths in the script.

## Model and Environment

The default model directory is `models/phone_asr/model/huper_recognizer/`, but the model included in this repository is under `model/best_by_target_metrics_0623/`. It contains `model.safetensors`, configuration, processor files, and vocabulary. Pass this directory explicitly with `--model-dir`. Install compatible versions of `torch`, `transformers`, `numpy`, `soundfile`, and `scipy` (see `environment_gpu.yml` or `requirements_min.txt`). Use `cpu` if CUDA is unavailable.

## Direct Inference

Run single-file or directory inference from the repository root, explicitly selecting the model included here (the shortcut wrappers default to a different model directory name):

```bash
python models/phone_asr/scripts/infer_huper.py \
  --root models/phone_asr \
  --model-dir models/phone_asr/model/best_by_target_metrics_0623 \
  --input /path/to/audio.wav \
  --output-dir models/phone_asr/outputs \
  --device cpu
```

For batch TSV inference, WAV files are sorted by filename and paired with one transcript line each. The number of lines must match the number of WAV files:

```bash
python models/phone_asr/scripts/infer_huper_to_tsv.py \
  --root models/phone_asr \
  --model-dir models/phone_asr/model/best_by_target_metrics_0623 \
  --wav-dir /path/to/wavs --text-txt /path/to/sentences.txt \
  --output-txt /path/to/predictions.tsv --device cpu
```

See `python models/phone_asr/scripts/infer_huper_to_csv.py --help` for the CSV entry point options. The TSV entry point replaces the model's CTC word-boundary marker (`|`) with spaces.
