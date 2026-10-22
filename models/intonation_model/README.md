# Intonation Model (3 Classes)

The inference experiment is in `exp8_span_tone_3class/`. It aligns audio with its transcript at the word level and predicts one of three tones: `rising`, `falling`, or `flat`. Words below all configured confidence thresholds are rejected and omitted from the output.

## Script Overview

- `exp8_span_tone_3class/infer_threshold_3class.py`: inference entry point. It reads an `id + sentence` text file and matching WAV files, runs WhisperX forced alignment, extracts filterbank features, and predicts word-level tones.
- `run_filtre_whispex_pred_3class.sh`: convenience wrapper for inference. It accepts the text file, WAV directory, output file, and three thresholds. Its model paths still point to an old server; update them before using the wrapper, or call the Python entry point directly.
- `reward_model3c.py`: neural network definition used by the experiment.
- `config3c.py` and `data_utils.py`: model configuration, preprocessing, tokenization, and feature utilities.
- `test_only.py` and `test_only_correctcases.py`: evaluation on configured test sets and analysis of correct cases. These are not general-purpose inference entry points; test data paths are configured in the scripts.
- `sweep_filter_dataset_3class.py` and `sweep_filter_dataset_3class_with_errors.py`: sweep thresholds over existing predictions or labeled data and produce filtered data or error analysis.
- `intonation_v4_0520_exp8_span_tone_3class_0622/`: model artifacts, including a checkpoint, `vocab.json`, and `stats.npz`.
- `whisperX-main/`: WhisperX source used for alignment. Inference also requires its dependencies and model resources.

## Direct Inference

Install PyTorch, NumPy, WhisperX, and the experiment dependencies. Each text line must contain `audio_id<TAB>English sentence`; the matching audio file must be named `<audio_id>.wav`.

There is currently a source import mismatch that prevents the inference script from running as-is: `infer_threshold_3class.py` imports `config` and `model`, while this directory contains `config3c.py` and `reward_model3c.py`. Align these module names or provide compatibility modules first. The default feature extractor path also points to an old server; pass `--extract-script` with the path to an available filterbank extractor. After addressing these source and path requirements, run:

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

Adjust thresholds for the use case. The TSV output columns are `id`, `word`, `tone_pred`, and `P(pred)`. Add `--output-json /path/to/predictions.json` to write detailed JSON output. Replace model and data paths with their deployment locations.
