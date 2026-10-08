# Linguistically Informed Fine-Grained Reinforcement Learning for TTS

This repository provides the resources accompanying the paper:

> **Linguistically Informed Fine-Grained Reinforcement Learning with Word-Level Optimization for Text-to-Speech**

The repository collects the key resources used in the fine-grained reinforcement learning experiments, including **reward models and their pretrained weights, reward computation procedures, and manually constructed evaluation sets**.

## Repository Contents

```text
Fine-Grained-GRPO/
├── models/
│   ├── intonation_model/
│   ├── phone_asr/
│   └── predict_pause_from_wav_V4/
├── rewards/
│   ├── asr_phone_norm_marked.py
│   ├── asr_phone_norm_marked_token.py
│   ├── asr_phone_norm_marked_token_speed.py
│   ├── intonation_prob_3class.py
│   ├── intonation_prob_3class_token.py
│   ├── predL2_whisper_pause.py
│   └── predL2_whisper_pause_token.py
└── test/
    ├── intonation/
    │   ├── tone80.txt
    │   └── tone80_label.txt
    ├── mos/
    │   └── mos50.txt
    ├── pause/
    │   ├── test_l2.txt
    │   └── test_l2_label.txt
    └── pronunciation/
        ├── test_280.txt
        └── test280_label.txt
```

### `models/`

This directory contains the pretrained weights of the reward models used in the experiments.

The available models include:

- `intonation_model/`: the word-level intonation reward model.
- `phone_asr/`: the phoneme-level pronunciation reward model.
- `predict_pause_from_wav_V4/`: the word-boundary pause reward model.

The model weights required by the reward computation scripts are provided in the corresponding directories.

### `rewards/`

The reward computation scripts cover the following GRPO strategies evaluated in the paper:

| GRPO Model | GRPO Strategy | Reward Script | Reward Model |
|---|---|---|---|
| Phone ASR-GRPO | Sent-level | `asr_phone_norm_marked.py` | `../models/phone_asr/` |
| Phone ASR-GRPO | Word-level | `asr_phone_norm_marked_token.py` | `../models/phone_asr/` |
| Phone ASR-GRPO | Word-level speedy | `asr_phone_norm_marked_token_speed.py` | `../models/phone_asr/` |
| Intonation-GRPO | Sent-level | `intonation_prob_3class.py` | `../models/intonation_model/` |
| Intonation-GRPO | Word-level | `intonation_prob_3class_token.py` | `../models/intonation_model/` |
| Pause-GRPO | Sent-level | `predL2_whisper_pause.py` | `../models/predict_pause_from_wav_V4/` |
| Pause-GRPO | Word-level | `predL2_whisper_pause_token.py` | `../models/predict_pause_from_wav_V4/` |

> **Note:** The sentence-level no-mask Whisper ASR baseline reported in the paper is not included in this directory. It uses the standard sentence-level ASR reward without target-aware masking.

---

## 1. Phone ASR-GRPO

The Phone ASR-GRPO reward evaluates pronunciation quality using a phoneme-level ASR model.

The corresponding reward model is located at:

```text
../models/phone_asr/
```

### Scripts

#### `asr_phone_norm_marked.py`

Implements the **sentence-level Phone ASR-GRPO** reward.

The reward is computed based on phoneme recognition results for the synthesized utterance. Target-aware masking can be used to focus the reward computation on the annotated target words.

#### `asr_phone_norm_marked_token.py`

Implements the **word/token-level Phone ASR-GRPO** reward.

The phoneme-level recognition results are aligned with target words, allowing the reward to be assigned to specific words rather than to the entire utterance.

#### `asr_phone_norm_marked_token_speed.py`

Implements the **word/token-level speedy Phone ASR-GRPO** reward.

This version follows the same word-level reward principle while using an accelerated reward computation procedure for more efficient processing.

### Correspondence with the Paper

| Script | Paper setting | Reward granularity |
|---|---|---|
| `asr_phone_norm_marked.py` | Sent-level | Sentence-level |
| `asr_phone_norm_marked_token.py` | Word-level | Word/token-level |
| `asr_phone_norm_marked_token_speed.py` | Word-level speedy | Word/token-level |

---

## 2. Intonation-GRPO

The Intonation-GRPO reward evaluates the intonation pattern of target words.

The corresponding intonation reward model is located at:

```text
../models/intonation_model/
```

The model predicts three intonation classes:

- Rising
- Falling
- Level

### Scripts

#### `intonation_prob_3class.py`

Implements the **sentence-level Intonation-GRPO** reward.

The intonation model predicts the intonation patterns of the relevant words, and the predictions are aggregated to obtain a sentence-level reward.

#### `intonation_prob_3class_token.py`

Implements the **word/token-level Intonation-GRPO** reward.

The reward is computed separately for target words, enabling fine-grained credit assignment based on word-level intonation quality.

### Correspondence with the Paper

| Script | Paper setting | Reward granularity |
|---|---|---|
| `intonation_prob_3class.py` | Sent-level | Sentence-level |
| `intonation_prob_3class_token.py` | Word-level | Word/token-level |

---

## 3. Pause-GRPO

The Pause-GRPO reward evaluates pause placement at word boundaries.

The corresponding pause prediction model is located at:

```text
../models/predict_pause_from_wav_V4/
```

The reward model predicts whether an appropriate pause occurs after a target word or word boundary.

### Scripts

#### `predL2_whisper_pause.py`

Implements the **sentence-level Pause-GRPO** reward.

The predicted pause information is aggregated to obtain a sentence-level reward.

#### `predL2_whisper_pause_token.py`

Implements the **word/token-level Pause-GRPO** reward.

The pause prediction is aligned with target word boundaries, allowing the reward to be assigned to individual target words.

### Correspondence with the Paper

| Script | Paper setting | Reward granularity |
|---|---|---|
| `predL2_whisper_pause.py` | Sent-level | Sentence-level |
| `predL2_whisper_pause_token.py` | Word-level | Word/token-level |

---

## 4. Reward Computation

The reward computation follows the linguistic aspect being optimized.

### Pronunciation Reward

The Phone ASR reward is based on phoneme-level recognition results. Compared with conventional word-level ASR rewards, phoneme-level recognition provides more fine-grained feedback for localized pronunciation errors.

For word-level optimization, the recognition results are aligned with the target words, and the reward is computed only for the annotated target regions.

### Intonation Reward

The intonation reward evaluates whether the synthesized speech exhibits the expected intonation pattern for each target word.

The target intonation can be one of:

```text
Rising
Falling
Level
```

For word-level optimization, rewards are computed independently for the annotated target words.

### Pause Reward

The pause reward evaluates whether an appropriate pause occurs at the target word boundary.

For word-level optimization, the reward is computed at individual word boundaries, enabling localized feedback for pause placement.

---

## 5. Relation to GRPO Experiments

The reward scripts correspond to the individual GRPO experiments reported in the paper.

```text
                         Fine-Grained GRPO
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
       Phone ASR-GRPO     Intonation-GRPO     Pause-GRPO
              │                 │                 │
       ┌──────┼──────┐      ┌────┴────┐      ┌────┴────┐
       │      │      │      │         │      │         │
      Sent   Word   Word   Sent      Word   Sent      Word
             │     Speedy
       │      │      │      │         │      │         │
       ▼      ▼      ▼      ▼         ▼      ▼         ▼
    asr_   asr_   asr_   intonation_ intonation_ predL2_ predL2_
    phone  phone  phone  prob_3class prob_3class whisper whisper
    norm_   norm_  norm_       .py       _token.py  _pause  _pause_
    marked  marked marked_                         .py     token.py
            token  token_speed
```

---

## 6. Evaluation Results

The following table summarizes the GRPO strategies evaluated in the paper.

| Model | GRPO Type | Seed-WER (%) ↓ | WER (%) ↓ | PER (%) ↓ | Phone Acc. (%) ↑ | Intonation Acc. (%) ↑ | No-Pause Acc. (%) ↑ | MOS ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline Models** | | | | | | | | |
| Before GRPO | – | 1.88 | 0.56 | 7.83 | 91.07 | 73.12 | 82 | 4.44 |
| Whisper ASR GRPO | Sent-level no mask | 1.85 | 0.44 | 7.41 | 91.79 | 74.15 | 84 | 4.46 |
| **Proposed Models** | | | | | | | | |
| **Phone ASR-GRPO** | Sent-level no mask | **1.72** | 0.51 | 7.26 | 91.53 | 69.75 | 86 | 4.46 |
| | Sent-level | 1.81 | 0.54 | 7.59 | 91.86 | 71.33 | 86 | 4.46 |
| | Word-level | 1.82 | **0.33** | 7.14 | 91.82 | 72.92 | 91 | 4.45 |
| | Word-level speedy | 1.85 | 0.39 | **7.06** | **92.14** | 69.74 | 86 | 4.46 |
| **Intonation-GRPO** | Sent-level | **1.88** | 0.50 | 7.27 | 91.78 | **80.89** | 89 | 4.45 |
| | Word-level | **1.71** | 0.39 | 7.37 | 91.07 | **79.21** | 88 | **4.47** |
| **Pause-GRPO** | Sent-level | 1.83 | 0.47 | 7.48 | 91.43 | 75.15 | **92** | 4.46 |
| | Word-level | 1.99 | 0.43 | 7.20 | 91.43 | 72.93 | **93** | 4.45 |
| **3 GRPO Fusion** | Sent-level | 1.88 | 0.36 | 7.27 | 91.74 | **79.52** | **92** | **4.47** |
| | Word-level | 1.87 | **0.35** | 7.36 | **92.36** | 77.29 | 91 | 4.46 |


### `test/`

This directory contains the manually constructed evaluation sets used to assess different aspects of speech generation quality.

Except for the MOS set, each evaluation set contains **test texts and their corresponding annotated texts**.

#### `intonation/`

Contains 80 test sentences for evaluating word-level intonation.

- `tone80.txt`: original test sentences.
- `tone80_label.txt`: test sentences annotated with prosodic boundaries where a rising or falling intonation is expected.

For example:

```text
After three hours of discussion↗, they finally reached a final decision on the matter↘.
```

Here, `↗` and `↘` indicate the expected rising and falling intonation, respectively.

#### `pause/`

Contains test sentences for evaluating pause placement.

- `test_l2.txt`: original test sentences.
- `test_l2_label.txt`: test sentences annotated with word groups where an inappropriate pause should not occur.

For example:

```text
One good month cannot 【make up for】 years of losses.
```

The brackets `【 】` mark the word group that should be produced without an internal pause.

#### `pronunciation/`

Contains 280 test sentences for evaluating pronunciation, with a particular focus on word-final consonants.

- `test_280.txt`: original test sentences.
- `test280_label.txt`: test sentences with the target words requiring particular attention to their final consonants marked by `【 】`.

For example:

```text
I knew the road【】was closed after the storm.
```

The brackets `【 】` indicate the word whose final consonant should receive particular attention during pronunciation evaluation.

#### `mos/`

Contains 50 sentences for evaluating the naturalness of synthesized speech.

- `mos50.txt`: the 50 sentences used for the MOS evaluation.

Unlike the other test sets, the MOS set does not contain additional word-level or boundary-level annotations.

### Notes

This repository is intended to facilitate research on **fine-grained reinforcement learning for text-to-speech synthesis**, particularly for improving pronunciation, intonation, and pause quality through linguistically informed reward modeling.

More details about the methodology, experimental settings, and results can be found in the accompanying paper.
