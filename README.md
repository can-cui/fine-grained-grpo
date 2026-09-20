# Linguistically Informed Fine-Grained Reinforcement Learning for TTS

This repository provides the resources accompanying the paper:

> **Linguistically Informed Fine-Grained Reinforcement Learning with Word-Level Optimization for Text-to-Speech**

The repository collects the key resources used in the fine-grained reinforcement learning experiments, including **reward models and their pretrained weights, reward computation procedures, and manually constructed evaluation sets**.

## Repository Contents

```text
Fine-Grained-GRPO/
├── models/
│   └── .gitkeep
├── rewards/
│   └── .gitkeep
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

This directory is **reserved for reward models resources** used in the GRPO. The relevant resources will be added in future updates.

### `rewards/`

This directory is **reserved for reward calculations resources** used for GRPO. The relevant resources will be added in future updates.

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
