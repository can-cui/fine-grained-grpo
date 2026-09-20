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
    └── ...
```

### `models/`

This directory is **reserved for model-related resources** used in the experiments.

### `rewards/`

This directory is **reserved for reward-model resources** used for fine-grained reinforcement learning, including:

- Phoneme-level pronunciation reward model
- Word-level intonation reward model
- Word-boundary-level pause reward model
- Corresponding pretrained model weights

### `test/`

This directory contains the manually constructed evaluation sets used to assess fine-grained speech generation quality, including test sets for:

- Pronunciation
- Final-consonant pronunciation
- Intonation
- Pause placement
- Speech naturalness

## Reward Computation

The repository also provides the implementation and resources for computing fine-grained rewards at the word level.

The reward signals are designed to evaluate different linguistic aspects of synthesized speech:

- **Pronunciation reward**: evaluates phoneme-level pronunciation accuracy.
- **Intonation reward**: evaluates word-level intonation patterns.
- **Pause reward**: evaluates pause placement at word boundaries.

These localized rewards are aligned with the corresponding words and used as fine-grained supervision during GRPO optimization.

## Notes

This repository is intended to facilitate research on **fine-grained reinforcement learning for text-to-speech synthesis**, particularly for improving pronunciation, intonation, and pause quality through linguistically informed reward modeling.

More details about the methodology, experimental settings, and results can be found in the accompanying paper.
