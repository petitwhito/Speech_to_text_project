# Speech-To-Text TP

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)
![Speech Recognition](https://img.shields.io/badge/field-Speech%20Recognition-brightgreen)
![Status](https://img.shields.io/badge/status-Completed-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Complete ASR pipeline: architectures from scratch then fine-tuning of pre-trained models.

## Structure

```
Speech_to_text_project/
├── 01_architectures_from_scratch/   # Section 1: MLP, CNN, RNN, Transformer, Optuna
│   ├── partie1/                     # MLP + MFCC + CTC
│   ├── partie2/                     # CNN + MelSpectrogram
│   ├── partie3/                     # LSTM / GRU / BiLSTM
│   ├── partie4/                     # Transformer
│   ├── partie5/                     # Hyperparameter tuning (Optuna)
│   └── RESULTS.md                   # Section 1 results
│
├── 02_finetuning_pretrained/        # Section 2: HuggingFace fine-tuning
│   ├── finetune_whisper.py          # Whisper fine-tuning
│   ├── finetune_wav2vec2.py         # wav2vec2 fine-tuning
│   └── RESULTS.md                   # Section 2 results
│
├── pyproject.toml                   # Dependencies (uv)
├── README.md
└── TP STT.pdf                       # Assignment
```

## Installation

```bash
# With uv (recommended)
uv sync

# Run scripts directly with uv
uv run python 02_finetuning_pretrained/finetune_whisper.py

# Or activate the venv
source .venv/bin/activate
python 02_finetuning_pretrained/finetune_whisper.py
```

## Section 1: Architectures from scratch

Comparison of different STT architectures with CTC loss on LibriSpeech dev-clean.

| Part | Architecture | Val Loss |
|------|--------------|----------|
| 1 | MLP + MFCC | 2.79 |
| 2 | CNN + MelSpectrogram | 2.34 |
| 3 | BiLSTM | 2.33 |
| 4 | Transformer | 5.83 |
| 5 | Optuna tuning | 2.97 |

Best model: **BiLSTM** with MelSpectrogram.

Details in `01_architectures_from_scratch/RESULTS.md`.

## Section 2: Fine-tuning pre-trained models

Fine-tuning of HuggingFace models on LibriSpeech dummy dataset (73 samples).

| Model | Final WER | Notes |
|-------|-----------|-------|
| Whisper-small | **20.41%** | Pre-trained for ASR, converges fast |
| wav2vec2-base | 100% | Needs more data (encoder only, no ASR head) |

Example prediction (Whisper):
```
Ground truth: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
Prediction:   MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
```

Scripts in `02_finetuning_pretrained/`.

## Resources

- [HF Audio Course](https://huggingface.co/learn/audio-course/)
- [Fine-tune wav2vec2](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)

## License

MIT License
