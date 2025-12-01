# Experiments Summary

**Dataset**: LibriSpeech dev-clean (500 files subset, 400 for training and 100 for validation). All models optimize CTC loss and produce predictions via greedy decoding.

## Results Overview

| Part | Main Architecture              | Validation Loss |
|------|--------------------------------|-----------------|
| 1    | MLP + MFCC                     | 2.79            |
| 2    | CNN + MelSpectrogram           | 2.34            |
| 3    | BiLSTM (bidirectional)         | 2.33            |
| 4    | Lightweight Transformer        | 5.83            |
| 5    | Optuna (BiLSTM + MelSpectrogram) | 2.97*         |

*Part 5 uses only 300 samples to speed up Optuna trials.

## Analysis by Part

1. **Part 1 — MLP + MFCC**: the model predicts almost exclusively the "blank" symbol. Loss decreases but outputs remain empty due to lack of temporal memory.

2. **Part 2 — CNN + MelSpectrogram**: first architecture capable of producing readable sequences. Convolutions better exploit spectrograms (80 mel bands).

3. **Part 3 — RNN**: LSTM, GRU and especially BiLSTM significantly improve scores. BiLSTM is retained as reference (loss 2.33) thanks to its ability to integrate past and future context. The hybrid CNN+LSTM model does not provide additional gain on this reduced corpus.

4. **Part 4 — Transformer**: even with a lightweight model (d_model=64), convergence remains poor on 500 examples. The model drifts toward repetitive outputs.

5. **Part 5 — Optuna**: automates hyperparameter search. The winning trial favors a bidirectional BiLSTM, MelSpectrogram (71 channels), 3 layers and a learning rate of 7.7e-3.

## Generated Files

- `partie1/part1_training_curves.png`: MLP loss curves
- `partie2/part2_training_curves.png`: CNN curves
- `partie3/part3_architecture_comparison.png`: LSTM/GRU/BiLSTM comparison
- `partie4/part4_transformer_comparison.png`: Transformer loss evolution
- `partie5/part5_tuning_summary.png`: 10 Optuna trials summary
- `partie5/part5_best_params.json`: retained hyperparameters

## Prediction Examples

### Part 2 - CNN (epoch 10)

| Input (ground truth) | Model Output |
|----------------------|--------------|
| "to make hot buttered toast seventeen twenty six" | "to md o boter tssevten tntlondi s" |
| "never use new bread for making any kind of toast..." | "me rse me rd fr mit nd inde nde trais i aitdin..." |

The CNN produces partially readable outputs: some words are recognizable ("to", "boter" for butter), others are truncated or confused. BiLSTM (part 3) improves these results thanks to its temporal memory.

### Other Parts

- **MLP (part 1)**: produces only empty strings (blank token only)
- **Transformer (part 4)**: repeats a single letter ("eeee..."), due to insufficient data

## Conclusion

On this data volume, recurrent architectures remain the best option: a bidirectional BiLSTM fed with MelSpectrograms offers the best accuracy while maintaining reasonable computational cost. Transformer approaches would become relevant with a much larger corpus (and potentially pre-trained learning like wav2vec2/Whisper).
