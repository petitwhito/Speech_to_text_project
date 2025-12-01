# Section 2: Fine-tuning Pre-trained Models

Fine-tuning HuggingFace speech models on LibriSpeech dummy dataset.

## Results

| Model | Final WER | Training Time |
|-------|-----------|---------------|
| Whisper-small | **20.41%** | ~3 min |
| wav2vec2-base | 100% | ~1 min |

Whisper converges well because it's already pre-trained for ASR. wav2vec2-base is only an audio encoder without ASR head, requiring much more data to learn transcription from scratch.

## Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `finetune_whisper.py` | openai/whisper-small | Seq2Seq fine-tuning |
| `finetune_wav2vec2.py` | facebook/wav2vec2-base | CTC fine-tuning |

## Usage

```bash
# With uv (recommended)
uv run python finetune_whisper.py
uv run python finetune_wav2vec2.py

# Or with activated venv
source ../.venv/bin/activate
python finetune_whisper.py
```

## Configuration

Both scripts use 73 samples from LibriSpeech dummy (60 train / 13 eval) and train for 200 steps.

Key parameters (editable in scripts):
- `MODEL_NAME`: pre-trained model to fine-tune
- `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES`: dataset size
- `OUTPUT_DIR`: where to save the fine-tuned model

## Example Output (Whisper)

```
Ground truth: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
Prediction:   MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
```

## Resources

- [Fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper)
- [Fine-tune wav2vec2](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [HF Audio Course](https://huggingface.co/learn/audio-course/)
