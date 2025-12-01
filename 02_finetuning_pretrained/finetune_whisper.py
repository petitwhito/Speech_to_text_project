"""
Fine-tuning Whisper on LibriSpeech
Based on HuggingFace tutorial: https://huggingface.co/blog/fine-tune-whisper
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "en"
TASK = "transcribe"
OUTPUT_DIR = "./whisper-finetuned"
MAX_TRAIN_SAMPLES = 60
MAX_EVAL_SAMPLES = 13


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech seq2seq with padding."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """Extract features and encode labels."""
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def compute_metrics(pred, processor, metric):
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    print(f"Loading model: {MODEL_NAME}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    print("Loading LibriSpeech validation split...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    total_samples = min(len(dataset), MAX_TRAIN_SAMPLES + MAX_EVAL_SAMPLES)
    print(f"Using {total_samples} samples...")
    samples = [dataset[i] for i in range(total_samples)]

    train_data = Dataset.from_list(samples[:MAX_TRAIN_SAMPLES])
    eval_data = Dataset.from_list(samples[MAX_TRAIN_SAMPLES:])

    train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
    eval_data = eval_data.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    print("Preprocessing...")
    train_data = train_data.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_data.column_names
    )
    eval_data = eval_data.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_data.column_names
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=200,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, wer_metric),
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)

    print("Final evaluation...")
    results = trainer.evaluate()
    print(f"Final WER: {results['eval_wer']:.2f}%")

    # Test inference
    print("\nTest inference:")
    test_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
    test_sample = test_dataset[0]

    import librosa
    audio_array = test_sample["audio"]["array"]
    sr = test_sample["audio"]["sampling_rate"]
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

    if torch.cuda.is_available():
        input_features = input_features.cuda()
        model = model.cuda()

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"Ground truth: {test_sample['text']}")
    print(f"Prediction:   {transcription}")


if __name__ == "__main__":
    main()

