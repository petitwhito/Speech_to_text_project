"""
Fine-tuning wav2vec2 on LibriSpeech (streaming mode - no disk storage)
Based on HuggingFace tutorial: https://huggingface.co/blog/fine-tune-wav2vec2-english
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np


MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./wav2vec2-finetuned"
MAX_TRAIN_SAMPLES = 60
MAX_EVAL_SAMPLES = 13


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch, processor):
    """Prepare a single batch for training."""
    audio = batch["audio"]
    
    batch["input_values"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # Encode text using tokenizer directly
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch


def compute_metrics(pred, processor, metric):
    """Compute WER metric."""
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    print(f"Loading model: {MODEL_NAME}")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    model.freeze_feature_encoder()
    
    # Load dataset (dummy version ~10MB, not the full 25GB dataset)
    print("Loading LibriSpeech dummy dataset...")
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation"
    )
    
    # Take samples
    total_samples = min(len(dataset), MAX_TRAIN_SAMPLES + MAX_EVAL_SAMPLES)
    print(f"Using {total_samples} samples...")
    samples = [dataset[i] for i in range(total_samples)]
    
    # Create train/eval splits
    train_data = Dataset.from_list(samples[:MAX_TRAIN_SAMPLES])
    eval_data = Dataset.from_list(samples[MAX_TRAIN_SAMPLES:])
    
    # Cast audio to 16kHz
    train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
    eval_data = eval_data.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")
    
    # Preprocess
    print("Preprocessing...")
    train_data = train_data.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_data.column_names
    )
    eval_data = eval_data.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_data.column_names
    )
    
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = evaluate.load("wer")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=50,
        max_steps=200,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    trainer = Trainer(
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
    
    input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values
    
    if torch.cuda.is_available():
        input_values = input_values.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    print(f"Ground truth: {test_sample['text']}")
    print(f"Prediction:   {transcription}")


if __name__ == "__main__":
    main()
