"""
Part 1: MLP + MFCC + CTC
Speech-to-Text implementation using Multi-Layer Perceptron with MFCC features and CTC loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class MFCCFeatureExtractor:
    """Extract MFCC features from audio."""
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': 40
            }
        )
    
    def extract(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        mfcc = self.mfcc_transform(waveform)
        return mfcc.squeeze(0).transpose(0, 1)


class TextEncoder:
    """Encode text to character indices."""
    
    def __init__(self):
        self.blank_idx = 0
        self.space_idx = 1
        self.chars = list('abcdefghijklmnopqrstuvwxyz')
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(self.chars)}
        self.char_to_idx[' '] = self.space_idx
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[self.blank_idx] = '<blank>'
        self.vocab_size = len(self.char_to_idx) + 1
    
    def encode(self, text):
        text = text.lower()
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        return torch.tensor(encoded, dtype=torch.long)
    
    def decode(self, indices):
        text = []
        for idx in indices:
            if idx == self.blank_idx:
                continue
            if idx in self.idx_to_char:
                text.append(self.idx_to_char[idx])
        return ''.join(text)


class SpeechDataset(Dataset):
    def __init__(self, audio_paths, transcripts, feature_extractor, text_encoder):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.feature_extractor = feature_extractor
        self.text_encoder = text_encoder
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        import soundfile as sf
        waveform, sample_rate = sf.read(self.audio_paths[idx])
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if sample_rate != self.feature_extractor.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.feature_extractor.sample_rate)
            waveform = resampler(waveform)
        features = self.feature_extractor.extract(waveform)
        transcript = self.text_encoder.encode(self.transcripts[idx])
        return features, transcript


def collate_fn(batch):
    features, transcripts = zip(*batch)
    feature_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    transcript_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)
    return features_padded, transcripts_padded, feature_lengths, transcript_lengths


class MLPEncoder(nn.Module):
    """MLP encoder for speech recognition."""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        """x: (batch, time, features) -> (time, batch, vocab_size)"""
        out = self.network(x)
        out = self.log_softmax(out)
        return out.transpose(0, 1)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, transcripts, feature_lengths, transcript_lengths in tqdm(dataloader, desc="Training"):
        features, transcripts = features.to(device), transcripts.to(device)
        log_probs = model(features)
        loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, text_encoder):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, transcripts, feature_lengths, transcript_lengths in dataloader:
            features, transcripts = features.to(device), transcripts.to(device)
            log_probs = model(features)
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def decode_predictions(log_probs, text_encoder, debug=False):
    """Greedy CTC decoding."""
    predictions = torch.argmax(log_probs, dim=2).transpose(0, 1)
    
    if debug:
        first_pred = predictions[0][:50]
        print(f"    [DEBUG] Unique: {torch.unique(first_pred).tolist()}, First 20: {first_pred[:20].tolist()}")
    
    decoded_texts = []
    for pred in predictions:
        decoded = []
        prev = None
        for p in pred.tolist():
            if p != prev and p != text_encoder.blank_idx:
                decoded.append(p)
            prev = p
        decoded_texts.append(text_encoder.decode(decoded))
    return decoded_texts


def load_librispeech_data(data_dir='../LibriSpeech/dev-clean', max_samples=500):
    """Load LibriSpeech dataset from local directory."""
    audio_paths, transcripts = [], []
    print(f"Loading LibriSpeech from {data_dir}...")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            audio_path = os.path.join(root, f'{file_id}.flac')
                            if os.path.exists(audio_path):
                                audio_paths.append(audio_path)
                                transcripts.append(text.lower())
                                if len(audio_paths) >= max_samples:
                                    print(f"Loaded {len(audio_paths)} samples")
                                    return audio_paths, transcripts
    print(f"Loaded {len(audio_paths)} samples")
    return audio_paths, transcripts


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    audio_paths, transcripts = load_librispeech_data(max_samples=500)
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    feature_extractor = MFCCFeatureExtractor()
    text_encoder = TextEncoder()
    print(f"Vocabulary size: {text_encoder.vocab_size}")
    
    train_dataset = SpeechDataset(train_paths, train_transcripts, feature_extractor, text_encoder)
    val_dataset = SpeechDataset(val_paths, val_transcripts, feature_extractor, text_encoder)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    model = MLPEncoder(feature_extractor.n_mfcc, [256, 256, 128], text_encoder.vocab_size)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    train_losses, val_losses = [], []
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device, text_encoder)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                features, transcripts_batch, _, _ = next(iter(val_loader))
                log_probs = model(features.to(device))
                predictions = decode_predictions(log_probs, text_encoder, debug=True)
                print("\nSample predictions:")
                for i in range(min(3, len(predictions))):
                    true_text = text_encoder.decode(transcripts_batch[i].tolist())
                    print(f"  True: '{true_text}'\n  Pred: '{predictions[i]}'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP + MFCC + CTC')
    plt.legend()
    plt.grid(True)
    plt.savefig('part1_training_curves.png')
    print("\nSaved: part1_training_curves.png")
    
    torch.save({'model_state_dict': model.state_dict()}, 'part1_model.pth')
    print("Saved: part1_model.pth")


if __name__ == '__main__':
    main()

