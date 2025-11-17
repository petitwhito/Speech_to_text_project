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
    """Extract MFCC features from audio"""
    
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
        """Extract MFCC features from waveform"""
        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract MFCC
        mfcc = self.mfcc_transform(waveform)
        
        # Transpose to (time, features) format
        mfcc = mfcc.squeeze(0).transpose(0, 1)
        
        return mfcc


class TextEncoder:
    """Encode text to character indices"""
    
    def __init__(self):
        # Define vocabulary: blank, space, a-z
        self.blank_idx = 0
        self.space_idx = 1
        self.chars = list('abcdefghijklmnopqrstuvwxyz')
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(self.chars)}
        self.char_to_idx[' '] = self.space_idx
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[self.blank_idx] = '<blank>'
        self.vocab_size = len(self.char_to_idx) + 1  # +1 for blank
    
    def encode(self, text):
        """Encode text to indices"""
        text = text.lower()
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
        return torch.tensor(encoded, dtype=torch.long)
    
    def decode(self, indices):
        """Decode indices to text"""
        text = []
        for idx in indices:
            if idx == self.blank_idx:
                continue
            if idx in self.idx_to_char:
                text.append(self.idx_to_char[idx])
        return ''.join(text)


class SpeechDataset(Dataset):
    """Custom dataset for speech recognition"""
    
    def __init__(self, audio_paths, transcripts, feature_extractor, text_encoder):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.feature_extractor = feature_extractor
        self.text_encoder = text_encoder
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio with soundfile (supports FLAC)
        import soundfile as sf
        waveform, sample_rate = sf.read(self.audio_paths[idx])
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        
        # Resample if necessary
        if sample_rate != self.feature_extractor.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.feature_extractor.sample_rate)
            waveform = resampler(waveform)
        
        # Extract features
        features = self.feature_extractor.extract(waveform)
        
        # Encode transcript
        transcript = self.text_encoder.encode(self.transcripts[idx])
        
        return features, transcript


def collate_fn(batch):
    """Collate function for DataLoader"""
    features, transcripts = zip(*batch)
    
    # Get lengths
    feature_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    transcript_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)
    
    return features_padded, transcripts_padded, feature_lengths, transcript_lengths


class MLPEncoder(nn.Module):
    """Simple MLP encoder for speech recognition"""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(MLPEncoder, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, features)
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # Apply MLP to each time step
        out = self.network(x)
        
        # Apply log softmax
        out = self.log_softmax(out)
        
        # CTC expects (time, batch, vocab_size)
        out = out.transpose(0, 1)
        
        return out


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for features, transcripts, feature_lengths, transcript_lengths in pbar:
            features = features.to(device)
            transcripts = transcripts.to(device)
            
            # Forward pass
            log_probs = model(features)
            
            # CTC Loss
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, text_encoder):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, transcripts, feature_lengths, transcript_lengths in dataloader:
            features = features.to(device)
            transcripts = transcripts.to(device)
            
            # Forward pass
            log_probs = model(features)
            
            # CTC Loss
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(log_probs, text_encoder, debug=False):
    """Decode CTC predictions using greedy decoding"""
    # log_probs: (time, batch, vocab_size)
    
    # Get argmax
    predictions = torch.argmax(log_probs, dim=2)  # (time, batch)
    predictions = predictions.transpose(0, 1)  # (batch, time)
    
    if debug:
        # Debug: show what's being predicted
        first_pred = predictions[0][:50]  # First 50 frames of first sample
        unique_vals = torch.unique(first_pred)
        print(f"    [DEBUG] Unique predictions in first 50 frames: {unique_vals.tolist()}")
        print(f"    [DEBUG] Blank index: {text_encoder.blank_idx}")
        print(f"    [DEBUG] First 20 predictions: {first_pred[:20].tolist()}")
    
    decoded_texts = []
    for pred in predictions:
        # Remove consecutive duplicates and blanks
        pred_list = pred.tolist()
        decoded = []
        prev = None
        for p in pred_list:
            if p != prev and p != text_encoder.blank_idx:
                decoded.append(p)
            prev = p
        
        text = text_encoder.decode(decoded)
        decoded_texts.append(text)
    
    return decoded_texts


def load_librispeech_data(data_dir='../LibriSpeech/dev-clean', max_samples=500):
    """Load LibriSpeech dataset"""
    audio_paths = []
    transcripts = []
    
    print(f"Loading LibriSpeech from {data_dir}...")
    
    # Walk through LibriSpeech directory structure
    for root, dirs, files in os.walk(data_dir):
        # Find transcript files
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                
                # Read transcripts
                with open(txt_path, 'r') as f:
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
    """Main training function"""
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load LibriSpeech data
    audio_paths, transcripts = load_librispeech_data(max_samples=500)
    
    # Split into train/val
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # Initialize components
    feature_extractor = MFCCFeatureExtractor()
    text_encoder = TextEncoder()
    
    print(f"Vocabulary size: {text_encoder.vocab_size}")
    
    # Create datasets
    train_dataset = SpeechDataset(train_paths, train_transcripts, feature_extractor, text_encoder)
    val_dataset = SpeechDataset(val_paths, val_transcripts, feature_extractor, text_encoder)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    input_size = feature_extractor.n_mfcc
    hidden_sizes = [256, 256, 128]
    output_size = text_encoder.vocab_size
    
    model = MLPEncoder(input_size, hidden_sizes, output_size)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10  # Reduced for faster testing
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device, text_encoder)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Show sample predictions
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                features, transcripts, _, _ = next(iter(val_loader))
                features = features.to(device)
                log_probs = model(features)
                predictions = decode_predictions(log_probs, text_encoder, debug=True)
                
                print("\nSample predictions:")
                
                for i in range(min(3, len(predictions))):
                    true_text = text_encoder.decode(transcripts[i].tolist())
                    print(f"  True: '{true_text}'")
                    print(f"  Pred: '{predictions[i]}'")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves - MLP + MFCC + CTC')
    plt.legend()
    plt.grid(True)
    plt.savefig('part1_training_curves.png')
    print("\nTraining curves saved to 'part1_training_curves.png'")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_encoder': text_encoder,
        'feature_extractor_config': {
            'n_mfcc': feature_extractor.n_mfcc,
            'sample_rate': feature_extractor.sample_rate
        }
    }, 'part1_model.pth')
    print("Model saved to 'part1_model.pth'")


if __name__ == '__main__':
    main()

