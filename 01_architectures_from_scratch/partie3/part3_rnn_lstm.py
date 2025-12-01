"""
Part 3: RNN (LSTM/GRU/BiLSTM)
Speech-to-Text implementation using Recurrent Neural Networks with CTC loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
        mfcc = mfcc.squeeze(0).transpose(0, 1)
        
        return mfcc


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


class LSTMEncoder(nn.Module):
    """LSTM-based encoder for speech recognition."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.3, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, lengths):
        """
        Args:
            x: (batch, time, features)
            lengths: (batch,)
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        packed_output, _ = self.lstm(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Fully connected
        output = self.fc(output)
        
        # Log softmax
        output = self.log_softmax(output)
        
        # CTC expects (time, batch, vocab_size)
        output = output.transpose(0, 1)
        
        return output


class GRUEncoder(nn.Module):
    """GRU-based encoder for speech recognition."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.3, bidirectional=True):
        super(GRUEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(gru_output_size, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, lengths):
        """
        Args:
            x: (batch, time, features)
            lengths: (batch,)
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # GRU forward
        packed_output, _ = self.gru(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Fully connected
        output = self.fc(output)
        
        # Log softmax
        output = self.log_softmax(output)
        
        # CTC expects (time, batch, vocab_size)
        output = output.transpose(0, 1)
        
        return output


class CNNRNNEncoder(nn.Module):
    """CNN + RNN hybrid encoder."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 rnn_type='lstm', dropout=0.3, bidirectional=True):
        super(CNNRNNEncoder, self).__init__()
        
        # CNN frontend for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # RNN backend
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=128,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=128,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layer
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(rnn_output_size, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, lengths):
        """
        Args:
            x: (batch, time, features)
            lengths: (batch,)
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # CNN expects (batch, features, time)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Back to (batch, time, features)
        
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN forward
        packed_output, _ = self.rnn(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Fully connected
        output = self.fc(output)
        
        # Log softmax
        output = self.log_softmax(output)
        
        # CTC expects (time, batch, vocab_size)
        output = output.transpose(0, 1)
        
        return output


def train_epoch(model, dataloader, optimizer, criterion, device):
    
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for features, transcripts, feature_lengths, transcript_lengths in pbar:
            features = features.to(device)
            transcripts = transcripts.to(device)
            feature_lengths = feature_lengths.to(device)
            
            # Forward pass
            log_probs = model(features, feature_lengths)
            
            # CTC Loss
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, transcripts, feature_lengths, transcript_lengths in dataloader:
            features = features.to(device)
            transcripts = transcripts.to(device)
            feature_lengths = feature_lengths.to(device)
            
            log_probs = model(features, feature_lengths)
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(log_probs, text_encoder):
    
    predictions = torch.argmax(log_probs, dim=2)
    predictions = predictions.transpose(0, 1)
    
    decoded_texts = []
    for pred in predictions:
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
    
    audio_paths = []
    transcripts = []
    
    print(f"Loading LibriSpeech from {data_dir}...")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                
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


def compare_architectures():
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load LibriSpeech data
    audio_paths, transcripts = load_librispeech_data(max_samples=500)
    
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    
    # Initialize components
    feature_extractor = MFCCFeatureExtractor()
    text_encoder = TextEncoder()
    
    # Create datasets
    train_dataset = SpeechDataset(train_paths, train_transcripts, feature_extractor, text_encoder)
    val_dataset = SpeechDataset(val_paths, val_transcripts, feature_extractor, text_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Model configurations
    configs = [
        ('LSTM', LSTMEncoder, {'bidirectional': True}),
        ('GRU', GRUEncoder, {'bidirectional': True}),
        ('BiLSTM', LSTMEncoder, {'bidirectional': True}),
        ('CNN+LSTM', CNNRNNEncoder, {'rnn_type': 'lstm', 'bidirectional': True}),
    ]
    
    results = {}
    
    for name, model_class, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print('='*60)
        
        # Initialize model
        if model_class == CNNRNNEncoder:
            model = model_class(
                input_size=feature_extractor.n_mfcc,
                hidden_size=128,
                num_layers=2,
                output_size=text_encoder.vocab_size,
                **kwargs
            )
        else:
            model = model_class(
                input_size=feature_extractor.n_mfcc,
                hidden_size=128,
                num_layers=2,
                output_size=text_encoder.vocab_size,
                **kwargs
            )
        
        model = model.to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        num_epochs = 10  # Reduced for testing
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1]
        }
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model_class.__name__,
            'config': kwargs
        }, f'part3_model_{name.lower().replace("+", "_")}.pth')
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f'{name} (Train)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=f'{name} (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('part3_architecture_comparison.png')
    print("\n\nComparison plot saved to 'part3_architecture_comparison.png'")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, result in results.items():
        print(f"{name:15s} - Final Val Loss: {result['final_val_loss']:.4f}")


if __name__ == '__main__':
    compare_architectures()

