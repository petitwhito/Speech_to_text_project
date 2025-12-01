"""
Part 4: Transformer Approach
Speech-to-Text implementation using Transformer architecture with CTC loss
Based on: https://keras.io/examples/audio/transformer_asr/
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
import math


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
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1)
        
        return mfcc


class TextEncoder:
    """Encode text to character indices"""
    
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


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for speech recognition"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, 
                 output_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time, features)
            mask: (batch, time) - padding mask
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # Project input to d_model dimensions
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask if provided
        if mask is not None:
            # Convert to transformer format (True = masked position)
            mask = ~mask.bool()
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Output projection
        x = self.output_proj(x)
        
        # Log softmax
        x = self.log_softmax(x)
        
        # CTC expects (time, batch, vocab_size)
        x = x.transpose(0, 1)
        
        return x


class ConformerBlock(nn.Module):
    """Conformer block: Convolution-augmented Transformer"""
    
    def __init__(self, d_model, nhead, dim_feedforward, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time, d_model)
        """
        # FF1
        x = x + 0.5 * self.ff1(self.norm1(x))
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out
        x = self.norm2(x)
        
        # Convolution
        conv_in = x.transpose(1, 2)  # (batch, d_model, time)
        conv_out = self.conv(conv_in)
        conv_out = conv_out.transpose(1, 2)  # (batch, time, d_model)
        x = x + conv_out
        x = self.norm3(x)
        
        # FF2
        x = x + 0.5 * self.ff2(self.norm4(x))
        
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for speech recognition"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward,
                 output_size, kernel_size=31, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_feedforward, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time, features)
            mask: (batch, time) - padding mask
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        # Project input
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, attn_mask)
        
        # Output projection
        x = self.output_proj(x)
        
        # Log softmax
        x = self.log_softmax(x)
        
        # CTC expects (time, batch, vocab_size)
        x = x.transpose(0, 1)
        
        return x


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
    """Collate function for DataLoader"""
    features, transcripts = zip(*batch)
    
    feature_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    transcript_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=0)
    
    # Create mask for padding
    max_len = features_padded.size(1)
    mask = torch.arange(max_len).expand(len(feature_lengths), max_len) < feature_lengths.unsqueeze(1)
    
    return features_padded, transcripts_padded, feature_lengths, transcript_lengths, mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for features, transcripts, feature_lengths, transcript_lengths, mask in pbar:
            features = features.to(device)
            transcripts = transcripts.to(device)
            mask = mask.to(device)
            
            # Forward pass
            log_probs = model(features, mask)
            
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
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, transcripts, feature_lengths, transcript_lengths, mask in dataloader:
            features = features.to(device)
            transcripts = transcripts.to(device)
            mask = mask.to(device)
            
            log_probs = model(features, mask)
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(log_probs, text_encoder):
    """Decode CTC predictions using greedy decoding"""
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
    """Load LibriSpeech dataset"""
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


def main():
    """Main training function"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load LibriSpeech data
    audio_paths, transcripts = load_librispeech_data(max_samples=500)
    
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # Initialize components
    feature_extractor = MFCCFeatureExtractor()
    text_encoder = TextEncoder()
    
    # Create datasets
    train_dataset = SpeechDataset(train_paths, train_transcripts, feature_extractor, text_encoder)
    val_dataset = SpeechDataset(val_paths, val_transcripts, feature_extractor, text_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Test Transformer only (Conformer too big for GPU)
    models_to_test = [
        ('Transformer', TransformerEncoder, {
            'd_model': 64,  # Reduced from 128
            'nhead': 4,
            'num_layers': 3,  # Reduced from 4
            'dim_feedforward': 256  # Reduced from 512
        })
    ]
    
    results = {}
    
    for model_name, model_class, config in models_to_test:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print('='*60)
        
        # Initialize model
        model = model_class(
            input_size=feature_extractor.n_mfcc,
            output_size=text_encoder.vocab_size,
            **config
        )
        model = model.to(device)
        
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower LR for Transformer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training
        num_epochs = 15  # More epochs for Transformer
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 3 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                
                # Show predictions
                model.eval()
                with torch.no_grad():
                    features, transcripts, _, _, mask = next(iter(val_loader))
                    features = features.to(device)
                    mask = mask.to(device)
                    log_probs = model(features, mask)
                    predictions = decode_predictions(log_probs, text_encoder)
                    
                    print("Sample predictions:")
                    for i in range(min(2, len(predictions))):
                        true_text = text_encoder.decode(transcripts[i].tolist())
                        print(f"  True: '{true_text}' | Pred: '{predictions[i]}'")
        
        results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, f'part4_model_{model_name.lower()}.pth')
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('part4_transformer_comparison.png')
    print("\n\nComparison plot saved to 'part4_transformer_comparison.png'")


if __name__ == '__main__':
    main()

