"""
Part 5: Hyperparameter Tuning
Using Optuna for hyperparameter optimization of speech recognition models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


class AudioFeatureExtractor:
    """Flexible audio feature extractor"""
    
    def __init__(self, feature_type='mfcc', sample_rate=16000, n_features=13, 
                 n_fft=400, hop_length=160):
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.n_features = n_features
        
        if feature_type == 'mfcc':
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_features,
                melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 40}
            )
        elif feature_type == 'melspec':
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_features
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def extract(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        features = self.transform(waveform)
        
        if self.feature_type == 'melspec':
            features = self.amplitude_to_db(features)
        
        features = features.squeeze(0).transpose(0, 1)
        return features


class AudioAugmentation:
    """Audio augmentation for training"""
    
    def __init__(self, time_stretch_rate=None, pitch_shift=None, noise_level=None):
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift = pitch_shift
        self.noise_level = noise_level
    
    def apply(self, waveform):
        if self.noise_level is not None and self.noise_level > 0:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # Time stretch and pitch shift would require more complex transforms
        # Simplified version here
        
        return waveform


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


class SpeechDataset(Dataset):
    """Custom dataset with augmentation"""
    
    def __init__(self, audio_paths, transcripts, feature_extractor, text_encoder, augmentation=None):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.feature_extractor = feature_extractor
        self.text_encoder = text_encoder
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        import soundfile as sf
        waveform, sample_rate = sf.read(self.audio_paths[idx])
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        
        if sample_rate != self.feature_extractor.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.feature_extractor.sample_rate)
            waveform = resampler(waveform)
        
        if self.augmentation is not None:
            waveform = self.augmentation.apply(waveform)
        
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


class FlexibleASRModel(nn.Module):
    """Flexible ASR model for hyperparameter tuning"""
    
    def __init__(self, input_size, output_size, architecture='lstm', 
                 hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
        super(FlexibleASRModel, self).__init__()
        
        self.architecture = architecture
        
        if architecture == 'lstm':
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif architecture == 'gru':
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(encoder_output_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, features)
        Returns:
            log_probs: (time, batch, vocab_size)
        """
        output, _ = self.encoder(x)
        output = self.fc(output)
        output = self.log_softmax(output)
        output = output.transpose(0, 1)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """Train model and return best validation loss"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for features, transcripts, feature_lengths, transcript_lengths in train_loader:
            features = features.to(device)
            transcripts = transcripts.to(device)
            
            optimizer.zero_grad()
            log_probs = model(features)
            loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, transcripts, feature_lengths, transcript_lengths in val_loader:
                features = features.to(device)
                transcripts = transcripts.to(device)
                
                log_probs = model(features)
                loss = criterion(log_probs, transcripts, feature_lengths, transcript_lengths)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
    
    return best_val_loss


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


def objective(trial, audio_paths, transcripts, device):
    """Optuna objective function"""
    
    # Hyperparameters to tune
    feature_type = trial.suggest_categorical('feature_type', ['mfcc', 'melspec'])
    # For MFCC, max is 40 (number of mel bins), for melspec can be higher
    if feature_type == 'mfcc':
        n_features = trial.suggest_int('n_features', 13, 40)
    else:
        n_features = trial.suggest_int('n_features', 40, 80)
    architecture = trial.suggest_categorical('architecture', ['lstm', 'gru'])
    hidden_size = trial.suggest_int('hidden_size', 64, 256, step=64)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    noise_level = trial.suggest_float('noise_level', 0.0, 0.1)
    
    # Split data
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    
    # Initialize components
    feature_extractor = AudioFeatureExtractor(feature_type=feature_type, n_features=n_features)
    text_encoder = TextEncoder()
    augmentation = AudioAugmentation(noise_level=noise_level)
    
    # Create datasets
    train_dataset = SpeechDataset(train_paths, train_transcripts, feature_extractor, 
                                   text_encoder, augmentation)
    val_dataset = SpeechDataset(val_paths, val_transcripts, feature_extractor, 
                                 text_encoder, None)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn)
    
    # Initialize model
    model = FlexibleASRModel(
        input_size=n_features,
        output_size=text_encoder.vocab_size,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train and get best validation loss
    try:
        best_val_loss = train_model(model, train_loader, val_loader, criterion, 
                                     optimizer, device, num_epochs=5)  # Reduced for speed
        return best_val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


def main():
    """Main hyperparameter tuning function"""
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load LibriSpeech data
    audio_paths, transcripts = load_librispeech_data(max_samples=300)  # Reduced for faster tuning
    
    # Create Optuna study
    print("\nStarting hyperparameter optimization with Optuna...")
    study = optuna.create_study(
        direction='minimize',
        study_name='asr_hyperparameter_tuning',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize (reduced trials for speed)
    study.optimize(
        lambda trial: objective(trial, audio_paths, transcripts, device),
        n_trials=10,  # Reduced from 20
        timeout=None,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials)
    }
    
    with open('part5_best_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBest parameters saved to 'part5_best_params.json'")
    
    # Plot optimization history
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image('part5_optimization_history.png')
        print("Optimization history saved to 'part5_optimization_history.png'")
    except Exception as e:
        print(f"Could not save optimization history plot: {e}")
    
    # Plot parameter importances
    try:
        fig2 = plot_param_importances(study)
        fig2.write_image('part5_param_importances.png')
        print("Parameter importances saved to 'part5_param_importances.png'")
    except Exception as e:
        print(f"Could not save parameter importances plot: {e}")
    
    # Create manual plots as fallback
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Optimization history
    plt.subplot(1, 2, 1)
    trial_numbers = [trial.number for trial in study.trials]
    values = [trial.value for trial in study.trials if trial.value is not None]
    plt.plot(trial_numbers[:len(values)], values, 'o-')
    plt.axhline(y=study.best_value, color='r', linestyle='--', label='Best')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Loss')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Trial values distribution
    plt.subplot(1, 2, 2)
    plt.hist(values, bins=20, edgecolor='black')
    plt.axvline(x=study.best_value, color='r', linestyle='--', label='Best')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')
    plt.title('Trial Values Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('part5_tuning_summary.png')
    print("Summary plots saved to 'part5_tuning_summary.png'")
    
    # Print trial statistics
    completed_trials = [t for t in study.trials if t.value is not None]
    print(f"\nTrial statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")


def grid_search_example():
    """Example of manual grid search (alternative to Optuna)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Grid Search - Using device: {device}\n")
    
    # Create data
    audio_paths, transcripts = create_dummy_data(num_samples=100)
    split_idx = int(0.8 * len(audio_paths))
    train_paths, val_paths = audio_paths[:split_idx], audio_paths[split_idx:]
    train_transcripts, val_transcripts = transcripts[:split_idx], transcripts[split_idx:]
    
    text_encoder = TextEncoder()
    
    # Define grid
    param_grid = {
        'feature_type': ['mfcc', 'melspec'],
        'n_features': [13, 40],
        'hidden_size': [64, 128],
        'num_layers': [2, 3],
        'learning_rate': [0.001, 0.0001]
    }
    
    # Generate all combinations
    from itertools import product
    
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Testing {len(combinations)} combinations...\n")
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"Combination {i+1}/{len(combinations)}: {params}")
        
        # Initialize components
        feature_extractor = AudioFeatureExtractor(
            feature_type=params['feature_type'],
            n_features=params['n_features']
        )
        
        # Create datasets
        train_dataset = SpeechDataset(train_paths, train_transcripts, 
                                       feature_extractor, text_encoder)
        val_dataset = SpeechDataset(val_paths, val_transcripts, 
                                     feature_extractor, text_encoder)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                                collate_fn=collate_fn)
        
        # Initialize model
        model = FlexibleASRModel(
            input_size=params['n_features'],
            output_size=text_encoder.vocab_size,
            architecture='lstm',
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers']
        )
        model = model.to(device)
        
        # Train
        criterion = nn.CTCLoss(blank=text_encoder.blank_idx, zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        best_val_loss = train_model(model, train_loader, val_loader, criterion, 
                                     optimizer, device, num_epochs=5)
        
        results.append({
            'params': params,
            'val_loss': best_val_loss
        })
        
        print(f"  Val Loss: {best_val_loss:.4f}\n")
    
    # Find best
    best_result = min(results, key=lambda x: x['val_loss'])
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS")
    print("="*60)
    print(f"Best validation loss: {best_result['val_loss']:.4f}")
    print(f"Best parameters: {best_result['params']}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--grid':
        grid_search_example()
    else:
        main()

