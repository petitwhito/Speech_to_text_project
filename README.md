# 🎧 Speech-To-Text - TP Jour 1

Ce projet implémente un pipeline complet de Speech-to-Text (STT) de A à Z, en explorant différentes architectures de deep learning.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Parties du TP](#parties-du-tp)
- [Utilisation](#utilisation)
- [Ressources](#ressources)

## 🎯 Vue d'ensemble

Ce TP couvre les aspects suivants du Speech-to-Text :

1. **Partie 1** : MLP + MFCC + CTC Loss
2. **Partie 2** : CNN + Spectrogrammes
3. **Partie 3** : RNN (LSTM/GRU/BiLSTM)
4. **Partie 4** : Transformers et Conformer
5. **Partie 5** : Optimisation d'hyperparamètres (Optuna)

## 🔧 Installation

### Prérequis

- Python 3.8+
- GPU recommandé (mais fonctionne aussi sur CPU)

### Installation des dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Installation avec GPU (CUDA)

```bash
# Pour CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📁 Structure du projet

```
Speech_to_text_project/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── TP STT.md                         # Sujet du TP (format markdown)
├── TP STT.pdf                        # Sujet du TP (format PDF)
│
├── part1_mlp_mfcc_ctc.py             # Partie 1: MLP + MFCC + CTC
├── part2_cnn_spectrogram.py          # Partie 2: CNN + Spectrogrammes
├── part3_rnn_lstm.py                 # Partie 3: RNN (LSTM/GRU/BiLSTM)
├── part4_transformer.py              # Partie 4: Transformers
└── part5_hyperparameter_tuning.py   # Partie 5: Tuning hyperparamètres
```

## 🚀 Parties du TP

### Partie 1 : MLP + MFCC + CTC

**Objectif** : Construire un pipeline STT minimal avec MLP.

**Features** :
- Extraction de MFCC (Mel-Frequency Cepstral Coefficients)
- Encodage caractère par caractère
- Architecture MLP simple
- Loss CTC (Connectionist Temporal Classification)

**Exécution** :
```bash
python part1_mlp_mfcc_ctc.py
```

**Résultats** :
- `part1_model.pth` : Modèle entraîné
- `part1_training_curves.png` : Courbes d'apprentissage

### Partie 2 : CNN + Spectrogrammes

**Objectif** : Améliorer l'extraction de features avec des CNN.

**Nouveautés** :
- Remplacement MFCC → Mel-Spectrogramme
- Couches convolutionnelles pour extraction de features
- Comparaison performances avec Partie 1

**Exécution** :
```bash
python part2_cnn_spectrogram.py
```

**Résultats** :
- `part2_model.pth` : Modèle CNN
- `part2_training_curves.png` : Courbes d'apprentissage

### Partie 3 : RNN (LSTM/GRU/BiLSTM)

**Objectif** : Explorer les architectures récurrentes pour capturer la temporalité.

**Architectures testées** :
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- BiLSTM (Bidirectional LSTM)
- CNN + LSTM hybride

**Exécution** :
```bash
python part3_rnn_lstm.py
```

**Résultats** :
- Modèles pour chaque architecture
- `part3_architecture_comparison.png` : Comparaison des architectures

**Aspects étudiés** :
- Capacité temporelle
- Stabilité de la CTC loss
- Vitesse d'entraînement

### Partie 4 : Transformers

**Objectif** : Implémenter une architecture Transformer pour ASR.

**Features** :
- Self-attention sur frames audio
- Positional encoding
- Architecture Transformer classique
- Architecture Conformer (Convolution-augmented Transformer)

**Exécution** :
```bash
python part4_transformer.py
```

**Résultats** :
- `part4_model_transformer.pth` : Modèle Transformer
- `part4_model_conformer.pth` : Modèle Conformer
- `part4_transformer_comparison.png` : Comparaison

### Partie 5 : Optimisation d'hyperparamètres

**Objectif** : Trouver les meilleurs hyperparamètres avec Optuna.

**Hyperparamètres optimisés** :
- Type de features (MFCC vs MelSpec)
- Nombre de features
- Architecture (LSTM vs GRU)
- Taille des couches cachées
- Nombre de couches
- Dropout
- Learning rate
- Batch size
- Augmentation audio (niveau de bruit)

**Exécution** :
```bash
# Optimisation avec Optuna (recommandé)
python part5_hyperparameter_tuning.py

# Grid Search (alternative)
python part5_hyperparameter_tuning.py --grid
```

**Résultats** :
- `part5_best_params.json` : Meilleurs hyperparamètres trouvés
- `part5_tuning_summary.png` : Résumé de l'optimisation
- `part5_optimization_history.png` : Historique (si plotly installé)
- `part5_param_importances.png` : Importance des paramètres (si plotly installé)

## 💻 Utilisation

### Entraînement rapide

Chaque script peut être exécuté indépendamment :

```bash
# Partie 1
python part1_mlp_mfcc_ctc.py

# Partie 2
python part2_cnn_spectrogram.py

# Partie 3
python part3_rnn_lstm.py

# Partie 4
python part4_transformer.py

# Partie 5
python part5_hyperparameter_tuning.py
```

### Données

Par défaut, les scripts génèrent des **données audio synthétiques** pour tester rapidement les architectures. Les données sont créées dans le dossier `data/dummy/`.

### Utiliser vos propres données

Pour utiliser vos propres données audio, modifiez la fonction `create_dummy_data()` dans chaque script :

```python
# Remplacer
audio_paths, transcripts = create_dummy_data(num_samples=100)

# Par vos propres données
audio_paths = ['chemin/vers/audio1.wav', 'chemin/vers/audio2.wav', ...]
transcripts = ['transcription 1', 'transcription 2', ...]
```

Formats audio supportés : WAV, MP3, FLAC, OGG

### Datasets recommandés

Pour des expériences réelles, utilisez ces datasets :

- **LibriSpeech** : [http://www.openslr.org/12/](http://www.openslr.org/12/)
- **Common Voice** : [https://commonvoice.mozilla.org/](https://commonvoice.mozilla.org/)
- **TIMIT** : [https://catalog.ldc.upenn.edu/LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)
- **VoxForge** : [http://www.voxforge.org/](http://www.voxforge.org/)

## 📊 Résultats attendus

Les scripts génèrent automatiquement :

1. **Modèles entraînés** (fichiers `.pth`)
2. **Courbes d'apprentissage** (fichiers `.png`)
3. **Prédictions d'exemple** (affichées dans le terminal)
4. **Comparaisons** entre architectures

### Exemple de sortie

```
Epoch 15/20 - Train Loss: 2.3456, Val Loss: 2.4567

Sample predictions:
  True: 'hello world'
  Pred: 'helo world'
  
  True: 'deep learning'
  Pred: 'deep learning'
```

## 🔬 Concepts clés

### MFCC vs Mel-Spectrogram

- **MFCC** : Coefficients cepstraux, représentation compacte
- **Mel-Spectrogram** : Représentation temps-fréquence, plus d'information

### CTC Loss

La **Connectionist Temporal Classification** permet d'aligner automatiquement les séquences audio et texte sans annotation temporelle précise.

**Avantages** :
- Pas besoin d'alignement manuel
- Gère des séquences de longueurs différentes
- Token "blank" pour gérer les silences

### Architectures

| Architecture | Avantages | Inconvénients |
|-------------|-----------|---------------|
| **MLP** | Simple, rapide | Pas de modélisation temporelle |
| **CNN** | Extraction de features locales | Champ réceptif limité |
| **LSTM/GRU** | Modélisation temporelle | Séquentiel, lent |
| **Transformer** | Parallélisable, long contexte | Coûteux en mémoire |
| **Conformer** | Combine CNN et attention | Complexe |

## 📚 Ressources

### Documentation officielle

- **PyTorch Audio** : [https://pytorch.org/audio/](https://pytorch.org/audio/)
- **Librosa** : [https://librosa.org/](https://librosa.org/)
- **Optuna** : [https://optuna.org/](https://optuna.org/)

### Tutoriels

- **MFCC vs Mel-Spectrogram** : [https://vtiya.medium.com/mfcc-vs-mel-spectrogram-8f1dc0abbc62](https://vtiya.medium.com/mfcc-vs-mel-spectrogram-8f1dc0abbc62)
- **Keras CTC ASR** : [https://keras.io/examples/audio/ctc_asr/](https://keras.io/examples/audio/ctc_asr/)
- **Understanding CTC** : [https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)
- **Transformer ASR** : [https://keras.io/examples/audio/transformer_asr/](https://keras.io/examples/audio/transformer_asr/)
- **HF Audio Course** : [https://huggingface.co/learn/audio-course/](https://huggingface.co/learn/audio-course/)

### Papers

- **CTC** : Graves et al., "Connectionist Temporal Classification"
- **wav2vec 2.0** : Baevski et al., 2020
- **Conformer** : Gulati et al., 2020
- **Whisper** : Radford et al., 2022

## 🛠️ Dépannage

### Problème : CUDA out of memory

**Solution** : Réduire le batch size

```python
train_loader = DataLoader(..., batch_size=4)  # Au lieu de 8
```

### Problème : CTC Loss = inf ou nan

**Solutions** :
1. Vérifier que `feature_lengths > transcript_lengths`
2. Utiliser `zero_infinity=True` dans CTCLoss
3. Réduire le learning rate
4. Ajouter gradient clipping

### Problème : Pas de GPU détecté

**Vérification** :
```python
import torch
print(torch.cuda.is_available())  # Devrait être True
print(torch.cuda.get_device_name(0))  # Nom de votre GPU
```

### Problème : Import error pour torchaudio

**Solution** :
```bash
pip uninstall torchaudio
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎓 Extensions possibles

Pour aller plus loin après le TP :

1. **Beam Search Decoding** : Améliorer le décodage CTC
2. **Language Model** : Ajouter un modèle de langue pour corriger les prédictions
3. **Data Augmentation** : SpecAugment, time stretching, pitch shifting
4. **Multi-GPU Training** : Distributed Data Parallel
5. **Quantization** : Optimiser pour l'inférence
6. **ONNX Export** : Déploiement optimisé

## 📝 Notes

- Les données synthétiques sont générées aléatoirement et ne permettent pas d'évaluer les performances réelles
- Pour des résultats significatifs, utilisez des datasets réels (LibriSpeech, Common Voice, etc.)
- Les architectures sont simplifiées pour des raisons pédagogiques
- Les modèles state-of-the-art utilisent des architectures beaucoup plus grandes

## 🤝 Contribution

Ce projet est un TP éducatif. Pour toute question ou suggestion :
- Consultez le fichier `TP STT.md` pour plus de détails
- Référez-vous aux ressources listées ci-dessus

## 📄 Licence

Ce projet est fourni à des fins éducatives.

---

**Bon courage pour le TP ! 🚀**

