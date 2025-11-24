Speech-To-Text – TP Jour 1
==========================

Objectif
--------
Mettre en œuvre un pipeline ASR complet et comparer plusieurs familles de modèles :
MLP, CNN, RNN (LSTM/GRU/BiLSTM), Transformer simplifié et script de tuning
Optuna. Chaque partie est autonome et documente les principaux choix
architecturaux (features audio, loss CTC, encodage des transcriptions).

Installation rapide
-------------------
```bash
python -m venv venv
source venv/bin/activate      
pip install -r requirements.txt
```
Les scripts ont été testés sous Python 3.10 avec PyTorch 2.9 + CUDA 12.

Organisation
------------
```
partie1/part1_mlp_mfcc_ctc.py          MLP + MFCC + CTC
partie2/part2_cnn_spectrogram.py       CNN + MelSpectrogram
partie3/part3_rnn_lstm.py              LSTM / GRU / BiLSTM / CNN+RNN
partie4/part4_transformer.py           Transformer allégé
partie5/part5_hyperparameter_tuning.py Recherche Optuna
RESULTATS_TP.txt                       Synthèse des conclusions
TP STT.md / TP STT.pdf                 Énoncé fourni par l’enseignant
```
Les sous-dossiers `partieX/` contiennent également les modèles et graphiques
obtenus lors des expérimentations locales (fichiers `.pth`, `.png`, etc.).

Utilisation
-----------
Exemple pour lancer chaque bloc :
```bash
cd partie1 && python part1_mlp_mfcc_ctc.py
cd partie2 && python part2_cnn_spectrogram.py
cd partie3 && python part3_rnn_lstm.py
cd partie4 && python part4_transformer.py
cd partie5 && python part5_hyperparameter_tuning.py
```
Les scripts chargent LibriSpeech (dev-clean) via `load_librispeech_data`. Pour
travailler sur un autre corpus, modifier cette fonction afin de renvoyer la
liste des chemins audio et les transcriptions associées. Les données audio ainsi
que les environnements virtuels sont ignorés par Git.

Résultats principaux
--------------------
Un résumé détaillé figure dans `RESULTATS_TP.txt`. Points clefs :
- Le MLP ne produit pas de séquences utiles (prédit essentiellement le blanc).
- Le CNN fournit les premières transcriptions lisibles en exploitant les
  spectrogrammes.
- Le BiLSTM bidirectionnel est l’architecture la plus précise sur ce jeu
de données réduit (Loss validation ≈ 2.33).
- Le Transformer allégé n’est pas compétitif sur seulement 500 échantillons.
- Optuna (partie 5) confirme l’intérêt des MelSpectrogram + BiLSTM et calibre
  automatiquement les hyperparamètres principaux.