# 🧠 Deep Learning – Mission 1 : Classification CIFAR-10

> Projet Fil Rouge | Deep Learning | Dr. Noulapeu N. A.
> Framework : TensorFlow / Keras | Application : Flet

---

## 📁 Structure du projet

```
cifar10_project/
│
├── data/                        # Données brutes (CIFAR-10 téléchargé automatiquement)
│
├── models/
│   ├── __init__.py
│   └── cnn_model.py             # Classe CustomCNN (API Subclassing Keras)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Pipeline tf.data.Dataset + normalisation
│   └── visualize.py             # Courbes, matrice de confusion, prédictions
│
├── outputs/                     # Générés à l'exécution
│   ├── cifar10_cnn.keras        # Modèle sauvegardé
│   ├── training_history.png     # Courbes Loss / Accuracy
│   ├── confusion_matrix.png     # Matrice de confusion normalisée
│   └── sample_predictions.png   # Grille d'exemples
│
├── train.py                     # 🚀 Script d'entraînement principal
├── evaluate.py                  # 📊 Script d'évaluation (modèle sauvegardé)
├── app_flet.py                  # 🖥️  Application Flet interactive
└── requirements.txt             # Dépendances
```

---

## 🏗️ Architecture du modèle (CustomCNN)

```
Input (32×32×3)
    │
    ▼
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    │
    ▼
Conv2D(32, 3×3, same) → BatchNorm → MaxPool(2×2)
    │
    ▼
Conv2D(64, 3×3, same) → BatchNorm → Conv2D(64, 3×3, same) → BatchNorm → MaxPool(2×2)
    │
    ▼
Conv2D(128, 3×3, same) → BatchNorm → Conv2D(128, 3×3, same) → BatchNorm → MaxPool(2×2)
    │
    ▼
Flatten → Dense(256, relu) → Dropout(0.4) → Dense(128, relu) → Dropout(0.2)
    │
    ▼
Dense(10, softmax)
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Entraînement

```bash
python train.py
# Options :
python train.py --epochs 60 --batch_size 64 --lr 0.001 --dropout 0.4
```

Génère automatiquement dans `outputs/` :
- `cifar10_cnn.keras` — meilleur modèle (EarlyStopping + ModelCheckpoint)
- `training_history.png` — courbes Train/Val Loss et Accuracy
- `confusion_matrix.png` — matrice de confusion sur 10 000 images de test
- `sample_predictions.png` — grille d'exemples annotés

---

## 📊 Évaluation

```bash
python evaluate.py
# Ou avec un autre modèle :
python evaluate.py --model_path outputs/cifar10_cnn.keras
```

---

## 🖥️ Application Flet

```bash
python app_flet.py
```

Fonctionnalités :
- Charger n'importe quelle image (PNG, JPG, WEBP…)
- Obtenir la classe prédite avec sa confiance
- Visualiser le **Top-5** des probabilités sous forme de barres

> ⚠️ Le modèle doit avoir été entraîné (`train.py`) avant de lancer l'application.

---

## 📈 Résultats attendus

| Métrique          | Valeur cible |
|-------------------|-------------|
| Test Accuracy     | ≥ 70%       |
| Optimiseur        | Adam        |
| Loss              | SparseCategoricalCrossentropy |
| Callbacks         | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 🔧 Hyperparamètres par défaut

| Paramètre    | Valeur |
|-------------|--------|
| Epochs max  | 50     |
| Batch size  | 64     |
| LR initial  | 0.001  |
| LR min      | 1e-6   |
| Dropout     | 0.4    |
| EarlyStopping patience | 8 époques |
| ReduceLR patience      | 4 époques |
