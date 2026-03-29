"""
Script d'évaluation – Charge un modèle .keras sauvegardé et teste ses performances.

Usage :
    python evaluate.py [--model_path outputs/cifar10_cnn.keras]
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

# ⚠️ Import obligatoire AVANT load_model : exécute le décorateur
# @register_keras_serializable qui permet à Keras de retrouver CustomCNN.
from models import CustomCNN  # noqa: F401

from utils import (
    load_and_preprocess_cifar10,
    plot_confusion_matrix,
    plot_sample_predictions,
    CIFAR10_CLASSES_FR,
)

SAVE_DIR = "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation du modèle CIFAR-10")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/cifar10_cnn.keras",
        help="Chemin vers le modèle sauvegardé (.keras ou .h5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ Modèle introuvable : {args.model_path}")
        print("   Lancez d'abord : python train.py")
        return

    print("=" * 60)
    print("  Évaluation du modèle CIFAR-10")
    print("=" * 60)

    # ── Chargement des données ────────────────────────────────────────────────
    print("\n📦 Chargement des données de test...")
    _, _, test_ds, x_test_raw, y_test_raw = load_and_preprocess_cifar10()

    # ── Chargement du modèle ──────────────────────────────────────────────────
    print(f"\n🔄 Chargement du modèle depuis {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    print("  ✔ Modèle chargé avec succès")

    # ── Évaluation ─────────────────────────────────────────────────────────────
    print("\n📊 Métriques globales :")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── Prédictions ───────────────────────────────────────────────────────────
    print("\n🔮 Génération des prédictions...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # ── Rapport de classification ─────────────────────────────────────────────
    print("\n📋 Rapport de classification détaillé :")
    print(classification_report(y_test_raw, y_pred, target_names=CIFAR10_CLASSES_FR))

    # ── Graphiques ────────────────────────────────────────────────────────────
    print("\n🎨 Génération des graphiques d'évaluation...")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_test_raw, y_pred, save_dir=SAVE_DIR)
    plot_sample_predictions(x_test_raw, y_test_raw, y_pred, n=16, save_dir=SAVE_DIR)

    print("\n✅ Évaluation terminée !")


if __name__ == "__main__":
    main()
