"""
Script principal d'entraînement – Mission 1 : Classification CIFAR-10.

Usage :
    python train.py [--epochs 50] [--batch_size 64] [--lr 0.001]
"""

import argparse
import os
from pathlib import Path

import tensorflow as tf

from models import CustomCNN
from utils  import (
    load_and_preprocess_cifar10,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)

# ── Reproductibilité ─────────────────────────────────────────────────────────
tf.random.set_seed(42)

SAVE_DIR    = "outputs"
MODEL_PATH  = "outputs/cifar10_cnn.keras"


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement CustomCNN sur CIFAR-10")
    parser.add_argument("--epochs",      type=int,   default=50,    help="Nombre max d'époques")
    parser.add_argument("--batch_size",  type=int,   default=64,    help="Taille du batch")
    parser.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate Adam")
    parser.add_argument("--dropout",     type=float, default=0.4,   help="Taux de Dropout")
    return parser.parse_args()


def build_callbacks(model_path: str) -> list:
    """Définit les callbacks d'entraînement."""
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Arrêt précoce si val_loss ne s'améliore plus pendant 8 époques
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        # Sauvegarde du meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Réduction du LR si stagnation
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        # Logs TensorBoard (optionnel : tensorboard --logdir outputs/logs)
        tf.keras.callbacks.TensorBoard(
            log_dir="outputs/logs",
            histogram_freq=0,
        ),
    ]
    return callbacks


def main():
    args = parse_args()
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Mission 1 – Classification CIFAR-10 avec CustomCNN")
    print("=" * 60)
    print(f"  Époques max   : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Dropout       : {args.dropout}")
    print("=" * 60)

    # ── 1. Données ────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, x_test_raw, y_test_raw = load_and_preprocess_cifar10(
        batch_size=args.batch_size
    )

    # ── 2. Modèle ─────────────────────────────────────────────────────────────
    print("\n🏗️  Construction du modèle...")
    model = CustomCNN(num_classes=10, dropout_rate=args.dropout)

    # Forcer la construction en appelant une forward pass
    model.build((None, 32, 32, 3))
    model.build_graph().summary()

    # ── 3. Compilation ────────────────────────────────────────────────────────
    print("\n⚙️  Compilation du modèle...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # ── 4. Entraînement ───────────────────────────────────────────────────────
    print("\n🚀 Démarrage de l'entraînement...\n")
    callbacks = build_callbacks(MODEL_PATH)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Évaluation finale ──────────────────────────────────────────────────
    print("\n📊 Évaluation sur le jeu de test...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

    if test_acc >= 0.70:
        print("  ✅ Objectif 70% ATTEINT !")
    else:
        print("  ⚠️  Objectif 70% non atteint – vérifiez les hyperparamètres.")

    # ── 6. Sauvegarde des graphiques ──────────────────────────────────────────
    print("\n🎨 Génération des graphiques...")

    plot_training_history(history, save_dir=SAVE_DIR)

    import numpy as np
    x_test_norm = x_test_raw.astype("float32") / 255.0
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    plot_confusion_matrix(y_test_raw, y_pred, save_dir=SAVE_DIR)
    plot_sample_predictions(x_test_raw, y_test_raw, y_pred, n=16, save_dir=SAVE_DIR)

    # ── 7. Sauvegarde du modèle ───────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n💾 Modèle sauvegardé → {MODEL_PATH}")
    print("\n✅ Entraînement terminé avec succès !")


if __name__ == "__main__":
    main()
