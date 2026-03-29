"""
Utilitaires de chargement et prétraitement des données CIFAR-10.
Utilise tf.data.Dataset pour un pipeline de données efficace.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple

# Labels des 10 classes CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_CLASSES_FR = [
    "Avion", "Automobile", "Oiseau", "Chat", "Cerf",
    "Chien", "Grenouille", "Cheval", "Bateau", "Camion"
]


def load_and_preprocess_cifar10(
    batch_size: int = 64,
    validation_split: float = 0.1,
    cache: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Charge CIFAR-10 et retourne des pipelines tf.data.Dataset.

    Args:
        batch_size       : Taille du batch pour l'entraînement.
        validation_split : Fraction des données d'entraînement réservée à la validation.
        cache            : Si True, met les données en cache mémoire.

    Returns:
        train_ds   : Dataset d'entraînement (batché, mélangé, préfetché).
        val_ds     : Dataset de validation.
        test_ds    : Dataset de test.
        x_test_raw : Images de test brutes (uint8) pour l'affichage.
        y_test_raw : Labels de test bruts.
    """
    print("📦 Chargement de CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Aplatir les labels (shape (N,1) → (N,))
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Conserver les images brutes pour l'affichage
    x_test_raw = x_test.copy()
    y_test_raw = y_test.copy()

    # ── Normalisation [0, 1] ─────────────────────────────────────────────────
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # ── Séparation Train / Validation ────────────────────────────────────────
    n_val = int(len(x_train) * validation_split)
    x_val, y_val     = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    print(f"  ✔ Train    : {len(x_train):,} images")
    print(f"  ✔ Val      : {len(x_val):,}   images")
    print(f"  ✔ Test     : {len(x_test):,}  images")

    # ── Création des tf.data.Dataset ─────────────────────────────────────────
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train), seed=42)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    if cache:
        train_ds = train_ds.cache()
        val_ds   = val_ds.cache()

    return train_ds, val_ds, test_ds, x_test_raw, y_test_raw
