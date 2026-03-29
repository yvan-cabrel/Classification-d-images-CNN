"""
Fonctions de visualisation : courbes d'entraînement, matrice de confusion,
et affichage d'échantillons CIFAR-10.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Pas de fenêtre graphique sur serveur
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

from utils.data_loader import CIFAR10_CLASSES_FR, CIFAR10_CLASSES


# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "train": "#4C72B0",
    "val":   "#DD8452",
    "grid":  "#E5E5E5",
    "bg":    "#F9F9F9",
}


def plot_training_history(history, save_dir: str = "outputs") -> str:
    """
    Génère et sauvegarde les courbes Loss & Accuracy (Train vs Validation).

    Args:
        history  : Objet History retourné par model.fit().
        save_dir : Répertoire de sauvegarde.

    Returns:
        Chemin du fichier PNG sauvegardé.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    metrics = [
        ("loss",     "sparse_categorical_accuracy", "Perte (Loss)", "Précision (Accuracy)"),
    ]

    # Loss
    ax = axes[0]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(history.history["loss"], color=COLORS["train"], lw=2, label="Train Loss")
    ax.plot(history.history["val_loss"], color=COLORS["val"], lw=2, linestyle="--", label="Val Loss")
    ax.set_title("Courbe de Perte", fontsize=14, fontweight="bold")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)

    # Accuracy – cherche la bonne clé (peut varier selon TF version)
    acc_key     = "accuracy" if "accuracy" in history.history else "sparse_categorical_accuracy"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_sparse_categorical_accuracy"

    ax = axes[1]
    ax.set_facecolor(COLORS["bg"])
    ax.plot(history.history[acc_key],     color=COLORS["train"], lw=2, label="Train Accuracy")
    ax.plot(history.history[val_acc_key], color=COLORS["val"],   lw=2, linestyle="--", label="Val Accuracy")
    ax.axhline(0.70, color="green", linestyle=":", lw=1.5, label="Objectif 70%")
    ax.set_title("Courbe de Précision", fontsize=14, fontweight="bold")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Historique d'Entraînement – CustomCNN (CIFAR-10)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Courbes d'entraînement sauvegardées → {save_path}")
    return save_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str = "outputs") -> str:
    """
    Calcule et affiche la matrice de confusion normalisée.

    Args:
        y_true   : Labels réels.
        y_pred   : Labels prédits.
        save_dir : Répertoire de sauvegarde.

    Returns:
        Chemin du fichier PNG sauvegardé.
    """
    from sklearn.metrics import confusion_matrix

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CIFAR10_CLASSES_FR, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CIFAR10_CLASSES_FR, fontsize=10)

    thresh = cm_norm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_title("Matrice de Confusion – CIFAR-10", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Classe réelle", fontsize=12)
    ax.set_xlabel("Classe prédite", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Matrice de confusion sauvegardée → {save_path}")
    return save_path


def plot_sample_predictions(
    x_test_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 16,
    save_dir: str = "outputs",
) -> str:
    """
    Affiche une grille de prédictions avec code couleur (vert=correct, rouge=erreur).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / "sample_predictions.png")

    indices = np.random.choice(len(x_test_raw), n, replace=False)
    cols = 4
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.patch.set_facecolor(COLORS["bg"])

    for idx, ax in zip(indices, axes.flat):
        ax.imshow(x_test_raw[idx])
        true_label = CIFAR10_CLASSES_FR[y_true[idx]]
        pred_label = CIFAR10_CLASSES_FR[y_pred[idx]]
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"Réel : {true_label}\nPrédit : {pred_label}", fontsize=8, color=color)
        ax.axis("off")

    plt.suptitle("Exemples de Prédictions (vert=correct, rouge=erreur)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Prédictions échantillon sauvegardées → {save_path}")
    return save_path
