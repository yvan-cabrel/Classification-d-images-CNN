import base64
import io
import os
import sys
import time
from pathlib import Path

import flet as ft
import numpy as np
from PIL import Image

# ───────────────────────────────
# GESTION DES CHEMINS (PYINSTALLER)
# ───────────────────────────────
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Variables globales
tf = None
model = None
CIFAR10_CLASSES_FR = [
    "✈️ Avion", "🚗 Automobile", "🐦 Oiseau", "🐱 Chat", "🦌 Cerf",
    "🐶 Chien", "🐸 Grenouille", "🐴 Cheval", "🚢 Bateau", "🚛 Camion",
]
MODEL_PATH = get_resource_path(os.path.join("outputs", "cifar10_cnn.keras"))

# ───────────────────────────────
# LOGIQUE IA
# ───────────────────────────────
def load_model_lazy():
    global tf, model
    if tf is None:
        import tensorflow as _tf
        tf = _tf
        from models.cnn_model import CustomCNN  # Indispensable pour le chargement
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return False
        model = tf.keras.models.load_model(MODEL_PATH)
    return True

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((32, 32))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def image_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ───────────────────────────────
# INTERFACE FLET
# ───────────────────────────────
def main(page: ft.Page):
    page.title = "🧠 CIFAR-10 Classifier Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1000
    page.window_height = 800
    page.bgcolor = "#0F1117"

    state = {"pil_image": None}
    ACCENT = "#6C63FF"
    SUCCESS = "#00C896"

    # --- Composants UI ---
    status_text = ft.Text("Prêt pour l'analyse", size=12, italic=True, color="#A0A3B1")
    
    # Barre de progression horizontale
    progress_bar = ft.ProgressBar(width=400, color=ACCENT, bgcolor="#333", visible=False)
    
    # Spinner circulaire
    spinner = ft.ProgressRing(width=20, height=20, stroke_width=3, color=SUCCESS, visible=False)

    img_placeholder = ft.Text("Cliquez pour charger une image (JPEG/PNG)", color="#A0A3B1")
    img_display = ft.Image(visible=False, border_radius=12, fit="contain")

    result_title = ft.Text("Résultat", size=22, weight="bold", visible=False)
    confidence_text = ft.Text("", size=16, color=SUCCESS)
    bars_container = ft.Column(spacing=10)

    # --- Actions ---
    def on_file_result(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            pil_img = Image.open(file_path)
            state["pil_image"] = pil_img
            img_display.src_base64 = image_to_b64(pil_img)
            img_display.visible = True
            img_placeholder.visible = False
            status_text.value = f"Fichier chargé : {os.path.basename(file_path)}"
            page.update()

    file_picker = ft.FilePicker(on_result=on_file_result)
    page.overlay.append(file_picker)

    def run_classification(e):
        if state["pil_image"] is None:
            status_text.value = "⚠️ Erreur : Aucune image sélectionnée"
            page.update()
            return

        # 1. Verrouiller l'UI et afficher le chargement
        btn_classify.disabled = True
        btn_load.disabled = True
        progress_bar.visible = True
        spinner.visible = True
        status_text.value = "Chargement du moteur TensorFlow..."
        page.update()

        # 2. Charger le modèle (Lazy Load)
        if not load_model_lazy():
            status_text.value = "❌ Erreur : Fichier .keras introuvable !"
            btn_classify.disabled = False
            progress_bar.visible = False
            spinner.visible = False
            page.update()
            return

        status_text.value = "Analyse des pixels en cours..."
        page.update()

        # 3. Prédire
        processed_img = preprocess_image(state["pil_image"])
        predictions = model.predict(processed_img, verbose=0)[0]
        
        # 4. Afficher les résultats
        top_indices = np.argsort(predictions)[::-1][:5]
        best_idx = top_indices[0]
        
        result_title.value = f"C'est un(e) : {CIFAR10_CLASSES_FR[best_idx]}"
        result_title.visible = True
        confidence_text.value = f"Confiance : {predictions[best_idx]*100:.1f}%"
        
        bars_container.controls.clear()
        for i in top_indices:
            score = predictions[i]
            bars_container.controls.append(
                ft.Column([
                    ft.Row([
                        ft.Text(CIFAR10_CLASSES_FR[i], width=120),
                        ft.Text(f"{score*100:.1f}%", size=12, color="#A0A3B1")
                    ], alignment="spaceBetween"),
                    ft.ProgressBar(value=score, color=SUCCESS if i == best_idx else ACCENT, bgcolor="#222")
                ])
            )

        # 5. Rétablir l'UI
        status_text.value = "✅ Analyse terminée avec succès"
        progress_bar.visible = False
        spinner.visible = False
        btn_classify.disabled = False
        btn_load.disabled = False
        page.update()

    # --- Mise en page ---
    btn_load = ft.ElevatedButton("📂 Charger Image", icon=ft.icons.IMAGE_OUTLINED, 
                                 on_click=lambda _: file_picker.pick_files())
    
    btn_classify = ft.FilledButton(
        text="🔍 Lancer la Classification",
        icon=ft.icons.PSYCHOLOGY,
        on_click=run_classification,
        style=ft.ButtonStyle(bgcolor=ACCENT, color=ft.colors.WHITE)
    )
    left_card = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Stack([img_placeholder, img_display]),
                alignment=ft.alignment.center,
                width=350, height=350, bgcolor="#1A1D27", border_radius=15,
                border=ft.border.all(1, "#333"),
                on_click=lambda _: file_picker.pick_files()
            ),
            ft.Row([btn_load, btn_classify], alignment="center"),
            ft.Row([spinner, status_text], alignment="center", spacing=10)
        ], horizontal_alignment="center", spacing=20),
        padding=20
    )

    right_card = ft.Container(
        content=ft.Column([
            progress_bar,
            result_title,
            confidence_text,
            ft.Divider(height=40, color="#333"),
            ft.Text("DÉTAILS DES PROBABILITÉS", size=10, weight="bold", color="#A0A3B1"),
            bars_container
        ], spacing=15),
        expand=True, padding=30, bgcolor="#1A1D27", border_radius=20
    )

    page.add(
        ft.Container(
            content=ft.Text("CIFAR-10 CLASSIFIER", size=30, weight="black", color=ACCENT),
            margin=ft.margin.only(bottom=20)
        ),
        ft.Row([left_card, right_card], alignment="start", vertical_alignment="start")
    )

if __name__ == "__main__":
    ft.app(target=main)