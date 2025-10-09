# app.py
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import gradio as gr


# =========================
# Config
# =========================
IMG_SIZE = 224
DEFAULT_CLASSES = ["Begnin", "Malignant"]  # orden por defecto (0, 1)
DEFAULT_MALIGNANT_IDX = 1

# Ruta al checkpoint (puedes sobreescribir con env var)
CHECKPOINT_PATH = os.getenv("MODEL_PATH", "models/basic_cnn_minFN.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (device == "cuda")


# =========================
# Modelo (tu BasicCNN)
# =========================
class BasicCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =========================
# Transforms (como tu val/test)
# =========================
val_test_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# =========================
# Carga de checkpoint
# =========================
def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {path}. "
            "Sube tu 'basic_cnn_minFN.pt' a 'models/' o define MODEL_PATH."
        )
    ckpt = torch.load(path, map_location=device)

    # Compatibilidad con distintas keys
    state = ckpt.get("model_state", None)
    if state is None:
        state = ckpt.get("model", None)
    if state is None:
        # quizá guardado directo con state_dict
        state = ckpt

    # Clases/mapeo si existen en el checkpoint
    class_to_idx = ckpt.get("class_to_idx", None)

    # Derivar la lista de clases en orden de índice
    if class_to_idx is not None and isinstance(class_to_idx, dict):
        # ordenar por índice (valor)
        classes_sorted = [name for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])]
        classes = classes_sorted
        # índice de Malignant (si existe ese nombre)
        malignant_idx = class_to_idx.get("Malignant", DEFAULT_MALIGNANT_IDX)
    else:
        classes = DEFAULT_CLASSES
        malignant_idx = DEFAULT_MALIGNANT_IDX

    return state, classes, malignant_idx, ckpt


# Instancia el modelo y carga pesos
model = BasicCNN(in_ch=3, num_classes=2).to(device)
state_dict, CLASSES, MALIGNANT_IDX, _ck = load_checkpoint(CHECKPOINT_PATH)
model.load_state_dict(state_dict)
model.eval()


# =========================
# Inferencia
# =========================
@torch.no_grad()
def predict(image: Image.Image, threshold: float = 0.50):
    """
    Devuelve:
      - etiqueta predicha (str)
      - prob. de la etiqueta predicha (float)
      - prob. de 'Malignant' (float)
      - detalle JSON (dict con probs y decisión)
    """
    if image is None:
        return "No image", 0.0, 0.0, {"error": "No image"}

    x = val_test_transforms(image.convert("RGB")).unsqueeze(0).to(device)

    if USE_AMP:
        with torch.cuda.amp.autocast():
            logits = model(x)
    else:
        logits = model(x)

    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy().tolist()
    p_mal = float(probs[MALIGNANT_IDX])

    pred_idx = int(p_mal >= threshold)  # 1 si >= threshold (Malignant), si no 0
    pred_name = CLASSES[pred_idx]
    pred_prob = float(probs[pred_idx])

    details = {
        CLASSES[0]: float(probs[0]),
        CLASSES[1]: float(probs[1]),
        "threshold": float(threshold),
        "decision": pred_name
    }
    return pred_name, pred_prob, p_mal, details


# =========================
# UI Gradio
# =========================
title = "Breast Cancer Classifier (BasicCNN)"
description = (
    "Sube una imagen (RGB). El modelo predice **Begnin** o **Malignant**. "
    "Ajusta el umbral para priorizar menos falsos negativos (baja el umbral)."
)
examples_note = (
    "Sugerencia: umbral por defecto 0.50. Si quieres reducir falsos negativos, prueba 0.40."
)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    gr.Markdown(f"**Clases detectadas:** {CLASSES} &nbsp;&nbsp;|&nbsp;&nbsp; **Malignant idx:** {MALIGNANT_IDX}")
    gr.Markdown(examples_note)

    with gr.Row():
        inp = gr.Image(type="pil", label="Imagen", sources=["upload", "clipboard"], height=320)
        with gr.Column():
            thr = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="Umbral (Malignant)")

    with gr.Row():
        out_label = gr.Label(label="Predicción")
        out_num = gr.Number(label="Prob. clase predicha", precision=4)
        out_mal = gr.Number(label="Prob. Malignant", precision=4)

    out_json = gr.JSON(label="Detalle de probabilidades")

    btn = gr.Button("Predecir", variant="primary")
    btn.click(predict, inputs=[inp, thr], outputs=[out_label, out_num, out_mal, out_json])

if __name__ == "__main__":
    # En Spaces no hace falta .launch(); lo dejamos igualmente para ejecución local.
    demo.launch()
