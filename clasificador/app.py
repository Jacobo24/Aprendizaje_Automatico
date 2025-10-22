import os, json, importlib.util, inspect
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import streamlit as st
import pandas as pd
import altair as alt

# ==============================
# CONFIG BÁSICA
# ==============================
MODEL_PATH  = os.getenv("MODEL_PATH",  "models/basic_cnn_minFN.pt")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE", "224"))
HEADER_GIF  = os.getenv("HEADER_GIF", "assets/header.gif")

# (opcional) para checkpoints con state_dict:
MODEL_DEF   = os.getenv("MODEL_DEF", "")      # p. ej. "models/basic_model.py"
CLASS_NAME  = os.getenv("CLASS_NAME", "")     # p. ej. "BasicCNN"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="¿Melanoma maligno o benigno?", page_icon="🫀", layout="centered")

# ==============================
# TEMA OSCURO - AZUL / CIAN
# ==============================
ACCENT="#00e5ff"         # cian brillante
ACCENT_TEXT="#001015"    # texto oscuro sobre cian
BG="#0b1220"             # fondo principal (azul muy oscuro)
BG_SIDEBAR="#101a30"     # fondo sidebar
TEXT="#e8f4ff"           # texto claro
CARD="#142038"           # tarjetas / cuadros
BORDER="#1f3559"         # bordes azulados

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; color:{TEXT}; }}
  .header-box {{ margin:-2.2rem -1rem 1.2rem -1rem; border-bottom:2px solid {ACCENT}; }}
  h1,h2,h3,h4,h5,h6 {{ color:{TEXT} !important; }}
  [data-testid="stSidebar"] > div {{ background:{BG_SIDEBAR}; color:{TEXT}; border-right:1px solid {BORDER}; }}
  .streamlit-expanderHeader {{
      color:{TEXT} !important;
      background:linear-gradient(90deg,{CARD},{BG_SIDEBAR});
      border:1px solid {BORDER};
      border-radius:8px;
  }}
  .streamlit-expanderContent {{
      background:{CARD} !important;
      color:{TEXT};
      border:1px solid {BORDER};
      border-top:none;
      border-radius:0 0 8px 8px;
  }}
  [data-testid="stFileUploaderDropzone"] {{
      background:{CARD};
      border:2px dashed {ACCENT};
  }}
  .stButton > button {{
      background:{ACCENT};
      color:{ACCENT_TEXT};
      border:0;
      border-radius:10px;
      padding:.65rem 1rem;
      font-weight:800;
  }}
  .stButton > button:hover {{ filter:brightness(0.9); }}
  .stDataFrame,.stTable,.stAlert {{ background:{CARD} !important; }}
  .stAlert {{ border:1px solid {BORDER}; }}
  .stTextInput input,.stNumberInput input {{
      color:{TEXT} !important;
      background:{CARD} !important;
      border:1px solid {BORDER} !important;
  }}
  hr {{ border-color:{BORDER}; }}
</style>
""", unsafe_allow_html=True)

# Badges
st.markdown("""
<style>
  .badge {
    display:inline-block; padding: .20rem .5rem; border-radius: 999px;
    font-size: .78rem; font-weight:700; letter-spacing:.2px;
    margin-right:.4rem; margin-bottom:.25rem; border:1px solid rgba(255,255,255,.08);
  }
  .badge-cyan   { background:#00e5ff22; color:#00e5ff; }
  .badge-green  { background:#22c55e22; color:#22c55e; }
  .badge-amber  { background:#f59e0b22; color:#f59e0b; }
  .badge-red    { background:#ef444422; color:#ef4444; }
  .muted        { color:#a9bdd6; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# HELPERS: compat width / use_container_width
# ==============================
def _supports_width(fn) -> bool:
    try:
        return 'width' in inspect.signature(fn).parameters
    except Exception:
        return False

def _image(img, **kwargs):
    if _supports_width(st.image):
        kwargs.pop('use_container_width', None)
        return st.image(img, **kwargs, width='stretch')
    else:
        kwargs.pop('width', None)
        return st.image(img, **kwargs, use_container_width=True)

def _dataframe(df, **kwargs):
    if _supports_width(st.dataframe):
        kwargs.pop('use_container_width', None)
        return st.dataframe(df, **kwargs, width='stretch')
    else:
        kwargs.pop('width', None)
        return st.dataframe(df, **kwargs, use_container_width=True)

def _altair_chart(chart, **kwargs):
    # Streamlit estable aún no acepta width en altair_chart -> fallback a use_container_width
    kwargs.pop('width', None)
    return st.altair_chart(chart, **kwargs, use_container_width=True)

# ==============================
# LABELS
# ==============================
def ensure_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(["Benigno","Maligno"], f, ensure_ascii=False, indent=2)
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if isinstance(labels, dict):  # soporta {"0": "Benigno", "1": "Maligno"}
        items = sorted(labels.items(), key=lambda kv: int(kv[0]))
        labels = [name for _, name in items]
    return list(labels)

labels = ensure_labels(LABELS_PATH)

# ==============================
# SIDEBAR: Configuración y ajustes de inferencia (sin TOP_K ni ImageNet)
# ==============================
with st.sidebar:
    exp = st.expander("⚙️ Configuración y detalles técnicos", expanded=False)
    with exp:
        st.write(f"**Dispositivo:** `{DEVICE}`")
        st.write(f"**Ruta del modelo:** `{MODEL_PATH}`")
        st.write(f"**Etiquetas:** `{LABELS_PATH}`  (clases: {len(labels)})")
        st.write(f"**Tamaño de imagen:** `{IMAGE_SIZE}`")

        st.markdown("---")
        st.subheader("Ajustes para igualar tu script")

        SS = st.session_state
        SS.setdefault("BINARY_MODE", True)          # renombrado: más claro
        SS.setdefault("THRESHOLD", 0.5)
        SS.setdefault("MALIGNANT_IDX", 1)
        SS.setdefault("RESIZE_STRATEGY", "Mantener ratio + CenterCrop")

        SS.BINARY_MODE = st.checkbox(
            "🔧 Modo binario",
            value=SS.BINARY_MODE,
            help="Actívalo si tu modelo es binario (1 ó 2 logits)."
        )
        SS.THRESHOLD = st.slider(
            "Umbral maligno",
            0.0, 1.0, float(SS.THRESHOLD), 0.01,
            help="Se aplica sobre p(Maligno)."
        )
        SS.MALIGNANT_IDX = st.number_input(
            "Índice de 'Maligno'",
            min_value=0, step=1, value=int(SS.MALIGNANT_IDX),
            help="Índice de la clase maligna si tu salida tiene 2 logits."
        )
        SS.RESIZE_STRATEGY = st.selectbox(
            "Redimensionado",
            ["Mantener ratio + CenterCrop", "Cuadrado directo sin crop"],
            index=0 if SS.RESIZE_STRATEGY.startswith("Mantener") else 1
        )

        st.caption("Normalización fija: mean=std=0.5.")

# ==============================
# CARGA MODELO (TorchScript o state_dict) + meta
# ==============================
def _import_model_class(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("user_model", module_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name, None)

class SimpleCNN(nn.Module):
    """Fallback muy simple por si no pasas tu clase real."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.head(x)

@st.cache_resource(show_spinner=True)
def load_model(path: str, num_classes_hint: Optional[int] = None):
    meta = {"kind": None, "arch": None, "used_class": None, "device": None, "notes": None}

    if not os.path.exists(path):
        st.error(f"No se encontró el archivo del modelo: {path}")
        st.stop()

    # 1) TorchScript
    try:
        model = torch.jit.load(path, map_location=DEVICE)
        model.eval()
        meta.update(kind="TorchScript", arch=type(model).__name__)
        try:
            meta["device"] = str(next(model.parameters()).device)
        except Exception:
            meta["device"] = str(DEVICE)
        return model, meta
    except Exception:
        pass

    # 2) nn.Module pickled
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, nn.Module):
        ckpt.eval()
        meta.update(kind="PickledModule", arch=type(ckpt).__name__)
        try:
            meta["device"] = str(next(ckpt.parameters()).device)
        except Exception:
            meta["device"] = str(DEVICE)
        return ckpt, meta

    # 3) Checkpoint dict
    if isinstance(ckpt, dict):
        state_dict = None
        for k in ["model_state_dict", "state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]; break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt  # state_dict puro

        num_classes = ckpt.get("num_classes", None)
        if num_classes is None:
            num_classes = num_classes_hint if num_classes_hint is not None else max(2, len(labels))

        model = None
        if MODEL_DEF and CLASS_NAME and os.path.exists(MODEL_DEF):
            Cls = _import_model_class(MODEL_DEF, CLASS_NAME)
            if Cls is not None:
                try:   model = Cls(num_classes=num_classes)
                except TypeError:  model = Cls()
                meta.update(kind="Checkpoint+Class", used_class=CLASS_NAME)

        if model is None:
            model = SimpleCNN(num_classes=num_classes)
            meta.update(kind="Checkpoint+Fallback", used_class="SimpleCNN",
                        notes="Se recomienda exportar TorchScript o proporcionar tu clase con MODEL_DEF/CLASS_NAME.")

        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                st.warning(f"Aviso: claves faltantes: {len(missing)}, inesperadas: {len(unexpected)} (strict=False).")

        model.to(DEVICE).eval()
        meta.update(arch=type(model).__name__, device=str(next(model.parameters()).device))
        return model, meta

    st.error("Formato de checkpoint no reconocido. Exporta a TorchScript o pasa tu clase con MODEL_DEF/CLASS_NAME.")
    st.stop()

# ==============================
# PREPROCESADO & PREDICCIÓN (normalización fija 0.5)
# ==============================
def build_transform(image_size: int):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)   # sin opción ImageNet
    if st.session_state.RESIZE_STRATEGY.startswith("Mantener"):
        tfs = [transforms.Resize(image_size, antialias=True), transforms.CenterCrop(image_size)]
    else:
        tfs = [transforms.Resize((image_size, image_size), antialias=True)]
    tfs += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(tfs)

@torch.inference_mode()
def predict(image: Image.Image, model, labels: List[str]):
    if image.mode != "RGB": image = image.convert("RGB")
    x = build_transform(IMAGE_SIZE)(image).unsqueeze(0).to(DEVICE)
    logits = model(x)
    if isinstance(logits, (list, tuple)): logits = logits[0]

    out_dim = logits.shape[-1] if logits.ndim >= 2 else 1

    if st.session_state.BINARY_MODE:
        # Suma = 1.0 garantizada
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if out_dim == 1:
            p_m = float(torch.sigmoid(logits.squeeze()).item())
            p_b = 1.0 - p_m
        elif out_dim == 2:
            p = F.softmax(logits.squeeze(), dim=-1)
            m_idx = int(st.session_state.MALIGNANT_IDX)
            b_idx = 1 - m_idx
            p_m = float(p[m_idx].item()); p_b = float(p[b_idx].item())
        else:
            p = F.softmax(logits.squeeze(), dim=-1)
            m_idx = int(st.session_state.MALIGNANT_IDX)
            b_idx = 1 - m_idx if p.numel() > 1 else 0
            p_m = float(p[m_idx].item()); p_b = float(p[b_idx].item())

        if labels and len(labels) >= 2:
            maligno_label = labels[int(st.session_state.MALIGNANT_IDX)]
            benigno_label = labels[1 - int(st.session_state.MALIGNANT_IDX)] if len(labels) > 1 else "Benigno"
        else:
            maligno_label, benigno_label = "Maligno", "Benigno"

        pred_label = maligno_label if p_m >= float(st.session_state.THRESHOLD) else benigno_label
        scores = {maligno_label: p_m, benigno_label: p_b}
        return pred_label, scores

    # Multiclase → softmax sobre todas las clases
    probs = F.softmax(logits, dim=-1).squeeze(0)
    num_classes = probs.shape[-1]
    _labels = labels if (labels and len(labels) == num_classes) else [f"class_{i}" for i in range(num_classes)]
    values = probs.tolist()
    scores = {_labels[i]: float(values[i]) for i in range(num_classes)}
    top_label = _labels[int(torch.argmax(probs).item())]
    return top_label, scores

def prob_bar_chart(df: pd.DataFrame):
    return (alt.Chart(df)
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("Clase:N", sort="-y", title=None),
                y=alt.Y("Probabilidad:Q", title="Probabilidad"),
                tooltip=["Clase", alt.Tooltip("Probabilidad:Q", format=".3f")]
            )
            .properties(height=280, background=BG)
            .configure_axis(labelColor=TEXT, titleColor=TEXT, grid=False, domainColor=BORDER)
            .configure_view(strokeWidth=0))

# ==============================
# CABECERA
# ==============================
st.markdown('<div class="header-box">', unsafe_allow_html=True)
if os.path.exists(HEADER_GIF):
    _image(HEADER_GIF)
else:
    st.warning("No se encontró el GIF de encabezado. Colócalo en 'assets/header.gif' o define HEADER_GIF.")
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# UI
# ==============================
st.title("¿Es tu melanoma maligno o benigno?")
st.write("Sube una foto de la lesión cutánea. La aplicación estima la **probabilidad** de cada clase y te muestra la más probable.")
st.caption("⚠️ Herramienta educativa. No sustituye un diagnóstico médico profesional.")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("📂 Sube una imagen (PNG/JPG/WEBP)", type=["png","jpg","jpeg","webp"])
with col2:
    if "show_info" not in st.session_state:
        st.session_state.show_info = False
    if st.button("ℹ️ ¿Cómo funciona?"):
        st.session_state.show_info = not st.session_state.show_info
    if st.session_state.show_info:
        st.info("El sistema usa un modelo entrenado con imágenes etiquetadas. Ajusta tu imagen al tamaño del modelo y calcula una probabilidad por clase. La mayor probabilidad es el resultado principal.")

if uploaded:
    image = Image.open(uploaded)
    _image(image, caption="Imagen cargada")

    with st.spinner("🧠 Analizando imagen..."):
        model, model_meta = load_model(MODEL_PATH, num_classes_hint=len(labels) if labels else None)
        top_label, scores = predict(image, model, labels)

        # Mostrar resultado con color dinámico (dentro del spinner → top_label ya existe)
        if "malig" in top_label.lower():  # detecta "maligno" o similar
            st.error(f"❌ Resultado: **{top_label}**")
        else:
            st.success(f"✅ Resultado: **{top_label}**")

        if scores:
            df = pd.DataFrame({"Clase": list(scores.keys()), "Probabilidad": list(scores.values())})
            df = df.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
            _dataframe(df, hide_index=True)
            _altair_chart(prob_bar_chart(df))
            st.caption(f"Suma de probabilidades: {sum(scores.values()):.6f}")


st.markdown("<hr/>", unsafe_allow_html=True)

# ==============================
# SIDEBAR: Badges del modelo
# ==============================
with st.sidebar:
    exp2 = st.expander("🔎 Estado del modelo", expanded=False)
    with exp2:
        if 'model_meta' in locals():
            kind = model_meta.get("kind", "desconocido")
            arch = model_meta.get("arch", "—")
            device = model_meta.get("device", "—")
            used_class = model_meta.get("used_class", "")
            notes = model_meta.get("notes", "")

            cls = {
                "TorchScript": "badge-cyan",
                "PickledModule": "badge-green",
                "Checkpoint+Class": "badge-amber",
                "Checkpoint+Fallback": "badge-red",
            }.get(kind, "badge")

            arch_label = "TorchScript (exportado)" if kind == "TorchScript" else arch

            st.markdown(
                f"""
                <div>
                  <span class="badge {cls}">{kind}</span>
                  <span class="badge badge-cyan">Arch: {arch_label}</span>
                  <span class="badge badge-cyan">Device: {device}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if used_class:
                st.markdown(f'<div class="muted">Clase usada: <strong>{used_class}</strong></div>', unsafe_allow_html=True)
            if notes:
                st.markdown(f'<div class="muted">{notes}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">El modelo se cargará tras la primera predicción.</div>', unsafe_allow_html=True)
