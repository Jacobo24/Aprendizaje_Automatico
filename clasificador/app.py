import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io


# ========= PALETA =========
COLOR_BURDEOS = "#390517"
COLOR_DORADO = "#A38560"
COLOR_VERDE = "#16302B"
COLOR_NEGRO_VERDOSO = "#03110D"
COLOR_GRIS = "#E0E0E0"

# ========= ESTILOS GLOBALES (HTML) =========
st.markdown(
    f"""
    <style>
    body {{
        background: #ffffff;
    }}
    .main-title {{
        color: {COLOR_NEGRO_VERDOSO};
        font-weight: 600;
    }}
    .card-ia {{
        background: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem;
        border: 2px solid {COLOR_DORADO};
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .tag {{
        display: inline-block;
        font-size: 0.8rem;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        background: {COLOR_DORADO};
        color: white;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }}
    .btn-green {{
        display: inline-block;
        background: linear-gradient(135deg, {COLOR_VERDE}, {COLOR_NEGRO_VERDOSO});
        color: #ffffff;
        padding: 0.7rem 1.2rem;
        border-radius: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        border: 2px solid {COLOR_DORADO};
        transition: all 0.3s ease;
    }}
    .btn-green:hover {{
        background: {COLOR_VERDE};
        transform: scale(1.03);
        box-shadow: 0 0 12px {COLOR_DORADO}55;
    }}
    .btn-red {{
        display: inline-block;
        background: linear-gradient(135deg, {COLOR_BURDEOS}, #200109);
        color: #ffffff;
        padding: 0.7rem 1.2rem;
        border-radius: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        border: 2px solid {COLOR_DORADO};
        transition: all 0.3s ease;
    }}
    .btn-red:hover {{
        background: {COLOR_BURDEOS};
        transform: scale(1.03);
        box-shadow: 0 0 12px {COLOR_DORADO}55;
    }}
    /* tablas */
    .stDataFrame, .stTable {{
        border: 2px solid {COLOR_DORADO}AA !important;
        border-radius: 0.8rem !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        background-color: {COLOR_VERDE}11;
    }}
    table.dataframe {{
        width: 100% !important;
        border-collapse: collapse !important;
        background-color: {COLOR_VERDE}22 !important;
    }}
    thead tr {{
        background-color: {COLOR_VERDE};
        color: #fdfdfd;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
    }}
    tbody tr {{
        background-color: #ffffff;
        color: {COLOR_NEGRO_VERDOSO};
        font-size: 0.9rem;
    }}
    tbody tr:nth-child(even) {{
        background-color: {COLOR_VERDE}08;
    }}
    tbody tr:hover {{
        background-color: {COLOR_VERDE}22;
        transition: background 0.3s ease;
    }}
    td, th {{
        border: 1px solid {COLOR_DORADO}33 !important;
        padding: 0.6rem 0.8rem !important;
        text-align: center !important;
    }}
    caption {{
        caption-side: top !important;
        color: {COLOR_DORADO};
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ========= RUTAS =========
BASE_DIR = Path(__file__).resolve().parent
IMG_SAN_PATH = BASE_DIR / ".." / "img" / "SAN.png"
IMG_BBVA_PATH = BASE_DIR / ".." / "img" / "bbva.png"

# ========= DATOS =========
SAN_ACTUAL = 8.826000213623047
BBVA_ACTUAL = 17.434999465942383

preds_san = [
    {"Date": "2025-11-03", "SAN_Close": 8.8338},
    {"Date": "2025-11-04", "SAN_Close": 8.8444},
    {"Date": "2025-11-05", "SAN_Close": 8.8027},
    {"Date": "2025-11-06", "SAN_Close": 8.8027},
    {"Date": "2025-11-07", "SAN_Close": 8.8709},
]

preds_bbva = [
    {"Date": "2025-11-03", "BBVA_Close": 17.7612},
    {"Date": "2025-11-04", "BBVA_Close": 17.6221},
    {"Date": "2025-11-05", "BBVA_Close": 17.4187},
    {"Date": "2025-11-06", "BBVA_Close": 17.4372},
    {"Date": "2025-11-07", "BBVA_Close": 17.5588},
]

def mejor_oportunidad(preds, actual, col_name):
    best = None
    for p in preds:
        fut = p[col_name]
        pct = (fut - actual) / actual * 100
        if best is None or pct > best["pct"]:
            best = {"date": p["Date"], "price": fut, "pct": pct}
    best["buy"] = best["pct"] > 0.3
    return best

best_san = mejor_oportunidad(preds_san, SAN_ACTUAL, "SAN_Close")
best_bbva = mejor_oportunidad(preds_bbva, BBVA_ACTUAL, "BBVA_Close")

# ========= SIDEBAR =========
st.sidebar.title("Inversión")
page = st.sidebar.radio("Secciones", ["Inicio", "Modelos", "Inversión"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Ref. SAN (31/10):** {SAN_ACTUAL:.3f} €")
st.sidebar.markdown(f"**Ref. BBVA (31/10):** {BBVA_ACTUAL:.3f} €")
st.sidebar.caption(f"Último acceso: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# pequeño separador burdeos
def separador():
    st.markdown(
        f"<hr style='border:0; border-top:2px solid {COLOR_BURDEOS}33; margin:1.4rem 0 1rem 0;'/>",
        unsafe_allow_html=True
    )

#======== Predecir =========
def cargar_df_bbva():
    df = pd.read_csv("../csv/bbva_enriched.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df

def cargar_df_san():
    df = pd.read_csv("../csv/santander_enriched.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df

def preparar_series_para_modelo(df, col_target="BBVA_Close", window_size=25):
    # nos quedamos solo con la columna objetivo
    serie = df[[col_target]].values.astype(float)
    scaler_y = MinMaxScaler()
    serie_scaled = scaler_y.fit_transform(serie)
    return serie, serie_scaled, scaler_y

def predecir_n_dias_univariante(model, serie_scaled, scaler_y, n_dias=5, window_size=25):
    """
    serie_scaled: array 2D (n, 1) ya escalado
    """
    historia = serie_scaled.flatten().tolist()
    preds = []

    for _ in range(n_dias):
        ventana = np.array(historia[-window_size:]).reshape(1, window_size, 1)
        pred_scaled = model.predict(ventana, verbose=0)
        pred_real = scaler_y.inverse_transform(pred_scaled)[0, 0]
        preds.append(pred_real)
        # añadimos el valor escalado a la historia para la siguiente vuelta
        historia.append(pred_scaled[0, 0])

    return preds


# ========= INICIO =========
if page == "Inicio":
    # ocultar inputs fantasma
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] > div:first-child {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # botones superiores centrados y más grandes
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; gap:2rem; margin:1.5rem 0 2rem 0;">
            <a href="#santander" style="text-decoration:none;">
                <div style="
                    background:{COLOR_DORADO};
                    color:white;
                    padding:1rem 2.5rem;
                    border-radius:9999px;
                    font-weight:700;
                    font-size:1.2rem;
                    letter-spacing:0.03em;
                    box-shadow:0 3px 10px rgba(0,0,0,0.15);
                    transition:all 0.3s ease;
                    text-align:center;">
                    Santander
                </div>
            </a>
            <a href="#bbva" style="text-decoration:none;">
                <div style="
                    background:{COLOR_VERDE};
                    color:white;
                    padding:1rem 2.5rem;
                    border-radius:9999px;
                    font-weight:700;
                    font-size:1.2rem;
                    letter-spacing:0.03em;
                    box-shadow:0 3px 10px rgba(0,0,0,0.15);
                    transition:all 0.3s ease;
                    text-align:center;">
                    BBVA
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="main-title">Panel de principal</h1>', unsafe_allow_html=True)
    st.write("Vista rápida usando las predicciones precalculadas (5 días).")

    # ====== noticias (2 filas x 2 columnas) ======
    separador()
    st.markdown(f"<h3 style='color:{COLOR_NEGRO_VERDOSO}; margin-bottom:0.6rem;'>Noticias</h3>", unsafe_allow_html=True)

    noticias_paths = [
        BASE_DIR / ".." / "img" / "noticia1.png",
        BASE_DIR / ".." / "img" / "noticia2.png",
        BASE_DIR / ".." / "img" / "noticia3.png",
        BASE_DIR / ".." / "img" / "noticia4.png",
    ]
    noticias_titulos = ["Mercados", "Santander", "BBVA", "IA y Finanzas"]

    # fila 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if noticias_paths[0].exists():
            st.image(str(noticias_paths[0]), use_container_width=True)
        st.markdown(
            f"<p style='text-align:left; margin-top:0.35rem; color:{COLOR_NEGRO_VERDOSO}; font-weight:500;'>{noticias_titulos[0]}</p>",
            unsafe_allow_html=True
        )
    with row1_col2:
        if noticias_paths[1].exists():
            st.image(str(noticias_paths[1]), use_container_width=True)
        st.markdown(
            f"<p style='text-align:left; margin-top:0.35rem; color:{COLOR_NEGRO_VERDOSO}; font-weight:500;'>{noticias_titulos[1]}</p>",
            unsafe_allow_html=True
        )

    # fila 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if noticias_paths[2].exists():
            st.image(str(noticias_paths[2]), use_container_width=True)
        st.markdown(
            f"<p style='text-align:left; margin-top:0.35rem; color:{COLOR_NEGRO_VERDOSO}; font-weight:500;'>{noticias_titulos[2]}</p>",
            unsafe_allow_html=True
        )
    with row2_col2:
        if noticias_paths[3].exists():
            st.image(str(noticias_paths[3]), use_container_width=True)
        st.markdown(
            f"<p style='text-align:left; margin-top:0.35rem; color:{COLOR_NEGRO_VERDOSO}; font-weight:500;'>{noticias_titulos[3]}</p>",
            unsafe_allow_html=True
        )

    # ====== bancos ======
    separador()
    st.markdown(f"<h3 style='color:{COLOR_NEGRO_VERDOSO};'>Bancos</h3>", unsafe_allow_html=True)

    # --- Santander (todo en un solo markdown) ---
    santander_html = f"""
<div id="santander" style="display:flex; justify-content:center; margin-top:0.9rem;">
  <div class="card-ia" style="max-width:520px; width:100%;">
    <span class="tag">Santander</span>
    <h3 style="margin-bottom:0.5rem;color:{COLOR_NEGRO_VERDOSO};">Situación</h3>
    <p>Precio actual (31/10): <b>{SAN_ACTUAL:.3f} €</b></p>
    <p>Mejor día previsto: <b>{best_san['date']}</b></p>
    <p>Objetivo ese día: <b>{best_san['price']:.3f} €</b></p>
    <p>Variación estimada: <b>{best_san['pct']:.2f}%</b></p>
    <div style="margin-top:1rem;">
"""
    # aquí metemos el botón SIN sangría
    if best_san["buy"]:
        santander_html += (
            f'<span style="display:inline-block;background:linear-gradient(135deg,{COLOR_VERDE},{COLOR_NEGRO_VERDOSO});'
            f'color:#fff;padding:1rem 1.8rem;border-radius:0.9rem;font-weight:600;border:2px solid {COLOR_DORADO};font-size:1rem;">'
            '✔ Mayor beneficio'
            '</span>'
        )
    else:
        santander_html += (
            f'<span style="display:inline-block;background:linear-gradient(135deg,{COLOR_BURDEOS},#200109);'
            f'color:#fff;padding:1rem 1.8rem;border-radius:0.9rem;font-weight:600;border:2px solid {COLOR_DORADO};font-size:1rem;">'
            '✖ No compensa'
            '</span>'
        )

    # cerrar divs
    santander_html += """
    </div>
  </div>
</div>
"""
    st.markdown(santander_html, unsafe_allow_html=True)

    # línea burdeos entre bancos
    separador()

    # --- BBVA (todo en un solo markdown) ---
    bbva_html = f"""
<div id="bbva" style="display:flex; justify-content:center; margin-top:0.2rem;">
  <div class="card-ia" style="max-width:520px; width:100%;">
    <span class="tag">BBVA</span>
    <h3 style="margin-bottom:0.5rem;color:{COLOR_NEGRO_VERDOSO};">Situación</h3>
    <p>Precio actual (31/10): <b>{BBVA_ACTUAL:.3f} €</b></p>
    <p>Mejor día previsto: <b>{best_bbva['date']}</b></p>
    <p>Objetivo ese día: <b>{best_bbva['price']:.3f} €</b></p>
    <p>Variación estimada: <b>{best_bbva['pct']:.2f}%</b></p>
    <div style="margin-top:1rem;">
"""
    if best_bbva["buy"]:
        bbva_html += (
            f'<span style="display:inline-block;background:linear-gradient(135deg,{COLOR_VERDE},{COLOR_NEGRO_VERDOSO});'
            f'color:#fff;padding:1rem 1.8rem;border-radius:0.9rem;font-weight:600;border:2px solid {COLOR_DORADO};font-size:1rem;">'
            '✔ Mayor beneficio'
            '</span>'
        )
    else:
        bbva_html += (
            f'<span style="display:inline-block;background:linear-gradient(135deg,{COLOR_BURDEOS},#200109);'
            f'color:#fff;padding:1rem 1.8rem;border-radius:0.9rem;font-weight:600;border:2px solid {COLOR_DORADO};font-size:1rem;">'
            '✖ No compensa'
            '</span>'
        )

    bbva_html += """
    </div>
  </div>
</div>
"""
    st.markdown(bbva_html, unsafe_allow_html=True)


# ========= MODELOS =========
elif page == "Modelos":
    st.markdown('<h1 class="main-title">Modelos y detalle</h1>', unsafe_allow_html=True)

    tab_san, tab_bbva = st.tabs(["Santander", "BBVA"])

    with tab_san:
        st.subheader("Predicción 5 días - Santander")
        if IMG_SAN_PATH.exists():
            st.image(str(IMG_SAN_PATH), caption="Gráfica SAN", use_container_width=True)
        else:
            st.info("No se encontró SAN.png en ..\\img\\")
        df_san = pd.DataFrame(preds_san)
        df_san["Actual_31_10"] = SAN_ACTUAL
        df_san["pct_change"] = (df_san["SAN_Close"] - SAN_ACTUAL) / SAN_ACTUAL * 100
        st.dataframe(df_san, use_container_width=True)

    with tab_bbva:
        st.subheader("Predicción 5 días - BBVA")
        if IMG_BBVA_PATH.exists():
            st.image(str(IMG_BBVA_PATH), caption="Gráfica BBVA", use_container_width=True)
        else:
            st.info("No se encontró bbva.png en ..\\img\\")
        df_bbva = pd.DataFrame(preds_bbva)
        df_bbva["Actual_31_10"] = BBVA_ACTUAL
        df_bbva["pct_change"] = (df_bbva["BBVA_Close"] - BBVA_ACTUAL) / BBVA_ACTUAL * 100
        st.dataframe(df_bbva, use_container_width=True)


# ========= INVERSIÓN =========
else:  # Inversión
    st.markdown(
        f"""
        <div style="margin-bottom:0.5rem;">
            <h1 class="main-title" style="margin-bottom:0.25rem;">Panel de inversión</h1>
            <p style="color:{COLOR_NEGRO_VERDOSO}; font-size:0.95rem; margin:0;">
                Vista rápida usando las predicciones precalculadas (5 días).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # contenedor con fondo verde suave
    st.markdown(
        f"""
        <div style="
            background:{COLOR_VERDE}10;
            border:2px solid {COLOR_DORADO};
            border-radius:1rem;
            padding:1.5rem;
            margin-top:1rem;
            box-shadow:0 4px 14px rgba(0,0,0,0.08);">
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        activo = st.selectbox("Activo", ["BBVA", "Santander"])
    with col2:
        dia_idx = st.number_input(
            "Día futuro (1 = primer día predicho, máx. 5)",
            min_value=1,
            max_value=5,
            value=1,
            step=1
        )

    st.markdown("---")

    # obtener precio actual y predicho según lo que haya elegido
    if activo == "BBVA":
        precio_actual = BBVA_ACTUAL
        prediccion_dia = preds_bbva[dia_idx - 1]["BBVA_Close"]
        fecha_dia = preds_bbva[dia_idx - 1]["Date"]
    else:
        precio_actual = SAN_ACTUAL
        prediccion_dia = preds_san[dia_idx - 1]["SAN_Close"]
        fecha_dia = preds_san[dia_idx - 1]["Date"]

    diferencia = prediccion_dia - precio_actual  # cálculo interno

    # ======= bloque visual de resultado =======
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:1rem;">
            <h3 style="color:{COLOR_NEGRO_VERDOSO}; margin-bottom:0.5rem;">{activo} · comparación</h3>
            <p style="color:{COLOR_NEGRO_VERDOSO}; font-size:0.95rem;">
                <b>Precio 31/10:</b> {precio_actual:.3f} €<br>
                <b>Estimado ({fecha_dia}):</b> {prediccion_dia:.3f} €
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======= recomendación =======
    if diferencia <= 0:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{COLOR_BURDEOS},#200109);
                        color:white; padding:1.2rem; border-radius:1rem;
                        border:2px solid {COLOR_DORADO};
                        text-align:center; font-weight:600;
                        box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                ✖ No te recomendamos invertir<br>
                <span style="font-size:0.9rem;opacity:0.85;">(el precio estimado es menor o igual que el actual)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,{COLOR_VERDE},{COLOR_NEGRO_VERDOSO});
                        color:white; padding:1.2rem; border-radius:1rem;
                        border:2px solid {COLOR_DORADO};
                        text-align:center; font-weight:600;
                        box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                ✔ Recomendamos invertir<br>
                <span style="font-size:0.9rem;opacity:0.85;">(el precio estimado es mayor que el actual)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{COLOR_NEGRO_VERDOSO};'>Define tu inversión</h4>", unsafe_allow_html=True)

        inversion = st.number_input(
            "¿Cuánto quieres invertir (€)?",
            min_value=0.01,
            max_value=1_000_000.00,
            value=100.00,
            step=1.00,
            format="%.2f",
            key="inversion_input"
        )

        boton_html = f"""
        <div style="text-align:center; margin-top:0.8rem;">
            <button style="
                background:linear-gradient(135deg,{COLOR_DORADO},{COLOR_VERDE});
                color:white;
                padding:0.9rem 2.2rem;
                border-radius:1rem;
                font-weight:700;
                font-size:1.05rem;
                border:none;
                box-shadow:0 4px 12px rgba(0,0,0,0.2);
                cursor:pointer;">Invertir</button>
        </div>
        """

        # usamos un pequeño contenedor HTML para mantener estilo uniforme
        invertir_btn = st.markdown(boton_html, unsafe_allow_html=True)

        if st.button("Confirmar inversión"):
            st.markdown(
                f"""
                <div style="background:{COLOR_DORADO}33; border:2px solid {COLOR_DORADO};
                            color:{COLOR_NEGRO_VERDOSO};
                            border-radius:1rem; padding:1rem; text-align:center;
                            margin-top:1rem; font-weight:600;">
                    Inversión con éxito<br>
                    Has invertido <b>{inversion:.2f} €</b> en <b>{activo}</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

    # cerrar contenedor principal
    st.markdown("</div>", unsafe_allow_html=True)
