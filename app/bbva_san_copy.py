import streamlit as st
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import datetime
from copy import deepcopy

# ============================================================
# 🧠 DEFINICIÓN DEL MODELO (idéntico al entrenamiento)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ============================================================
# ⚙️ CARGAR MODELO
# ============================================================
@st.cache_resource
def load_model(model_name, input_size):
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir.parent / "models" / model_name
    model = LSTMModel(input_size=input_size)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# 🏦 CONFIGURACIÓN APP
# ============================================================
# ============================================================
# 🎨 ESTILO CORPORATIVO LIMPIO (fondo blanco, sin emojis)
# ============================================================
st.set_page_config(page_title="Predicción LSTM — BBVA y Santander", layout="wide")

st.markdown("""
    <style>
    /* Fondo general y fuente */
    .stApp {
        background-color: #ffffff;
        color: #1b1f23;
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    /* Título */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #004481; /* Azul BBVA */
        text-align: center;
        margin-bottom: 0px;
        margin-top: -20px;
    }

    /* Subtítulo */
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #4a4a4a;
        margin-bottom: 35px;
    }

    /* Línea divisoria */
    hr {
        border: none;
        height: 1px;
        background-color: #d9d9d9;
        margin: 25px 0;
    }

    /* Botones */
    .stButton>button {
        background-color: #004481;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        height: 2.6em;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0068b4;
    }

    /* Cuadros de texto informativos */
    .stAlert {
        background-color: #f3f6fa;
        border-left: 5px solid #004481;
        color: #1b1f23;
        font-size: 15px;
    }

    /* Select y slider */
    .stSelectbox, .stSlider label {
        color: #004481 !important;
        font-weight: 600;
    }

    /* Gráficos centrados */
    .plotly-chart {
        margin: auto;
    }

    /* Subtítulos (como Escenario macroeconómico) */
    .stSubheader {
        color: #004481;
        font-weight: 600;
    }
    /* Forzar color negro en etiquetas de selectbox y sliders */
label, .stSelectbox label, div[data-baseweb="select"] label {
    color: #000000 !important;
    font-weight: 700 !important;
}

    </style>
""", unsafe_allow_html=True)

# Encabezado elegante (sin emojis)
st.markdown("<h1 class='main-title'>Predicción de precios — BBVA y Santander</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Modelo LSTM con ajuste macroeconómico dinámico (inflación y tipos BCE)</p>", unsafe_allow_html=True)

st.markdown("App que replica el flujo del notebook: fusión, limpieza y predicción automática.")

# ============================================================
# SELECCIÓN DE BANCO Y CARGA DE CSV
# ============================================================
bank = st.selectbox("Selecciona el banco:", ["BBVA", "SAN"])
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "data" / "csv" / "clean"

# ============================================================
# 1️⃣ CARGA Y FUSIÓN (idéntica al notebook)
# ============================================================
df_stock = pd.read_csv(data_dir / f"{bank}_macro_extended.csv", parse_dates=["Date"]).sort_values("Date")
df_infl = pd.read_csv(data_dir / "euro_inflation_2000_2025_final.csv", parse_dates=["date"])
df_rates = pd.read_csv(data_dir / "euro_interest_2000_2025_clean.csv", parse_dates=["date"])

# Medias móviles
df_stock["sma_7"] = df_stock["close"].rolling(window=7).mean()
df_stock["sma_30"] = df_stock["close"].rolling(window=30).mean()
df_stock["sma_60"] = df_stock["close"].rolling(window=60).mean()
df_stock = df_stock.bfill()

# Fusión inflación
df_infl = df_infl.rename(columns={"date": "Date", "inflation": "inflation_rate"})
df_merged = pd.merge_asof(
    df_stock.sort_values("Date"),
    df_infl.sort_values("Date"),
    on="Date",
    direction="backward"
)

# Fusión tipos BCE
df_rates = df_rates.rename(columns={
    "date": "Date",
    "deposit_rate": "deposit_rate",
    "marginal_rate": "marginal_rate",
    "refinancing_rate": "refinancing_rate"
})
df_final = pd.merge_asof(
    df_merged.sort_values("Date"),
    df_rates.sort_values("Date"),
    on="Date",
    direction="backward"
)

# Limpieza columnas innecesarias
for col in ["adj close", "sma_5", "sma_10"]:
    if col in df_final.columns:
        df_final = df_final.drop(columns=col)

# ============================================================
# 2️⃣ LIMPIEZA DE 'volume', 'range', 'return'
# ============================================================
cols_to_clean = ["volume", "range", "return"]
for col in cols_to_clean:
    df_final[col] = (
        df_final[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
    df_final[col] = df_final[col].replace(0, np.nan)
    df_final[col] = df_final[col].interpolate(method="linear").bfill().ffill()

# ============================================================
# 3️⃣ CREAR COLUMNA BINARIA DE DIVIDENDOS
# ============================================================
if "dividends" in df_final.columns:
    df_final["dividends_f"] = np.where(df_final["dividends"] > 0, 1, 0)
    df_final = df_final.drop(columns=["dividends"])
else:
    df_final["dividends_f"] = 0.0

# ============================================================
# 4️⃣ REEMPLAZO DE NaN Y ORDEN FINAL
# ============================================================
df_final = df_final.fillna(0).sort_values("Date").set_index("Date")

feature_cols = [
    "open","high","low","volume","range","return","vol_rel","eventos_negativos",
    "sma_7","sma_30","sma_60","inflation_rate","deposit_rate",
    "marginal_rate","refinancing_rate","dividends_f"
]
input_size = len(feature_cols)

# ============================================================
# 🎛️ Selector de días futuros + escenario macro
# ============================================================
st.markdown("---")
forecast_days = st.slider(
    "Selecciona el número de días futuros a predecir:",
    min_value=1,
    max_value=60,
    value=7,
    step=1
)

st.subheader("Escenario para variables macroeconómicas (inflación y tipos)")

macro_mode = st.selectbox(
    "Modo de predicción de exógenas futuras:",
    ["Mantener último valor (ffill)", "Escenario manual", "Deriva simple (drift)"],
    index=0,
    help="Selecciona cómo se comportarán la inflación y los tipos de interés en los días futuros."
)

# 💬 Descripción dinámica del modo seleccionado
if macro_mode == "Mantener último valor (ffill)":
    st.info(
        "🔹 **Modo: Mantener último valor (ffill)**\n\n"
        "Este modo congela la inflación y los tipos de interés en su último valor conocido.\n"
        "Por ejemplo, si el 31/10/2025 la inflación era **2.5 %** y el tipo de depósito **4.0 %**, "
        "todas las predicciones futuras usarán exactamente esos mismos valores."
    )

elif macro_mode == "Escenario manual":
    st.info(
        "🟢 **Modo: Escenario manual**\n\n"
        "Permite ajustar manualmente los valores macroeconómicos para simular un cambio inmediato.\n"
        "Por ejemplo, puedes subir la inflación +0.5 p.p. y reducir los tipos BCE -25 pbs "
        "para ver cómo afectaría ese escenario a la predicción del precio."
    )

elif macro_mode == "Deriva simple (drift)":
    st.info(
        "🟠 **Modo: Deriva simple (drift)**\n\n"
        "Aplica una tendencia progresiva diaria (subida o bajada gradual) en inflación y tipos.\n"
        "Por ejemplo, una deriva de **−1 pbs/día** reducirá los tipos un 0.01 % cada día futuro, "
        "simulando una bajada paulatina de los tipos de interés."
    )


# Parámetros para "Escenario manual"
inflation_shift_pp = 0.0
deposit_shift_bp = marginal_shift_bp = refi_shift_bp = 0

if macro_mode == "Escenario manual":
    inflation_shift_pp = st.number_input("Ajuste inflación (p.p.)", -2.0, 2.0, 0.0, 0.05)
    deposit_shift_bp   = st.number_input("Ajuste deposit rate (pbs)", -100, 100, 0, 5)
    marginal_shift_bp  = st.number_input("Ajuste marginal rate (pbs)", -100, 100, 0, 5)
    refi_shift_bp      = st.number_input("Ajuste refinancing rate (pbs)", -100, 100, 0, 5)

# Parámetros para "Deriva simple"
inflation_drift_pp_per_day = 0.0
rate_drift_bp_per_day = 0

if macro_mode == "Deriva simple (drift)":
    inflation_drift_pp_per_day = st.number_input("Deriva diaria inflación (p.p./día)", -0.1, 0.1, 0.0, 0.005)
    rate_drift_bp_per_day = st.number_input("Deriva diaria tipos (pbs/día)", -10, 10, 0, 1)

# ============================================================
# 🧩 Función para generar exógenas futuras coherentes
# ============================================================
macro_vars = ["inflation_rate","deposit_rate","marginal_rate","refinancing_rate"]

def exogenous_for_date(d, df_hist, day_index):
    last_vals = {col: float(df_hist[col].ffill().iloc[-1]) if col in df_hist.columns else 0.0 for col in macro_vars}

    if macro_mode == "Mantener último valor (ffill)":
        return last_vals
    elif macro_mode == "Escenario manual":
        return {
            "inflation_rate": last_vals["inflation_rate"] + inflation_shift_pp,
            "deposit_rate": last_vals["deposit_rate"] + deposit_shift_bp / 100.0,
            "marginal_rate": last_vals["marginal_rate"] + marginal_shift_bp / 100.0,
            "refinancing_rate": last_vals["refinancing_rate"] + refi_shift_bp / 100.0
        }
    elif macro_mode == "Deriva simple (drift)":
        drift_pp = day_index * (rate_drift_bp_per_day / 100.0)
        return {
            "inflation_rate": last_vals["inflation_rate"] + day_index * inflation_drift_pp_per_day,
            "deposit_rate": last_vals["deposit_rate"] + drift_pp,
            "marginal_rate": last_vals["marginal_rate"] + drift_pp,
            "refinancing_rate": last_vals["refinancing_rate"] + drift_pp
        }
    return last_vals


# ============================================================
# 5️⃣ CARGAR MODELO
# ============================================================
model = load_model("best_bbva_lstm.pth" if bank=="BBVA" else "best_san_lstm.pth", input_size)
model.eval()

# ============================================================
# 🟠 PREDICCIÓN HISTÓRICA (vectorizada y compatible CPU/GPU)
# ============================================================
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Usando dispositivo: {device}")

model = load_model("best_bbva_lstm.pth" if bank == "BBVA" else "best_san_lstm.pth", input_size)
model.to(device)
model.eval()

print("🔹 Iniciando cálculo de predicción histórica...")

# Escalado
scaler = MinMaxScaler().fit(df_final[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])
close_min, close_max = scaler.data_min_[3], scaler.data_max_[3]
print("✅ Escalador entrenado. close_min:", close_min, "close_max:", close_max)

def scale_df(df):
    df_scaled = df.copy()
    df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]] = \
        scaler.transform(df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])
    df_scaled[["inflation_rate","deposit_rate","marginal_rate","refinancing_rate"]] /= 10.0
    return df_scaled

lookback = 60
target_col = "close"
df_scaled = scale_df(df_final)
print("✅ Dataset escalado. Filas totales:", len(df_scaled))

# Crear secuencias
def create_sequences(df, lookback):
    Xs, ys = [], []
    for i in range(len(df) - lookback):
        Xs.append(df.iloc[i:i+lookback][feature_cols].values)
        ys.append(df.iloc[i+lookback][target_col])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(df_scaled, lookback)
print("✅ Secuencias creadas. X:", X_seq.shape, "y:", y_seq.shape)

# 🔥 Vectorizado y compatible GPU
print("🚀 Realizando predicción vectorizada...")
X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)

with torch.no_grad():
    preds_scaled = model(X_tensor).cpu().numpy().flatten()

print("✅ Predicciones completadas:", len(preds_scaled))

# Desescalar
preds_real = [close_min + p*(close_max - close_min) for p in preds_scaled]
y_real = [close_min + y*(close_max - close_min) for y in y_seq]
print("✅ Desescalado completado.")

# ============================================================
# 🔁 Mostrar todo el histórico disponible (desde 2000)
# ============================================================
dates_hist = df_final.index[lookback:]
dates_tail = dates_hist  # usamos TODO el rango de fechas
y_real_tail = y_real     # todos los valores reales
y_pred_tail = preds_real # todas las predicciones del modelo
print(f"✅ Datos históricos listos para graficar. Fechas: {dates_tail[0]} → {dates_tail[-1]}")

# ============================================================
# 🔮 FORECAST FUTURO (autorregresivo flexible)
# ============================================================
scaler = MinMaxScaler().fit(df_final[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])
close_min, close_max = scaler.data_min_[3], scaler.data_max_[3]

def scale_df(df):
    df_scaled = df.copy()
    df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]] = \
        scaler.transform(df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])
    df_scaled[["inflation_rate","deposit_rate","marginal_rate","refinancing_rate"]] /= 10.0
    return df_scaled

lookback = 60
target_col = "close"
window = df_final.tail(lookback).copy()
start_date = df_final.index[-1] + pd.Timedelta(days=1)
future_dates = pd.bdate_range(start_date, periods=forecast_days)

# ============================
# 🔥 Forecast rápido y estable
#   (sin acumulación de /10 en macro)
# ============================
print(f"⚡ Forecast rápido de {forecast_days} días…")

# Índices auxiliares
idx = {c: feature_cols.index(c) for c in feature_cols}
scaled_cols = ["open","high","low","close","volume","sma_7","sma_30","sma_60"]
scaled_idx = [idx[c] for c in ["open","high","low","volume","sma_7","sma_30","sma_60"]]  # 'close' va aparte
macro_idx  = [idx[c] for c in ["inflation_rate","deposit_rate","marginal_rate","refinancing_rate"]]

# Ventana en escala REAL (numpy) y vector de cierres reales
window_real = window[feature_cols].to_numpy(dtype=np.float32)
close_real  = window["close"].to_numpy(dtype=np.float32)

def make_scaled_from_real(win_real: np.ndarray, close_vec: np.ndarray) -> np.ndarray:
    """Devuelve copia ESCALADA de la ventana real sin modificar la original."""
    win_scaled = win_real.copy()

    # Construir matriz 8 col para el scaler (open,high,low,close,volume,sma_7,sma_30,sma_60)
    temp8 = np.column_stack([
        win_real[:, idx["open"]],
        win_real[:, idx["high"]],
        win_real[:, idx["low"]],
        close_vec,  # close real (el scaler se entrenó con él)
        win_real[:, idx["volume"]],
        win_real[:, idx["sma_7"]],
        win_real[:, idx["sma_30"]],
        win_real[:, idx["sma_60"]],
    ])
    scaled8 = scaler.transform(
        pd.DataFrame(temp8, columns=scaled_cols)
    ).astype(np.float32)

    # Insertar las 7 columnas escaladas en sus posiciones (sin 'close' porque no es feature)
    win_scaled[:, idx["open"]]   = scaled8[:, 0]
    win_scaled[:, idx["high"]]   = scaled8[:, 1]
    win_scaled[:, idx["low"]]    = scaled8[:, 2]
    win_scaled[:, idx["volume"]] = scaled8[:, 4]
    win_scaled[:, idx["sma_7"]]  = scaled8[:, 5]
    win_scaled[:, idx["sma_30"]] = scaled8[:, 6]
    win_scaled[:, idx["sma_60"]] = scaled8[:, 7]

    # Macro: dividir /10 UNA sola vez (no acumulativa)
    win_scaled[:, macro_idx] = win_real[:, macro_idx] / 10.0

    return win_scaled

forecast_preds_real = []
last_date = df_final.index[-1]

with torch.no_grad():
    for i in range(forecast_days):
        # 1) Construir ventana ESCALADA a partir de la REAL (sin acumulaciones)
        win_scaled = make_scaled_from_real(window_real, close_real)
        x_t = torch.tensor(win_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # 2) Predecir en [0,1] y desescalar a €
        y_next_scaled = model(x_t).cpu().numpy().reshape(-1)[0]
        y_next_real   = close_min + y_next_scaled * (close_max - close_min)
        forecast_preds_real.append(float(y_next_real))

        # 3) Fabricar nueva fila REAL para avanzar la ventana
        prev_close = float(close_real[-1])
        open_d = prev_close
        high_d = max(prev_close, y_next_real) * 1.01
        low_d  = min(prev_close, y_next_real) * 0.99
        vol_d  = float(window["volume"].tail(20).mean())

        # exógenas reales para la fecha futura
        future_day = last_date + pd.tseries.offsets.BDay(i+1)
        exo = exogenous_for_date(future_day, df_final, i)

        new_row_real = np.array([
            open_d,                 # open
            high_d,                 # high
            low_d,                  # low
            vol_d,                  # volume
            (high_d - low_d) / max(open_d, 1e-6),                  # range
            (y_next_real - prev_close) / max(prev_close, 1e-6),    # return
            1.0,                    # vol_rel (aprox)
            0.0,                    # eventos_negativos
            np.mean(np.append(close_real, y_next_real)[-7:]),      # sma_7 (real)
            np.mean(np.append(close_real, y_next_real)[-30:]),     # sma_30
            np.mean(np.append(close_real, y_next_real)[-60:]),     # sma_60
            exo.get("inflation_rate", 0.0),
            exo.get("deposit_rate", 0.0),
            exo.get("marginal_rate", 0.0),
            exo.get("refinancing_rate", 0.0),
            0.0,                    # dividends_f
        ], dtype=np.float32)

        # 4) Avanzar ventana REAL y vector de cierres
        window_real = np.vstack([window_real, new_row_real])[-lookback:]
        close_real  = np.append(close_real, y_next_real)[-lookback:]

print(f"✅ Forecast OK. Último predicho: {forecast_preds_real[-1]:.2f} €")

# ============================================================
# 6️⃣ GRÁFICO INTERACTIVO (histórico completo + forecast)
# ============================================================
last_real_close = df_final["close"].iloc[-1]
last_real_date = df_final.index[-1].strftime("%Y-%m-%d")
st.markdown(f"**Último cierre real:** {last_real_close:.2f} €  —  📅 *{last_real_date}*")

fig = make_subplots(specs=[[{"secondary_y": False}]])

# Línea azul corporativa – precio real histórico
fig.add_trace(go.Scatter(
    x=dates_tail, y=y_real_tail,
    mode="lines",
    name="Cierre real (€)",
    line=dict(color="#004481", width=2.2)
))

# Línea naranja sobria – predicción del modelo sobre el histórico
fig.add_trace(go.Scatter(
    x=dates_tail, y=y_pred_tail,
    mode="lines",
    name="Predicción histórica LSTM (€)",
    line=dict(color="#E67E22", width=1.8, dash="dot")
))

# Línea verde – predicción futura
fig.add_trace(go.Scatter(
    x=future_dates, y=forecast_preds_real,
    mode="lines+markers",
    name=f"Predicción futura ({forecast_days} días)",
    line=dict(color="#27AE60", width=2.5, dash="solid"),
    marker=dict(size=6, color="#27AE60", line=dict(width=0))
))

# Línea gris clara – conexión entre último real y primera predicción
fig.add_trace(go.Scatter(
    x=[dates_tail[-1], future_dates[0]],
    y=[y_pred_tail[-1], forecast_preds_real[0]],
    mode="lines",
    line=dict(color="#000000", width=1, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))

# Configuración estética del gráfico (texto negro, botones blancos)
fig.update_layout(
    title=f"Predicción LSTM — {bank} (2000–2025 + {forecast_days} días futuros)",
    title_font=dict(size=20, color="#000000", family="Inter, Segoe UI"),
    
    xaxis_title="Fecha",
    yaxis_title="Precio (€)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#000000"),
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25, font=dict(size=12, color="#000000")),
    height=650,
    plot_bgcolor="#f7f9fb",   # Fondo gris claro de la gráfica
    paper_bgcolor="#fdfdfd",  # Fondo del recuadro
    dragmode="pan"
)

# Ejes con texto negro y cuadrículas suaves
fig.update_xaxes(
    showgrid=True,
    gridcolor="#000000",
    tickformat="%Y-%m-%d",
    color="#000000",  # ← Fechas y ticks del eje X en negro
    tickfont=dict(color="#000000", size=12),   # ← Números del eje X (fechas) en negro

    title_font=dict(color="#000000"),
    rangeslider_visible=True,
    rangeselector=dict(
        bgcolor="#2C3E50",        # Fondo oscuro de los botones
        activecolor="#004481",    # Azul BBVA cuando están activos
        font=dict(color="#FFFFFF"),  # ← Texto de los botones en blanco
        buttons=list([
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1A", step="year", stepmode="backward"),
            dict(count=5, label="5A", step="year", stepmode="backward"),
            dict(step="all", label="Todo")
        ])
    )
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="#000000",
    tickfont=dict(color="#000000", size=12),   # ← Números del eje X (fechas) en negro

    color="#000000",  # ← Números de la columna izquierda (precio) en negro
    title_font=dict(color="#000000")
    
)

fig.update_traces(hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} €")

plotly_config = {"scrollZoom": False, "displayModeBar": True}
st.plotly_chart(fig, use_container_width=True, config=plotly_config)

# ============================================================
# 💰 Simulación interactiva de inversión (optimizada)
# ============================================================

st.markdown("---")
st.subheader("Simulación de inversión")

# Inicializar variables persistentes
if "inversion_total" not in st.session_state:
    st.session_state.inversion_total = 0.0
if "ultimo_precio" not in st.session_state:
    st.session_state.ultimo_precio = float(last_real_close)

precio_actual = st.session_state.ultimo_precio
fecha_actual = last_real_date

# Mostrar precio actual
st.markdown(f"**💵 Precio actual de {bank} ({fecha_actual}): {precio_actual:.2f} €**")

# Crear layout para los botones
col1, col2 = st.columns(2)
with col1:
    comprar = st.button("🟢 Comprar", use_container_width=True)
with col2:
    vender = st.button("🔴 Vender", use_container_width=True)

# Si se pulsa "Comprar"
if comprar:
    monto = st.number_input("Introduce la cantidad que deseas invertir (€):", min_value=0.0, step=100.0, key="monto_compra")

    if monto > 0:
        acciones = monto / precio_actual
        st.success(
            f"✅ Vas a invertir **{monto:.2f} €** en **{bank}** a **{precio_actual:.2f} €** por acción "
            f"(≈ {acciones:.2f} acciones)."
        )

        confirmar = st.button("Confirmar inversión", key="confirmar_compra", use_container_width=True)
        if confirmar:
            st.session_state.inversion_total += monto
            st.session_state.ultima_accion = f"Compra de {monto:.2f} € confirmada."
            st.rerun()  # 🔁 Refresca solo la interfaz, sin recalcular el modelo

# Si se pulsa "Vender"
elif vender:
    monto = st.number_input("Introduce el valor que deseas vender (€):", min_value=0.0, step=100.0, key="monto_venta")

    if monto > 0:
        if monto > st.session_state.inversion_total:
            st.error(f"🚫 No puedes vender {monto:.2f} € porque solo tienes invertidos {st.session_state.inversion_total:.2f} €.")
        else:
            acciones = monto / precio_actual
            st.warning(
                f"⚠️ Vas a vender **{monto:.2f} €** de **{bank}** a **{precio_actual:.2f} €** "
                f"(≈ {acciones:.2f} acciones)."
            )

            confirmar = st.button("Confirmar venta", key="confirmar_venta", use_container_width=True)
            if confirmar:
                st.session_state.inversion_total -= monto
                st.session_state.ultima_accion = f"Venta de {monto:.2f} € confirmada."
                st.rerun()

# Mostrar total invertido (actualizado en tiempo real)
st.markdown("---")
st.markdown(
    f"<div style='background-color:#004481;padding:15px;border-radius:10px;color:white;font-size:18px;font-weight:600;text-align:center;'>"
    f"💼 Total invertido actualmente: {st.session_state.inversion_total:.2f} €"
    f"</div>",
    unsafe_allow_html=True
)

# Mostrar último movimiento (si existe)
if "ultima_accion" in st.session_state:
    st.info(st.session_state.ultima_accion)
