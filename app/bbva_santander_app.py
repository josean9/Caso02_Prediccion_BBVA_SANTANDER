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
st.set_page_config(page_title="Predicción LSTM — BBVA y Santander", layout="wide")
st.title("📈 Predicción de precios — BBVA y Santander")
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
df_stock = df_stock.fillna(method="bfill")

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
# 5️⃣ CARGAR MODELO Y HACER PREDICCIÓN
# ============================================================
model = load_model("best_bbva_lstm.pth" if bank=="BBVA" else "best_san_lstm.pth", input_size)
model.eval()

# ============================================================
# 🔮 PREDICCIÓN COMPLETA: histórica + forecast
# ============================================================
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

lookback = 60
target_col = "close"
feature_cols = [
    "open","high","low","volume","range","return","vol_rel","eventos_negativos",
    "sma_7","sma_30","sma_60","inflation_rate","deposit_rate",
    "marginal_rate","refinancing_rate","dividends_f"
]

# ------------------ Escalado igual que en entrenamiento ------------------
scaler = MinMaxScaler()
scaler.fit(df_final[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])

close_min = scaler.data_min_[3]  # índice de 'close'
close_max = scaler.data_max_[3]

def scale_df(df):
    df_scaled = df.copy()
    df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]] = \
        scaler.transform(df_scaled[["open","high","low","close","volume","sma_7","sma_30","sma_60"]])
    df_scaled[["inflation_rate","deposit_rate","marginal_rate","refinancing_rate"]] /= 10.0
    return df_scaled

df_scaled = scale_df(df_final)

# ============================================================
# 1️⃣ PREDICCIÓN HISTÓRICA (día a día alineada)
# ============================================================
def create_sequences(df, lookback):
    Xs, ys = [], []
    for i in range(len(df) - lookback):
        Xs.append(df.iloc[i:i+lookback][feature_cols].values)
        ys.append(df.iloc[i+lookback][target_col])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(df_scaled, lookback)

model.eval()
preds_scaled = []
with torch.no_grad():
    for i in range(len(X_seq)):
        x_t = torch.tensor(X_seq[i], dtype=torch.float32).unsqueeze(0)
        y_pred = model(x_t).cpu().numpy().flatten()[0]
        preds_scaled.append(y_pred)

# Desescalar a euros
preds_real = [close_min + p*(close_max - close_min) for p in preds_scaled]
y_real = [close_min + y*(close_max - close_min) for y in y_seq]

dates_hist = df_final.index[lookback:]
n_tail = 100
dates_tail = dates_hist[-n_tail:]
y_real_tail = y_real[-n_tail:]
y_pred_tail = preds_real[-n_tail:]

# ============================================================
# 2️⃣ FORECAST FUTURO (autorregresivo, 1–7 nov)
# ============================================================
from copy import deepcopy

def recompute_derived(df):
    df = df.copy()
    df["return"] = df[target_col].pct_change()
    df["sma_7"]  = df[target_col].rolling(7, min_periods=1).mean()
    df["sma_30"] = df[target_col].rolling(30, min_periods=1).mean()
    df["sma_60"] = df[target_col].rolling(60, min_periods=1).mean()
    vol_ma10 = df["volume"].rolling(10, min_periods=1).mean().replace(0, np.nan)
    df["vol_rel"] = (df["volume"] / vol_ma10.replace(0, np.nan)).fillna(method="ffill").fillna(1.0)
    base = df["open"].replace(0, np.nan)
    df["range"] = ((df["high"] - df["low"]) / base).fillna(0.0)
    return df

window = df_final.tail(lookback).copy()
forecast_days = 7
future_dates = pd.bdate_range("2025-11-01", periods=forecast_days)
forecast_preds_real = []

for d in future_dates:
    win_scaled = scale_df(window)
    x_t = torch.tensor(win_scaled[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_next_scaled = model(x_t).cpu().numpy().reshape(-1)[0]
    y_next_real = close_min + y_next_scaled*(close_max - close_min)
    forecast_preds_real.append(y_next_real)

    prev_close = window["close"].iloc[-1]
    new_row = deepcopy(window.iloc[-1])
    new_row["open"] = prev_close
    new_row["close"] = y_next_real
    new_row["high"] = max(prev_close, y_next_real) * 1.01
    new_row["low"] = min(prev_close, y_next_real) * 0.99
    window = pd.concat([window, pd.DataFrame([new_row], index=[d])])
    window = recompute_derived(window)
    window = window.tail(lookback)

# ============================================================
# 3️⃣ GRÁFICO INTERACTIVO
# ============================================================
fig = make_subplots(specs=[[{"secondary_y": False}]])

# Real
fig.add_trace(go.Scatter(
    x=dates_tail, y=y_real_tail,
    mode="lines", name="Cierre real (€)",
    line=dict(color="#1f77b4", width=2)
))

# Predicción histórica
fig.add_trace(go.Scatter(
    x=dates_tail, y=y_pred_tail,
    mode="lines", name="Predicción histórica LSTM (€)",
    line=dict(color="#ff7f0e", width=2, dash="dot")
))

# Forecast futuro
fig.add_trace(go.Scatter(
    x=future_dates, y=forecast_preds_real,
    mode="lines+markers", name="Predicción futura (1–7 nov)",
    line=dict(color="#00e676", width=2, dash="dash"),
    marker=dict(size=8)
))

# Conexión (solo línea estética, sin hover ni leyenda)
fig.add_trace(go.Scatter(
    x=[dates_tail[-1], future_dates[0]],
    y=[y_pred_tail[-1], forecast_preds_real[0]],
    mode="lines",
    line=dict(color="gray", width=1, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))


fig.update_layout(
    title=f"📈 Predicción LSTM — {bank} (últimos 100 días + forecast 1–7 nov 2025)",
    xaxis_title="Fecha", yaxis_title="Precio (€)",
    hovermode="x unified", template="plotly_dark",
    legend=dict(orientation="h", y=-0.25), height=600
)
fig.update_xaxes(rangeslider_visible=True, tickformat="%Y-%m-%d")
fig.update_traces(hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} €")
st.plotly_chart(fig, use_container_width=True)
