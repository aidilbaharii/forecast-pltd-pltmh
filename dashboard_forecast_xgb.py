# =========================================================
# DASHBOARD FORECASTING BEBAN H+1 - PLTD REMA & PLTMH PANTAN CUACA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os, joblib

# ---------- Konfigurasi halaman ----------
st.set_page_config(page_title="Forecast H+1 PLTD & PLTMH", layout="wide")
plt.rcParams["font.size"] = 11

# ---------- Auto refresh tiap 1 jam ----------
REFRESH_INTERVAL = 3600
st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_INTERVAL}'>", unsafe_allow_html=True)
st.info(f"⏳ Halaman auto-refresh setiap **{int(REFRESH_INTERVAL/60)} menit**.")

# ---------- Ambil data dari Google Sheet ----------
st.title("📊 Dashboard Forecast Beban H+1 — PLTD REMA & PLTMH PANTAN CUACA")

sheet_url = "https://docs.google.com/spreadsheets/d/19RPYUYHcorItlqUp6vUvnnE6IF3MAiqIWPGnui4YDaw/export?format=csv&gid=0"

@st.cache_data(show_spinner=True, ttl=300)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

try:
    data = load_data(sheet_url)
    st.success("✅ Data berhasil diambil dari Google Sheet!")
except Exception as e:
    st.error(f"❌ Gagal mengambil data: {e}")
    st.stop()

if st.button("🔄 Refresh Data Sekarang"):
    st.cache_data.clear()
    st.rerun()

st.write("🧾 Kolom terbaca:", list(data.columns))

# ---------- Validasi kolom ----------
if not ({"DATE", "TIME"} <= set(data.columns)):
    st.error("❌ Kolom DATE dan TIME tidak ditemukan di data.")
    st.stop()

# Gabungkan DATE + TIME
data["Datetime"] = pd.to_datetime(data["DATE"] + " " + data["TIME"], errors="coerce")
data = data.sort_values("Datetime").reset_index(drop=True)

num_cols = [
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"
]
missing = [c for c in num_cols if c not in data.columns]
if missing:
    st.error(f"❌ Kolom numerik hilang: {missing}")
    st.stop()

for c in num_cols:
    data[c] = data[c].astype(str).str.replace(",", ".").str.replace(" ", "")
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=["Datetime"] + num_cols).copy()

# ---------- Fitur waktu ----------
data["hour"] = data["Datetime"].dt.hour
data["dayofweek"] = data["Datetime"].dt.dayofweek
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

# ---------- Fitur lag & rolling ----------
def add_lag_roll(df, col):
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag24"] = df[col].shift(24)
    df[f"{col}_roll3"] = df[col].shift(1).rolling(3).mean()
    df[f"{col}_roll24"] = df[col].shift(1).rolling(24).mean()
    return df

for c in ["TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]:
    data = add_lag_roll(data, c)

# ---------- Fitur pola harian (7 hari terakhir) ----------
data["avg_hourly_rema"] = data.groupby("hour")["TOTAL_BEBAN_BUS_REMA_KW"].transform(lambda x: x.rolling(24*7, min_periods=1).mean())
data["avg_hourly_bkj"] = data.groupby("hour")["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].transform(lambda x: x.rolling(24*7, min_periods=1).mean())

data = data.dropna().reset_index(drop=True)

# ---------- Siapkan fitur & target ----------
feat_pltd = [
    "hour", "dayofweek", "hour_sin", "hour_cos",
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_REMA_KW_lag1", "TOTAL_BEBAN_BUS_REMA_KW_lag24",
    "TOTAL_BEBAN_BUS_REMA_KW_roll3", "TOTAL_BEBAN_BUS_REMA_KW_roll24",
    "avg_hourly_rema"
]
feat_pltmh = [
    "hour", "dayofweek", "hour_sin", "hour_cos",
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag1", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag24",
    "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll3", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll24",
    "avg_hourly_bkj"
]

X_pltd = data[feat_pltd].copy()
X_pltmh = data[feat_pltmh].copy()
y_pltd = np.log1p(data["TOTAL_BEBAN_BUS_REMA_KW"])
y_pltmh = np.log1p(data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"])

# ---------- Scaling & Training (cached) ----------
@st.cache_resource(show_spinner=True)
def train_models(Xp, yp, Xm, ym):
    scaler_p = StandardScaler()
    scaler_m = StandardScaler()
    Xp_scaled = scaler_p.fit_transform(Xp)
    Xm_scaled = scaler_m.fit_transform(Xm)

    model_p = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
    model_m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
    model_p.fit(Xp_scaled, yp)
    model_m.fit(Xm_scaled, ym)
    return model_p, model_m, scaler_p, scaler_m

st.info("⏳ Melatih model (cached)...")
model_pltd, model_pltmh, scaler_pltd, scaler_pltmh = train_models(X_pltd, y_pltd, X_pltmh, y_pltmh)
st.success("✅ Model siap digunakan!")

# ---------- Baseline (profil harian 3 hari terakhir) ----------
def recent_hourly_profile(series, hours=24*3):
    last = series.tail(hours)
    tmp = data.tail(hours).copy()
    tmp["_y"] = last.values
    prof = tmp.groupby(tmp["hour"])["_y"].mean().reindex(range(24), fill_value=tmp["_y"].mean())
    return prof.values

baseline_rema = recent_hourly_profile(data["TOTAL_BEBAN_BUS_REMA_KW"])
baseline_bkj  = recent_hourly_profile(data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"])

# ---------- Forecast 24 jam ke depan ----------
st.info("📈 Membuat prediksi H+1 (24 jam ke depan)...")

last_block = data.iloc[-24:].copy()
forecast_rows = []
current_time = last_block["Datetime"].iloc[-1]

for i in range(24):
    cur_time = current_time + timedelta(hours=1)
    cur_hour = cur_time.hour
    cur_dow = cur_time.dayofweek

    row = {
        "hour": cur_hour,
        "dayofweek": cur_dow,
        "hour_sin": np.sin(2*np.pi*cur_hour/24),
        "hour_cos": np.cos(2*np.pi*cur_hour/24),
        "V_BUS_PC": last_block["V_BUS_PC"].iloc[-1],
        "TOTAL_P_PC_KW": last_block["TOTAL_P_PC_KW"].iloc[-1],
        "V_BUS_REMA": last_block["V_BUS_REMA"].iloc[-1],
        "TOTAL_P_REMA_KW": last_block["TOTAL_P_REMA_KW"].iloc[-1],
        "TOTAL_BEBAN_BUS_REMA_KW_lag1": last_block["TOTAL_BEBAN_BUS_REMA_KW"].iloc[-1],
        "TOTAL_BEBAN_BUS_REMA_KW_lag24": last_block["TOTAL_BEBAN_BUS_REMA_KW"].iloc[-24],
        "TOTAL_BEBAN_BUS_REMA_KW_roll3":  last_block["TOTAL_BEBAN_BUS_REMA_KW"].tail(3).mean(),
        "TOTAL_BEBAN_BUS_REMA_KW_roll24": last_block["TOTAL_BEBAN_BUS_REMA_KW"].tail(24).mean(),
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag1": last_block["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].iloc[-1],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag24": last_block["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].iloc[-24],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll3":  last_block["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].tail(3).mean(),
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll24": last_block["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].tail(24).mean(),
        "avg_hourly_rema": data.loc[data["hour"]==cur_hour, "avg_hourly_rema"].tail(24*7).mean(),
        "avg_hourly_bkj":  data.loc[data["hour"]==cur_hour, "avg_hourly_bkj"].tail(24*7).mean(),
    }

    X_row_p = pd.DataFrame([{k: row[k] for k in feat_pltd}])
    X_row_m = pd.DataFrame([{k: row[k] for k in feat_pltmh}])

    pred_p = np.expm1(model_pltd.predict(scaler_pltd.transform(X_row_p)))[0]
    pred_m = np.expm1(model_pltmh.predict(scaler_pltmh.transform(X_row_m)))[0]

    # Baseline correction
    pred_p = 0.7 * pred_p + 0.3 * baseline_rema[cur_hour]
    pred_m = 0.7 * pred_m + 0.3 * baseline_bkj[cur_hour]

    total_load = pred_p + pred_m  # 💡 Kolom total prediksi beban

    forecast_rows.append({
        "Datetime": cur_time,
        "Jam ke-": cur_hour + 1,
        "WO PLTD (kW)": round(pred_p, 2),
        "WO PLTMH (kW)": round(pred_m, 2),
        "Total Prediksi Beban (kW)": round(total_load, 2)
    })

    appended = {
        "Datetime": cur_time,
        "V_BUS_PC": row["V_BUS_PC"],
        "TOTAL_P_PC_KW": row["TOTAL_P_PC_KW"],
        "V_BUS_REMA": row["V_BUS_REMA"],
        "TOTAL_P_REMA_KW": row["TOTAL_P_REMA_KW"],
        "TOTAL_BEBAN_BUS_REMA_KW": pred_p,
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW": pred_m,
        "hour": cur_hour,
        "dayofweek": cur_dow
    }
    last_block = pd.concat([last_block, pd.DataFrame([appended])], ignore_index=True)
    current_time = cur_time

result = pd.DataFrame(forecast_rows)
forecast_date = (datetime.now() + timedelta(days=1)).date()

# ---------- Tampilkan tabel ----------
st.subheader(f"📊 Hasil Prediksi Beban H+1 ({forecast_date.strftime('%d %B %Y')})")
st.dataframe(result, use_container_width=True, height=460)

# ---------- Grafik ----------
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(result["Jam ke-"], result["WO PLTD (kW)"], marker="o", label="PLTD")
ax.plot(result["Jam ke-"], result["WO PLTMH (kW)"], marker="s", label="PLTMH")
ax.plot(result["Jam ke-"], result["Total Prediksi Beban (kW)"], marker="^", linestyle="--", label="Total Beban")
ax.set_title(f"Peramalan Beban H+1 ({forecast_date.strftime('%d %b %Y')})")
ax.set_xlabel("Jam ke-")
ax.set_ylabel("Beban (kW)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig, use_container_width=True)

# ---------- Download ----------
csv = result.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Hasil Prediksi (CSV)", csv, "forecast_hplus1.csv", "text/csv")

st.caption("📘 Sumber: Google Sheet | Model: Gradient Boosting Regressor | Fitur lengkap + kolom total prediksi beban | Auto-refresh tiap 1 jam.")


