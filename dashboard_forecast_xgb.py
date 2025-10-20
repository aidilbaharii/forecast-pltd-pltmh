# =========================================================
# DASHBOARD FORECASTING BEBAN H+1 - PLTD REMA & PLTMH PANTAN CUACA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# 🔁 Auto-refresh Streamlit (setiap 1 jam)
# ==============================
REFRESH_INTERVAL = 3600
st.write(f"⏳ Halaman akan auto-refresh setiap {REFRESH_INTERVAL/60:.0f} menit.")
st.markdown(f"<meta http-equiv='refresh' content='{REFRESH_INTERVAL}'>", unsafe_allow_html=True)

# ==============================
# 1️⃣ Ambil data dari Google Sheet
# ==============================
st.title("📊 Dashboard Forecast Beban H+1 - PLTD REMA & PLTMH PANTAN CUACA")

sheet_url = "https://docs.google.com/spreadsheets/d/19RPYUYHcorItlqUp6vUvnnE6IF3MAiqIWPGnui4YDaw/export?format=csv&gid=0"

try:
    data = pd.read_csv(sheet_url)
    st.success("✅ Data berhasil diambil dari Google Sheet!")
except Exception as e:
    st.error(f"❌ Gagal mengambil data: {e}")
    st.stop()

if st.button("🔄 Refresh Data Sekarang"):
    st.experimental_rerun()

# ==============================
# 2️⃣ Persiapan & pembersihan data
# ==============================
st.write("🧾 Kolom yang terbaca:", list(data.columns))

# Gabungkan DATE + TIME menjadi datetime penuh
if "DATE" in data.columns and "TIME" in data.columns:
    data["Datetime"] = pd.to_datetime(data["DATE"] + " " + data["TIME"], errors="coerce")
else:
    st.error("❌ Kolom DATE dan TIME tidak ditemukan di Google Sheet.")
    st.stop()

# Sortir berdasarkan waktu
data = data.sort_values("Datetime").reset_index(drop=True)

# Pastikan kolom numerik dibersihkan
numeric_cols = [
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"
]
for c in numeric_cols:
    data[c] = data[c].astype(str).str.replace(",", ".").astype(float)

# ==============================
# 3️⃣ Tambahkan fitur waktu (hour, sin/cos, lag, rolling)
# ==============================
data["hour"] = data["Datetime"].dt.hour
data["dayofweek"] = data["Datetime"].dt.dayofweek
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

# Tambahkan lag dan rolling window
target_cols = ["TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]
for target in target_cols:
    data[f"{target}_lag1"] = data[target].shift(1)
    data[f"{target}_lag24"] = data[target].shift(24)
    data[f"{target}_roll3"] = data[target].shift(1).rolling(3).mean()
    data[f"{target}_roll24"] = data[target].shift(1).rolling(24).mean()

data = data.dropna().reset_index(drop=True)

# ==============================
# 4️⃣ Training model untuk PLTD dan PLTMH
# ==============================
st.info("⏳ Melatih model Gradient Boosting Regressor ...")

features = [
    "hour", "dayofweek", "hour_sin", "hour_cos",
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_REMA_KW_lag1", "TOTAL_BEBAN_BUS_REMA_KW_lag24",
    "TOTAL_BEBAN_BUS_REMA_KW_roll3", "TOTAL_BEBAN_BUS_REMA_KW_roll24"
]

X_pltd = data[features]
y_pltd = data["TOTAL_BEBAN_BUS_REMA_KW"]

features_pltmh = [
    "hour", "dayofweek", "hour_sin", "hour_cos",
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag1", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag24",
    "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll3", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll24"
]
X_pltmh = data[features_pltmh]
y_pltmh = data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]

model_pltd = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
model_pltmh = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
model_pltd.fit(X_pltd, y_pltd)
model_pltmh.fit(X_pltmh, y_pltmh)

st.success("✅ Model selesai dilatih!")

# ==============================
# 5️⃣ Prediksi 24 jam ke depan (iteratif)
# ==============================
st.info("📈 Membuat prediksi H+1 (24 jam ke depan)...")

last_data = data.iloc[-24:].copy()
forecast_rows = []
current_time = last_data["Datetime"].iloc[-1]

for i in range(24):
    cur_time = current_time + timedelta(hours=1)
    cur_hour = cur_time.hour
    cur_dow = cur_time.dayofweek

    cur = pd.DataFrame({
        "hour": [cur_hour],
        "dayofweek": [cur_dow],
        "hour_sin": [np.sin(2 * np.pi * cur_hour / 24)],
        "hour_cos": [np.cos(2 * np.pi * cur_hour / 24)],
        "V_BUS_PC": [last_data["V_BUS_PC"].iloc[-1]],
        "TOTAL_P_PC_KW": [last_data["TOTAL_P_PC_KW"].iloc[-1]],
        "V_BUS_REMA": [last_data["V_BUS_REMA"].iloc[-1]],
        "TOTAL_P_REMA_KW": [last_data["TOTAL_P_REMA_KW"].iloc[-1]],
        "TOTAL_BEBAN_BUS_REMA_KW_lag1": [last_data["TOTAL_BEBAN_BUS_REMA_KW"].iloc[-1]],
        "TOTAL_BEBAN_BUS_REMA_KW_lag24": [last_data["TOTAL_BEBAN_BUS_REMA_KW"].iloc[-24]],
        "TOTAL_BEBAN_BUS_REMA_KW_roll3": [last_data["TOTAL_BEBAN_BUS_REMA_KW"].tail(3).mean()],
        "TOTAL_BEBAN_BUS_REMA_KW_roll24": [last_data["TOTAL_BEBAN_BUS_REMA_KW"].tail(24).mean()],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag1": [last_data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].iloc[-1]],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_lag24": [last_data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].iloc[-24]],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll3": [last_data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].tail(3).mean()],
        "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW_roll24": [last_data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"].tail(24).mean()]
    })

    pred_pltd = model_pltd.predict(cur[features])[0]
    pred_pltmh = model_pltmh.predict(cur[features_pltmh])[0]

    cur["TOTAL_BEBAN_BUS_REMA_KW"] = pred_pltd
    cur["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"] = pred_pltmh
    cur["Datetime"] = cur_time

    forecast_rows.append(cur)
    last_data = pd.concat([last_data, cur], ignore_index=True)
    current_time = cur_time

result = pd.concat(forecast_rows, ignore_index=True)
forecast_date = (datetime.now() + timedelta(days=1)).date()

# ==============================
# 6️⃣ Tampilkan hasil
# ==============================
result["Jam ke-"] = result["Datetime"].dt.hour + 1
st.subheader(f"📊 Hasil Prediksi Beban H+1 ({forecast_date.strftime('%d %B %Y')})")

tabel = result[["Datetime", "Jam ke-", "TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]]
tabel.columns = ["Tanggal & Jam", "Jam ke-", "Prediksi PLTD (kW)", "Prediksi PLTMH (kW)"]
st.dataframe(tabel, use_container_width=True)

# Grafik
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tabel["Jam ke-"], tabel["Prediksi PLTD (kW)"], marker="o", label="PLTD")
ax.plot(tabel["Jam ke-"], tabel["Prediksi PLTMH (kW)"], marker="s", label="PLTMH")
ax.set_title(f"Peramalan Beban H+1 ({forecast_date.strftime('%d %b %Y')})")
ax.set_xlabel("Jam ke-")
ax.set_ylabel("Beban (kW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Download hasil
csv = tabel.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Hasil Prediksi (CSV)", csv, "forecast_hplus1.csv", "text/csv")

st.caption("📘 Data sumber: Google Sheet | Model: Gradient Boosting Regressor | Auto-refresh tiap 1 jam.")

