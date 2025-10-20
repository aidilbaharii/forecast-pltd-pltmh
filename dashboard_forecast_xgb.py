# =========================================================
# DASHBOARD FORECASTING BEBAN H+1 - PLTD REMA & PLTMH PANTAN CUACA
# =========================================================

import streamlit as st
import pandas as pd
import datetime
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# 🔁 Auto-refresh versi native Streamlit (tanpa library eksternal)
# ==============================
import time

# Auto-refresh setiap 1 jam (3600 detik)
# Streamlit akan me-refresh halaman setelah interval selesai
REFRESH_INTERVAL = 3600  # detik
st.write(f"⏳ Halaman akan auto-refresh setiap {REFRESH_INTERVAL/60:.0f} menit.")
st.markdown(
    f"""
    <meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
    """,
    unsafe_allow_html=True
)


# ==============================
# 1️⃣ Ambil data dari Google Sheet
# ==============================
st.title("📊 Dashboard Forecast Beban H+1 PLTD REMA & PLTMH PANTAN CUACA")

sheet_url = "https://docs.google.com/spreadsheets/d/19RPYUYHcorItlqUp6vUvnnE6IF3MAiqIWPGnui4YDaw/export?format=csv&gid=0"

try:
    data = pd.read_csv(sheet_url)
    st.success("✅ Data berhasil diambil dari Google Sheet!")
except Exception as e:
    st.error(f"❌ Gagal mengambil data: {e}")
    st.stop()

# Tombol manual refresh
if st.button("🔄 Refresh Data Sekarang"):
    st.experimental_rerun()

# ==============================
# 2️⃣ Bersihkan dan siapkan data
# ==============================
cols = [
    "V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW",
    "TOTAL_BEBAN_BUS_REMA_KW", "TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"
]

# Ganti koma jadi titik dan ubah ke float
for c in cols:
    data[c] = data[c].astype(str).str.replace(",", ".").str.strip()
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Hapus baris kosong
data = data.dropna(subset=cols)

# Coba ambil tanggal terakhir dari data
if 'Tanggal' in data.columns:
    last_date = pd.to_datetime(data['Tanggal'].dropna().iloc[-1])
else:
    last_date = datetime.date.today()

forecast_date = last_date + datetime.timedelta(days=1)

st.info(f"📅 Tanggal peramalan beban: **{forecast_date.strftime('%A, %d %B %Y')}**")
st.write("🧾 Kolom yang terbaca dari Google Sheet:", list(data.columns))

# Tambahkan fitur waktu
import datetime

# Kalau tidak ada kolom tanggal, buat otomatis dari index terakhir
if "Tanggal" in data.columns:
    data["Tanggal"] = pd.to_datetime(data["Tanggal"], errors="coerce")
else:
    data["Tanggal"] = datetime.date.today()

# Kalau tidak ada kolom Jam, buat kolom jam urut (0–23)
if "Jam" in data.columns:
    data["Jam"] = pd.to_datetime(data["Jam"], format="%H:%M", errors="coerce").dt.hour.fillna(0)
else:
    data["Jam"] = list(range(len(data)))  # asumsi tiap baris = 1 jam

# Fitur sin/cos untuk representasi waktu periodik (0–24 jam)
import numpy as np
data["sin_hour"] = np.sin(2 * np.pi * data["Jam"] / 24)
data["cos_hour"] = np.cos(2 * np.pi * data["Jam"] / 24)


# Input (fitur) dan target
X = data[["V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW", "sin_hour", "cos_hour"]]
y_pltd = data["TOTAL_BEBAN_BUS_REMA_KW"]
y_pltmh = data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]

# ==============================
# 3️⃣ Latih model
# ==============================
st.info("⏳ Melatih model Gradient Boosting Regressor untuk PLTD dan PLTMH...")

model_pltd = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model_pltmh = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

model_pltd.fit(X, y_pltd)
model_pltmh.fit(X, y_pltmh)

st.success("✅ Model berhasil dilatih!")

# ==============================
# 4️⃣ Prediksi beban H+1 (24 jam ke depan)
# ==============================
import datetime

forecast_date = (datetime.date.today() + datetime.timedelta(days=1))
future_hours = np.arange(0, 24)

# Ambil rata-rata kondisi terakhir untuk nilai fitur non-waktu
mean_values = X.mean()

# Buat dataframe prediksi dengan kombinasi sin/cos jam
future_data = pd.DataFrame({
    "sin_hour": np.sin(2 * np.pi * future_hours / 24),
    "cos_hour": np.cos(2 * np.pi * future_hours / 24),
    "V_BUS_PC": mean_values["V_BUS_PC"],
    "TOTAL_P_PC_KW": mean_values["TOTAL_P_PC_KW"],
    "V_BUS_REMA": mean_values["V_BUS_REMA"],
    "TOTAL_P_REMA_KW": mean_values["TOTAL_P_REMA_KW"],
})

# Prediksi 24 jam ke depan
pred_pltd = model_pltd.predict(future_data)
pred_pltmh = model_pltmh.predict(future_data)

# Gabungkan hasil
result = pd.DataFrame({
    "Tanggal": forecast_date,
    "Jam ke-": range(1, 25),
    "Prediksi PLTD (kW)": pred_pltd.round(2),
    "Prediksi PLTMH (kW)": pred_pltmh.round(2)
})

# ==============================
# 5️⃣ Tampilkan tabel & grafik
# ==============================
st.subheader("📈 Hasil Prediksi Beban Harian (H+1)")
st.dataframe(result, use_container_width=True)

# Plot grafik
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result["Jam ke-"], result["Prediksi PLTD (kW)"], marker="o", color="blue", label="PLTD")
ax.plot(result["Jam ke-"], result["Prediksi PLTMH (kW)"], marker="s", color="orange", label="PLTMH")
ax.set_xlabel("Jam ke-")
ax.set_ylabel("Beban (kW)")
ax.set_title(f"Peramalan Beban H+1 ({forecast_date.strftime('%d %b %Y')})")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ==============================
# 6️⃣ Tambahan: Download hasil prediksi
# ==============================
csv = result.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Hasil Prediksi (CSV)", csv, "forecast_hplus1.csv", "text/csv")

st.caption("📘 Data sumber: Google Sheet DB STREAMLIT | Model: Gradient Boosting Regressor | Auto-refresh setiap 1 jam.")




