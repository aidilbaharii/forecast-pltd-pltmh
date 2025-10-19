import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Ambil data dari Google Sheet
# ==============================
st.title("📊 Dashboard Forecast Beban H+1 PLTD REMA dan PLTMH PANTAN CUACA")

sheet_url = "https://docs.google.com/spreadsheets/d/19RPYUYHcorItlqUp6vUvnnE6IF3MAiqIWPGnui4YDaw/export?format=csv&gid=0"
data = pd.read_csv(sheet_url)

st.success("✅ Data berhasil diambil dari Google Sheet!")

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

# Input (fitur) dan target
X = data[["V_BUS_PC", "TOTAL_P_PC_KW", "V_BUS_REMA", "TOTAL_P_REMA_KW"]]
y_pltd = data["TOTAL_BEBAN_BUS_REMA_KW"]
y_pltmh = data["TOTAL_BEBAN_BUS_BLANGKEJEREN_KW"]

# ==============================
# 3️⃣ Latih model XGBoost
# ==============================
st.info("⏳ Melatih model XGBoost untuk PLTD dan PLTMH...")

model_pltd = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model_pltmh = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

model_pltd.fit(X, y_pltd)
model_pltmh.fit(X, y_pltmh)

st.success("✅ Model berhasil dilatih!")

# ==============================
# 4️⃣ Prediksi beban H+1 (24 jam ke depan)
# ==============================
X_pred = X.tail(24)
pred_pltd = model_pltd.predict(X_pred)
pred_pltmh = model_pltmh.predict(X_pred)

result = pd.DataFrame({
    "Jam ke-": range(1, len(pred_pltd) + 1),
    "Prediksi PLTD (kW)": pred_pltd.round(2),
    "Prediksi PLTMH (kW)": pred_pltmh.round(2)
})

# ==============================
# 5️⃣ Tampilkan tabel & grafik
# ==============================
st.subheader("📈 Hasil Prediksi Beban Harian (H+1)")
st.dataframe(result, use_container_width=True)

# Plot grafik
# Plot grafik
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result["Jam ke-"], result["Prediksi PLTD (kW)"], marker="o", color="blue", label="PLTD")
ax.plot(result["Jam ke-"], result["Prediksi PLTMH (kW)"], marker="s", color="orange", label="PLTMH")
ax.set_xlabel("Jam ke-")
ax.set_ylabel("Beban (kW)")
ax.set_title("Peramalan Beban H+1 per Jam (PLTD vs PLTMH)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ==============================
# 6️⃣ Tambahan: Download hasil prediksi
# ==============================
csv = result.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Hasil Prediksi (CSV)", csv, "forecast_hplus1.csv", "text/csv")

st.caption("Data sumber: Google Sheet DB STREAMLIT. Model: XGBoost Regressor.")

