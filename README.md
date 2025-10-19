# 🔋 Forecasting Beban Harian PLTD dan PLTMH - UP2D ACEH

Proyek ini dibuat untuk memprediksi beban listrik harian (H+1) antara **PLTD REMA** dan **PLTMH PANTAN CUACA** 
berdasarkan data operasi sistem kelistrikan yang diambil dari Profile beban harian ULP Blang Kejeren.

## 🚀 Fitur Utama
- Menarik data langsung dari Google Sheets
- Melatih model *Gradient Boosting Regressor* untuk forecasting
- Menampilkan hasil dalam bentuk tabel dan grafik interaktif
- Dapat dijalankan lokal (VSCode / Streamlit) atau di *Streamlit Cloud*

## 🧠 Algoritma yang digunakan
Model utama menggunakan:
- **Gradient Boosting Regressor (sklearn)**
- Dapat dikembangkan menjadi *XGBoost* untuk akurasi lebih tinggi

## 📦 Instalasi
Clone repositori ini dan install dependensi:
```bash
pip install -r requirements.txt
