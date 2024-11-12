# Proyek Analisis Harga Sewa Rumah

Proyek ini melakukan analisis data mengenai harga penyewaan rumah menggunakan dataset yang telah disediakan. Analisis ini mencakup pemrosesan data, visualisasi, dan pembuatan dashboard menggunakan Streamlit.

## Struktur Berkas

- `data/`: Berisi Dataset yang digunakan dalam analisis
- `dashboard/`: Berisi berkas untuk dashboard Streamlit.
- `Latihan_Prediksi_Harga_Rumah.ipynb`: Berkas Jupyter Notebook yang berisi analisis data.
- `requirements.txt`: Daftar library yang diperlukan untuk menjalankan proyek.
- `README.md`: Dokumen ini.
- `url.txt`: Berisi informasi tambahan

## Membuat Virtual Environment

```
python -m venv env
source env/bin/activate       # Untuk MacOS/Linux
env\Scripts\activate          # Untuk Windows
```

## Instalasi Library

```
pip install -r requirements.txt
```

## Menjalankan Dashboard Streamlit

```
streamlit run dashboard/latihan_prediksi_harga_rumah.py
```
