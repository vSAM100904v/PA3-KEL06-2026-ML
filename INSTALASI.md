# Panduan Instalasi & Menjalankan MamaCare AI v2.0

## Prasyarat

- **Python 3.10 atau lebih baru** — cek dengan `python --version`
- **pip** — biasanya sudah ikut saat install Python

---

## Langkah 1 — Siapkan Folder Proyek

Buat satu folder, lalu masukkan semua file berikut ke dalamnya:

```
mamacare/
├── mamacare_train_v2.py
├── mamacare_app_v2.py
├── api_v2.py
├── dataset_anc_bumil_v2.xlsx
└── requirements_v2.txt
```

---

## Langkah 2 — Buat Virtual Environment (Direkomendasikan)

Buka terminal, masuk ke folder proyek, lalu jalankan:

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan — Windows
venv\Scripts\activate

# Aktifkan — Mac / Linux
source venv/bin/activate
```

> Setelah aktif, terminal akan menampilkan `(venv)` di depan prompt.

---

## Langkah 3 — Install Dependensi

```bash
pip install -r requirements_v2.txt
```

Tunggu hingga selesai (sekitar 2–5 menit tergantung koneksi).

---

## Langkah 4 — Training Model (Jalankan Sekali)

```bash
python mamacare_train_v2.py
```

Proses ini akan:
- Membaca `dataset_anc_bumil_v2.xlsx`
- Melatih 2 model (Random Forest Overall + MultiOutputClassifier)
- Menghasilkan 7 file `.pkl` dan 3 gambar visualisasi di folder yang sama

Selesai dalam sekitar **5–10 menit**. Jika berhasil, output terakhir di terminal adalah:

```
✅ TRAINING v2.0 SELESAI
```

> File `.pkl` yang dihasilkan: `model_risk_overall.pkl`, `model_risk_types.pkl`, `scaler_v2.pkl`, `feature_cols_v2.pkl`, `label_map_v2.pkl`, `risk_type_names.pkl`, `feature_importances_v2.pkl`

---

## Langkah 5 — Jalankan Aplikasi

Pilih salah satu atau keduanya sesuai kebutuhan.

### Opsi A — Aplikasi Web (Streamlit)

```bash
streamlit run mamacare_app_v2.py
```

Browser akan terbuka otomatis ke `http://localhost:8501`

---

### Opsi B — REST API (FastAPI)

```bash
uvicorn api_v2:app --host 0.0.0.0 --port 8001 --reload
```

API berjalan di `http://localhost:8001`

Dokumentasi interaktif tersedia di `http://localhost:8001/docs`

---

## Catatan

- File `.pkl` hanya perlu dibuat **sekali**. Selanjutnya langsung ke Langkah 5.
- Jika ada error `ModuleNotFoundError`, pastikan virtual environment sudah aktif dan `pip install` sudah selesai.
- Jika port 8001 sudah dipakai, ganti dengan port lain, contoh: `--port 8002`
