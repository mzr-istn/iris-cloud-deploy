---
title: Iris Multi-Cloud Deployment
emoji: 🌸
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Proyek Multi-Cloud Deployment Model ML Iris (`model_iris.joblib`)

Repositori ini berisi satu paket proyek lengkap untuk menjalankan dan men-*deploy* model klasifikasi bunga Iris (`.joblib`) di **4 lingkungan berbeda** tanpa mengubah file model:
1. **Komputer Lokal** (REST API, Streamlit, atau Gradio)
2. **Render** (sebagai REST API menggunakan FastAPI)
3. **Streamlit Community Cloud** (sebagai Web App interaktif)
4. **Hugging Face Spaces** (sebagai Web App interaktif menggunakan Gradio)

---

## Struktur File Proyek

* `train.py` : Skrip untuk melatih ulang model Iris dan menyimpan file `model_iris.joblib`.
* `model_iris.joblib` : File model terlatih (siap pakai).
* `api_render.py` : Server REST API menggunakan **FastAPI** (untuk Lokal & **Render**).
* `app_streamlit.py` : Antarmuka web menggunakan **Streamlit** (untuk Lokal & **Streamlit Community Cloud**).
* `app.py` : Antarmuka web menggunakan **Gradio** (untuk Lokal & **Hugging Face Spaces**).
* `requirements.txt` : Daftar pustaka Python agar versi lingkungan lokal dan cloud sama.

---

## 1. Menjalankan di Komputer Lokal

1. Buat *virtual environment* dan instal dependensi:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt
   ```
2. *(Opsional)* Latih ulang model untuk menghasilkan `model_iris.joblib`:
   ```bash
   python train.py
   ```
3. Pilih salah satu mode untuk dijalankan di lokal:
   * **Mode REST API (FastAPI):** `python api_render.py` -> buka `http://localhost:8000/docs`
   * **Mode Web UI (Streamlit):** `streamlit run app_streamlit.py` -> buka `http://localhost:8501`
   * **Mode Web UI (Gradio):** `python app.py` -> buka `http://localhost:7860`

---

## 2. Deployment ke Render (REST API Gratis)

1. Unggah seluruh file ini ke repositori **GitHub** Anda.
2. Masuk ke [render.com](https://render.com) -> **New +** -> **Web Service** -> hubungkan repositori GitHub Anda.
3. Isi konfigurasi berikut:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn api_render:app --host 0.0.0.0 --port $PORT`
   * **Instance Type:** `Free`
4. Klik **Deploy Web Service**. Akses dokumentasi API publik di `<URL-RENDER-ANDA>/docs`.

---

## 3. Deployment ke Streamlit Community Cloud (Gratis)

1. Gunakan repositori **GitHub** yang sama.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io) menggunakan akun GitHub.
3. Klik **New app**, pilih repositori dan branch Anda, lalu atur:
   * **Main file path:** `app_streamlit.py`
4. Klik **Deploy!**.

---

## 4. Deployment ke Hugging Face Spaces (Gratis)

File `README.md` ini sudah dilengkapi konfigurasi header YAML untuk Hugging Face Spaces (`sdk: gradio`, `app_file: app.py`).

1. Buat akun di [huggingface.co](https://huggingface.co) -> klik **New Space**.
2. Beri nama Space, pilih **SDK: Gradio**, dan pilih **CPU basic (Free)**.
3. Unggah semua file proyek (`app.py`, `model_iris.joblib`, `requirements.txt`, `README.md`) melalui tab **Files** -> **Add file** -> **Upload files** (atau via `git push`).
4. Hugging Face akan otomatis membangun dan menjalankan aplikasi Anda.
