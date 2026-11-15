# 🌸 Iris Classifier - App Alternatives Guide

Repositori ini menyediakan **3 alternatif aplikasi Streamlit** yang dapat dijalankan di lokal maupun di-deploy ke **Streamlit Community Cloud**.

---

## 📋 Perbandingan Aplikasi

| Fitur | `app_streamlit.py` | `app_dashboard.py` | `app_api.py` | `app_explorer.py` |
|-------|:--:|:--:|:--:|:--:|
| **Tipe** | Simple | Professional | API-like | Educational |
| **Kompleksitas** | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Single Prediction** | ✅ | ✅ | ✅ | ✅ |
| **Batch CSV Upload** | ❌ | ✅ | ❌ | ❌ |
| **JSON Response** | ❌ | ❌ | ✅ | ❌ |
| **Model Info** | ❌ | ✅ | ✅ | ✅ |
| **Visualizations** | ❌ | ✅ | ❌ | ✅ |
| **Decision Boundary** | ❌ | ❌ | ❌ | ✅ |
| **Feature Analysis** | ❌ | ❌ | ❌ | ✅ |
| **Mobile Friendly** | ✅ | ❌ | ✅ | ❌ |
| **Load Time** | Cepat ⚡ | Medium ⚡⚡ | Cepat ⚡ | Medium ⚡⚡ |

---

## 🚀 Cara Menjalankan di Lokal

### Prasyarat
```bash
cd iris-multi-cloud-deployment
source venv/bin/activate  # atau venv\Scripts\activate di Windows
pip install -r requirements.txt
```

### Jalankan Salah Satu Aplikasi

**1️⃣ Simple UI (Default)**
```bash
streamlit run app_streamlit.py
```
→ Buka `http://localhost:8501`

**2️⃣ Dashboard (Profesional)**
```bash
streamlit run app_dashboard.py
```
→ Buka `http://localhost:8501`

**3️⃣ API Simulator**
```bash
streamlit run app_api.py
```
→ Buka `http://localhost:8501`

**4️⃣ Explorer (Educational)**
```bash
streamlit run app_explorer.py
```
→ Buka `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Community Cloud

### Setup Pertama Kali

1. **Push ke GitHub:**
   ```bash
   git add .
   git commit -m "Add alternative Streamlit apps"
   git push origin main
   ```

2. **Login ke Streamlit Cloud:**
   - Buka https://share.streamlit.io
   - Klik **"New app"**
   - Pilih repository dan branch Anda

3. **Deploy Aplikasi yang Diinginkan:**

#### Deploy `app_dashboard.py` (Rekomendasi)
```
Repository: mzr-istn/iris-cloud-deploy
Branch: main
Main file path: app_dashboard.py
```
→ URL: `https://share.streamlit.io/mzr-istn/iris-cloud-deploy/main/app_dashboard.py`

#### Deploy `app_api.py` (API Testing)
```
Repository: mzr-istn/iris-cloud-deploy
Branch: main
Main file path: app_api.py
```

#### Deploy `app_explorer.py` (Learning)
```
Repository: mzr-istn/iris-cloud-deploy
Branch: main
Main file path: app_explorer.py
```

### Mengganti Deployed App
- Di Streamlit Cloud, klik **"..."** → **"Settings"**
- Ubah **"Main file path"** ke app yang diinginkan
- Klik **"Save"** dan tunggu redeploy

---

## 📊 Deskripsi Lengkap Setiap Aplikasi

### 1️⃣ `app_streamlit.py` - Simple UI (Default)
**Gunakan untuk:** Produksi sederhana, mobile-friendly

**Fitur:**
- ✅ Input slider untuk single prediction
- ✅ Probabilitas per spesies
- ✅ Visualisasi bar chart confidence
- ✅ Cepat & responsive
- ✅ Cocok untuk demonstrasi cepat

**Use Case:**
- Demo sederhana
- Presentation
- Tutorial pembelajaran
- Mobile access

**Load Time:** ~2 detik

---

### 2️⃣ `app_dashboard.py` - Professional Dashboard
**Gunakan untuk:** Production-grade application

**Fitur:**
- ✅ **Tab 1 - Single Prediction:** Input slider + chart confidence
- ✅ **Tab 2 - Batch CSV:** Upload CSV dengan multiple rows
  - Prediksi semua sekaligus
  - Download hasil sebagai CSV
- ✅ **Tab 3 - Model Info:** Detail model, features, ranges
- ✅ **Tab 4 - Visualizations:** 
  - Prediction heatmap
  - Distribution chart

**Use Case:**
- Business application
- ML operations (MLOps)
- Production deployment
- Batch processing workflow

**Load Time:** ~4 detik

---

### 3️⃣ `app_api.py` - API Simulator
**Gunakan untuk:** Testing, API development, integration

**Fitur:**
- ✅ JSON input/output format
- ✅ API documentation
- ✅ cURL example
- ✅ Download response as JSON
- ✅ Similar to FastAPI response

**Use Case:**
- Test API sebelum production
- Development & testing
- API integration documentation
- Learning REST API concepts

**Load Time:** ~2 detik

**Contoh Response:**
```json
{
  "status": "success",
  "data": {
    "input": {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    "prediction": {
      "species": "Iris Setosa",
      "species_id": 0,
      "confidence": 95.5
    },
    "probabilities": {
      "iris_setosa": 95.5,
      "iris_versicolor": 4.2,
      "iris_virginica": 0.3
    }
  }
}
```

---

### 4️⃣ `app_explorer.py` - Interactive Explorer
**Gunakan untuk:** Education, model understanding, research

**Fitur:**
- ✅ **Tab 1 - Dataset Explorer:**
  - Sample dataset preview
  - Feature statistics
  - Species distribution
- ✅ **Tab 2 - Decision Boundaries:**
  - Visualize model separation
  - Select feature pairs
  - Analyze boundary decisions
- ✅ **Tab 3 - Feature Importance:**
  - Ranking features
  - Importance percentages
  - Visual chart
- ✅ **Tab 4 - Learning Center:**
  - Iris flower facts
  - ML concepts explanation
  - Quick test examples

**Use Case:**
- Educational purpose
- Model interpretability
- Research & analysis
- ML course material
- Understanding classification

**Load Time:** ~4 detik

---

## 🔄 Workflow Rekomendasi

### Untuk Development:
1. Mulai dengan `app_streamlit.py` - simple & fast
2. Test logic dengan `app_api.py` - verify responses
3. Build UI dengan `app_dashboard.py` - production-ready

### Untuk Deployment:
- **Demo/Presentation:** `app_streamlit.py`
- **Production:** `app_dashboard.py`
- **Internal Testing:** `app_api.py`
- **Team Learning:** `app_explorer.py`

### Untuk Education:
1. Start dengan `app_explorer.py` - understand the model
2. Move to `app_streamlit.py` - see it in action
3. Explore `app_dashboard.py` - production patterns

---

## 📦 Requirements

Semua aplikasi menggunakan `requirements.txt` yang sama:

```
scikit-learn==1.5.2
joblib==1.4.2
numpy==1.26.4
pandas==2.2.3
streamlit==1.39.0
```

**Catatan:** 
- Tidak ada Gradio (hanya untuk `app.py`)
- Tidak ada FastAPI/Uvicorn (hanya untuk `api_render.py`)
- Semua apps 100% Streamlit compatible

---

## 🐛 Troubleshooting

### App loading lambat di Streamlit Cloud?
- ✅ Gunakan `app_streamlit.py` atau `app_api.py` (lebih ringan)
- ✅ Pastikan `model_iris.joblib` sudah ter-upload

### CSV upload tidak berfungsi?
- ✅ Pastikan CSV memiliki header: `sepal_length,sepal_width,petal_length,petal_width`
- ✅ Gunakan `app_dashboard.py` (hanya ini yang support CSV)

### API simulator tidak menampilkan response?
- ✅ Pastikan semua input dalam range yang valid
- ✅ Gunakan `app_api.py` untuk testing API format

---

## 📞 Support & Questions

Untuk dokumentasi lengkap model & deployment, lihat `README.md`

---

**Happy Classifying! 🌸**
