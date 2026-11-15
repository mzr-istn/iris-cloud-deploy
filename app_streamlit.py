import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="centered",
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_iris.joblib")
    return joblib.load(model_path)

artifact = load_model()
model = artifact["model"]
target_names = artifact["target_names"]

st.title("🌸 Prediksi Spesies Bunga Iris")
st.write(
    "Aplikasi Machine Learning menggunakan **Random Forest (`model_iris.joblib`)** "
    "yang dapat dijalankan di **Lokal** maupun **Streamlit Community Cloud**."
)

st.sidebar.header("Input Ukuran Bunga (cm)")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"],
)

st.subheader("Data Input Pengguna")
st.dataframe(input_df, use_container_width=True, hide_index=True)

if st.button("Prediksi Spesies", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_idx = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    predicted_species = target_names[pred_idx].capitalize()

    st.success(f"Hasil Prediksi Spesies: **Iris {predicted_species}**")

    prob_df = pd.DataFrame({
        "Spesies": [name.capitalize() for name in target_names],
        "Probabilitas (%)": [round(float(p) * 100, 2) for p in probabilities],
    })
    st.subheader("Tingkat Keyakinan Model")
    st.bar_chart(prob_df.set_index("Spesies"))
