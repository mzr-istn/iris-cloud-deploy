import os
import joblib
import numpy as np
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_iris.joblib")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probabilities = model.predict_proba(features)[0]
    return {
        f"Iris {name.capitalize()}": float(prob)
        for name, prob in zip(target_names, probabilities)
    }


demo = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Sepal Length (cm)"),
        gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Sepal Width (cm)"),
        gr.Slider(1.0, 7.0, value=1.4, step=0.1, label="Petal Length (cm)"),
        gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Petal Width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Hasil Prediksi & Probabilitas"),
    title="🌸 Klasifikasi Bunga Iris (Lokal & Hugging Face Spaces)",
    description="Masukkan ukuran sepal dan petal bunga Iris untuk memprediksi spesiesnya menggunakan model_iris.joblib.",
    examples=[
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.0, 2.7, 4.5, 1.5],  # Versicolor
        [6.9, 3.1, 5.4, 2.1],  # Virginica
    ],
)

if __name__ == "__main__":
    demo.launch()
