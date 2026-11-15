import os
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Iris Classification REST API",
    description="API Prediksi Spesies Bunga Iris (Lokal & Render Cloud)",
    version="1.0.0",
)

# Memuat model dari file joblib (menggunakan relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_iris.joblib")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]


class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Panjang sepal (cm)")
    sepal_width: float = Field(..., example=3.5, description="Lebar sepal (cm)")
    petal_length: float = Field(..., example=1.4, description="Panjang petal (cm)")
    petal_width: float = Field(..., example=0.2, description="Lebar petal (cm)")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Iris Classification API",
        "endpoints": {"docs": "/docs", "predict": "/predict"},
    }


@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]])

    pred_idx = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "predicted_class": target_names[pred_idx],
        "class_index": pred_idx,
        "probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(target_names, probabilities)
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_render:app", host="0.0.0.0", port=port, reload=False)
