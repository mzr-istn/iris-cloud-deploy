import os
import json
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Iris API Simulator",
    page_icon="🔌",
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

st.title("🔌 Iris Classification API Simulator")
st.markdown("**Test API-like predictions** - Similar to REST API response format")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Request Body")
    st.markdown("**JSON Input Format:**")
    
    sepal_length = st.number_input("sepal_length", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.number_input("sepal_width", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.number_input("petal_length", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.number_input("petal_width", 0.1, 2.5, 0.2, 0.1)
    
    request_data = {
        "sepal_length": float(sepal_length),
        "sepal_width": float(sepal_width),
        "petal_length": float(petal_length),
        "petal_width": float(petal_width)
    }
    
    st.json(request_data)

with col2:
    st.subheader("📤 Response")
    
    if st.button("🚀 Send Request", type="primary", use_container_width=True):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_idx = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        
        response = {
            "status": "success",
            "data": {
                "input": request_data,
                "prediction": {
                    "species": f"Iris {target_names[pred_idx].capitalize()}",
                    "species_id": int(pred_idx),
                    "confidence": round(float(np.max(probabilities)) * 100, 2)
                },
                "probabilities": {
                    f"iris_{name}": round(float(prob) * 100, 2)
                    for name, prob in zip(target_names, probabilities)
                }
            }
        }
        
        st.json(response)
        
        st.divider()
        st.markdown("**Copy Response:**")
        st.code(json.dumps(response, indent=2), language="json")
        
        st.download_button(
            "📥 Download as JSON",
            data=json.dumps(response, indent=2),
            file_name="iris_prediction.json",
            mime="application/json"
        )

st.divider()
st.markdown("---")

st.subheader("📚 API Documentation")
st.markdown("""
### Endpoint: `/predict`
**Method:** POST

#### Request Body
```json
{
    "sepal_length": float,
    "sepal_width": float,
    "petal_length": float,
    "petal_width": float
}
```

#### Response
```json
{
    "status": "success",
    "data": {
        "input": { ... },
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

#### Example cURL Request
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
""")

st.caption("🔌 Iris API Simulator | For testing before deploying to production API server")
