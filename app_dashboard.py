import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

st.set_page_config(
    page_title="Iris Classifier Dashboard",
    page_icon="🌸",
    layout="wide",
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_iris.joblib")
    return joblib.load(model_path)

artifact = load_model()
model = artifact["model"]
target_names = artifact["target_names"]

st.title("🌸 Iris Flower Classification Dashboard")
st.markdown("**Professional ML Model Deployment** - Local & Cloud Compatible")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Prediction",
    "📤 Batch Prediction (CSV)",
    "📋 Model Info",
    "📈 Visualizations"
])

# ============= TAB 1: Single Prediction =============
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Flower Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1, key="sl")
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1, key="sw")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1, key="pl")
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1, key="pw")
        
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        
    with col2:
        st.subheader("Model Prediction")
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            predicted_species = target_names[int(prediction)].capitalize()
            
            st.success(f"### Predicted Species: **Iris {predicted_species}**", icon="✅")
            
            prob_df = pd.DataFrame({
                "Species": [f"Iris {name.capitalize()}" for name in target_names],
                "Confidence (%)": [round(float(p) * 100, 2) for p in probabilities],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.bar_chart(prob_df.set_index("Species"))

# ============= TAB 2: Batch Prediction =============
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    st.markdown("**CSV Format:** sepal_length,sepal_width,petal_length,petal_width")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(df)} rows**")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                if all(col in df.columns for col in required_cols):
                    features = df[required_cols].values
                    predictions = model.predict(features)
                    probabilities = model.predict_proba(features)
                    
                    result_df = df.copy()
                    result_df["predicted_species"] = [target_names[int(p)].capitalize() for p in predictions]
                    result_df["confidence"] = [round(np.max(probs) * 100, 2) for probs in probabilities]
                    
                    st.success("✅ Predictions completed!")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="iris_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"❌ CSV must have columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ============= TAB 3: Model Info =============
with tab3:
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Species Classes", len(target_names))
    with col3:
        st.metric("Features Used", 4)
    
    st.divider()
    st.markdown("### Iris Species Classes")
    for i, species in enumerate(target_names):
        st.write(f"**{i+1}. Iris {species.capitalize()}**")
    
    st.divider()
    st.markdown("### Input Features")
    features_info = {
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Unit": ["cm", "cm", "cm", "cm"],
        "Min": [4.0, 2.0, 1.0, 0.1],
        "Max": [8.0, 4.5, 7.0, 2.5]
    }
    st.dataframe(pd.DataFrame(features_info), use_container_width=True, hide_index=True)

# ============= TAB 4: Visualizations =============
with tab4:
    st.subheader("Model Predictions Visualization")
    
    # Generate sample predictions for all combinations
    sl_range = np.linspace(4, 8, 5)
    sw_range = np.linspace(2, 4.5, 5)
    
    results = []
    for sl in sl_range:
        for sw in sw_range:
            pred = model.predict([[sl, sw, 3.5, 1.3]])[0]
            results.append({"Sepal Length": sl, "Sepal Width": sw, "Species": target_names[int(pred)].capitalize()})
    
    viz_df = pd.DataFrame(results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Prediction Heatmap (Sepal Dimensions)")
        pivot_data = viz_df.pivot_table(
            values="Species",
            index="Sepal Width",
            columns="Sepal Length",
            aggfunc="first"
        )
        st.dataframe(pivot_data, use_container_width=True)
    
    with col2:
        st.markdown("### Distribution Chart")
        species_count = viz_df["Species"].value_counts()
        st.bar_chart(species_count)

st.divider()
st.markdown("---")
st.caption("🚀 Iris Flower Classifier | Deployable to Streamlit Cloud & Local")
