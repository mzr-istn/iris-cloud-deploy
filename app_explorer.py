import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_iris.joblib")
    return joblib.load(model_path)

@st.cache_data
def load_sample_data():
    from sklearn import datasets
    iris_data = datasets.load_iris()
    return pd.DataFrame(
        iris_data.data,
        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ), iris_data.target, iris_data.target_names

artifact = load_model()
model = artifact["model"]
target_names = artifact["target_names"]
sample_df, sample_target, target_names_sklearn = load_sample_data()

st.title("🔬 Iris Flower Explorer & Model Analysis")
st.markdown("**Educational & Interactive ML Model Exploration**")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌸 Sample Dataset",
    "🧠 Model Decision Boundaries",
    "📊 Feature Importance",
    "🎓 Learning Center"
])

# ============= TAB 1: Sample Dataset =============
with tab1:
    st.subheader("Iris Dataset Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(sample_df))
    with col2:
        st.metric("Features", sample_df.shape[1])
    with col3:
        st.metric("Classes", len(target_names))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Preview")
        st.dataframe(sample_df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### Feature Statistics")
        stats_df = sample_df.describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    st.markdown("### Distribution by Species")
    sample_df_with_target = sample_df.copy()
    sample_df_with_target['species'] = sample_target
    sample_df_with_target['species_name'] = sample_df_with_target['species'].map(
        {i: name.capitalize() for i, name in enumerate(target_names)}
    )
    
    for species_id, species_name in enumerate(target_names):
        count = (sample_target == species_id).sum()
        st.write(f"**Iris {species_name.capitalize()}:** {count} samples")

# ============= TAB 2: Decision Boundaries =============
with tab2:
    st.subheader("Model Decision Boundaries Analysis")
    st.markdown("Visualize how the model separates different iris species")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("Feature X-axis", 
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            index=2)
    with col2:
        feature_y = st.selectbox("Feature Y-axis",
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            index=3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_feature_1 = st.selectbox("Fixed Feature 1", 
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            index=0)
        fixed_value_1 = st.slider(f"{fixed_feature_1} (fixed)", 
            float(sample_df[fixed_feature_1].min()),
            float(sample_df[fixed_feature_1].max()),
            float(sample_df[fixed_feature_1].mean()), 0.1)
    
    with col2:
        fixed_feature_2 = st.selectbox("Fixed Feature 2",
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            index=1)
        fixed_value_2 = st.slider(f"{fixed_feature_2} (fixed)",
            float(sample_df[fixed_feature_2].min()),
            float(sample_df[fixed_feature_2].max()),
            float(sample_df[fixed_feature_2].mean()), 0.1)
    
    # Create mesh for decision boundary
    x_min, x_max = sample_df[feature_x].min() - 0.5, sample_df[feature_x].max() + 0.5
    y_min, y_max = sample_df[feature_y].min() - 0.5, sample_df[feature_y].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Create feature matrix for mesh
    mesh_features = np.zeros((xx.ravel().shape[0], 4))
    feature_indices = {
        'sepal_length': 0, 'sepal_width': 1,
        'petal_length': 2, 'petal_width': 3
    }
    
    mesh_features[:, feature_indices[feature_x]] = xx.ravel()
    mesh_features[:, feature_indices[feature_y]] = yy.ravel()
    mesh_features[:, feature_indices[fixed_feature_1]] = fixed_value_1
    mesh_features[:, feature_indices[fixed_feature_2]] = fixed_value_2
    
    Z = model.predict(mesh_features).reshape(xx.shape)
    
    st.success(f"✅ Decision boundary computed for {feature_x} vs {feature_y}")
    st.info(f"Fixed values: {fixed_feature_1}={fixed_value_1:.1f}, {fixed_feature_2}={fixed_value_2:.1f}")

# ============= TAB 3: Feature Importance =============
with tab3:
    st.subheader("Feature Importance in Model")
    
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': (importances * 100).round(2)
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Feature Ranking")
            for idx, row in importance_df.iterrows():
                st.write(f"**{idx+1}. {row['Feature']}:** {row['Percentage']}%")
        
        with col2:
            st.markdown("### Importance Chart")
            st.bar_chart(importance_df.set_index('Feature')['Percentage'])
        
        st.divider()
        st.markdown("### What this means:")
        st.markdown(f"""
- **{importance_df.iloc[0]['Feature'].title()}** is the most important feature ({importance_df.iloc[0]['Percentage']}%)
- **{importance_df.iloc[-1]['Feature'].title()}** is the least important feature ({importance_df.iloc[-1]['Percentage']}%)
- The model relies most on petal measurements for classification
        """)
    else:
        st.warning("⚠️ Feature importance not available for this model type")

# ============= TAB 4: Learning Center =============
with tab4:
    st.subheader("🎓 Learning Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### About Iris Flowers")
        st.markdown("""
**Iris Setosa**
- Small petals
- Typically: petal_length < 2 cm
- Sepal dimensions more uniform

**Iris Versicolor**
- Medium-sized petals
- Typically: 3 cm < petal_length < 5 cm
- Mix of characteristics

**Iris Virginica**
- Large petals
- Typically: petal_length > 5 cm
- Larger overall dimensions
        """)
    
    with col2:
        st.markdown("### Machine Learning Concepts")
        st.markdown("""
**Random Forest Classifier**
- Ensemble learning method
- Multiple decision trees
- Combines predictions

**Feature Engineering**
- Sepal: Outer leaf-like structure
- Petal: Inner colorful structure
- 4 measurements per flower

**Classification Task**
- Input: 4 measurements
- Output: One of 3 species
- Confidence score for each
        """)
    
    st.divider()
    st.markdown("### Quick Test Examples")
    
    examples = {
        "🔴 Typical Setosa": [5.1, 3.5, 1.4, 0.2],
        "🟠 Typical Versicolor": [5.9, 2.7, 4.2, 1.3],
        "🟣 Typical Virginica": [6.9, 3.1, 5.4, 2.1],
    }
    
    for label, features in examples.items():
        if st.button(f"Test: {label}"):
            pred = model.predict([features])[0]
            probs = model.predict_proba([features])[0]
            predicted = target_names[int(pred)].capitalize()
            confidence = round(np.max(probs) * 100, 2)
            st.success(f"✅ Predicted: **Iris {predicted}** ({confidence}% confidence)")

st.divider()
st.caption("🔬 Iris Explorer | Educational tool for understanding ML classification models")
