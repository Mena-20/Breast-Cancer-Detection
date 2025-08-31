import streamlit as st
import numpy as np
from joblib import load
from PIL import Image
import os

# Page Config
st.set_page_config(
    page_title="Breast Cancer Diagnosis App",
    page_icon=":medical_symbol:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        padding: 1rem 0;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
    }
    .model-selector {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #6c757d;
    }
    .arabic-text {
        text-align: center;
        font-style: italic;
        color: #ff4b4b;
        font-family: cursive;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained models
@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), 'Models')
    models = {
        'logistic': load(os.path.join(model_path, 'logistic_model.joblib')),
        'svm': load(os.path.join(model_path, 'svm_model.joblib')),
        'tree': load(os.path.join(model_path, 'tree_model.joblib')),
        'random_forest': load(os.path.join(model_path, 'RF_model.joblib')),
        'scaler': load(os.path.join(model_path, 'scaler.joblib'))
    }
    
    # Check which models support predict_proba
    for name, model in models.items():
        if hasattr(model, '_final_estimator'):
            models[name] = model._final_estimator
    
    return models

models = load_models()

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='header'><h1>Breast Cancer Diagnosis Assistant</h1><p class='lead'>Predict tumor malignancy with machine learning models</p></div>", unsafe_allow_html=True)
with col2:
    try:
        image = Image.open("pexels-pixabay-256262.jpg")
        st.image(image, width=350, caption="Developed by Ragab")
    except:
        st.image("https://via.placeholder.com/150", width=250, caption="Developer Image")

st.markdown("---")

# Feature Input Section
st.markdown("## 🔬 Tumor Characteristics")
st.markdown("Please provide the tumor features below:")

feature_columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

feature_descriptions = {
    "radius_mean": "Mean of distances from center to points on the perimeter",
    "texture_mean": "Standard deviation of gray-scale values",
    "perimeter_mean": "Tumor perimeter length",
    "area_mean": "Area of the tumor",
    "smoothness_mean": "Local variation in radius lengths",
    "compactness_mean": "Perimeter² / area - 1.0",
    "concavity_mean": "Severity of concave portions of the contour",
    "concave_points_mean": "Number of concave portions of the contour",
    "symmetry_mean": "Symmetry of the tumor shape",
    "fractal_dimension_mean": "Coastline approximation - 1"
}

input_values = []
cols = st.columns(3)
for i, feature in enumerate(feature_columns):
    col = cols[i % 3]
    with col:
        help_text = feature_descriptions.get(feature, "Tumor feature measurement")
        val = st.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help=help_text,
            key=feature
        )
        input_values.append(val)

# Model Selection Section
st.markdown("## 🤖 Machine Learning Model Selection")
with st.container():
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    model_choice = st.radio(
        "Select prediction model:",
        options=["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest"],
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Section
st.markdown("## 🔍 Prediction Results")
if st.button("Run Diagnosis", type="primary", use_container_width=True):
    user_data = np.array([input_values])
    scaled_data = models['scaler'].transform(user_data)
    
    model_map = {
        "Logistic Regression": models['logistic'],
        "Support Vector Machine": models['svm'],
        "Decision Tree": models['tree'],
        "Random Forest": models['random_forest']
    }
    
    model = model_map[model_choice]
    prediction = model.predict(scaled_data)
    
    try:
        probability = model.predict_proba(scaled_data)[0]
        show_probability = True
    except AttributeError:
        show_probability = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        if show_probability:
            st.markdown("### Model Confidence")
            st.write(f"Probability of Benign: {probability[0]*100:.2f}%")
            st.write(f"Probability of Malignant: {probability[1]*100:.2f}%")
            st.progress(probability[1])
        else:
            st.markdown("### Model Information")
            st.warning("This model doesn't support probability estimates")
    
    with col2:
        if prediction[0] == 0:
            st.markdown("<div class='result-card benign'><h2>Diagnosis Result</h2><h1>Benign Tumor</h1><p>No malignancy detected</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-card malignant'><h2>Diagnosis Result</h2><h1>Malignant Tumor</h1><p>Cancer detected - Please consult a specialist</p></div>", unsafe_allow_html=True)
    
    st.markdown("### Model Details")
    st.write(f"Algorithm used: {model_choice}")
    st.write("Note: This prediction is for informational purposes only and should not replace professional medical advice.")

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>🔬 Breast Cancer Diagnosis Assistant | Developed by 3amk Ragab</p>
    <p>Powered by Machine Learning | For educational purposes</p>
</div>
""", unsafe_allow_html=True)
