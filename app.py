import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.title("🩺 Breast Cancer Diagnosis Predictor")

st.write(
    "This application uses a machine learning model to classify "
    "tumor measurements as **Malignant** or **Benign**."
)

st.info(
    "⚠️ Educational project only. This application is not a medical "
    "diagnostic tool and should not be used for real medical decisions."
)

# Feature names
feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

# Organize inputs into three sections
st.subheader("Enter Tumor Measurements")

col1, col2, col3 = st.columns(3)

inputs = []

for i, feature in enumerate(feature_names):

    if i < 10:
        with col1:
            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )

    elif i < 20:
        with col2:
            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )

    else:
        with col3:
            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"feature_{i}"
            )

    inputs.append(value)

st.divider()

# Prediction button
if st.button("🔍 Predict Diagnosis", use_container_width=True):

    input_data = np.array(inputs).reshape(1, -1)

    # Apply the same scaling used during training
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.error("🔴 Prediction: Malignant (Cancerous)")
    else:
        st.success("🟢 Prediction: Benign (Non-cancerous)")

# Sidebar
st.sidebar.title("About the Project")

st.sidebar.write(
    "This project was developed as part of the "
    "BIOTECHTREK AI in Healthcare & Drug Discovery Bootcamp."
)

st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write("**Number of Features:** 30")
st.sidebar.write("**Accuracy:** 95.61%")
st.sidebar.write("**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset")

st.sidebar.divider()

st.sidebar.caption(
    "For educational purposes only. "
    "Not intended for clinical diagnosis."
)
