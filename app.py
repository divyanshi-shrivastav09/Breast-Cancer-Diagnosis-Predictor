
import streamlit as st
import joblib
import numpy as np


# Load trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")


# Page title
st.title("Breast Cancer Diagnosis Predictor")

st.write(
    "Enter the 30 tumor measurements below "
    "to predict whether the tumor is malignant or benign."
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


# Create input fields
inputs = []

for feature in feature_names:
    value = st.number_input(
        feature,
        value=0.0
    )
    inputs.append(value)


# Prediction button
if st.button("Predict"):

    # Convert input into NumPy array
    input_data = np.array(inputs).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Display result
    if prediction == 0:
        st.error("Prediction: Malignant (Cancerous)")
    else:
        st.success("Prediction: Benign (Non-cancerous)")
