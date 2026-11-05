import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib
from PIL import Image
import os

st.set_page_config(page_title="Plant Leaf Disease Dashboard", layout="wide")

st.title("🍃 Plant Leaf Disease Classification Dashboard")
st.write("Upload an image of a crop leaf to detect the type of disease using the trained MobileNetV2 and Logistic Regression models.")

# Load labels and models safely
@st.cache_resource
def load_assets():
    label_encoder = joblib.load("leaf_label_encoder.pkl")
    lr_model = joblib.load("leaf_logistic_regression.pkl")
    mobilenet_model = tf.keras.models.load_model("plant_leaf_disease_model.keras")
    return label_encoder, lr_model, mobilenet_model

try:
    label_encoder, lr_model, mobilenet_model = load_assets()
    classes = label_encoder.classes_
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Leaf Image")
        st.image(img_rgb, use_column_width=True)
        
    with col2:
        st.subheader("Model Predictions")
        
        # 1. Prep for Deep Learning Model (MobileNetV2)
        img_dl = cv2.resize(img_rgb, (224, 224))
        img_dl_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_dl.astype("float32"))
        img_dl_batch = np.expand_dims(img_dl_preprocessed, axis=0)
        
        # MobileNetV2 Prediction
        dl_probs = mobilenet_model.predict(img_dl_batch)[0]
        dl_pred_idx = np.argmax(dl_probs)
        dl_confidence = dl_probs[dl_pred_idx]
        dl_class = classes[dl_pred_idx]
        
        # 2. Prep for HSV + Logistic Regression
        img_hsv = cv2.resize(img_rgb, (224, 224))
        hsv = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().reshape(1, -1)
        
        # Logistic Regression Prediction
        lr_probs = lr_model.predict_proba(hist)[0]
        lr_pred_idx = np.argmax(lr_probs)
        lr_confidence = lr_probs[lr_pred_idx]
        lr_class = classes[lr_pred_idx]
        
        # Display Results
        st.markdown("### 🤖 MobileNetV2 Deep Learning Result")
        st.metric(label="Predicted Class", value=dl_class)
        st.metric(label="Confidence Score", value=f"{dl_confidence * 100:.2f}%")
        
        st.write("--- ")
        
        st.markdown("### 📊 HSV + Logistic Regression Result")
        st.metric(label="Predicted Class", value=lr_class)
        st.metric(label="Confidence Score", value=f"{lr_confidence * 100:.2f}%")
