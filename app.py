import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import json
import time
from streamlit_lottie import st_lottie

# Page Setup
st.set_page_config(page_title="Rice Type Classifier", page_icon="🍚", layout="centered")

# Load model and labels
model = load_model("J:/aimicrodeegree/HEENA/New_Rice_Classification_Model/rice_classification_streamlit/rice_classification_cnn.h5")
labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load Lottie animations
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

animations = [
    load_lottie("rain.json"),
    load_lottie("ricegrain.json"),
    load_lottie("ricecake.json")
]

# Animation loop state
if "anim_index" not in st.session_state:
    st.session_state.anim_index = 0
    st.session_state.last_switch = time.time()

# Switch every 1.4 seconds
if time.time() - st.session_state.last_switch > 1.4:
    st.session_state.anim_index = (st.session_state.anim_index + 1) % len(animations)
    st.session_state.last_switch = time.time()

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        /* Remove all top spacing */
        html, body, .main, .block-container {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        body {
            background-color: #f7f9fc;
        }

        .fade-container {
            height: 220px;
            display: flex;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.6s ease-in-out;
            margin-bottom: 20px;
        }

        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        .title {
            font-size: 40px;
            color: #1e3a5f;
            text-align: center;
            font-weight: 700;
            margin-bottom: 10px;
            animation: slideGlow 1.5s ease-in-out;
        }

        @keyframes slideGlow {
            0% {
                transform: translateY(-20px);
                opacity: 0;
                text-shadow: none;
            }
            50% {
                transform: translateY(5px);
                text-shadow: 0 0 20px #3b82f6;
            }
            100% {
                transform: translateY(0);
                opacity: 1;
                text-shadow: 0 0 10px #06b6d4;
            }
        }

        .section-title {
            font-size: 24px;
            color: #1e3a5f;
            font-weight: 600;
            margin-top: 30px;
        }

        .result-box {
            background-color: #e6f4ea;
            border-left: 6px solid #22c55e;
            padding: 1.5rem;
            border-radius: 10px;
            color: #1f2937;
            font-size: 18px;
            margin-top: 20px;
        }

        .confidence-bar {
            background-color: #d1d5db;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 15px;
        }

        .confidence-fill {
            height: 20px;
            background: linear-gradient(90deg, #22c55e, #86efac);
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ---------------- UI ----------------

# Lottie animation in a fixed-size box with fade
st.markdown("<div class='fade-container'>", unsafe_allow_html=True)
st_lottie(animations[st.session_state.anim_index], height=200, key="lottie-loop")
st.markdown("</div>", unsafe_allow_html=True)

# Title stays independent
st.markdown("<div class='title'>🍚 Rice Type Classifier</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Upload Section
st.markdown("<div class='section-title'>📤 Upload a Rice Grain Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ---------------- Prediction Section ----------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocessing
    img_resized = img.resize((100, 100))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    predicted_class = labels[class_index]
    confidence = float(np.max(pred)) * 100

    # Results
    st.markdown("<div class='section-title'>🔍 Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='result-box'><b>Predicted Type:</b> {predicted_class}<br><b>Confidence:</b> {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )

    # Confidence Bar
    st.markdown(
        f"<div class='confidence-bar'><div class='confidence-fill' style='width: {confidence}%;'></div></div>",
        unsafe_allow_html=True
    )

    # Refresh Button
    st.markdown("<a href='/' class='refresh-button'>🔄 Try Another Image</a>", unsafe_allow_html=True)

else:
    st.info("Please Upload an Image to Get Started.")
    time.sleep(1)
    st.rerun()
