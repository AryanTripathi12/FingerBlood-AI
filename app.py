import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FingerBlood AI",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MODEL CONFIG
# ============================================================

CLASS_NAMES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

IMG_SIZE = (128, 128)

# Try possible model locations
MODEL_PATHS = [
    "model/model.h5",
    "model/model.keras",
    "models/model.h5",
    "models/model.keras",
    "model.h5",
    "model.keras"
]

MODEL_PATH = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(120, 30, 60, 0.18),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(60, 40, 120, 0.15),
                transparent 35%
            ),
            #080a10;
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit default elements */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* ---------- HERO ---------- */

    .hero {
        padding: 45px 35px;
        margin-bottom: 35px;

        border-radius: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(35, 20, 35, 0.95),
                rgba(17, 18, 27, 0.98)
            );

        border: 1px solid rgba(255,255,255,0.10);

        box-shadow:
            0 20px 70px rgba(0,0,0,0.45);
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -2px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #ff6b81
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: #aeb3c2;
        font-size: 16px;
        line-height: 1.7;
    }

    .developer {
        color: #ff647c;
        font-weight: 700;
    }


    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 35px;
        margin-bottom: 15px;
    }


    /* ---------- CARDS ---------- */

    .card {
        background: rgba(22, 24, 33, 0.92);

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 20px;

        padding: 28px;

        margin-bottom: 25px;

        box-shadow:
            0 10px 35px rgba(0,0,0,0.25);
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .card-text {
        color: #b8bdc9;
        font-size: 16px;
        line-height: 1.8;
    }

    .highlight {
        color: #ff637b;
        font-weight: 700;
    }


    /* ---------- STAT CARDS ---------- */

    .stat-card {
        background: rgba(20,22,30,0.95);

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 18px;

        padding: 25px 15px;

        text-align: center;

        min-height: 135px;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.25);
    }

    .stat-value {
        font-size: 30px;
        font-weight: 800;

        color: #ff657d;
    }

    .stat-label {
        margin-top: 8px;

        color: #aeb3c2;

        font-size: 14px;
    }


    /* ---------- UPLOAD ---------- */

    .upload-info {
        color: #aeb3c2;
        margin-bottom: 12px;
    }


    /* ---------- RESULT ---------- */

    .result-card {
        padding: 35px;

        border-radius: 22px;

        text-align: center;

        background:
            linear-gradient(
                135deg,
                rgba(40,20,30,0.95),
                rgba(20,22,32,0.98)
            );

        border: 1px solid rgba(255,90,110,0.20);

        box-shadow:
            0 15px 50px rgba(0,0,0,0.35);
    }

    .result-label {
        color: #9da3b3;
        font-size: 15px;
    }

    .blood-group {
        font-size: 65px;
        font-weight: 900;

        margin: 8px 0;

        color: #ff6178;

        text-shadow:
            0 0 30px rgba(255,80,100,0.25);
    }

    .confidence {
        font-size: 20px;
        color: #ffffff;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;

        margin-top: 50px;

        padding-top: 25px;

        border-top:
            1px solid rgba(255,255,255,0.08);

        color: #777d8d;

        font-size: 13px;
    }

    .footer-name {
        color: #ff647c;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if MODEL_PATH is None:
        return None

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


model = load_model()


# ============================================================
# HERO SECTION WITH PHOTO
# ============================================================

st.markdown('<div class="hero">', unsafe_allow_html=True)

hero_col1, hero_col2 = st.columns([2.5, 1], gap="large")

with hero_col1:
    st.markdown(
        """
        <div style="font-size: 38px; margin-bottom: 5px;">🩸</div>
        <div class="hero-title">FingerBlood AI</div>
        <div class="hero-subtitle">
            Advanced Fingerprint-Based Blood Group Classification Platform.
            <br>
            Powered by state-of-the-art <b>Convolutional Neural Networks (CNN)</b> for fast, non-invasive screening.
            <br><br>
            Developed by 
            <span class="developer">Aryan Tripathi</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_col2:
    profile_path = "profile.jpg"
    if os.path.exists(profile_path):
        st.image(profile_path, width=180)
    else:
        st.markdown(
            """
            <div style="width: 150px; height: 150px; border-radius: 50%; background: linear-gradient(135deg, #ff6b81, #781e3c); display: flex; align-items: center; justify-content: center; font-size: 45px; margin: 0 auto; box-shadow: 0 0 25px rgba(255,50,80,0.3);">
                👨‍💻
            </div>
            <div style="text-align: center; color: #aeb3c2; font-size: 12px; margin-top: 10px;">
                Add 'profile.jpg' to folder
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔬 About the Project
    </div>

    <div class="card">
        <div class="card-title">
            🧠 FingerBlood AI Architecture
        </div>
        <div class="card-text">
            FingerBlood AI is a computer vision and deep learning
            project that analyzes ridge patterns in fingerprint images using a
            <span class="highlight">Convolutional Neural Network (CNN)</span>
            pipeline to classify them accurately into eight primary blood-group categories.
            <br><br>
            The system supports full classification across:
            <br>
            <span class="highlight">A+, A-, AB+, AB-, B+, B-, O+, O-</span>
            <br><br>
            Upload a fingerprint image below to let the trained model
            generate real-time predicted blood groups alongside confidence metrics and full class probability distributions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📊 Model Overview
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-value">86.62%</div>
            <div class="stat-label">Test Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-value">8</div>
            <div class="stat-label">Blood Groups</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-value">128×128</div>
            <div class="stat-label">Input Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-value">CNN</div>
            <div class="stat-label">Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📤 Upload Fingerprint
    </div>
    <div class="upload-info">
        Upload a fingerprint sample image in JPG, JPEG, PNG, or BMP format for analysis.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a fingerprint image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed"
)


# ============================================================
# MODEL NOT FOUND WARNING
# ============================================================

if model is None:
    st.warning(
        """
        ⚠️ Trained model weights not found.

        Please save your trained `.h5` or `.keras` model file into:
        `model/model.h5` or `model/model.keras`
        """
    )


# ============================================================
# PREDICTION PIPELINE
# ============================================================

if uploaded_file is not None and model is not None:

    # Convert to Grayscale ("L") to match model's expected 1-channel shape
    image = Image.open(uploaded_file).convert("L")

    st.markdown(
        """
        <div class="section-title">
            🖼️ Uploaded Fingerprint Sample
        </div>
        """,
        unsafe_allow_html=True
    )

    col_img, col_result = st.columns([1, 1], gap="medium")

    with col_img:
        st.image(
            image,
            caption="Target Fingerprint",
            use_container_width=True
        )

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Shape: (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 128, 128, 1)

    # Run Prediction
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index] * 100

    with col_result:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Blood Group</div>
                <div class="blood-group">{predicted_class}</div>
                <div class="confidence">
                    Model Confidence: <b>{confidence:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            📈 Probability Distribution
        </div>
        """,
        unsafe_allow_html=True
    )

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        percentage = float(probability) * 100
        st.write(f"**{class_name}** — {percentage:.2f}%")
        st.progress(min(float(probability), 1.0))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🩸 <b>FingerBlood AI</b>
        <br><br>
        Fingerprint-Based Blood Group Classification System
        <br>
        Designed & Developed with ❤️ by <span class="footer-name">Aryan Tripathi</span>
        <br><br>
        AI/ML • Computer Vision • Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)