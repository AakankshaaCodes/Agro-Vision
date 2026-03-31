import os
import streamlit as st

# Reduce TensorFlow startup noise in terminal output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AgroVision 🌱", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    color: #1b5e20;
}

/* Glass Card (Light) */
.glass {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(8px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Heading */
.title {
    color: #1b5e20;
    text-align: center;
}

/* Upload Box */
.upload-box {
    border: 2px dashed #66bb6a;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
    background: #f1f8e9;
}
.upload-box:hover {
    background: #dcedc8;
}

/* Button */
.stButton>button {
    background: #43a047;
    color: white;
    border-radius: 10px;
}

/* Progress */
.stProgress > div > div > div {
    background-color: #43a047;
}

/* Floating animation */
@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
    100% {transform: translateY(0px);}
}

.float {
    animation: float 3s ease-in-out infinite;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_ai_model():
    return load_model("agroguard_model.h5")

try:
    model = load_ai_model()
except:
    model = None
    st.error("Model not loaded")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 AgroVision")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📤 Upload Image",
    "📊 Detection Result",
    "ℹ️ About"
])

# ---------------- HOME ----------------
if page == "🏠 Home":

    st.markdown("<h1 class='title'>🌱 AgroVision</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Smart Crop Disease Detection</h4>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='glass float'>🤖 AI Detection</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass float'>⚡ Fast Results</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='glass float'>🌾 Farmer Friendly</div>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
elif page == "📤 Upload Image":

    st.markdown("<h2 class='title'>Upload Crop Image</h2>", unsafe_allow_html=True)

    st.markdown("<div class='upload-box'>📤 Drag & Drop Image</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload crop image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Preview", width="stretch")
        st.session_state["image"] = uploaded_file

# ---------------- RESULT ----------------
elif page == "📊 Detection Result":

    st.markdown("<h2 class='title'>Detection Result</h2>", unsafe_allow_html=True)

    if "image" in st.session_state and model is not None:

        file = st.session_state["image"]

        img = image.load_img(file, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)/255.0

        with st.spinner("Analyzing crop health... 🌱"):
            prediction = model.predict(img_array)

        class_names = ["fungus", "healthy", "pest", "weed"]
        result = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        if result == "healthy":
            st.success("🌱 Healthy Crop")
        elif result == "pest":
            st.warning("🐛 Pest Detected")
        else:
            st.error(f"⚠️ {result} Detected")

        st.progress(int(confidence * 100))
        st.write(f"Confidence: {confidence:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Upload image first")

# ---------------- ABOUT ----------------
elif page == "ℹ️ About":

    st.markdown("<h2 class='title'>About AgroVision</h2>", unsafe_allow_html=True)

    st.write("""
    AgroVision helps farmers detect crop diseases using AI.
    Simple, fast, and effective.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center style='color:#1b5e20;'>Built with ❤️ for farmers | AgroVision AI</center>",
    unsafe_allow_html=True
)