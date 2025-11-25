import streamlit as st
from PIL import Image
import io
import os
import sys
import requests

# ---- Page Config ----
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ---- Inject CSS ----
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- Add project root ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- Load components ----
from app.components.upload_component import upload_mri
from app.components.display_component import show_results

# ---- Sidebar ----
st.sidebar.title("NeuroScan AI")
st.sidebar.markdown("Brain Tumor Detection & Segmentation")

logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

# ---- Title ----
st.title("🧠 NeuroScan AI — MRI Analysis")

BACKEND_URL = "http://localhost:8000/predict"

# ---- Session State ----
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

# ---- Upload ----
uploaded_img = upload_mri()

# ---- CASE 1: No image → clear predictions ----
if uploaded_img is None:
    st.session_state.prediction_result = None
    st.session_state.last_image_hash = None
    st.stop()

# ---- CASE 2: New image uploaded → reset predictions ----
img_bytes_raw = io.BytesIO()
uploaded_img.save(img_bytes_raw, format="PNG")
img_bytes = img_bytes_raw.getvalue()

current_hash = hash(img_bytes)

if st.session_state.last_image_hash != current_hash:
    st.session_state.prediction_result = None
    st.session_state.last_image_hash = current_hash

# ---- Show image ----
col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded MRI Scan")
    st.image(uploaded_img, width=350)

# ---- If no prediction yet → Call backend ----
if st.session_state.prediction_result is None:

    with st.spinner("🔍 Analyzing MRI scan… please wait..."):
        try:
            files = {"file": ("mri.png", img_bytes, "image/png")}
            r = requests.post(BACKEND_URL, files=files, timeout=60)

            if r.status_code != 200:
                st.error("❌ Backend returned an error.")
                st.stop()

            backend_result = r.json()

            # 🔥 Translate backend keys → UI keys
            parsed = {
                "prediction": backend_result.get("label", "N/A"),
                "confidence": float(backend_result.get("confidence", 0)),
                "mask": None,
                "heatmap": None,
                "overlay": None
            }

            st.session_state.prediction_result = parsed

        except Exception as e:
            st.error(f"❌ Error connecting to backend: {str(e)}")
            st.stop()

# ---- Display results ----
if st.session_state.prediction_result:
    show_results(st.session_state.prediction_result)
