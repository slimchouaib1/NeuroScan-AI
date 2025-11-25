import streamlit as st
from PIL import Image
import io
import os
import sys

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

# Ensure project root is on sys.path so `import app...` works when Streamlit runs this file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- Import components normally ----
from app.components.upload_component import upload_mri
from app.components.display_component import show_results

# (moved page config above CSS injection to satisfy Streamlit requirement)

# ---- Sidebar ----
st.sidebar.title("NeuroScan AI")
st.sidebar.markdown("Brain Tumor Detection & Segmentation")

logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

# ---- Main Title ----
st.title("🧠 NeuroScan AI — Brain MRI Analysis")
st.markdown("Upload an MRI scan and receive detection, segmentation, and explainability results.")

# ---- Upload ----
uploaded_img = upload_mri()

# ---- When the user uploads an image ----
if uploaded_img is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI Scan")
        st.image(uploaded_img, width=350)

    # Convert to bytes
    img_bytes = io.BytesIO()
    uploaded_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    st.info("⚙️ Processing image... (Backend not yet connected)")

    # Fake response for demo (use absolute paths to assets folder)
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    fake_response = {
        "prediction": "Tumor Detected",
        "confidence": 0.92,
        "mask": os.path.join(assets_dir, "sample_mask.png"),
        "heatmap": os.path.join(assets_dir, "sample_heatmap.png"),
        "overlay": os.path.join(assets_dir, "sample_overlay.png"),
    }

    show_results(fake_response)
