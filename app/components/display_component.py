import streamlit as st
from PIL import Image
import os


def safe_image(path, caption, width):
    """Display an image if it exists, otherwise show a placeholder message."""
    if path is None:
        st.info(f"No {caption} provided")
        return

    # if path is a bytes-like object or PIL image, Streamlit can display directly
    try:
        if hasattr(path, "read"):
            img = Image.open(path)
            st.image(img, caption=caption, width=width)
            return
    except Exception:
        pass

    # if path is a file path
    if isinstance(path, str) and os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption, width=width)
        return

    st.warning(f"Missing or invalid image for: {caption}")


def show_results(response):
    """Render model response dict into the Streamlit UI."""
    st.subheader("🧪 Model Results")

    prediction = response.get("prediction", "N/A")
    confidence = response.get("confidence", 0.0) or 0.0

    st.markdown(f"### 🧠 Prediction: **{prediction}**")
    st.markdown(f"### 🔍 Confidence: **{confidence*100:.2f}%**")

    col1, col2, col3 = st.columns(3)

    with col1:
        safe_image(response.get("mask"), "Segmentation Mask", 300)

    with col2:
        safe_image(response.get("heatmap"), "Grad-CAM Heatmap", 300)

    with col3:
        safe_image(response.get("overlay"), "Overlay (Mask + MRI)", 300)
