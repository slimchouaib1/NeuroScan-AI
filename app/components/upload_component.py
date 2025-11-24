import streamlit as st
from PIL import Image

def upload_mri():
    """Streamlit file uploader that returns a PIL Image or None."""
    uploaded = st.file_uploader(
        "Upload an MRI Image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            return img
        except Exception:
            st.error("Failed to read uploaded image. Please upload a valid PNG/JPG file.")
            return None

    return None

