import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

@st.cache_resource
def load_model():
    return YOLO('trained_yolo11l_model20.pt')

def main():
    st.title("Cheating or Normal")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an Image :)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model(image)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detection Results", use_container_width=True)

if __name__ == "__main__":
    main()
