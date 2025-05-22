import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO('trained_yolo11l_model20.pt')  
    return model

def main():
    st.title("Cheating or Normal")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        results = model(image)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detection Results", use_column_width=True)

if __name__ == "__main__":
    main()
