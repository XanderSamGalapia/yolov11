import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO('trained_yolo11l_model20.pt', map_location='cpu')  # Update path as needed
    model.eval()
    return model

model = load_model()


st.title("Cheating or Normal :o")
st.write("Upload an image and let the model make predictions!")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


    img_array = np.array(image.resize((224, 224))) / 255.0  # Resize if needed
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float()  # shape: [1, 3, H, W]


    with torch.no_grad():
        output = model(img_tensor)

        prediction = output.argmax(dim=1).item()

    # 🧠 Display result
    st.success(f"Model Prediction: {prediction}")
