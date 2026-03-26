import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

model = YOLO("runs/detect/train5/weights/best.pt")

st.title("🚗 Car Damage Detection System")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Damage"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)

        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_column_width=True)