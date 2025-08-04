import streamlit as st
from PIL import Image
import numpy as np
import os
import shutil
from ultralytics import YOLO
import streamlit as st

def make_predictions(image_path, model_path):
    # load the yolo model
    yolo_model = YOLO(model_path)
    results = yolo_model.predict(image_path, save = True)
    return results


def run_app():
    # constants
    IMAGE_NAME = "uploaded.png"
    model_path = "/workspaces/Plastic_detection/best.pt"
    IMAGE_ADDRESS = "https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/10/policy-scenarios-for-eliminating-plastic-pollution-by-2040_28eb9536/76400890-en.jpg"
    
    # UI
    st.title("Plastic Detection")
    st.image(IMAGE_ADDRESS, caption = "Plastic Detection")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Open the uploaded file using PIL
        image = Image.open(uploaded_file)

        # Display the image
        image.save(IMAGE_NAME)

        # get predictions
        with st.spinner("Getting Predictions......"):
            mask_response = make_predictions(IMAGE_NAME,model_path)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image)

            with col2:
                st.subheader("Mask")
                if mask_response:
                    st.image(mask_response)
                else:
                    st.error("Error Getting Predictions", icon="🚨")

run_app()                    