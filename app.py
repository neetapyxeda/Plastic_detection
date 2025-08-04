import streamlit as st
from PIL import Image
import numpy as np
from model_utils import load_yolo_model
import os
import shutil

def make_predictions(image_path, MODEL_NAME):
    # load the yolo model
    yolo_model = load_yolo_model(MODEL_NAME)
    results = yolo_model.predict(image_path, save = True)
    try:
        if os.path.exists(PRED_IMAGE_PATH):
            shutil.move(PRED_IMAGE_PATH, PRED_MOVE_NAME)
            os.rmdir(DIRECTORY)
            return True
        else:
            return False
    except Exception as error:
        print(str(error))

        return False


def run_app():
    # constants
    IMAGE_NAME = "uploaded.png"
    MODEL_NAME = "best.pt"
    IMAGE_ADDRESS = "https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/10/policy-scenarios-for-eliminating-plastic-pollution-by-2040_28eb9536/76400890-en.jpg"
    PRED_IMAGE_PATH = "runs/segment/predict/uploaded.png"
    DIRECTORY = "runs/segment/predict"
    PRED_MOVE_NAME = "pred_image.png"
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
            mask_response = make_predictions(IMAGE_NAME,MODEL_NAME)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image)

            with col2:
                st.subheader("Mask")
                if mask_response:
                    st.image(PRED_MOVE_NAME)
                else:
                    st.error("Error Getting Predictions", icon="🚨")

run_app()                    