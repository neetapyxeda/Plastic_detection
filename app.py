import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

def make_predictions(image_path, model_path):
    try:
        # Load the YOLO model
        yolo_model = YOLO(model_path)

        # Perform prediction; setting save=False to avoid saving files
        results = yolo_model.predict(image_path, save=False)

        # Extract the first detection result
        if results:
            detection = results
            return detection
        else:
            return None
    except Exception as e:
        st.error(f"Failed to make predictions: {str(e)}")
        return None

def run_app():
    # Constants
    IMAGE_NAME = "uploaded.png"
    MODEL_PATH = "best.pt"
    IMAGE_ADDRESS = "https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/10/policy-scenarios-for-eliminating-plastic-pollution-by-2040_28eb9536/76400890-en.jpg"

    # UI
    st.title("Plastic Detection")
    st.image(IMAGE_ADDRESS, caption="Plastic Detection Example")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Open and save the uploaded file using PIL
        image = Image.open(uploaded_file)
        image.save(IMAGE_NAME)

        # Get predictions with a spinner
        with st.spinner("Getting Predictions..."):
            mask_response = make_predictions(IMAGE_NAME, MODEL_PATH)

            # Display images and results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image)

            with col2:
                st.subheader("Detection Result")
                if mask_response:
                    # Assuming `mask_response` has an image-like output
                    st.image(mask_response)
                else:
                    st.error("Error Getting Predictions", icon="🚨")

if __name__ == "__main__":
    run_app()