from ultralytics import YOLO
import streamlit as st

MODEL_NAME = "best.pt"

@st.cache_resource
def load_yolo_model(image_path):

    return YOLO(MODEL_NAME)