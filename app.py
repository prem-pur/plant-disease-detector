# ui/app.py
"""
Streamlit demo for plant disease classifier (PyTorch).
Run:
    streamlit run ui/app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import json
from src.model_loader import load_model
from src.utils import get_default_transform, load_class_names, get_device
from src.gradcam import GradCAM, apply_colormap_on_image

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("Plant Leaf Disease Classifier")
st.write("Upload an image and the model will predict the disease and show a Grad-CAM overlay.")

# Sidebar options
st.sidebar.header("Settings")
MODEL_PATH = st.sidebar.text_input("Model path", value="model/plant_disease.pth")
MODEL_NAME = st.sidebar.selectbox("Backbone", options=["resnet50", "efficientnet_b0"], index=0)
NUM_CLASSES = st.sidebar.number_input("num_classes (0 = infer if possible)", min_value=0, value=0)
CLASS_NAMES_PATH = st.sidebar.text_input("class_names.json", value="model/plant_disease.json")
IMAGE_SIZE = st.sidebar.number_input("image_size", min_value=64, max_value=1024, value=224)

# Load class names
try:
    class_names = load_class_names(CLASS_NAMES_PATH)
except Exception as e:
    class_names = None
    st.sidebar.warning(f"Could not load class names: {e}")

# Load model button (to avoid reloading on every action)
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

if st.sidebar.button("Load model"):
    try:
        nc = None if NUM_CLASSES == 0 else int(NUM_CLASSES)
        model, device = load_model(MODEL_PATH, model_name=MODEL_NAME, num_classes=nc, device=get_device())
        st.session_state['model'] = model
        st.session_state['device'] = device
        st.session_state['model_loaded'] = True
        st.sidebar.success(f"Model loaded on {device}")
        # determine default target layer depending on backbone
        if MODEL_NAME == "resnet50":
            st.session_state['target_layer_name'] = "layer4[-1].conv3"
        else:
            st.session_state['target_layer_name'] = "features[-1]"  # approximate
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)
    if not st.session_state.get('model_loaded', False):
        st.info("Load the model first using the left sidebar 'Load model' button.")
    else:
        model = st.session_state['model']
        device = st.session_state['device']
        tf = get_default_transform(IMAGE_SIZE)
        inp = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        label = class_names[idx] if class_names else str(idx)
        st.markdown(f"**Prediction:** {label}  —  **{probs[idx]:.3f}**")

        # Grad-CAM (if user wants)
        if st.checkbox("Show Grad-CAM overlay"):
            # pick target layer for resnet50
            if MODEL_NAME == "resnet50":
                target_layer = model.layer4[-1].conv3
            elif MODEL_NAME == "efficientnet_b0":
                # for torchvision efficientnet_b0, the feature extractor ends with features[-1]
                try:
                    target_layer = model.features[-1]
                except Exception:
                    target_layer = list(model.modules())[-1]
            else:
                target_layer = list(model.modules())[-1]

            gcam = GradCAM(model, target_layer)
            cam, cls = gcam(inp, None)
            # resize cam to original image size
            cam_resized = cv2.resize(cam, (img.width, img.height))
            overlay = apply_colormap_on_image(np.array(img), cam_resized, alpha=0.4)
            st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
