import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

# =====================
# 1. Load .env
# =====================

IMAGE_SIZE = int (224)
MODEL_PATH = 'human_classification1.keras'

# =====================
# 2. Load Model (cache)
# =====================
@st.cache_resource
def load_my_model():
    st.write(f" Đang tải model từ: `{MODEL_PATH}` ...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

try:
    model = load_my_model()
    INPUT_SHAPE = model.input_shape
except Exception as e:
    st.error(f" Lỗi khi load model: {e}")
    st.stop()

# =====================
# 3. Preprocess Image
# =====================
def preprocess_image(image: Image.Image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = image.convert("RGB")

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))

    return img_array

# =====================
# 4. Streamlit UI
# =====================
st.set_page_config(
    page_title="Human vs Non-Human AI",
    page_icon="</>",
    layout="wide"
)
st.markdown(
    """
    <h1 style="text-align:center;"> Human vs Non-Human Classifier</h1>
    <p style="text-align:center; color:gray;">
        Deep Learning model using ResNet50
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader(
    " Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing a human or non-human object"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption=" Uploaded Image",
            use_container_width=True
        )

    with col2:
        with st.spinner(" Model is thinking..."):
            img_tensor = preprocess_image(image)

            if isinstance(INPUT_SHAPE, list) and len(INPUT_SHAPE) >= 2:
                prediction_raw = model.predict(
                    [img_tensor, img_tensor], verbose=0
                )
            else:
                prediction_raw = model.predict(
                    img_tensor, verbose=0
                )

            prediction = float(prediction_raw[0][0])

        is_human = prediction >= 0.5
        label = "Human " if is_human else "Non-Human "
        confidence = prediction if is_human else 1.0 - prediction

        st.subheader(" Prediction Result")

        if is_human:
            st.success(f"**{label}**")
        else:
            st.warning(f"**{label}**")

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f} %"
        )

        st.progress(confidence)

        with st.expander("Model raw output"):
            st.write(f"Raw score: `{prediction:.6f}`")
            st.write(f"Threshold: `0.5`")

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:14px;">
         Powered by TensorFlow & Streamlit <br>
        Made with <3 for Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)
