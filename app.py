import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# Load Model
model = load_model("cnn_cancer_model.h5")

def prediction(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    class_label = "Malignant" if pred > 0.5 else "Benign"

    return class_label, pred

# Header
st.title("🩺 Skin Cancer Detection System")
st.markdown(
    """
    Welcome to the **AI-powered Skin Cancer Detection System**.

    Upload a skin lesion image and the deep learning model will analyze it
    to determine whether the lesion is:

    - ✅ **Benign** (Non-Cancerous)
    - ⚠️ **Malignant** (Potentially Cancerous)

    ---
    """
)

# Disclaimer
st.warning(
    "⚠️ This application is for educational purposes only and should not be used as a substitute for professional medical diagnosis."
)

# File Upload
uploaded_image = st.file_uploader(
    "📤 Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    st.image(
        uploaded_image,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing Image..."):
        label, confidence = prediction(uploaded_image)

    st.success("Analysis Complete")

    if label == "Malignant":
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.write(f"**Model Confidence Score:** {confidence:.4f}")

    st.markdown("---")
    st.markdown(
        """
        ### About this Model
        This CNN-based model has been trained to classify skin lesion images into:
        
        - **Benign**
        - **Malignant**
        
        The prediction is generated using deep learning techniques and image pattern recognition.
        """
    )

# Footer
st.markdown("---")
st.caption("Developed using Streamlit, TensorFlow, and CNN")