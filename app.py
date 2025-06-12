# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained Keras model
model = load_model("plant_disease_model.h5")

# Class labels used by the model
class_names = ['Early Blight', 'Healthy', 'Late Blight']

# Streamlit page configuration
st.set_page_config(
    page_title="Plant Disease Detection System",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Basic styling for centered text
st.markdown("""
    <style>
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar with instructions
# ----------------------------
with st.sidebar:
    st.title("Instructions")
    st.markdown("""
    **How to use this system:**

    - Upload a plant leaf image (JPG or PNG format).
    - The system will analyze it using a deep learning model.
    - You will get the predicted disease class and confidence level.

    **Supported classes:**
    - Early Blight
    - Healthy
    - Late Blight
    """)

# ----------------------------
# Title and description (centered)
# ----------------------------
st.markdown("<h1 class='center-text'>Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text'>Upload a plant leaf image to detect whether it's healthy or affected by a disease using deep learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Upload image section
# ----------------------------
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess the image
    resized_img = image_pil.resize((128, 128))
    img_array = image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Display prediction result
    st.markdown("<h3 class='center-text'>Prediction Result</h3>", unsafe_allow_html=True)
    st.success(f"Predicted Class: {predicted_class}")
    st.markdown(f"<p class='center-text'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

    # Confidence bar chart
    st.markdown("### Prediction Confidence")
    st.bar_chart({
        "Confidence (%)": {class_names[i]: prediction[i] * 100 for i in range(len(class_names))}
    })

else:
    st.info("Please upload a valid leaf image to begin analysis.")

# ----------------------------
# Footer with credit
# ----------------------------
st.markdown("---")
st.markdown("<div class='center-text' style='color: grey;'>Made using Streamlit by Devesh Jha</div>", unsafe_allow_html=True)
