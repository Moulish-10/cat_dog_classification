import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('trained_model_2.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((100, 100))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input shape
    return image

# Streamlit UI
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🐱🐶 Cat vs Dog Classifier 🐶🐱</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("ℹ️ About")
st.sidebar.write("This AI model classifies images as either **Cat** or **Dog** .")

uploaded_file = st.file_uploader("📤 Upload an image of a cat or dog", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

    # Set a confidence threshold (e.g., 80%) for "Something Else"
    threshold = 80  # Adjust this if needed

    if confidence >= threshold:
        if prediction > 0.5:
            st.success(f"🐶 This is a **Dog**! (Confidence: {confidence}%)")
        else:
            st.success(f"🐱 This is a **Cat**! (Confidence: {confidence}%)")
    else:
        st.warning(f"🤔 This doesn't look like a Cat or Dog! (Confidence: {confidence}%)")

# Footer
st.markdown(
    "<br><p style='text-align: center;'>Made with ❤️ using TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)
