import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("attention_unet_best_model.keras")

# Streamlit UI
st.title("🩺 Breast Cancer Detection App")
st.write("Upload an image to predict if it's benign or malignant.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    """Load and preprocess image for model prediction."""
    img = image.convert("L")  # Convert to grayscale
    img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Resize
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    
    # Reshape to (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    return img_array

# If an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Show image preview
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_array)
    prediction_value = float(prediction.flatten()[0])  # Extract single value

    # Display result
    result = "🔴 Malignant (Cancer Detected)" if prediction_value > 0.5 else "🟢 Benign (No Cancer Detected)"
    st.subheader("Prediction Result:")
    st.markdown(f"**{result}**")

    # Show confidence score
    st.write(f"**Confidence:** {prediction_value:.4f}")

