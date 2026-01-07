import os
import sys

# 1. Get the path to the project root (one level up from 'app')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. Add it to Python's "search path"
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import tensorflow as tf
import numpy as np
from sqlalchemy import text
from PIL import Image, ImageOps
from database.db_manager import engine

# 1. Page Configuration
st.set_page_config(page_title="AI Fashion Classifier", layout="centered")
st.title("🧥 Fashion MNIST Classifier")
st.write("Upload an image of a clothing item to see the AI's prediction.")


# 2. Load the Model (Saved in Block A/B)
@st.cache_resource
def load_my_model():
    # Ensure this path matches where you saved your model in the notebook
    return tf.keras.models.load_model("models/my_mnist_model.keras")


model = load_my_model()

# 3. Define Class Names (Same as HOML Ch 10)
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 4. Image Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 5. Preprocessing Pipeline
    # Convert to grayscale, resize to 28x28, and normalize
    # 5. Preprocessing Pipeline
    st.write("### AI Perspective")
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28))

    # Optional: Only invert if the background is white
    img = ImageOps.invert(img)

    st.image(img, caption="What the AI sees (28x28)", width=200)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # NEW: Debug View (Show what the AI actually "sees")
    # We display the 28x28 image scaled up so you can see the blur
    # 6. Inference
    if st.button("Classify Image"):
        predictions = model.predict(img_array)
        score = np.max(predictions)
        class_idx = np.argmax(predictions)
        result = class_names[class_idx]

        # 7. Display Results
        st.write("### Results")
        # If confidence is too low, don't risk a wrong guess
        if confidence < 0.50:
            st.warning(
                f"⚠️ Low Confidence ({confidence:.2f}). Flagging for Human Review."
            )
            # In a real app, this would write to a 'manual_review' SQL table
        else:
            st.success(f"**Prediction:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}")

            # THE FEEDBACK LOOP
            try:
                with engine.connect() as conn:
                    with conn.begin():
                        conn.execute(
                            text(
                                "INSERT INTO inference_logs (predicted_class, confidence_score) VALUES (:p, :c)"
                            ),
                            {"p": result, "c": float(score)},
                        )
                st.success("✅ Prediction logged to SQL for Day 5 analysis!")
            except Exception as e:
                st.error(f"Logging failed: {e}")
