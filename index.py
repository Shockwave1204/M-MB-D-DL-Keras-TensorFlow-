import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from contextlib import suppress
import streamlit_shadcn_ui as ui

# Function to load model with custom objects if necessary
@st.cache_resource()
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load multiple models
model_paths = {
    "DenseNet169": "densenet169_final.keras",
    "DenseNet201": "densenet201_final.keras",
    "DenseNet121": "densenet121_final.keras",
    "VGG16": "vgg16_final.keras",
    "ResNet50": "resnet50_final.keras",
    "ResNet152": "resnet152_final.keras",
}

models = {name: load_model(path) for name, path in model_paths.items()}

# Define class labels
class_labels = {0: "Benign", 1: "Malignant"}  # Adjust according to your dataset


def main():
    # Create navigation bar
    st.sidebar.title("Dashboard")
    page = st.sidebar.radio("Go to", ["Home", "About Us"])

    if page == "Home":
        st.title("Melanoma Malignant and Benign Classification App")
        st.write("Upload an image and select a model. The selected model will predict the class.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        selected_model = st.selectbox("Select Model", list(models.keys()))

        classify_button = ui.button(text="Classify", key="styled_btn_tailwind", className="bg-green-400 text-white")

        if uploaded_file is not None and classify_button:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Classifying...")

            # Resize image to 224x224
            input_shape = (224, 224)
            image = image.resize(input_shape)

            # Convert image to numpy array and normalize
            image = np.array(image) / 255.0  

            # Expand dimensions to match the input shape expected by the model
            image = np.expand_dims(image, axis=0)

            with suppress():
                # Predict class probabilities using the selected model
                prediction = models[selected_model].predict(image)
            
            # Get the predicted class label
            pred_class = np.argmax(prediction)

            # Map predicted class label to class name
            predicted_label = class_labels[pred_class]

            # Get the probability of the predicted class
            confidence = prediction[0][pred_class]

            st.write(f"Predicted Class: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}")
    elif page == "About Us":
        st.title("About Us")

        cols = st.columns(2)
        with cols[0]:
            ui.metric_card("MD LIKHON MIA", "203-15-3916", "57_D, CSE, likhon15-3916@diu.edu.bd")

        with cols[1]:
            ui.metric_card("Eshita Akter", "203-15-3922", "57_D, CSE, eshita15-3922@diu.edu.bd")
        # You can add more content about your team, project, etc.

if __name__ == "__main__":
    main()
