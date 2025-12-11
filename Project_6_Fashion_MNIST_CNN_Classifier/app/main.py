#  Importing the dependencies
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/fashion_mnist_model.h5"
st.write("Model path:", model_path)

# Loading the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define the classes labels for Fashion MNIST dataset
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def pre_process(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28,1)
    return img_array

# Streamlit App
st.title("Fashion MNIST Classifier")

uploaded_image = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((100,100))
        st.image(resized_image)

    with col2:
        if st.button("Classify"):
            # Preprocess the uploaded image
            img_array = pre_process(uploaded_image)

            # Make the prediction using the pretrained model
            result = model.predict(img_array)
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f"Prediction : {prediction}")