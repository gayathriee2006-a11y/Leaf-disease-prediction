import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model_path = "model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)
