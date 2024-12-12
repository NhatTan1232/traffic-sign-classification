import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

st.title('Traffic signs classification')
model = tf.keras.models.load_model('model.h5')

st.header('Upload a traffic sign image')
uploaded_file = st.file_uploader("Choose an image file", type=(['png', 'jpg', 'jpeg']))

labels = [
    "Speed limit (5km/h)",
    "Speed limit (15km/h)",
    "Speed limit (30km/h)",
    "Speed limit (40km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "Don't Go straight or left",
    "Don't Go straight or right",
    "Don't Go straight",
    "Don't Go left",
    "Don't Go left or right",
    "Don't Go right",
    "Don't overtake from left",
    "No U-turn",
    "No car",
    "No horn",
    "Speed limit (40km/h)",
    "Speed limit (50km/h)",
    "Go straight or right",
    "Go straight",
    "Go left",
    "Go left or right",
    "Go right",
    "Keep left",
    "Keep right",
    "Roundabout mandatory",
    "Watch out for cars",
    "Horn",
    "Bicycles crossing",
    "U-turn",
    "Road divider",
    "Traffic signals",
    "Danger ahead",
    "Zebra crossing",
    "Bicycles crossing",
    "Children crossing",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Downhill ahead",
    "Uphill ahead",
    "Slow",
    "Go right or straight",
    "Go left or straight",
    "Village ahead",
    "Zigzag curve",
    "Train crossing",
    "Under construction",
    "Continuous curves ahead",
    "Fences",
    "Heavy vehicle accidents",
    "Stop",
    "Give way", 
    "No stopping", 
    "No entry", 
    "Yield", 
    "Check"
]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Test image')

    if st.button('Predict'):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))
        cv2.equalizeHist(image)
        image = image / 255.0
        image = image.reshape(1, 100, 100, 1)

        result = model.predict(image)
        result = np.argmax(result, axis=1)

        st.header('Result')
        st.text(labels[result[0]])
