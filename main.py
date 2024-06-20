import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from sklearn import svm
import joblib

# Load data
data = np.load('data/processed/processed_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load models
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
svm_model = joblib.load('models/svm_model.pkl')

def predict_with_cnn(image):
    image = cv2.resize(image, (128, 128))
    image = image.reshape(1, 128, 128, 1)
    prediction = cnn_model.predict(image)
    return np.argmax(prediction, axis=1)[0]

def predict_with_svm(image):
    image = cv2.resize(image, (128, 128))
    image = image.reshape(1, -1)
    prediction = svm_model.predict(image)
    return prediction[0]

st.title('Mammography Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)
    st.image(image, channels="GRAY")

    cnn_pred = predict_with_cnn(image)
    svm_pred = predict_with_svm(image)

    st.write(f"Prediction with CNN: {cnn_pred}")
    st.write(f"Prediction with SVM: {svm_pred}")
