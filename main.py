import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from sklearn import svm
import joblib
import os
from pathlib import Path
import requests
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to download file from OneDrive
def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Successfully downloaded file to {local_filename}")
        return local_filename
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        return None

# OneDrive URLs (replace these with your actual OneDrive shared links)
processed_data_url = "https://1drv.ms/u/s!AhyCheI--Ucdn7E2GA0RBOZIEB4-eQ?e=ch74hd"
cnn_model_url = "YOUR_CNN_MODEL_ONEDRIVE_URL"
svm_model_url = "YOUR_SVM_MODEL_ONEDRIVE_URL"

# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()
logging.info(f"Created temporary directory: {temp_dir}")

# Download and load the processed data
processed_data_path = os.path.join(temp_dir, 'processed_data.npz')
if download_file(processed_data_url, processed_data_path):
    logging.info(f"Attempting to load data from {processed_data_path}")
    try:
        # Try loading with numpy
        data = np.load(processed_data_path, allow_pickle=True)
        X_test = data['X_test']
        y_test = data['y_test']
        logging.info("Successfully loaded data with numpy")
    except Exception as np_error:
        logging.error(f"Error loading with numpy: {np_error}")
        try:
            # If numpy fails, try loading with joblib
            data = joblib.load(processed_data_path)
            X_test = data['X_test']
            y_test = data['y_test']
            logging.info("Successfully loaded data with joblib")
        except Exception as joblib_error:
            logging.error(f"Error loading with joblib: {joblib_error}")
            st.error("Failed to load the processed data. Please check the file format and try again.")
            st.stop()
else:
    st.error("Failed to download the processed data. Please check the URL and try again.")
    st.stop()

# Download and load the models
cnn_model_path = os.path.join(temp_dir, 'cnn_model.h5')
svm_model_path = os.path.join(temp_dir, 'svm_model.pkl')

if download_file(cnn_model_url, cnn_model_path) and download_file(svm_model_url, svm_model_path):
    try:
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        svm_model = joblib.load(svm_model_path)
        logging.info("Successfully loaded both models")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        st.stop()
else:
    st.error("Failed to download the model files. Please check the URLs and try again.")
    st.stop()

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

def interpret_prediction(prediction):
    if prediction in [1, 2]:  # Assuming 1 and 2 represent different types of cancer
        return "Cancer"
    else:
        return "No Cancer"

st.title('Mammography Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 0)
    st.image(image, channels="GRAY")

    cnn_pred = predict_with_cnn(image)
    svm_pred = predict_with_svm(image)

    st.write(f"Prediction with CNN: {interpret_prediction(cnn_pred)} (Class {cnn_pred})")
    st.write(f"Prediction with SVM: {interpret_prediction(svm_pred)} (Class {svm_pred})")

    st.write("\nNote: Class 0 typically represents 'No Cancer', while Classes 1 and 2 represent different types of cancer (e.g., calcification and mass).")

st.write(f"\nUsing temporary directory for models and data: {temp_dir}")

# Display logs in Streamlit
with st.expander("View Logs"):
    st.text(logging.getLogger().handlers[0].stream.getvalue())