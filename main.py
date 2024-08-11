import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from sklearn import svm
import joblib
import os
from pathlib import Path
import requests


# Define the URL of the file
onedrive_url = "https://1drv.ms/u/s!AhyCheI--Ucdn7E2GA0RBOZIEB4-eQ?e=ch74hd"
local_filename = Path("processed_data.npz")

# Download the file from OneDrive
response = requests.get(onedrive_url)
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Try loading with numpy
try:
    data = np.load(local_filename, allow_pickle=True)
except (pickle.UnpicklingError, ValueError) as e:
    print(f"Error loading file with numpy: {e}")
    # If numpy fails, try loading with joblib
    try:
        data = joblib.load(local_filename)
    except Exception as e:
        print(f"Error loading file with joblib: {e}")
        data = None

# Function to get the model directory
def get_model_dir():
    # Try to get the model directory from an environment variable
    model_dir = os.getenv('MAMMOGRAPHY_MODEL_DIR')
    if model_dir:
        return Path(model_dir)
    
    # If not set, use the OneDrive path
    onedrive_path = Path(os.path.expanduser("~")) / "OneDrive" / "1. Baker-Richmond" / "1. Edu" / "1. RVB" / "1. Duke" / "0. 540" / "code" / "Module_1" / "models"
    if onedrive_path.exists():
        return onedrive_path
    
    # If OneDrive path doesn't exist, use a default path relative to the script
    return Path(__file__).parent / "models"

# Get model and data directories
model_dir = get_model_dir()
data_dir = Path(__file__).parent / "data" / "processed"

# Load data
data = np.load(data_dir / 'processed_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load models
cnn_model = tf.keras.models.load_model(model_dir / 'cnn_model.h5')
svm_model = joblib.load(model_dir / 'svm_model.pkl')

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

st.write(f"\nUsing model directory: {model_dir}")
st.write(f"Using data directory: {data_dir}")