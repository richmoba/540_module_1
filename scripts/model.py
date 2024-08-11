import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn import svm
import joblib
import os
from pathlib import Path
import requests
import tempfile

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def get_model_dir():
    return Path(tempfile.mkdtemp())

def train_cnn(X_train, y_train):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(model_dir / 'cnn_model.h5')
    print(f"CNN model saved successfully in {model_dir}")

def train_svm(X_train, y_train):
    model = svm.SVC(kernel='linear')
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_dir / 'svm_model.pkl')
    print(f"SVM model saved successfully in {model_dir}")

if __name__ == "__main__":
    # OneDrive URL for processed data (replace with your actual URL)
    processed_data_url = "https://1drv.ms/u/s!AhyCheI--Ucdn7E2GA0RBOZIEB4-eQ?e=ch74hd"
    
    # Create a temporary directory to store downloaded files
    temp_dir = tempfile.mkdtemp()
    
    # Download and load the processed data
    processed_data_path = os.path.join(temp_dir, 'processed_data.npz')
    download_file(processed_data_url, processed_data_path)
    data = np.load(processed_data_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']

    train_cnn(X_train, y_train)
    train_svm(X_train, y_train)