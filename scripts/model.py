import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn import svm
import joblib
import os
from pathlib import Path

def get_model_dir():
    model_dir = os.getenv('MAMMOGRAPHY_MODEL_DIR')
    if model_dir:
        return Path(model_dir)
    
    onedrive_path = Path(os.path.expanduser("~")) / "OneDrive" / "1. Baker-Richmond" / "1. Edu" / "1. RVB" / "1. Duke" / "0. 540" / "code" / "Module_1" / "models"
    if onedrive_path.exists():
        return onedrive_path
    
    return Path(__file__).parent / "models"

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
    data_dir = Path(__file__).parent / "data" / "processed"
    data = np.load(data_dir / 'processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']

    train_cnn(X_train, y_train)
    train_svm(X_train, y_train)