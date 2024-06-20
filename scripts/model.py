import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn import svm
import joblib

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
    model.save('models/cnn_model.h5')

def train_svm(X_train, y_train):
    model = svm.SVC(kernel='linear')
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    joblib.dump(model, 'models/svm_model.pkl')

if __name__ == "__main__":
    data = np.load('data/processed/processed_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']

    train_cnn(X_train, y_train)
    train_svm(X_train, y_train)
