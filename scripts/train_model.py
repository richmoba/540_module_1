# train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def load_data(npz_file):
    data = np.load(npz_file)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def train_cnn(X_train, y_train):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: normal, benign, malignant
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('D:\\540_module_1\\models\\cnn_model.h5')

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    joblib.dump(model, 'D:\\540_module_1\\models\\svm_model.pkl')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data('D:\\540_module_1\\data\\processed\\processed_data.npz')
    
    # Reshape data for CNN input
    X_train_cnn = X_train.reshape(-1, 128, 128, 1)
    X_test_cnn = X_test.reshape(-1, 128, 128, 1)
    
    # Train and evaluate CNN
    train_cnn(X_train_cnn, y_train)
    cnn_model = tf.keras.models.load_model('D:\\540_module_1\\models\\cnn_model.h5')
    cnn_predictions = cnn_model.predict(X_test_cnn)
    cnn_accuracy = np.mean(np.argmax(cnn_predictions, axis=1) == y_test)
    print(f"CNN Accuracy: {cnn_accuracy}")

    # Train and evaluate SVM
    train_svm(X_train, y_train)
    svm_model = joblib.load('D:\\540_module_1\\models\\svm_model.pkl')
    svm_predictions = svm_model.predict(X_test.reshape(len(X_test), -1))
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"SVM Accuracy: {svm_accuracy}")