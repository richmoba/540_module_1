import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, image_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(1 if 'cancer' in label else 0)
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        feature, _ = hog(image, visualize=True)
        hog_features.append(feature)
    return np.array(hog_features)

if __name__ == "__main__":
    data_folder = 'data/raw/cbis-ddsm'
    images, labels = load_images_from_folder(data_folder)
    hog_features = extract_hog_features(images)

    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

    np.savez('data/processed/features.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
