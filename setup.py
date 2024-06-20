import os
import subprocess
import sys

# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
        raise

# List of required packages
required_packages = [
    "pydicom",
    "SimpleITK",
    "itk",
    "pandas",    # Ensure pandas is installed
    "numpy",     # Ensure numpy is installed
    "opencv-python",  # Ensure OpenCV is installed
    "scikit-learn"    # Ensure scikit-learn is installed
]

# Attempt to install required packages if not already installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install_package(package)

# Now import the modules
import shutil
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Try to import GDCM and set a flag if it's available
try:
    import gdcm
    gdcm_available = True
except ImportError as e:
    print(f"GDCM import failed: {e}. GDCM is not available.")
    gdcm_available = False

import pydicom
import SimpleITK as sitk
import itk

from scripts.make_dataset import preprocess_and_split_data

if __name__ == "__main__":
    # Define the path to the metadata file and output directory
    project_root = 'e:\\Project'  # Change this to your project root on the E: drive
    metadata_file_path = os.path.join(project_root, 'data\\raw\\CBIS-DDSM\\metadata.csv')
    output_directory = os.path.join(project_root, 'data\\processed')

    # Verify paths
    print(f"Metadata file path: {metadata_file_path}")
    print(f"Output directory: {output_directory}")

    # Call the preprocessing function
    preprocess_and_split_data(metadata_file_path, output_directory)
