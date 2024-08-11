#Richmond baker 540 module 1
# setup.py


import os   # Added to import os module
import subprocess   # Added to import subprocess module
import sys  # Added to import sys module

# Function to install a package
def install_package(package):   # Added to install a package
    try:    # Added to try to install a package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])    # Added to install a package
    except subprocess.CalledProcessError as e:  # Added to handle exceptions
        print(f"Failed to install {package}. Error: {e}")   # Added to print error message
        raise

# List of required packages
required_packages = [
    "pydicom",
    "SimpleITK",
    "itk",
    "pandas",
    "numpy",
    "opencv-python",
    "scikit-learn",
    "tensorflow"
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
    project_root = 'D:\\540_module_1'  # Updated to D: drive
    metadata_file_path = os.path.join(project_root, 'data\\raw\\CBIS-DDSM\\metadata.csv')
    output_directory = os.path.join(project_root, 'data\\processed')

    # Verify paths
    print(f"Metadata file path: {metadata_file_path}")
    print(f"Output directory: {output_directory}")

    # Call the preprocessing function
    preprocess_and_split_data(metadata_file_path, output_directory)