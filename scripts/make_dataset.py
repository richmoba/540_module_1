import os
import shutil
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Try to import the required DICOM libraries and set flags
try:
    import pydicom
    pydicom_available = True
except ImportError:
    pydicom_available = False
    print("pydicom is not available. Install it using `pip install pydicom`.")

try:
    import SimpleITK as sitk
    simpleitk_available = True
except ImportError:
    simpleitk_available = False
    print("SimpleITK is not available. Install it using `pip install SimpleITK`.")

try:
    import itk
    itk_available = True
except ImportError:
    itk_available = False
    print("ITK is not available. Install it using `pip install itk`.")

# Handle the GDCM import separately since it often has issues on Windows
try:
    import gdcm
    gdcm_available = True
except ImportError as e:
    gdcm_available = False
    print(f"GDCM import failed: {e}. GDCM is not available.")

# Function to read DICOM using pydicom
def read_dicom_with_pydicom(file_path):
    if not pydicom_available:
        return None
    try:
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array
        return image
    except Exception as e:
        print(f"Error reading DICOM file {file_path} with pydicom: {e}")
        return None

# Function to read DICOM using SimpleITK
def read_dicom_with_simpleitk(file_path):
    if not simpleitk_available:
        return None
    try:
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        return image_array
    except Exception as e:
        print(f"Error reading DICOM file {file_path} with SimpleITK: {e}")
        return None

# Function to read DICOM using GDCM
def read_dicom_with_gdcm(file_path):
    if not gdcm_available:
        return None
    
    try:
        reader = gdcm.ImageReader()
        reader.SetFileName(file_path)
        if not reader.Read():
            raise ValueError("Failed to read DICOM file")
        
        image = reader.GetImage()
        image_array = image.GetBufferAsUint8()
        return np.frombuffer(image_array, dtype=np.uint8).reshape(image.GetDimension(0), image.GetDimension(1))
    except Exception as e:
        print(f"Error reading DICOM file {file_path} with GDCM: {e}")
        return None

# Function to read DICOM using ITK
def read_dicom_with_itk(file_path):
    if not itk_available:
        return None
    try:
        image = itk.imread(file_path)
        array = itk.GetArrayFromImage(image)
        return array
    except Exception as e:
        print(f"Error reading DICOM file {file_path} with ITK: {e}")
        return None

# Function to try multiple DICOM readers in sequence
def read_dicom_file(file_path):
    readers = [
        read_dicom_with_pydicom,
        read_dicom_with_simpleitk,
        read_dicom_with_gdcm if gdcm_available else lambda x: None,  # Skip GDCM if not available
        read_dicom_with_itk
    ]
    
    for reader in readers:
        image = reader(file_path)
        if image is not None:
            return image
    print(f"Failed to read DICOM file {file_path} with all available readers.")
    return None

# Function to process the DICOM files and prepare the dataset
def copy_files_and_load_images(metadata_file, temp_dir):
    images = []
    labels = []

    # Load the CSV metadata file
    metadata_df = pd.read_csv(metadata_file)

    # Filter for rows with 'ROI mask images' if needed
    roi_mask_df = metadata_df[metadata_df['Series Description'].str.contains('ROI mask images', na=False)]

    for index, row in roi_mask_df.iterrows():
        file_path = row['File Location']
        
        # Debugging information
        print(f"Original file location from metadata: {file_path}")

        # Normalize the path by removing any redundant components
        if file_path.startswith('.'):
            file_path = file_path[2:]  # Remove the leading './'

        # Ensure 'CBIS-DDSM' is included only once in the final path
        file_path = file_path.lstrip("\\/")  # Strip leading slashes
        if file_path.startswith('CBIS-DDSM'):
            file_path = file_path[len('CBIS-DDSM'):]  # Remove leading 'CBIS-DDSM' if present
        file_path = file_path.lstrip("\\/")  # Strip any leading slashes again

        # Construct the full path considering the project root directory
        base_path = os.path.dirname(metadata_file)  # Get the directory of the metadata file
        full_file_path = os.path.join(base_path, 'CBIS-DDSM', file_path)

        # Normalize the path to remove any redundant segments
        full_file_path = os.path.normpath(full_file_path)
        
        print(f"Resolved file path: {full_file_path}")
        
        if os.path.exists(full_file_path):
            try:
                # Copy the file to a temporary location
                temp_file_path = os.path.join(temp_dir, os.path.basename(full_file_path))
                shutil.copy2(full_file_path, temp_file_path)  # Use copy2 to preserve metadata
                
                # Attempt to read the file from the temporary location
                image = read_dicom_file(temp_file_path)
                if image is not None:
                    # Resize image to a standard size (optional)
                    image = cv2.resize(image, (128, 128))  # Example size, adjust as needed
                    images.append(image)

                    # Infer label based on some logic or column in metadata
                    if 'benign' in file_path.lower():
                        labels.append('benign')
                    elif 'malignant' in file_path.lower():
                        labels.append('malignant')
                    else:
                        labels.append('normal')  # Adjust this as per your dataset

            except Exception as e:
                print(f"Error processing file {full_file_path} after copying to {temp_file_path}: {e}")
        else:
            print(f"File not found: {full_file_path}")

    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    return np.array(images), np.array(labels)

# Function to preprocess and split the dataset
def preprocess_and_split_data(metadata_file, output_dir):
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    images, labels = copy_files_and_load_images(metadata_file, temp_dir)
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No images or labels found. Please check the metadata file and ensure paths are correct.")
    
    # Normalize images (optional)
    images = images / 255.0  # Scale pixel values to [0, 1]
    
    # Convert labels to numerical values if they are categorical
    label_map = {'normal': 0, 'benign': 1, 'malignant': 2}  # Example mapping
    labels = np.array([label_map[label] for label in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    np.savez(os.path.join(output_dir, 'processed_data.npz'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir)
    print("Preprocessing and splitting completed successfully.")
