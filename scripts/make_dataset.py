import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pydicom
import SimpleITK as sitk
import subprocess

def find_dcm_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                return os.path.join(root, file)
    return None

def check_file_access(file_path):
    try:
        if not os.path.exists(file_path):
            return f"File does not exist: {file_path}"
        
        if not os.path.isfile(file_path):
            return f"Path is not a file: {file_path}"
        
        if not os.access(file_path, os.R_OK):
            return f"File is not readable: {file_path}"
        
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
            return "File is accessible and readable"
        except IOError as e:
            return f"IOError when trying to read file: {e}"
    except Exception as e:
        return f"Error checking file access: {str(e)}"

def run_icacls(file_path):
    try:
        result = subprocess.run(['icacls', file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running icacls: {str(e)}"

def read_dicom_file(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        return dicom.pixel_array
    except Exception as e:
        print(f"Error reading with pydicom: {e}")
        try:
            image = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            print(f"Error reading with SimpleITK: {e}")
            return None

def copy_files_and_load_images(metadata_file, output_dir):
    images = []
    labels = []
    errors = []
    processed_files = 0
    skipped_files = 0

    metadata_df = pd.read_csv(metadata_file, sep=',')
    roi_mask_df = metadata_df[metadata_df['Series Description'] == 'ROI mask images']

    for index, row in roi_mask_df.iterrows():
        file_path = row['File Location']
        
        print(f"Processing file {index + 1} of {len(roi_mask_df)}")
        print(f"Original file location from metadata: {file_path}")

        base_path = os.path.dirname(metadata_file)
        full_dir_path = os.path.normpath(os.path.join(base_path, file_path))
        
        dcm_file_path = find_dcm_file(full_dir_path)
        
        if dcm_file_path is None:
            error_msg = f"No DCM file found in directory: {full_dir_path}"
            print(error_msg)
            errors.append(error_msg)
            skipped_files += 1
            continue

        print(f"Found DCM file: {dcm_file_path}")
        
        access_status = check_file_access(dcm_file_path)
        print(f"File access status: {access_status}")
        
        icacls_output = run_icacls(dcm_file_path)
        print(f"ICACLS output:\n{icacls_output}")
        
        if "File is accessible and readable" in access_status:
            try:
                image = read_dicom_file(dcm_file_path)
                
                if image is not None:
                    image = cv2.resize(image, (128, 128))
                    images.append(image)

                    if 'Calc-Test' in dcm_file_path or 'Calc-Training' in dcm_file_path:
                        labels.append('calcification')
                    elif 'Mass-Test' in dcm_file_path or 'Mass-Training' in dcm_file_path:
                        labels.append('mass')
                    else:
                        labels.append('normal')

                    processed_files += 1
                else:
                    error_msg = f"Failed to read DICOM file: {dcm_file_path}"
                    print(error_msg)
                    errors.append(error_msg)
                    skipped_files += 1
            except Exception as e:
                error_msg = f"Error processing file {dcm_file_path}. Error: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                skipped_files += 1
        else:
            error_msg = f"File access issue: {dcm_file_path}. Status: {access_status}"
            print(error_msg)
            errors.append(error_msg)
            skipped_files += 1

    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    print(f"Encountered {len(errors)} errors.")
    
    with open(os.path.join(output_dir, 'error_log.txt'), 'w') as f:
        for error in errors:
            f.write(f"{error}\n")
    
    return np.array(images), np.array(labels)

def preprocess_and_split_data(metadata_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    images, labels = copy_files_and_load_images(metadata_file, output_dir)
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No images or labels found. Please check the metadata file and ensure paths are correct.")
    
    images = images / 255.0
    
    label_map = {'normal': 0, 'calcification': 1, 'mass': 2}
    labels = np.array([label_map[label] for label in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    np.savez(os.path.join(output_dir, 'processed_data.npz'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    print("Preprocessing and splitting completed successfully.")

if __name__ == "__main__":
    project_root = 'D:\\540_module_1'
    metadata_file_path = os.path.join(project_root, 'data\\raw\\CBIS-DDSM\\metadata.csv')
    output_directory = os.path.join(project_root, 'data\\processed')

    print(f"Metadata file path: {metadata_file_path}")
    print(f"Output directory: {output_directory}")

    preprocess_and_split_data(metadata_file_path, output_directory)