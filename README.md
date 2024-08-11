# 540_module_1
540_module_1
# Medical Image Classification Project
Youtube video location https://youtu.be/Eb6bSmw19Gk

## Overview
This project aims to classify mammography images to identify the presence of abnormalities such as masses and calcifications. The dataset used is the CBIS-DDSM, which must be manually downloaded from the CBIS-DDSM organization on TCIA.

## Project Structure
project/
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── scripts/
│ ├── make_dataset.py
│ ├── build_features.py
│ ├── model.py
├── models/
├── data/
│ ├── raw/
│ │  ├── CBIS-DDSM/
│ ├── processed/
│ ├── outputs/
├── notebooks/
├── .gitignore


## Setup Instructions
0. git clone https://github.com/richmoba/540_module_1.git 
1. Manually download the CBIS-DDSM dataset from the CBIS-DDSM organization on TCIA.
2. https://www.cancerimagingarchive.net/collection/cbis-ddsm/
Note: ensure that the CBIS-DDSM folder is not read only.
3. Place the downloaded dataset in the `data/raw` directory. The directory structure should look similar to this:
data/raw/
├── CBIS-DDSM/
├── Calc-Test_P_00038_LEFT_MLO_1/
├── 08-29-2017-DDSM-NA-11739/
├── 1.000000-ROI mask images-88680/
├── image1.jpg
├── image2.jpg
└── ...

4. Install the required dependencies using `pip install -r requirements.txt`.
5. Run the setup script: `python setup.py`.
6. Run the model script: 'python  scripts\model.py
6. Start the Streamlit interface: `streamlit run main.py`.

## Running the Project
The project includes a visual interface to upload and classify new images. The interface will display results from both the SVM and CNN models.

## Model Evaluation
Both classical machine learning (SVM) and deep learning (CNN) approaches are evaluated. The models are compared using accuracy, precision, recall, and a naive approach.
