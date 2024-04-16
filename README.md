# Installation
The main branch works with python=3.7+, PyTorch=1.13+, cuda=11.7

## Step 0.  Install MMCV using MIM
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

## Step 1. Install MMSegmentation.
pip install "mmsegmentation>=1.0.0"

## Step 2. Install other requirements
pip install -r requirements.txt

# Usage
## Step 0. Use AI model to inference images

The model and config file can be find [here](https://drive.google.com/drive/folders/1jfPfhizYpO7O-2XCnOmOKgmWpcWz-bGI?usp=drive_link)

inference.py
Args:
    inputImages(str): file path or folder path for the input images 
    predoutFolder(str): folder for the predicted mask
    maskedoutFolder(str): folder for the image with predicted mask
    isCPUonly(bool): True for using CPU, False for using GPU
        
## Step 1. Calculate FP width and GBM width
measurement4mask.py
Args:
    imagefilepath(str): folder path for the input images, files in imagefilepath and segfilepath should match perfectly except the suffix
    segfilepath(str): folder path for the predicted masks, files in imagefilepath and segfilepath should match perfectly except the suffix
    outCSV(str): csv file path for the measurements
    outfig(str): folder path for the masked images with GBM skeleton and FP width
    outjson(str): json file path for the FP width"# TEAMKidney" 
