#!/bin/bash

############ Download ObMan Annotations and Splits ############
# Set target directories
ann_dir="data/ObMan/annotations"
splits_dir="data/ObMan/splits"

# Create directories
mkdir -p "$ann_dir"
mkdir -p "$splits_dir"

# Download annotations folder from Google Drive
echo "Downloading annotations to $ann_dir ..."
gdown --folder https://drive.google.com/drive/folders/1DBzG9J0uLzCy4A6W6Uq6Aq4JNAHiiNJQ -O "$ann_dir"

# Download split JSON files from GitHub
echo "Downloading train/test split files to $splits_dir ..."
wget -c https://raw.githubusercontent.com/zerchen/gSDF/05101b5bde6765e9168026cff853b74a1412c125/datasets/obman/splits/train_87k.json -O "$splits_dir/train_87k.json"
wget -c https://raw.githubusercontent.com/zerchen/gSDF/05101b5bde6765e9168026cff853b74a1412c125/datasets/obman/splits/test_6k.json -O "$splits_dir/test_6k.json"

echo "ObMan annotations and splits successfully downloaded."
############ End of ObMan setup ############