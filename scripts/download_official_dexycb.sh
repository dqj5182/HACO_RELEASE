#!/bin/bash

# Set target directories
data_dir="data/DexYCB/data"
splits_dir="data/DexYCB/splits"

# Create directories
mkdir -p "$data_dir"
mkdir -p "$splits_dir"

# Download, extract, and remove archive
gdown https://drive.google.com/uc?id=1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk -O "$data_dir/20200709-subject-01.tar.gz"
tar -xzf "$data_dir/20200709-subject-01.tar.gz" -C "$data_dir" && rm "$data_dir/20200709-subject-01.tar.gz"

gdown https://drive.google.com/uc?id=1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62 -O "$data_dir/20200813-subject-02.tar.gz"
tar -xzf "$data_dir/20200813-subject-02.tar.gz" -C "$data_dir" && rm "$data_dir/20200813-subject-02.tar.gz"

gdown https://drive.google.com/uc?id=1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq -O "$data_dir/20200820-subject-03.tar.gz"
tar -xzf "$data_dir/20200820-subject-03.tar.gz" -C "$data_dir" && rm "$data_dir/20200820-subject-03.tar.gz"

gdown https://drive.google.com/uc?id=14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA -O "$data_dir/20200903-subject-04.tar.gz"
tar -xzf "$data_dir/20200903-subject-04.tar.gz" -C "$data_dir" && rm "$data_dir/20200903-subject-04.tar.gz"

gdown https://drive.google.com/uc?id=1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS -O "$data_dir/20200908-subject-05.tar.gz"
tar -xzf "$data_dir/20200908-subject-05.tar.gz" -C "$data_dir" && rm "$data_dir/20200908-subject-05.tar.gz"

gdown https://drive.google.com/uc?id=1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ -O "$data_dir/20200918-subject-06.tar.gz"
tar -xzf "$data_dir/20200918-subject-06.tar.gz" -C "$data_dir" && rm "$data_dir/20200918-subject-06.tar.gz"

gdown https://drive.google.com/uc?id=1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0 -O "$data_dir/20200928-subject-07.tar.gz"
tar -xzf "$data_dir/20200928-subject-07.tar.gz" -C "$data_dir" && rm "$data_dir/20200928-subject-07.tar.gz"

gdown https://drive.google.com/uc?id=1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv -O "$data_dir/20201002-subject-08.tar.gz"
tar -xzf "$data_dir/20201002-subject-08.tar.gz" -C "$data_dir" && rm "$data_dir/20201002-subject-08.tar.gz"

gdown https://drive.google.com/uc?id=1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH -O "$data_dir/20201015-subject-09.tar.gz"
tar -xzf "$data_dir/20201015-subject-09.tar.gz" -C "$data_dir" && rm "$data_dir/20201015-subject-09.tar.gz"

gdown https://drive.google.com/uc?id=1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB- -O "$data_dir/20201022-subject-10.tar.gz"
tar -xzf "$data_dir/20201022-subject-10.tar.gz" -C "$data_dir" && rm "$data_dir/20201022-subject-10.tar.gz"

# Download split JSON files from GitHub (gSDF)
echo "Downloading train/test split files to $splits_dir ..."
wget -c https://raw.githubusercontent.com/zerchen/gSDF/05101b5bde6765e9168026cff853b74a1412c125/datasets/dexycb/splits/train_s0_29k.json -O "$splits_dir/train_s0_29k.json"
wget -c https://raw.githubusercontent.com/zerchen/gSDF/05101b5bde6765e9168026cff853b74a1412c125/datasets/dexycb/splits/test_s0_5k.json -O "$splits_dir/test_s0_5k.json"

echo "All files downloaded, extracted, and archives removed."