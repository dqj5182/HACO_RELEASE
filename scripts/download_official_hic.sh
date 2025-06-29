#!/bin/bash

# Set target directory
save_dir="data/HIC/data"
mkdir -p "$save_dir"
cd "$save_dir" || exit

# Download and unzip Hand_Hand sequences
for seq_idx in 01 02 03 04 05 06 07 08 09 10 11; do
    echo "Downloading Hand_Hand sequence $seq_idx..."
    wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Hand_Hand___All_Files/"$seq_idx".zip
    unzip "$seq_idx".zip
    rm "$seq_idx".zip
done

# Download and unzip Hand_Object sequences
for seq_idx in 15 16 17 18 19 20 21; do
    echo "Downloading Hand_Object sequence $seq_idx..."
    wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/Hand_Object___All_Files/"$seq_idx".zip
    unzip "$seq_idx".zip
    rm "$seq_idx".zip
done

# Download the MANO-compatible parameter file
echo "Downloading MANO-compatible parameter file..."
wget http://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/Dataset/MANO_compatible/IJCV16___Results_MANO___parms_for___joints21.zip
unzip IJCV16___Results_MANO___parms_for___joints21.zip
rm IJCV16___Results_MANO___parms_for___joints21.zip

echo "All files downloaded, unzipped, and cleaned up in $save_dir."

# Download HIC.json
gdown https://drive.google.com/uc?id=1oqquzJ7DY728M8zQoCYvvuZEBh8L8zkQ -O data/HIC/data/HIC.json