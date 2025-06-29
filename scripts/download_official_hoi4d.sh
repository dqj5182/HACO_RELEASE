#!/bin/bash

# Target directory
target_dir="data/HOI4D/data/datalist"
mkdir -p "$target_dir"

# Download files
wget https://raw.githubusercontent.com/leolyliu/HOI4D-Instructions/main/prepare_4Dseg/datalists/train_all.txt -O "$target_dir/train_all.txt"
wget https://raw.githubusercontent.com/leolyliu/HOI4D-Instructions/main/prepare_4Dseg/datalists/test_all.txt -O "$target_dir/test_all.txt"

echo "Download complete: Files saved to $target_dir"