#!/bin/bash

# Target directory
TARGET_DIR="release_checkpoint"
mkdir -p "$TARGET_DIR"

# Base URL of the Hugging Face dataset repo (using 'resolve/main')
BASE_URL="https://huggingface.co/datasets/dqj5182/haco-checkpoints/resolve/main/NeurIPS2025"

# List of files to download (add more as needed)
FILES=(
  "haco_neurips_hamer_checkpoint.ckpt"
  "haco_neurips_handoccnet_checkpoint.ckpt"
  "haco_neurips_vit_l_checkpoint.ckpt"
  "haco_neurips_vit_b_checkpoint.ckpt"
  "haco_neurips_vit_s_checkpoint.ckpt"
  "haco_neurips_hrnet_w48_checkpoint.ckpt"
  "haco_neurips_hrnet_w32_checkpoint.ckpt"
  "haco_neurips_resnet_152_checkpoint.ckpt"
  "haco_neurips_resnet_50_checkpoint.ckpt"
  "haco_neurips_resnet_101_checkpoint.ckpt"
  "haco_neurips_resnet_34_checkpoint.ckpt"
  "haco_neurips_resnet_18_checkpoint.ckpt"
)

# Download each file directly to the target directory
for file in "${FILES[@]}"; do
  echo "Downloading $file to $TARGET_DIR..."
  wget -c "$BASE_URL/$file" -O "$TARGET_DIR/$file"
done

echo "All files downloaded to $TARGET_DIR"