#!/bin/bash

TARGET_DIR="asset"
FILE_URL="https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/demo/asset/example_videos.zip"
ARCHIVE_NAME="$TARGET_DIR/example_videos.zip"

mkdir -p "$TARGET_DIR"

echo "Downloading example_videos.zip..."
wget -c "$FILE_URL" -O "$ARCHIVE_NAME"

echo "Unzipping into $TARGET_DIR..."
unzip -o "$ARCHIVE_NAME" -d "$TARGET_DIR"

# Remove zip after extraction
rm "$ARCHIVE_NAME"

echo "Done. Extracted to $TARGET_DIR"