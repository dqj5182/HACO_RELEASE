#!/bin/bash
set -e

# HaMeR
mkdir -p data/base_data/pretrained_models/hamer
wget -P data/base_data/pretrained_models/hamer https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar -xzf data/base_data/pretrained_models/hamer/hamer_demo_data.tar.gz -C data/base_data/pretrained_models/hamer
mv data/base_data/pretrained_models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt data/base_data/pretrained_models/hamer/hamer.ckpt
rm -rf data/base_data/pretrained_models/hamer/hamer_demo_data.tar.gz data/base_data/pretrained_models/hamer/_DATA

# HandOccNet
mkdir -p data/base_data/pretrained_models/handoccnet
gdown https://drive.google.com/uc?id=1JXOcWgn6Bx173BhDH99EH6sZ7oOW05Hh -O data/base_data/pretrained_models/handoccnet/snapshot_demo.pth.tar

# HRNet
mkdir -p data/base_data/pretrained_models/hrnet
gdown https://drive.google.com/uc?id=1aTXmxKAJVLsXbvM-TmQ0ZjJxP868G73q -O data/base_data/pretrained_models/hrnet/hrnet_w32-36af842e.pth
gdown https://drive.google.com/uc?id=1qm5-QfHTz5Ia71ByZ1Haq5zJpyEbZRoc -O data/base_data/pretrained_models/hrnet/hrnet_w48-8ef0771d.pth

# Pose2Pose
mkdir -p data/base_data/pretrained_models/pose2pose/hand
gdown https://drive.google.com/uc?id=15wYR8psO2U3ZhFYQEH1-DWc81XkWvK2Y -O data/base_data/pretrained_models/pose2pose/hand/snapshot_12.pth.tar
