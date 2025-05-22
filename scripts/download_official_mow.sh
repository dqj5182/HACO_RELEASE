#!/bin/bash
set -e

# Download and extract MOW dataset
mkdir -p data/MOW
wget --show-progress -P data/MOW https://zhec.github.io/rhoi/mow.zip
unzip -q data/MOW/mow.zip -d data/MOW
mkdir -p data/MOW/data
mv data/MOW/mow/images data/MOW/data/
mv data/MOW/mow/models data/MOW/data/
rm -rf data/MOW/__MACOSX data/MOW/mow data/MOW/mow.zip

# Download poses.json
wget --show-progress -O data/MOW/data/poses.json https://raw.githubusercontent.com/ZheC/MOW/b2acbb4fac40acc4c286833da895fc9f23e58bb6/poses.json