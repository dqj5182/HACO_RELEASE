# Create target directory for DexYCB
mkdir -p data/DexYCB/data
gdown https://drive.google.com/uc?id=1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu -O data/DexYCB/data/YCB_Video_Models.zip

# Unzip in DexYCB
unzip data/DexYCB/data/YCB_Video_Models.zip -d data/DexYCB/data
rm data/DexYCB/data/YCB_Video_Models.zip

# Copy to H2O3D
mkdir -p data/H2O3D/YCB_object_models
cp -r data/DexYCB/data/models data/H2O3D/YCB_object_models/models