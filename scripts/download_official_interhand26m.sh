#!/bin/bash

########### Download InterHand2.6M images ############
# Set target directory for images
save_images_dir="data/InterHand26M/images"
mkdir -p "$save_images_dir"
cd "$save_images_dir" || exit 1

base_url="https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/InterHand2.6M/InterHand2.6M.images.5.fps.v1.0/"

# Download all part files
for part1 in a b; do
  for part2 in {a..z}; do
    if [[ "$part1" == "b" && "$part2" == "s" ]]; then
      break
    fi
    filename="InterHand2.6M.images.5.fps.v1.0.tar.part${part1}${part2}"
    echo "Downloading $filename ..."
    wget -c "${base_url}${filename}"
  done
done

# Download CHECKSUM and helper scripts
wget -c "${base_url}InterHand2.6M.images.5.fps.v1.0.tar.CHECKSUM"
wget -c "${base_url}unzip.sh"
wget -c "${base_url}verify_download.py"

# Run verification
echo "Running verify_download.py..."
python3 verify_download.py || { echo "Checksum verification failed"; exit 1; }

# Run unzip
echo "Running unzip.sh..."
bash unzip.sh || { echo "Unzip failed"; exit 1; }

cd "../../.." || exit 1

# Move extracted images into the target directory root
extracted_subdir="$save_images_dir/InterHand2.6M_5fps_batch1/images"
if [ -d "$extracted_subdir" ]; then
  echo "Moving images to $save_images_dir ..."
  mv "$extracted_subdir"/* "$save_images_dir"
  rm -r "$save_images_dir/InterHand2.6M_5fps_batch1"
else
  echo "Expected directory $extracted_subdir not found."
  exit 1
fi

echo "InterHand2.6M image data downloaded and extracted to $save_images_dir"
########### End of image download ############



############ Download InterHand2.6M annotations ############
save_ann_dir="data/InterHand26M/annotations"
mkdir -p "$save_ann_dir"

echo "Downloading annotations to $save_ann_dir ..."
gdown --folder https://drive.google.com/drive/folders/12RNG9slv9i_TsXSoZ6pQAq-Fa98eGLoy -O "$save_ann_dir"

# Move contents up if nested under 'annotations'
if [ -d "$save_ann_dir/annotations" ]; then
  mv "$save_ann_dir/annotations/"* "$save_ann_dir"
  rmdir "$save_ann_dir/annotations"
fi

echo "InterHand2.6M annotations downloaded to $save_ann_dir"
############ End of annotations download ############