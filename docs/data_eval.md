## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data  
|   |-- base_data
|   |-- MOW
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- HIC
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- RICH
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- Hi4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
```
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/train/data/base_data.tar.gz) by running:
```
bash scripts/download_train_base_data.sh
```
#### preprocessed_data
* Download `preprocessed_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data) by running:
```
bash scripts/download_train_preprocessed_data.sh
```
#### MOW dataset
```
${ROOT} 
|-- data
|   |-- MOW
|   |   |-- data
|   |   |   |-- images
|   |   |   |-- masks
|   |   |   |-- poses.json
|   |   |-- preprocessed_data
|   |   |   |-- test
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `images`, `poses.json` by running:
```
bash scripts/download_official_mow.sh
```
* `masks`, `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### HIC dataset
```
${ROOT} 
|-- data
|   |-- HIC
|   |   |-- data
|   |   |   |-- 01
|   |   |   |-- ...
|   |   |   |-- 12
|   |   |   |-- IJCV16___Results_MANO___parms_for___joints21
|   |   |   |-- HIC.json
|   |   |-- preprocessed_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_hic.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### RICH dataset
```
${ROOT} 
|-- data
|   |-- RICH
|   |   |-- data
|   |   |   |-- hsc
|   |   |   |-- images_jpg_subset
|   |   |   |-- multicam2world
|   |   |   |-- scan_calibration
|   |   |-- preprocessed_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_rich.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### Hi4D dataset
```
${ROOT} 
|-- data
|   |-- Hi4D
|   |   |-- data
|   |   |   |-- pair00
|   |   |   |-- ...
|   |   |   |-- pair37
|   |   |-- preprocessed_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- test.json
|   |   |   |-- test_pid.json
|   |   |-- dataset.py
```
* Download `data` by running after download request from [official website](https://yifeiyin04.github.io/Hi4D):
```
bash scripts/download_official_hi4d.sh
bash scripts/extract_official_hi4d.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).