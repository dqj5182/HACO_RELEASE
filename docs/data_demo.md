## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data  
|   |-- base_data
|   |   |-- demo_data
|   |   |   |-- hand_landmarker.task
|   |   |-- human_models
|   |   |   |-- mano
|   |   |   |   |-- MANO_LEFT.pkl
|   |   |   |   |-- MANO_RIGHT.pkl
|   |   |   |   |-- V_regressor_84.npy
|   |   |   |   |-- V_regressor_336.npy
|   |   |-- pretrained_models
|   |   |   |-- hamer
|   |   |   |-- handoccnet
|   |   |   |-- hrnet
|   |   |   |-- pose2pose
```
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data/blob/main/demo/data/base_data.tar.gz) by running (if not working, try [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/EUmlgxCPqwpEvIhma80VZsoBnHrIPXzbsmJzoQpP-saj-A?e=fSxPEi)):
```
bash scripts/download_demo_base_data.sh
```