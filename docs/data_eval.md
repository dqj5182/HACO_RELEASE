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
|   |-- MOW
|   |   |-- data
|   |   |   |-- images
|   |   |   |-- masks
|   |   |   |-- models
|   |   |   |-- poses.json
|   |   |   |-- watertight_models
|   |   |-- preprocessed_data
|   |   |   |-- test
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |-- dataset.py
```
* Download `base_data` from [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/EUmlgxCPqwpEvIhma80VZsoBnHrIPXzbsmJzoQpP-saj-A?e=fSxPEi).
* Download [MOW](https://zhec.github.io/rhoi/) data from official GitHub ([images](https://github.com/ZheC/MOW), [models](https://github.com/ZheC/MOW), [poses.json](https://github.com/ZheC/MOW)) and OneDrive ([masks](https://1drv.ms/u/c/bf7e2a9a100f1dba/Ef2YhwccS4tPt1WrAAP4-iMBjcaSUgawDMnf_HDpqoTeNw?e=eQYJ4e), [watertight_models](https://1drv.ms/u/c/bf7e2a9a100f1dba/EW5YXeXtk3NBnX9PcvJtGIABj_9c1FW2RdrcppDgRzqHhg?e=ryUqCf), [preprocessed_data](https://1drv.ms/u/c/bf7e2a9a100f1dba/ESkqLhHk9gFHo4HH2uA9akABgYuS2wLgWfr4YJMRmagezQ?e=DoGFso), [splits](https://1drv.ms/u/c/bf7e2a9a100f1dba/EW60jCPiuNNOjkmCUdqlBbEBact_Ums22dwBoQoFMkUV6w?e=2lxpJd)). For GitHub data, you can directly download them by running:
```
bash scripts/download_official_mow.sh
```
* Download initial checkpoints by running:
```
bash scripts/download_initial_checkpoints.sh
```