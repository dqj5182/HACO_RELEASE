## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data
|   |-- base_data
|   |-- ARCTIC
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- Decaf
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- DexYCB
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- toolkit
|   |   |-- dataset.py
|   |-- H2O
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- H2O3D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- YCB_object_models
|   |   |-- dataset.py
|   |-- Hi4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- HIC
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- HO3D
|   |   |-- annotations
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- HOI4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- InterHand26M
|   |   |-- annotations
|   |   |-- images
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- MOW
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- ObMan
|   |   |-- annotations
|   |   |-- data
|   |   |-- object_models
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- PROX
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
|   |-- RICH
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- splits
|   |   |-- dataset.py
```
#### base_data
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/train/data/base_data.tar.gz) by running:
```
bash scripts/download_train_base_data.sh
```
#### preprocessed_data
* Download `preprocessed_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data) by running:
```
bash scripts/download_train_preprocessed_data.sh
```
#### ARCTIC dataset
```
${ROOT} 
|-- data
|   |-- ARCTIC
|   |   |-- data
|   |   |   |-- cropped_images
|   |   |   |-- images
|   |   |   |-- meta
|   |   |   |-- splits
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download `data` from [official GitHub repository](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md).
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### Decaf dataset
```
${ROOT} 
|-- data
|   |-- Decaf
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- cameras
|   |   |   |   |-- ...
|   |   |   |   |-- videos
|   |   |   |-- test
|   |   |   |   |-- cameras
|   |   |   |   |-- ...
|   |   |   |   |-- videos
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- contact_data
|   |   |   |   |-- db_bb.npz
|   |   |   |   |-- db_contact.npz
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_decaf.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### DexYCB dataset
```
${ROOT} 
|-- data
|   |-- DexYCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ...
|   |   |   |-- 20201022-subject-10
|   |   |   |-- models
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train_s0_29k.json
|   |   |   |-- test_s0_5k.json
|   |   |-- toolkit
|   |   |-- dataset.py
```
* Download `data` except `data/models` and `splits` by running:
```
bash scripts/download_official_dexycb.sh
```
* Download `data/models` by running:
```
bash scripts/download_official_ycb_models.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### H2O dataset
```
${ROOT} 
|-- data
|   |-- H2O
|   |   |-- data
|   |   |   |-- label_split
|   |   |   |-- object
|   |   |   |-- subject1_ego
|   |   |   |-- subject2_ego
|   |   |   |-- subject3_ego
|   |   |   |-- subject4_ego
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download `label_split.zip`, `object.zip`, `subject1_ego_v1_1.tar.gz`, `subject2_ego_v1_1.tar.gz`, `subject3_ego_v1_1.tar.gz`, `subject4_ego_v1_1.tar.gz` from [official website](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) and place them in `data`.
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### H2O3D dataset
```
${ROOT} 
|-- data
|   |-- H2O3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |-- evaluation
|   |   |   |-- train.txt
|   |   |   |-- evaluation.txt
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- contact_data
|   |   |-- YCB_object_models
|   |   |-- dataset.py
```
* Download and extract `H2O-3D v1` from [official website](https://github.com/shreyashampali/ho3d) and place it in `data`.
* If you have not already run this script for DexYCB, download `YCB_object_models` by running:
```
bash scripts/download_official_ycb_models.sh
```
* `preprocessed_data` is already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
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
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |   |-- test.json
|   |   |   |-- train_pid.json
|   |   |   |-- test_pid.json
|   |   |-- dataset.py
```
* Download `data` by running after download request from [official website](https://yifeiyin04.github.io/Hi4D):
```
bash scripts/download_official_hi4d.sh
bash scripts/extract_official_hi4d.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
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
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_hic.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### HO3D dataset
```
${ROOT} 
|-- data
|   |-- HO3D
|   |   |-- annotations
|   |   |   |-- HO3D_train_data.json
|   |   |   |-- HO3D_evaluation_data.json
|   |   |-- data
|   |   |   |-- train
|   |   |   |-- evaluation
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- dataset.py
```
* Download and extract `HO-3D v2` from [official website](https://github.com/shreyashampali/ho3d) and place it in `data`.
* Download `annotation files` from [HandOccNet](https://github.com/namepllet/HandOccNet) and place it in `annotations`.
* `preprocessed_data` is already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### HOI4D dataset
```
${ROOT} 
|-- data
|   |-- HOI4D
|   |   |-- data
|   |   |   |-- datalist
|   |   |   |-- HOI4D_annotations
|   |   |   |-- HOI4D_CAD_models
|   |   |   |-- HOI4D_cameras
|   |   |   |-- HOI4D_color
|   |   |   |-- HOI4D_Handpose
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download and extract `HOI4D_annotations.zip`, `HOI4D_CAD_Model_for_release.zip`, `camera_params.zip`, `HOI4D_release.zip`, `HOI4D_Hand_pose.zip` from [official website](https://hoi4d.github.io/) after download request and place them in `data`.
* Download `datalist` by running:
```
bash scripts/download_official_hoi4d.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### InterHand2.6M dataset
```
${ROOT} 
|-- data
|   |-- InterHand26M
|   |   |-- annotations
|   |   |   |-- train
|   |   |   |   |-- InterHand2.6M_train_camera.json
|   |   |   |   |-- ...
|   |   |   |   |-- InterHand2.6M_train_MANO_NeuralAnnot.json
|   |   |   |   |-- InterHand2.6M_train_data.pkl
|   |   |   |-- test
|   |   |   |   |-- InterHand2.6M_test_camera.json
|   |   |   |   |-- ...
|   |   |   |   |-- InterHand2.6M_test_MANO_NeuralAnnot.json
|   |   |   |-- val
|   |   |   |   |-- InterHand2.6M_val_camera.json
|   |   |   |   |-- ...
|   |   |   |   |-- InterHand2.6M_val_MANO_NeuralAnnot.json
|   |   |   |-- skeleton.txt
|   |   |   |-- subject.txt
|   |   |-- images
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- val
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download `annotations`, `images` by running:
```
bash scripts/download_official_interhand26m.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### MOW dataset
```
${ROOT} 
|-- data
|   |-- MOW
|   |   |-- data
|   |   |   |-- images
|   |   |   |-- masks
|   |   |   |-- models
|   |   |   |-- poses.json
|   |   |   |-- watertight_models
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `images`, `models`, `poses.json` by running:
```
bash scripts/download_official_mow.sh
```
* `masks`, `watertight_models`, `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### ObMan dataset
```
${ROOT} 
|-- data
|   |-- ObMan
|   |   |-- annotations
|   |   |   |-- obman_train.json
|   |   |   |-- obman_test.json
|   |   |-- data
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- val
|   |   |-- object_models
|   |   |   |-- watertight_meshes
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train_87k.json
|   |   |   |-- test_6k.json
|   |   |-- dataset.py
```
* Download and extract `obman.zip` from [official website](https://www.di.ens.fr/willow/research/obman/data/) after download request and place it in `data`.
* Download `annotations` and `splits` by running:
```
bash scripts/download_gsdf_obman.sh
```
* `object_models`, `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).
#### PROX dataset
```
${ROOT} 
|-- data
|   |-- PROX
|   |   |-- data
|   |   |   |-- quantitative
|   |   |   |   |-- body_segments
|   |   |   |   |-- ...
|   |   |   |   |-- sdf
|   |   |   |   |-- vicon2scene.json
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_prox.sh
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
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |   |-- test
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- splits
|   |   |   |-- train.json
|   |   |   |-- test.json
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_rich.sh
```
* `preprocessed_data`, `splits` are already downloaded from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data).