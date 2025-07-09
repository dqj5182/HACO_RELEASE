## Data
You need to follow directory structure of the `data` as below.
```
${ROOT} 
|-- data
|   |-- base_data
|   |-- ARCTIC
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- Decaf
|   |   |-- data
|   |   |-- preprocessed_data
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
|   |   |-- dataset.py
|   |-- H2O3D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- YCB_object_models
|   |   |-- dataset.py
|   |-- Hi4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- HIC
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- HO3D
|   |   |-- annotations
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- HOI4D
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
|   |-- InterHand26M
|   |   |-- annotations
|   |   |-- images
|   |   |-- preprocessed_data
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
|   |   |-- dataset.py
|   |-- RICH
|   |   |-- data
|   |   |-- preprocessed_data
|   |   |-- dataset.py
```
#### base_data
* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/train/data/base_data.tar.gz) by running (if not working, try [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/Ea4_PNhv7ZJAkZDCbbn56ywBFLUqE_eivVxercSw3dZO1w?e=N9NSQj)):
```
bash scripts/download_train_base_data.sh
```

#### preprocessed_data
* Download `preprocessed_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data) by running:
```
bash scripts/download_preprocessed_data.sh
```
* If HuggingFace does not work, download `preprocessed_data` from OneDrive ([ARCTIC](https://1drv.ms/u/c/bf7e2a9a100f1dba/ERDI4bt3TxVAuzhZ_qmcBggBmm1lxl-8p5agUdHzSzjvcQ?e=gUcg7i), [Decaf](https://1drv.ms/u/c/bf7e2a9a100f1dba/EWzRc5lBMI5OhSZBc8IIY34BEY-ESzVoZx_I4ENH4e4NdA?e=Mk7Tnu), [DexYCB](https://1drv.ms/u/c/bf7e2a9a100f1dba/Ea_9KQy-IEVLgGyPLOKjuhMBidzy8GoYXw5NVdgwl-7PlQ?e=C9zWNp), [H2O](https://1drv.ms/u/c/bf7e2a9a100f1dba/Eaf_Bjf_y2JAoSSLTQqBKJQBNOYISOOwbmkE7Vz5JwTLGw?e=FRWEIS), [H2O3D](https://1drv.ms/u/c/bf7e2a9a100f1dba/EQm4V907IIZFhLCvjwXqZGoBZ1q3RTT-y84dgfmJNrzfkg?e=G1cmVH), [Hi4D](https://1drv.ms/u/c/bf7e2a9a100f1dba/ES929M9KL99PmSxXvbHd1jUBlEBQY03DRwXT6iAOrg2Xtw?e=2PlMi1), [HIC](https://1drv.ms/u/c/bf7e2a9a100f1dba/ESxGJUbApOJItOT8OdrjDFgBe7-ADUjCOkJfUz0JoUhtGg?e=BKd4j2), [HO3D](https://1drv.ms/u/c/bf7e2a9a100f1dba/ETxW6n3z4PJFiXcM_NAWSqQBdZo_f5HpVoVLO0OYPtBmIg?e=Pl1nNy), [HOI4D](https://1drv.ms/u/c/bf7e2a9a100f1dba/EcHQG8MXSUpBpfbZvswEtIMBLnPOkSJeqaNC6nGqmfV3sw?e=ZzV6dF), [InterHand2.6M](https://1drv.ms/u/c/bf7e2a9a100f1dba/EUPp2IdGdkBCqOe7K9oGuVoBLnETEYIhfQpI-wz2fNPB4Q?e=TUJtPg), [MOW](https://1drv.ms/u/c/bf7e2a9a100f1dba/ESkqLhHk9gFHo4HH2uA9akABgYuS2wLgWfr4YJMRmagezQ?e=jfThFd), [ObMan](https://1drv.ms/u/c/bf7e2a9a100f1dba/EQA_uJoKMCZMlMaHIPkeEOYBCGONtYsDNAJ2FDtD3iGEzA?e=vYkkG2), [PROX](https://1drv.ms/u/c/bf7e2a9a100f1dba/ER9bCCmZVC5AvqWNljeXp3wBMiWRt_YOhwkuQVbi8tKCoQ?e=PC7uMs), [RICH](https://1drv.ms/u/c/bf7e2a9a100f1dba/EYD4nVEt76NPqgXY1xmU9WcBgwDWXoIZLaxTD6f8UvS8nQ?e=NkP3P4)).
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
|   |   |-- dataset.py
```
* Download `data` from [official GitHub repository](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md).
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
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_decaf.sh
```
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
* Download `data/models` by running (if not working, please visit [here](https://rse-lab.cs.washington.edu/projects/posecnn/)):
```
bash scripts/download_official_ycb_models.sh
```
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
|   |   |-- dataset.py
```
* Download `label_split.zip`, `object.zip`, `subject1_ego_v1_1.tar.gz`, `subject2_ego_v1_1.tar.gz`, `subject3_ego_v1_1.tar.gz`, `subject4_ego_v1_1.tar.gz` from [official website](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) and place it in `data`.
#### H2O3D dataset
```
${ROOT} 
|-- data
|   |-- H2O3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |-- evaluation
|   |   |-- preprocessed_data
|   |   |   |-- train
|   |   |   |   |-- annot_data
|   |   |   |   |-- contact_data
|   |   |-- YCB_object_models
|   |   |-- dataset.py
```
* Download and extract `h2o3d_v1.zip` from [official website](https://github.com/shreyashampali/ho3d) and place it in `data`.
* If you have not already run this script for DexYCB, download `YCB_object_models` by running:
```
bash scripts/download_official_ycb_models.sh
```
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
|   |   |-- dataset.py
```
* Download `data` by running after download request (if not working, please directly download from [official website](https://yifeiyin04.github.io/Hi4D)):
```
bash scripts/download_official_hi4d.sh
bash scripts/extract_official_hi4d.sh
```
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
|   |   |-- dataset.py
```
* Download `data` by running (if not working, please refer to [InterWild](https://github.com/facebookresearch/InterWild)):
```
bash scripts/download_official_hic.sh
```
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
* Download and extract `HO3D_v2.zip` from [official website](https://github.com/shreyashampali/ho3d) and place it in `data`.
* Download `annotation files` from [HandOccNet](https://github.com/namepllet/HandOccNet) and place it in `annotations`.
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
|   |   |-- dataset.py
```
* Download and extract `HOI4D_annotations.zip`, `HOI4D_CAD_Model_for_release.zip`, `camera_params.zip`, `HOI4D_release.zip`, `HOI4D_Hand_pose.zip` from [official website](https://hoi4d.github.io/) after download request and place it in `data`.
* Download `datalist` by running:
```
bash scripts/download_official_hoi4d.sh
```
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
|   |   |-- dataset.py
```
* Download `annotations`, `images` by running:
```
bash scripts/download_official_interhand26m.sh
```
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
|   |   |-- dataset.py
```
* Download `images`, `models`, `poses.json` by running:
```
bash scripts/download_official_mow.sh
```
* `masks`, `watertight_models`, `splits` are already downloaded as part of `preprocessed_data`. But, you can also download them directly from OneDrive ([masks](https://1drv.ms/u/c/bf7e2a9a100f1dba/Ef2YhwccS4tPt1WrAAP4-iMBjcaSUgawDMnf_HDpqoTeNw?e=4A0OF8), [watertight_models](https://1drv.ms/u/c/bf7e2a9a100f1dba/EW5YXeXtk3NBnX9PcvJtGIABj_9c1FW2RdrcppDgRzqHhg?e=A5l0iI), [splits](https://1drv.ms/u/c/bf7e2a9a100f1dba/EW60jCPiuNNOjkmCUdqlBbEBact_Ums22dwBoQoFMkUV6w?e=A0TcjM)) if needed.
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
* Download `annotations` and `splits` by running (if not working, please visit [here](https://github.com/zerchen/gSDF)):
```
bash scripts/download_gsdf_obman.sh
```
* Donwload `object_models` from [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/EfOzLZbNp0VHvic5LyR4xMEBDrvq0c9jwoexfIlRx-NAsA?e=qgdWNx).
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
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_prox.sh
```
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
|   |   |-- dataset.py
```
* Download `data` by running:
```
bash scripts/download_official_rich.sh
```