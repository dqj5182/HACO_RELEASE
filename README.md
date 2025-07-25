<div align="center">

# HACO: Learning Dense Hand Contact Estimation <br> from Imbalanced Data

<b>[Daniel Sungho Jung](https://dqj5182.github.io/)</b>, <b>[Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)</b> 

<img src="asset/logo_cvlab.png" width="30%">

<b>Seoul National University</b>

<a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
<a href='https://haco-release.github.io/'><img src='https://img.shields.io/badge/Project_Page-HACO-green' alt='Project Page'></a>
<a href="https://arxiv.org/pdf/2505.11152"><img src='https://img.shields.io/badge/Paper-HACO-blue' alt='Paper PDF'></a>
<a href="https://arxiv.org/abs/2505.11152"><img src='https://img.shields.io/badge/arXiv-HACO-red' alt='Paper PDF'></a>

<a href="https://huggingface.co/datasets/dqj5182/haco-checkpoints"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/dqj5182/HACO"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> <a href="https://huggingface.co/papers/2505.11152"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-Huggingface-orange"></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2505-11152/dense-contact-estimation-on-mow)](https://paperswithcode.com/sota/dense-contact-estimation-on-mow?p=2505-11152)



<h2>ArXiv 2025</h2>

<img src="./asset/teaser.png" alt="Logo" width="75%">

</div>

_**HACO** is a framework for **dense hand contact estimation** that addresses **class and spatial imbalance issues** in training on large-scale datasets. Based on **14 datasets** that span **hand-object**, **hand-hand**, **hand-scene**, and **hand-body interaction**, we build a powerful model that learns dense hand contact in diverse scenarios._



## Installation
* We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. Install PyTorch >=1.11.0 and Python >= 3.8.0. Our latest HACO model is tested on Python 3.8.20, PyTorch 1.11.0, CUDA 11.3.
* Setup the environment.
``` 
# Initialize conda environment
conda create -n haco python=3.8 -y
conda activate haco

# Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install all remaining packages
pip install -r requirements.txt
```


## Data
You need to follow our directory structure of the `data`.
* For quick demo: See [`docs/data_demo.md`](docs/data_demo.md).
* For evaluation: See [`docs/data_eval.md`](docs/data_eval.md).
* For training: See [`docs/data_train.md`](docs/data_train.md).

Then, download the official checkpoints and place them in the `release_checkpoint` from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-checkpoints/tree/main) by running (if not working, try [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/Ef18aU5ItbFDgW1sSv3P0l0BGTzN6PlsCnm0q5ecpTWIfQ?e=Y40qsN)):
```
bash scripts/download_haco_checkpoints.sh
```


## Quick demo (Image)
To run HACO on demo images using the [WiLoR](https://github.com/rolpotamias/WiLoR) or [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) hand detector, please run:
```
python demo.py --backbone {BACKBONE_TYPE} --detector {DETECTOR_TYPE} --checkpoint {CKPT_PATH} --input_path {INPUT_PATH}
```

For example,
```
# ViT-H (Default, HaMeR initialized) backbone
python demo.py --backbone hamer --detector wilor --checkpoint release_checkpoint/haco_final_hamer_checkpoint.ckpt --input_path asset/example_images

# ViT-B (ImageNet initialized) backbone
python demo.py --backbone vit-b-16 --detector wilor --checkpoint release_checkpoint/haco_final_vit_b_checkpoint.ckpt --input_path asset/example_images
```

> Note: The demo includes post-processing to reduce noise in small or sparse contact areas.

## Quick demo (Video)
Before the video demo, please download example videos from [HuggingFace](https://huggingface.co/datasets/dqj5182/haco-data/blob/main/demo/asset/example_videos.zip) and save at `asset/example_videos` by running (if not working, try [OneDrive](https://1drv.ms/u/c/bf7e2a9a100f1dba/ERsk_D-EubxBi1Usu2bW2hABwy9nxzRxAHutXDxmv85TLw?e=rIjOI7)):
```
bash scripts/download_demo_example_videos.sh
```

To run HACO on demo videos using the [WiLoR](https://github.com/rolpotamias/WiLoR) or [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) hand detector, please run:
```
python demo_video.py --backbone {BACKBONE_TYPE} --checkpoint {CKPT_PATH} --input_path {INPUT_PATH}
```

For example,
```
# ViT-H (Default, HaMeR initialized) backbone
python demo_video.py --backbone hamer --checkpoint release_checkpoint/haco_final_hamer_checkpoint.ckpt --input_path asset/example_videos

# ViT-B (ImageNet initialized) backbone
python demo_video.py --backbone vit-b-16 --checkpoint release_checkpoint/haco_final_vit_b_checkpoint.ckpt --input_path asset/example_videos
```

> Note: The demo includes post-processing for both spatial smoothing of small contact areas and temporal smoothing across frames to ensure stable contact predictions and hand detections.




## Running HACO
### Train
To train HACO, please run:
```
python train.py --backbone {BACKBONE_TYPE}
```

For example,
```
# ViT-H (Default, HaMeR initialized) backbone
python train.py --backbone hamer

# ViT-B (ImageNet initialized) backbone
python train.py --backbone vit-b-16
```

### Test
To evaluate HACO on [MOW](https://github.com/ZheC/MOW) dataset, please run:
```
python test.py --backbone {BACKBONE_TYPE} --checkpoint {CKPT_PATH}
```

For example,
```
# ViT-H (Default, HaMeR initialized) backbone
python test.py --backbone hamer --checkpoint release_checkpoint/haco_final_hamer_checkpoint.ckpt

# ViT-L (ImageNet initialized) backbone
python test.py --backbone vit-l-16 --checkpoint release_checkpoint/haco_final_vit_l_checkpoint.ckpt

# ViT-B (ImageNet initialized) backbone
python test.py --backbone vit-b-16 --checkpoint release_checkpoint/haco_final_vit_b_checkpoint.ckpt

# ViT-S (ImageNet initialized) backbone
python test.py --backbone vit-s-16 --checkpoint release_checkpoint/haco_final_vit_s_checkpoint.ckpt

# FPN (HandOccNet initialized) backbone
python test.py --backbone handoccnet --checkpoint release_checkpoint/haco_final_handoccnet_checkpoint.ckpt

# HRNet-W48 (ImageNet initialized) backbone
python test.py --backbone hrnet-w48 --checkpoint release_checkpoint/haco_final_hrnet_w48_checkpoint.ckpt

# HRNet-W32 (ImageNet initialized) backbone
python test.py --backbone hrnet-w32 --checkpoint release_checkpoint/haco_final_hrnet_w32_checkpoint.ckpt

# ResNet-152 (ImageNet initialized) backbone
python test.py --backbone resnet-152 --checkpoint release_checkpoint/haco_final_resnet_152_checkpoint.ckpt

# ResNet-101 (ImageNet initialized) backbone
python test.py --backbone resnet-101 --checkpoint release_checkpoint/haco_final_resnet_101_checkpoint.ckpt

# ResNet-50 (ImageNet initialized) backbone
python test.py --backbone resnet-50 --checkpoint release_checkpoint/haco_final_resnet_50_checkpoint.ckpt

# ResNet-34 (ImageNet initialized) backbone
python test.py --backbone resnet-34 --checkpoint release_checkpoint/haco_final_resnet_34_checkpoint.ckpt

# ResNet-18 (ImageNet initialized) backbone
python test.py --backbone resnet-18 --checkpoint release_checkpoint/haco_final_resnet_18_checkpoint.ckpt
```



## Technical Q&A
* `ImportError: cannot import name 'bool' from 'numpy'`: Please just comment out the line `from numpy import bool, int, float, complex, object, unicode, str, nan, inf`.
* `np.int was a deprecated alias for the builtin int. To avoid this error in existing code, use int by itself. Doing this will not modify any behavior and is safe. When replacing np.int, you may wish to use e.g. np.int64 or np.int32 to specify the precision. If you wish to review your current use, check the release note link for additional information`: Please refer to [here](https://github.com/scikit-optimize/scikit-optimize/issues/1171).


## Acknowledgement
We thank:
* [DECO](https://openaccess.thecvf.com/content/ICCV2023/papers/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.pdf) for human-scene contact estimation.
* [CB Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) for inspiration on VCB Loss.
* [HaMeR](https://openaccess.thecvf.com/content/CVPR2024/papers/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.pdf) for Transformer-based regression architecture.



## Reference
```  
@article{jung2025haco,    
title = {Learning Dense Hand Contact Estimation from Imbalanced Data},
author = {Jung, Daniel Sungho and Lee, Kyoung Mu},
journal = {arXiv preprint arXiv:2505.11152},  
year = {2025}  
}  
```