<div align="center">

# HACO: Learning Dense Hand Contact Estimation <br> from Imbalanced Data

<b><b>[Daniel Sungho Jung](https://dqj5182.github.io/)</b>, <b>[Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)</b> 

<p align="center">
    <img src="asset/logo_cvlab.png" height=55>
</p>

<b>Seoul National University</b>

<a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
<a href='https://haco-release.github.io/'><img src='https://img.shields.io/badge/Project_Page-HACO-green' alt='Project Page'></a>
<a href="https://arxiv.org/pdf/2505.11152"><img src='https://img.shields.io/badge/Paper-HACO-blue' alt='Paper PDF'></a>
<a href="https://arxiv.org/abs/2505.11152"><img src='https://img.shields.io/badge/arXiv-HACO-red' alt='Paper PDF'></a>


<h2>ArXiv 2025</h2>

<img src="./asset/teaser.png" alt="Logo" width="75%">

</div>

_**HACO** is a framework for dense hand contact estimation that addresses class and spatial imbalance issues in training on large-scale datasets. Based on 14 datasets that span hand-object, hand-hand, hand-scene, and hand-body interaction, we build a powerful model that learns dense hand contact in diverse scenarios._
<br/>


## Code


### We are in the process of organizing the codebase, with the demo and inference code scheduled for release by the end of May and the training code in June.


## Acknowledgement
We thank:
* [DECO](https://openaccess.thecvf.com/content/ICCV2023/papers/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.pdf) for human-scene contact estimation.
* [CB Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) for inspiration on VCB Loss.
* [HaMeR](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Pavlakos_Reconstructing_Hands_in_CVPR_2024_supplemental.pdf) for Transformer-based regression architecture.



## Reference
```  
@article{jung2025haco,    
title = {Learning Dense Hand Contact Estimation from Imbalanced Data},
author = {Jung, Daniel Sungho and Lee, Kyoung Mu},
journal = {arXiv preprint arXiv:2505.11152},  
year = {2025}  
}  
```