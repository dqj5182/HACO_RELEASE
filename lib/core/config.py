import os
import torch
import numpy as np
from easydict import EasyDict as edict

from lib.core.logger import ColorLogger
from lib.utils.log_utils import init_dirs


cfg = edict()


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_name = ['ObMan', 'DexYCB', 'HO3D', 'MOW', 'H2O3D', 'HOI4D', 'H2O', 'ARCTIC', 'InterHand26M', 'HIC', 'PROX', 'RICH', 'Decaf', 'Hi4D']
cfg.DATASET.test_name = 'MOW' # ONLY TEST ONE DATASET AT A TIME
cfg.DATASET.workers = 2
cfg.DATASET.random_seed = 314
cfg.DATASET.ho_bbox_expand_ratio = 1.3
cfg.DATASET.hand_bbox_expand_ratio = 1.3
cfg.DATASET.ho_big_bbox_expand_ratio = 2.0
cfg.DATASET.hand_scene_bbox_expand_ratio = 2.5
cfg.DATASET.obj_bbox_expand_ratio = 1.5


""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.seed = 314
cfg.MODEL.input_img_shape = (256, 256)
cfg.MODEL.img_mean = (0.485, 0.456, 0.406)
cfg.MODEL.img_std = (0.229, 0.224, 0.225)
# MANO
cfg.MODEL.human_model_path = 'data/base_data/human_models'
# Contact
cfg.MODEL.contact_means_path = 'data/base_data/contact_data/dexycb/contact_means_dexycb.npy'
# Backbone
cfg.MODEL.backbone_type = ''
cfg.MODEL.hamer_backbone_pretrained_path = 'data/base_data/pretrained_models/hamer/hamer.ckpt'
cfg.MODEL.hrnet_w32_backbone_config_path = 'data/base_data/pretrained_models/hrnet/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
cfg.MODEL.hrnet_w32_backbone_pretrained_path = 'data/base_data/pretrained_models/hrnet/hrnet_w32-36af842e.pth'
cfg.MODEL.hrnet_w48_backbone_config_path = 'data/base_data/pretrained_models/hrnet/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
cfg.MODEL.hrnet_w48_backbone_pretrained_path = 'data/base_data/pretrained_models/hrnet/hrnet_w48-8ef0771d.pth'
cfg.MODEL.handoccnet_backbone_pretrained_path = 'data/base_data/pretrained_models/handoccnet/snapshot_demo.pth.tar'
# Multi-level joint regressor
cfg.MODEL.V_regressor_336_path = 'data/base_data/human_models/mano/V_regressor_336.npy'
cfg.MODEL.V_regressor_84_path = 'data/base_data/human_models/mano/V_regressor_84.npy'
# Hand Detector
cfg.MODEL.hand_landmarker_path = 'data/base_data/demo_data/hand_landmarker.task'


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch = 24
cfg.TRAIN.epoch = 10
cfg.TRAIN.lr = 1e-5
cfg.TRAIN.weight_decay = 0.0001
cfg.TRAIN.milestones = (5, 10)
cfg.TRAIN.step_size = 10
cfg.TRAIN.gamma = 0.9
cfg.TRAIN.betas = (0.9, 0.95)
cfg.TRAIN.print_freq = 5

cfg.TRAIN.loss_weight = 1.0


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch = 1


""" CAMERA """
cfg.CAMERA = edict()

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True
logger = None


def update_config(backbone_type='', exp_dir='', ckpt_path=''):
    if backbone_type == '':
        backbone_type = 'hamer'
    cfg.MODEL.backbone_type = backbone_type

    global logger
    log_dir = os.path.join(exp_dir, 'log')
    try:
        init_dirs([log_dir])
        logger = ColorLogger(log_dir)
        logger.info("Logger initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        logger = None