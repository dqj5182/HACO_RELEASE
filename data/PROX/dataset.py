import os
import cv2
import json
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact



# Main dataloader code for PROX dataset
class PROX(Dataset):
    def __init__(self, transform, data_split):
        super(PROX, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'prox'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'PROX')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        # SMPL-X to MANO mapping
        smplx_mano_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smplx_to_mano.pkl')

        with open(smplx_mano_mapping_path, 'rb') as f:
            self.smplx_to_mano_mapping = pickle.load(f)
            self.smplx_to_mano_mapping_r = self.smplx_to_mano_mapping["right_hand"]

        # Camera
        with open(os.path.join(self.data_dir, 'quantitative/calibration/Color.json'), 'r') as f:
            calibration = json.load(f)
            self.cam_param = {'focal': calibration['f'], 'princpt': calibration['c']}

        # Organize db id based on split
        db_split_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            self.db = json.load(f)


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        seq_name = aid.split('/')[-3]
        img_name = aid.split('/')[-1]
        sample_id = f'{seq_name}-{img_name}'

        orig_img_path = os.path.join(os.path.join(self.data_dir, 'quantitative', 'recordings', seq_name, 'Color', f'{img_name}.jpg'))
        
        orig_img = load_img(orig_img_path)
        orig_img = cv2.flip(orig_img, 1) # only for PROX dataset
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        
        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_hand_r = annot_data['bbox_ho']

        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################
        

        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand_r, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
        crop_img = img.copy()  

        # Transform for 3D HMR
        if ('resnet' in cfg.MODEL.backbone_type or 'hrnet' in cfg.MODEL.backbone_type or 'handoccnet' in cfg.MODEL.backbone_type):
            img = self.transform(img.astype(np.float32)/255.0)
        elif (cfg.MODEL.backbone_type in ['hamer']) or ('vit' in cfg.MODEL.backbone_type):
            normalize_img = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
            img = img.transpose(2, 0, 1) / 255.0
            img = normalize_img(torch.from_numpy(img)).float()
        else:
            raise NotImplementedError
        ############################### PROCESS CROP AND AUGMENTATION ################################


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)
        

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)