import os
import cv2
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact, mask2bbox



# Main dataloader code for MOW dataset
class MOW(Dataset):
    def __init__(self, transform, data_split):
        super(MOW, self).__init__()
        self.__dict__.update(locals())

        self.transfrom = transform
        dataset_name = 'mow'

        self.data_split = data_split
        self.root_path = root_path = 'data/MOW'

        self.data_dir = os.path.join(self.root_path, 'data')
        self.split_dir = os.path.join(self.root_path, 'splits')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        with open(os.path.join(self.data_dir, 'poses.json')) as f:
            self.db = {a['image_id']: a for a in json.load(f)}

        # Organize db id based on split
        db_split_path = os.path.join(self.split_dir, f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            self.split = json.load(f)
            self.length = len(self.split)


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        sample_id = self.split[index]
        ann = self.db[sample_id]
        image_id = ann['image_id']

        img_path = os.path.join(self.data_dir, 'images', f'{image_id}.jpg')
        orig_img = load_img(img_path)

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        mask_ho_path = os.path.join(self.data_dir, 'masks/both', f'{image_id}.jpg')
        mask_ho = (cv2.imread(mask_ho_path) > 128)[:, :, 0]
        bbox_ho = mask2bbox(mask_ho, expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)

        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ###############################
        # Crop image
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_ho, self.data_split, enforce_flip=False)
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
        ############################### PROCESS CROP AND AUGMENTATION ###############################


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)