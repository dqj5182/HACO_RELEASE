import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(split, index):
    aid = split[index]
    seq_name = aid.split('_')[0]
    cam_name = aid.split('_')[1]
    img_name = aid.split('_')[2]
    sample_id = aid
    return sample_id



# Main dataloader code for Decaf dataset
class Decaf(Dataset):
    def __init__(self, transform, data_split):
        super(Decaf, self).__init__()
        self.__dict__.update(locals())

        self.transfrom = transform
        dataset_name = 'decaf'

        self.data_split = data_split
        self.root_path = root_path = 'data/Decaf'
        self.data_dir = os.path.join(self.root_path, 'data')

        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        self.mano_joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')
        
        seq_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        cam_list = ['084', '100', '102', '108', '110', '111', '121', '122']

        # Make db
        contact_rh_data = {seq: np.load(os.path.join(self.data_dir, self.data_split, 'contacts', seq, 'contacts_rh.npy')) for seq in seq_list}
        
        bb_rh_data = {
            f"{seq}_{cam}": np.load(os.path.join(self.data_dir, self.data_split, 'right_hand_bbs', seq, f'{cam}.npy'))
            for seq in seq_list
            for cam in cam_list
        }

        # Organize db id based on split
        db_split_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            db_keys = json.load(f)

        self.split = list(db_keys)

        # Sort contact by difficulty (Balanced contact sampling)
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id = get_sample_id(self.split, split_idx)
                if each_sample_id in sample_id_to_split_id:
                    raise KeyError(f"Key '{key}' already exists in the dictionary.")
                else:
                    sample_id_to_split_id[each_sample_id] = self.split[split_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(sample_id_to_split_id, self.contact_data_path, contact_means_path)

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list if key in [*sample_id_to_split_id]]
            self.split = new_split


    def __len__(self):
        return len(self.split)


    def __getitem__(self, index):
        aid = self.split[index]
        seq_name = aid.split('_')[0]
        cam_name = aid.split('_')[1]
        img_name = aid.split('_')[2]
        sample_id = aid

        orig_img_path = os.path.join(self.data_dir, self.data_split, 'images', seq_name, cam_name, f'{img_name}.jpg')
        
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        db_contact_path = os.path.join(self.root_path, 'preprocessed_data', self.data_split, 'db_contact', f'{sample_id}.npy')
        db_bb_path = os.path.join(self.root_path, 'preprocessed_data', self.data_split, 'db_bb', f'{sample_id}.npy')
        db_contact = np.load(db_contact_path)
        db_bb = np.load(db_bb_path)

        contact_rh = db_contact.astype(np.float32)
        bbox_rh = db_bb.tolist() # GT bbox is in [x_min, y_min, x_max, y_max]
        bbox_rh = np.array([bbox_rh[0], bbox_rh[1], bbox_rh[2]-bbox_rh[0], bbox_rh[3]-bbox_rh[1]]) # Change from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
        contact_data = dict(contact_h=contact_rh)
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_rh, self.data_split, enforce_flip=False)
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