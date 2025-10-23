import os
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
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(db, db_pid, index):
    aid = db[index]
    pid = db_pid[index]
    pair_name = aid.split('/')[-5]
    action_name = aid.split('/')[-4]
    cam_name = aid.split('/')[-2]
    img_name = aid.split('/')[-1].split('.jpg')[0]
    sample_id = f'{pair_name}-{action_name}-{cam_name}-{img_name}-{pid}'
    return sample_id



# Main dataloader code for Hi4D dataset
class Hi4D(Dataset):
    def __init__(self, transform, data_split):
        super(Hi4D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'hi4d'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'Hi4D')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # SMPL-X to MANO mapping
        smpl_smplx_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smpl_to_smplx.pkl')
        smplx_mano_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smplx_to_mano.pkl')

        with open(smpl_smplx_mapping_path, 'rb') as f:
            self.smpl_to_smplx_mapping = pickle.load(f)

        with open(smplx_mano_mapping_path, 'rb') as f:
            self.smplx_to_mano_mapping = pickle.load(f)
            self.smplx_to_mano_mapping_r = self.smplx_to_mano_mapping["right_hand"]

        # Organize db id based on split
        split_db_id_file_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(split_db_id_file_path, 'r') as f:
            self.db = json.load(f)
        split_db_pid_file_path = os.path.join(self.root_path, 'splits', f'{self.data_split}_pid.json')
        with open(split_db_pid_file_path, 'r') as f:
            self.db_pid = json.load(f)
        
        # Sort contact by difficulty (Balanced contact sampling)
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_db_id = {}
            for db_idx in range(len(self.db)):
                each_sample_id = get_sample_id(self.db, self.db_pid, db_idx)
                if each_sample_id in sample_id_to_db_id:
                    raise KeyError(f"Key '{key}' already exists in the dictionary.")
                else:
                    sample_id_to_db_id[each_sample_id] = self.db[db_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(sample_id_to_db_id, self.contact_data_path, contact_means_path)

            new_db = [sample_id_to_db_id[key] for key in sample_id_difficulty_list]
            self.db = new_db


    def __len__(self):
        return len(self.db)


    def __getitem__(self, index):
        aid = self.db[index]
        pid = self.db_pid[index]
        pair_name = aid.split('/')[-5]
        action_name = aid.split('/')[-4]
        cam_name = aid.split('/')[-2]
        img_name = aid.split('/')[-1].split('.jpg')[0]
        sample_id = f'{pair_name}-{action_name}-{cam_name}-{img_name}-{pid}'

        orig_img_path = os.path.join(self.data_dir, pair_name, action_name, 'images', cam_name, f'{img_name}.jpg')
        
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        annot_data = np.load(annot_data_path, allow_pickle=True)
        mano_r_contact_0 = annot_data['mano_r_contact_0']
        mano_r_contact_1 = annot_data['mano_r_contact_1']
        bbox_hand_r_0 = annot_data['bbox_hand_r_0']
        bbox_hand_r_1 = annot_data['bbox_hand_r_1']
        
        if pid == 0:
            contact_h = mano_r_contact_0.astype(np.float32)
        else:
            contact_h = mano_r_contact_1.astype(np.float32)

        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ################################
        if pid == 0:
            img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand_r_0, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
        else:
            img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand_r_1, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
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