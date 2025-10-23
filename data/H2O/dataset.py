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
    subject_name = aid.split('/')[0] + '_ego' # we only use ego split for H2O dataset
    seq_name = aid.split('/')[1]
    obj_id = aid.split('/')[2]
    cam_name = aid.split('/')[3]
    img_name = aid.split('/')[5]
    img_id = img_name.split('.png')[0]

    sample_id = f'{subject_name}-{seq_name}-{obj_id}-{cam_name}-{img_id}'
    return sample_id



# Main dataloader code for H2O dataset
class H2O(Dataset):
    def __init__(self, transform, data_split):
        super(H2O, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'h2o'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'H2O')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # Organize db id based on split
        db_split_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            self.split = json.load(f)

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
        subject_name = aid.split('/')[0] + '_ego' # we only use ego split for H2O dataset
        seq_name = aid.split('/')[1]
        obj_id = aid.split('/')[2]
        cam_name = aid.split('/')[3]
        img_name = aid.split('/')[5]
        img_id = img_name.split('.png')[0]

        sample_id = f'{subject_name}-{seq_name}-{obj_id}-{cam_name}-{img_id}'

        orig_img_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'rgb', f'{img_id}.png')

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        
        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_hand_r = annot_data['bbox_hand_r']

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