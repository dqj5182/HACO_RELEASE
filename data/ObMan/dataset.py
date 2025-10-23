import os
import json
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(db, split, index):
    aid = split[index]
    ann = db.anns[aid]
    img_data = db.loadImgs(ann['image_id'])[0]
    sample_id = img_data['file_name']
    return sample_id, img_data



# Main dataloader code for ObMan dataset
class ObMan(Dataset):
    def __init__(self, transform, data_split):
        super(ObMan, self).__init__()
        self.__dict__.update(locals())
        
        self.transfrom = transform
        dataset_name = 'obman'

        if data_split == 'train':
            data_split_name = 'train_87k'
        elif data_split == 'test':
            data_split_name = 'test_6k'
        else:
            raise NotImplementedError
        
        self.data_split = data_split
        self.root_path = root_path = 'data/ObMan'

        self.data_dir = os.path.join(self.root_path, 'data')
        self.split_dir = os.path.join(self.root_path, 'splits')
        self.annot_dir = os.path.join(self.root_path, 'annotations')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        with open(os.path.join(self.split_dir , f'{data_split_name}.json'), 'r') as f:
            self.split = json.load(f)

        self.split = [int(idx) for idx in self.split]

        self.anno_file = os.path.join(self.annot_dir, f'{dataset_name}_{data_split}.json')
        self.img_source = os.path.join(self.data_dir, data_split, 'rgb')
        self.seg_source = os.path.join(self.data_dir, data_split, 'segm')
        self.input_img_shape = cfg.MODEL.input_img_shape # should be (256, 256)

        self.cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
                                  [0., 0., 1.]]).astype(np.float32)
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)

        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            }
                        }
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')

        # Organize db id based on split
        self.db = COCO(self.anno_file)
        self.start_point = 0
        self.end_point = len(self.split)
        self.length = self.end_point - self.start_point

        # Sort contact by difficulty (Balanced contact sampling)
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id, _ = get_sample_id(self.db, self.split, split_idx)
                if each_sample_id in sample_id_to_split_id:
                    raise KeyError(f"Key '{key}' already exists in the dictionary.")
                else:
                    sample_id_to_split_id[each_sample_id] = self.split[split_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(sample_id_to_split_id, self.contact_data_path, contact_means_path)

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list]
            self.split = new_split


    def __len__(self):
        return len(self.split)


    def __getitem__(self, index):
        sample_id, img_data = get_sample_id(self.db, self.split, index)

        # Base path
        img_path = os.path.join(self.img_source, img_data['file_name'] + '.jpg')
        seg_path = os.path.join(self.seg_source, img_data['file_name'] + '.png')

        # Full image
        orig_img = load_img(img_path)
        orig_img_shape = orig_img.shape[:2]

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        bbox_ho = np.array([0, 0, 256, 256])
        
        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        contact_h = contact_data['contact_h']
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ################################
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
        ############################### PROCESS CROP AND AUGMENTATION ################################


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)