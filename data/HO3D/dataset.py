import os
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
    image_id = split[index]
    ann_ids = db.getAnnIds(imgIds=[image_id])
    ann = db.loadAnns(ann_ids)[0]
    img = db.loadImgs(image_id)[0]
    sample_id = str(image_id)
    return sample_id



# Main dataloader code for HO3D dataset
class HO3D(Dataset):
    def __init__(self, transform, data_split):
        super(HO3D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'ho3d'

        if data_split == 'train':
            self.data_split_name = 'train'
        elif data_split == 'test':
            self.data_split_name = 'evaluation'
        else:
            raise NotImplementedError

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'HO3D')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.annot_dir = os.path.join(self.root_path, 'annotations')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        self.joints_name = ('wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index4', 'middle4', 'ring4', 'pinky4')
        self.mano_joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')

        # Organize db id based on split
        self.db = COCO(os.path.join(self.annot_dir, f"HO3D_{self.data_split_name}_data.json"))
        self.split = self.db.getImgIds()

        # Sort contact by difficulty (Balanced contact sampling)
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id = get_sample_id(self.db, self.split, split_idx)
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
        image_id = self.split[index]
        ann_ids = self.db.getAnnIds(imgIds=[image_id])
        ann = self.db.loadAnns(ann_ids)[0]
        img = self.db.loadImgs(image_id)[0]
        orig_img_path = os.path.join(self.data_dir, self.data_split_name, img['file_name'])
        meta_path = os.path.join(self.data_dir, self.data_split_name, img['file_name'].replace('/rgb/', '/meta/').replace('.png', '.pkl'))
        img_shape = (img['height'], img['width'])
        sample_id = str(image_id)

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_ho = annot_data['bbox_ho']
        cam_param = annot_data['cam_param']

        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_ho, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
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