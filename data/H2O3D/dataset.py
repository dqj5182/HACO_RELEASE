import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def swap_coord_sys(arr):
    if not isinstance(arr, np.ndarray):
        arr =  np.array(arr)
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return arr.dot(coordChangMat.T)


def get_sample_id(split, index):
    aid = split[index].replace('/', '-')
    image_id = aid
    seq_name = split[index].split('/')[0]
    file_name = split[index].split('/')[1]
    file_path = split[index]

    sample_id = image_id
    return sample_id



# Main dataloader code for H2O3D dataset
class H2O3D(Dataset):
    def __init__(self, transform, data_split):
        super(H2O3D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'h2o3d'

        if data_split == 'train':
            self.data_split_name = 'train'
        elif data_split == 'test':
            self.data_split_name = 'evaluation'
        else:
            raise NotImplementedError

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'H2O3D')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # Organize db id based on split
        with open(os.path.join(self.data_dir, f'{self.data_split_name}.txt'), encoding='utf-8') as f:
            self.split = [line.strip() for line in f]

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

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list]
            self.split = new_split


    def __len__(self):
        return len(self.split)


    def __getitem__(self, index):
        aid = self.split[index].replace('/', '-')
        image_id = aid
        seq_name = self.split[index].split('/')[0]
        file_name = self.split[index].split('/')[1]
        file_path = self.split[index]
        sample_id = image_id    

        orig_img = load_img(os.path.join(self.data_dir, self.data_split_name, seq_name, 'rgb', f'{file_name}.jpg'))
        orig_img_shape = orig_img.shape[:2]
        img_h, img_w = orig_img_shape

        mano_valid = np.ones((1), dtype=np.float32)

        
        ################################## LOAD ANNOTATION DATA #####################################
        with open(os.path.join(self.data_dir, self.data_split_name, seq_name, 'meta', f'{file_name}.pkl'), 'rb') as f:
            meta_data = pickle.load(f)

        joint_cam_r, joint_cam_l = meta_data['rightHandJoints3D'], meta_data['leftHandJoints3D']
        hand_joint_valid_r, hand_joint_valid_l = meta_data['jointValidRight'], meta_data['jointValidLeft']
        hand_valid_r, hand_valid_l = hand_joint_valid_r.sum() > 0, hand_joint_valid_l.sum() > 0

        joint_cam_r, joint_cam_l = swap_coord_sys(joint_cam_r), swap_coord_sys(joint_cam_l) # swap_coord_sys is ok for this scenario

        cam_k = meta_data['camMat']
        cam_param = {'focal': [cam_k[0][0], cam_k[0][0]], 'princpt': [cam_k[0][2], cam_k[1][2]]}

        joint_img_r = cam2pixel(joint_cam_r, cam_param['focal'], cam_param['princpt'])[:, :2] # this is only for bbox extraction
        joint_img_l = cam2pixel(joint_cam_l, cam_param['focal'], cam_param['princpt'])[:, :2] # this is only for bbox extraction

        bbox_hand_r = get_bbox(joint_img_r[hand_joint_valid_r==1, :], np.ones(len(joint_img_r[hand_joint_valid_r==1, :])), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio) # expand 1.3 as joints are very tight bbox
        bbox_hand_l = get_bbox(joint_img_l[hand_joint_valid_l==1, :], np.ones(len(joint_img_l[hand_joint_valid_l==1, :])), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio) # expand 1.3 as joints are very tight bbox
        
        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        contact_h = contact_data['contact_h']
        ################################## LOAD ANNOTATION DATA #####################################


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans_r, bb2img_trans_r, rot_r, do_flip_r, _ = augmentation_contact(orig_img.copy(), bbox_hand_r, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
        
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


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)