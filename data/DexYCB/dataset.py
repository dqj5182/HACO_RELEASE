import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from data.DexYCB.toolkit.dex_ycb import _SUBJECTS, _YCB_CLASSES 
from data.DexYCB.toolkit.factory import get_dataset

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.transforms import cam2pixel
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(dataset, split, index):
    aid = split[index]
    image_id = aid
    label_dict = dataset[image_id]

    img_path = label_dict['color_file']
    sample_id = '_'.join([str(int(img_path.split('/')[-4].split('-')[-1])), img_path.split('/')[-3], img_path.split('/')[-2], str(int(img_path.split('/')[-1].split('_')[-1].split('.')[0]))])
    return sample_id, label_dict



# Main dataloader code for DexYCB dataset
class DexYCB(Dataset):
    def __init__(self, transform, data_split, mode=["scale"], apply_Xtg=False, max_voxels=512, voxel_dim=16):
        super(DexYCB, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform

        dataset_name = 'dexycb'
        self.data_split = data_split
        self.root_path = root_path = 'data/DexYCB'

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # Load dataset from original DexYCB toolkit
        split_name = 'train_s0_29k' if data_split == 'train' else 'test_s0_5k'
        stage_split = '_'.join(split_name.split('_')[:-1])

        self.dataset = get_dataset(stage_split.split('_')[1] + '_' + stage_split.split('_')[0])

        # Organize db id based on split
        with open(os.path.join(root_path, 'splits', split_name + '.json'), 'r') as f:
            self.split = json.load(f)
        self.split = [int(idx) for idx in self.split]

        self.start_point = 0
        self.end_point = len(self.split)
        self.length = self.end_point - self.start_point
        self.input_img_shape = cfg.MODEL.input_img_shape
        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            }
                        }
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')

        # Sort contact by difficulty (Balanced contact sampling)
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id, _ = get_sample_id(self.dataset, self.split, split_idx)
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
        sample_id, label_dict = get_sample_id(self.dataset, self.split, index)        
        subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
        video_id = '_'.join(sample_id.split('_')[1:3])
        cam_id = sample_id.split('_')[-2]
        frame_id = sample_id.split('_')[-1].rjust(6, '0')
        ycb_id = label_dict['ycb_ids'][label_dict['ycb_grasp_ind']]

        img_path = os.path.join(self.root_path, 'data', subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
        seg_path = os.path.join(self.root_path, 'data', subject_id, video_id, cam_id, 'labels_' + frame_id + '.npz')
        obj_rest_mesh_path = os.path.join('data/DexYCB/data/models', _YCB_CLASSES[ycb_id], 'textured_simple.obj')

        mano_valid = np.ones((1), dtype=np.float32)


        ################################## LOAD ANNOTATION DATA #####################################
        sample_id1, sample_id2, sample_id3, sample_id4, sample_id5 = sample_id.split('_')

        # Hand parameters
        label_file_path = label_dict['label_file']
        label = np.load(label_file_path, mmap_mode='r')

        # Sanity check
        label_file_name1, label_file_name2, label_file_name3, label_file_name4 = label_file_path.split('/')[-4], label_file_path.split('/')[-3], label_file_path.split('/')[-2], label_file_path.split('/')[-1]
        assert label_file_name1 == f'{sample_id2}-subject-{int(sample_id1):02d}' and label_file_name2 == f'{sample_id2}_{sample_id3}' and label_file_name3 == sample_id4 and label_file_name4 == f'labels_{int(sample_id5):06d}.npz'

        # Camera intrinsic parameter
        cam_param = {'focal': [label_dict['intrinsics']['fx'], label_dict['intrinsics']['fy']], 'princpt': [label_dict['intrinsics']['ppx'], label_dict['intrinsics']['ppy']]}

        # MANO params
        hand_pose = np.array(label['pose_m'][0, 0:48])
        hand_shape = np.array(label_dict['mano_betas'])
        hand_trans = np.array(label['pose_m'][0, 48:51])

        # Hand joints
        joint_cam = label['joint_3d'][0]
        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])[:, :2] # this is only for bbox extraction
        joint_valid = np.ones(len(joint_cam))

        # Full image
        orig_img = load_img(img_path)
        orig_img_shape = orig_img.shape[:2] # (img_h, img_w)

        # Segmentation
        hand_seg = (label['seg'] == 255)
        obj_seg = (label['seg'] == ycb_id)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_ho = annot_data['bbox_ho']
        cam_param = annot_data['cam_param']

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