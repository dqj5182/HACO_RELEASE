import os
import cv2
import json
import trimesh
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from data.DexYCB.toolkit.dex_ycb import _SUBJECTS, _SERIALS, _YCB_CLASSES 
from data.DexYCB.toolkit.factory import get_dataset

from lib.core.config import cfg
from lib.utils.preprocessing import augmentation_contact
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation, inv_mano_global_orient
from lib.utils.preprocessing import process_human_model_output_orig
from lib.utils.func_utils import load_img, get_bbox, pca_to_axis_angle
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.human_models import mano
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(dataset, split, index):
    aid = split[index]
    image_id = aid
    label_dict = dataset[image_id]

    img_path = label_dict['color_file']
    sample_id = '_'.join([str(int(img_path.split('/')[-4].split('-')[-1])), img_path.split('/')[-3], img_path.split('/')[-2], str(int(img_path.split('/')[-1].split('_')[-1].split('.')[0]))])
    return sample_id, label_dict


class DexYCB(Dataset):
    def __init__(self, transform, data_split, mode=["scale"], apply_Xtg=False, max_voxels=512, voxel_dim=16):
        super(DexYCB, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform

        dataset_name = 'dexycb'
        self.data_split = data_split
        self.root_path = root_path = 'data/DexYCB'

        # Load dataset from original DexYCB toolkit
        split_name = 'train_s0_29k' if data_split == 'train' else 'test_s0_5k'
        stage_split = '_'.join(split_name.split('_')[:-1])

        self.dataset = get_dataset(stage_split.split('_')[1] + '_' + stage_split.split('_')[0])

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

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)



        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id, _ = get_sample_id(self.dataset, self.split, split_idx)
                if each_sample_id in sample_id_to_split_id:
                    raise KeyError(f"Key '{key}' already exists in the dictionary.")
                else:
                    sample_id_to_split_id[each_sample_id] = self.split[split_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(self.contact_data_path, contact_means_path)

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list]
            self.split = new_split


    def __len__(self):
        return len(self.split)

        
    def __getitem__(self, index):
        sample_id, label_dict = get_sample_id(self.dataset, self.split, index)
        
        # Organize id
        subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
        video_id = '_'.join(sample_id.split('_')[1:3])
        cam_id = sample_id.split('_')[-2]
        frame_id = sample_id.split('_')[-1].rjust(6, '0')
        ycb_id = label_dict['ycb_ids'][label_dict['ycb_grasp_ind']]

        # Base path
        img_path = os.path.join(self.root_path, 'data', subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
        seg_path = os.path.join(self.root_path, 'data', subject_id, video_id, cam_id, 'labels_' + frame_id + '.npz')
        obj_rest_mesh_path = os.path.join('data/DexYCB/data/models', _YCB_CLASSES[ycb_id], 'textured_simple.obj')

        ###################################### READ ANNOTATION #######################################
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

        mano_valid = np.ones((1), dtype=np.float32)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_ho = annot_data['bbox_ho']
            cam_param = annot_data['cam_param']
        else:
            # Convert use_pca=True to use_pca=False (this is for unified MANO setting for HACO)
            hand_pose = pca_to_axis_angle(hand_pose[None])[0].detach().cpu().numpy()

            mano_param = {'pose': hand_pose, 'shape': hand_shape, 'trans': hand_trans[:, None], 'hand_type': 'right'}
            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape, mano_trans = process_human_model_output_orig(mano_param, cam_param)
            mano_mesh_img = cam2pixel(mano_mesh_cam, cam_param['focal'], cam_param['princpt'])[:, :2]

            # Object rot, trans
            obj_transform = label['pose_y'][label_dict['ycb_grasp_ind']]
            obj_rot, obj_trans = obj_transform[:3, :3], obj_transform[:, 3:]

            # Object bbox
            grasp_obj_id = label_dict['ycb_ids'][label_dict['ycb_grasp_ind']]
            obj_rest_mesh = trimesh.load(obj_rest_mesh_path, process=False, skip_materials=True)
            obj_rest_corners = trimesh.bounds.corners(obj_rest_mesh.bounds)
            obj_cam = (obj_rot @ obj_rest_corners.transpose(1, 0) + obj_trans).transpose(1, 0)
            obj_img = cam2pixel(obj_cam, cam_param['focal'], cam_param['princpt'])[:, :2]

            # Bounding box
            bbox_hand = get_bbox(joint_img[joint_valid==1, :], np.ones(len(joint_img[joint_valid==1, :])), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)
            bbox_obj = get_bbox(obj_img, np.ones(len(obj_img)), expansion_factor=cfg.DATASET.obj_bbox_expand_ratio)
            bbox_ho = get_bbox(np.concatenate((joint_img[joint_valid==1, :], obj_img), axis=0), np.ones(len(joint_img[joint_valid==1, :])+len(obj_img)), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)
            
            if True:
                annot_data = dict(sample_id=sample_id, mano_param=mano_param, cam_param=cam_param, joint_cam=joint_cam, joint_img=joint_img, joint_valid=joint_valid, obj_cam=obj_cam, obj_img=obj_img, bbox_hand=bbox_hand, bbox_obj=bbox_obj, bbox_ho=bbox_ho, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)
        ###################################### READ ANNOTATION #######################################


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

        if not self.use_preprocessed_data and (self.data_split != 'train'):
            # Hand canonical space
            hand_vert_can, homo_transform_to_can, homo_transform_from_can = inv_mano_global_orient(torch.from_numpy(mano_mesh_cam).float(), torch.from_numpy(mano_joint_cam[mano.joints_name.index('Wrist')]).float(), torch.from_numpy(mano_pose[:3]).float(), torch.from_numpy(mano_trans).float())
            hand_vert_can, homo_transform_to_can, homo_transform_from_can = hand_vert_can.detach().cpu().numpy(), homo_transform_to_can.detach().cpu().numpy(), homo_transform_from_can.detach().cpu().numpy()

            # Get hand mesh for depth rendering
            mesh_hand_watertight = trimesh.Trimesh(mano_mesh_cam, mano.watertight_face['right'])

            # Get object mesh
            obj_rest_mesh = trimesh.load(obj_rest_mesh_path, process=False, skip_materials=True)
            homo_obj_verts = np.ones((obj_rest_mesh.vertices.shape[0], 4))
            homo_obj_verts[:, :3] = obj_rest_mesh.vertices

            obj_verts = np.dot(obj_transform, homo_obj_verts.transpose(1, 0)).transpose(1, 0)
            obj_verts = obj_verts[:, :3]
            mesh_obj = trimesh.Trimesh(obj_verts, obj_rest_mesh.faces)

            # Contact data
            mesh_hand_can = trimesh.Trimesh(hand_vert_can, mano.layer['right'].faces)
            obj_vert_can = apply_homogeneous_transformation(torch.from_numpy(mesh_obj.vertices).float(), torch.from_numpy(homo_transform_to_can))
            mesh_obj_can = trimesh.Trimesh(obj_vert_can, mesh_obj.faces)

            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_can, mesh_obj_can, cfg.MODEL.c_thres)

            contact_data = dict(contact_h=contact_h)
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)

        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, orig_img=orig_img, mano_valid=mano_valid, mano_mesh_cam=mano_mesh_cam, mano_mesh_img=mano_mesh_img, cam_param=cam_param)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)



if __name__ == "__main__":
    dataset_name = 'DexYCB'
    data_split = 'train'
    task = 'debug'

    transform = transforms.ToTensor()

    dataset = eval(dataset_name)(transform, data_split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.DATASET.workers, pin_memory=False)
    iterator = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

    if task == 'debug':
        for idx, data in iterator:
            pass
    elif task == 'save_contact_means':
        contact_means_save_root_path = f'data/base_data/contact_data/{dataset_name.lower()}'
        os.makedirs(contact_means_save_root_path, exist_ok=True)
        contact_means_save_path = os.path.join(contact_means_save_root_path, f'contact_means_{dataset_name.lower()}.npy')
        contact_h_list = []

        for idx, data in iterator:
            contact_h = data['targets_data']['contact_data']['contact_h'][0].tolist()
            contact_h_list.append(contact_h)

        contact_h_list = np.array(contact_h_list, dtype=np.float32)
        contact_means = np.mean(contact_h_list, axis=0)
        np.save(contact_means_save_path, contact_means)
    elif task == 'save_contact_data':
        contact_data_save_root_path = f'data/{dataset_name}/preprocessed_data/{data_split}/contact_data'
        os.makedirs(contact_data_save_root_path, exist_ok=True)

        for idx, data in iterator:
            sample_id = data['meta_info']['sample_id'][0]
            contact_h = data['targets_data']['contact_data']['contact_h'][0].tolist()
            contact_h = np.array(contact_h, dtype=int)

            contact_data_save_path = os.path.join(contact_data_save_root_path, f'{sample_id}.npy')
            np.save(contact_data_save_path, contact_h)
    else:
        raise NotImplementedError