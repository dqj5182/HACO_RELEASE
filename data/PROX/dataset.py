import os
import os.path as osp
import numpy as np
import cv2
import json
import glob
import pickle
import trimesh
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact


def get_sample_id(db, index):
    aid = db[index]
    seq_name = aid.split('/')[-3]
    img_name = aid.split('/')[-1]
    sample_id = f'{seq_name}-{img_name}'
    return sample_id


class PROX(Dataset):
    def __init__(self, transform, data_split):
        super(PROX, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'prox'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'PROX')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.db = glob.glob(os.path.join(self.data_dir, 'quantitative/fittings/mosh/*/images/*'))

        # SMPL-X to MANO mapping
        smplx_mano_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smplx_to_mano.pkl')

        with open(smplx_mano_mapping_path, 'rb') as f:
            self.smplx_to_mano_mapping = pickle.load(f)
            self.smplx_to_mano_mapping_l = self.smplx_to_mano_mapping["left_hand"]
            self.smplx_to_mano_mapping_r = self.smplx_to_mano_mapping["right_hand"]

        # Camera
        with open(os.path.join(self.data_dir, 'quantitative/calibration/Color.json'), 'r') as f:
            calibration = json.load(f)
            self.cam_param = {'focal': calibration['f'], 'princpt': calibration['c']}

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)


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

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand_r = annot_data['bbox_ho']
            cam_param = annot_data['cam_param']

            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
        else:
            # Camera
            with open(os.path.join(self.data_dir, 'quantitative', 'cam2world', f'vicon.json'), 'r') as f:
                c2w_homo = np.array(json.load(f))

            # Body mesh
            mesh_body_path = os.path.join(self.data_dir, 'quantitative', 'fittings', 'mosh', seq_name, 'meshes', img_name, '000.ply')
            mesh_body = trimesh.load(mesh_body_path)
            mesh_body_cam = mesh_body.copy()
            mesh_hand_r_cam = mesh_body_cam.vertices[self.smplx_to_mano_mapping_r]
            mesh_hand_r_img = cam2pixel(mesh_hand_r_cam, self.cam_param['focal'], self.cam_param['princpt'])
            bbox_hand_r = get_bbox(mesh_hand_r_img, np.ones(len(mesh_hand_r_img)), expansion_factor=cfg.DATASET.hand_scene_bbox_expand_ratio)

            mesh_body_world = apply_homogeneous_transformation_np(mesh_body.vertices, c2w_homo)
            mesh_body = trimesh.Trimesh(mesh_body_world, mesh_body.faces)

            mesh_hand_r_world = mesh_body_world[self.smplx_to_mano_mapping_r]
            mesh_hand_r = trimesh.Trimesh(mesh_hand_r_world, mano.layer['right'].faces)

            # Scene mesh
            scene_mesh_path = os.path.join(self.data_dir, 'quantitative', 'scenes', f'vicon.ply')
            scene_mesh = trimesh.load(scene_mesh_path)

            if False:
                annot_data = dict(sample_id=sample_id, mano_param={}, cam_param=self.cam_param, joint_cam=mesh_hand_r_cam, joint_img=mesh_hand_r_img, joint_valid=np.array([]), obj_cam=np.array([]), obj_img=np.array([]), bbox_hand=np.array([]), bbox_obj=np.array([]), bbox_ho=bbox_hand_r, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)

            # Contact
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_r, scene_mesh, cfg.MODEL.c_thres)
            contact_data = dict(contact_h=contact_h)

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


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)



if __name__ == "__main__":
    dataset_name = 'PROX'
    data_split = 'train' # This dataset only has train set
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