import os
import os.path as osp
import numpy as np
import cv2
import glob
import pickle
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
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
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


class Hi4D(Dataset):
    def __init__(self, transform, data_split):
        super(Hi4D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'hi4d'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'Hi4D')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        self.db = [item for item in glob.glob(os.path.join(self.data_dir, '*/*/images/*/*')) if '.mp4' not in item]
        self.db = [item for item in self.db for _ in range(2)]
        self.db_pid = [i % 2 for i in range(len(self.db))] # person id

        # SMPL-X to MANO mapping
        smpl_smplx_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smpl_to_smplx.pkl')
        smplx_mano_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smplx_to_mano.pkl')

        with open(smpl_smplx_mapping_path, 'rb') as f:
            self.smpl_to_smplx_mapping = pickle.load(f)

        with open(smplx_mano_mapping_path, 'rb') as f:
            self.smplx_to_mano_mapping = pickle.load(f)
            self.smplx_to_mano_mapping_l = self.smplx_to_mano_mapping["left_hand"]
            self.smplx_to_mano_mapping_r = self.smplx_to_mano_mapping["right_hand"]

        self.db_new = []
        self.db_pid_new = []
        for db_idx, each_db in enumerate(self.db):
            aid = self.db[db_idx]
            pid = self.db_pid[db_idx]
            pair_name = aid.split('/')[-5]
            action_name = aid.split('/')[-4]
            cam_name = aid.split('/')[-2]
            img_name = aid.split('/')[-1].split('.jpg')[0]
            sample_id = f'{pair_name}-{action_name}-{cam_name}-{img_name}-{pid}'

            annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

            annot_data = np.load(annot_data_path, allow_pickle=True)
            mano_r_contact_0 = annot_data['mano_r_contact_0']
            mano_r_contact_1 = annot_data['mano_r_contact_1']

            if pid == 0:
                contact_h = mano_r_contact_0.astype(np.float32)
            else:
                contact_h = mano_r_contact_1.astype(np.float32)

            if contact_h.sum() == 0.:
                continue

            self.db_new.append(self.db[db_idx])
            self.db_pid_new.append(self.db_pid[db_idx])

        self.db = self.db_new
        self.db_pid = self.db_pid_new

        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_db_id = {}
            for db_idx in range(len(self.db)):
                each_sample_id = get_sample_id(self.db, self.db_pid, db_idx)
                if each_sample_id in sample_id_to_db_id:
                    raise KeyError(f"Key '{key}' already exists in the dictionary.")
                else:
                    sample_id_to_db_id[each_sample_id] = self.db[db_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(self.contact_data_path, contact_means_path)

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

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            mano_r_contact_0 = annot_data['mano_r_contact_0']
            mano_r_contact_1 = annot_data['mano_r_contact_1']
            bbox_hand_r_0 = annot_data['bbox_hand_r_0']
            bbox_hand_r_1 = annot_data['bbox_hand_r_1']
        else:
            # Camera
            camera_path = os.path.join(self.data_dir, pair_name, action_name, 'cameras', 'rgb_cameras.npz')
            camera_data = np.load(camera_path)
            cam_idx = camera_data['ids'].tolist().index(int(cam_name)) # TODO: CHECK WHETHER THIS IS STABLE
            cam_extr = camera_data['extrinsics'][cam_idx][:,:3]
            cam_transl = camera_data['extrinsics'][cam_idx][:,3]  # Translation Vector (T)
            cam_intr = camera_data['intrinsics'][cam_idx]
            cam_param = {'focal': [cam_intr[0][0], cam_intr[1][1]], 'princpt': [cam_intr[0][2], cam_intr[1][2]]}

            # Contact
            smpl_data_path = os.path.join(self.data_dir, pair_name, action_name, 'smpl', f'{img_name}.npz')
            smpl_data = np.load(smpl_data_path)

            smpl_verts_0, smpl_verts_1 = smpl_data['verts'][0], smpl_data['verts'][1]
            smpl_contact_0, smpl_contact_1 = np.array(smpl_data['contact'][0] > 0, dtype=int), np.array(smpl_data['contact'][1] > 0, dtype=int)
            
            # Transform vertices to camera space
            smpl_verts_0_cam = (cam_extr @ smpl_verts_0.T).T + cam_transl
            smpl_verts_1_cam = (cam_extr @ smpl_verts_1.T).T + cam_transl

            # Get hand vertices
            smpl_verts_0_img = cam2pixel(smpl_verts_0_cam, cam_param['focal'], cam_param['princpt'])
            smpl_verts_1_img = cam2pixel(smpl_verts_1_cam, cam_param['focal'], cam_param['princpt'])
            
            smplx_verts_0_img = np.matmul(self.smpl_to_smplx_mapping['matrix'], smpl_verts_0_img)
            smplx_verts_1_img = np.matmul(self.smpl_to_smplx_mapping['matrix'], smpl_verts_1_img)
            
            mano_r_verts_0_img = smplx_verts_0_img[self.smplx_to_mano_mapping_r]
            mano_r_verts_1_img = smplx_verts_1_img[self.smplx_to_mano_mapping_r]

            bbox_hand_r_0 = get_bbox(mano_r_verts_0_img, np.ones(len(mano_r_verts_0_img)), expansion_factor=cfg.DATASET.hand_scene_bbox_expand_ratio)
            bbox_hand_r_1 = get_bbox(mano_r_verts_1_img, np.ones(len(mano_r_verts_1_img)), expansion_factor=cfg.DATASET.hand_scene_bbox_expand_ratio)

            # Get hand contact
            smplx_contact_0 = np.matmul(self.smpl_to_smplx_mapping['matrix'], smpl_contact_0)
            smplx_contact_1 = np.matmul(self.smpl_to_smplx_mapping['matrix'], smpl_contact_1)

            mano_r_contact_0 = smplx_contact_0[self.smplx_to_mano_mapping_r]
            mano_r_contact_1 = smplx_contact_1[self.smplx_to_mano_mapping_r]

            if True:
                annot_data = dict(sample_id=sample_id, mano_r_contact_0=mano_r_contact_0, mano_r_contact_1=mano_r_contact_1, bbox_hand_r_0=bbox_hand_r_0, bbox_hand_r_1=bbox_hand_r_1, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)

        if pid == 0:
            contact_h = mano_r_contact_0.astype(np.float32)
        else:
            contact_h = mano_r_contact_1.astype(np.float32)


        contact_valid = np.ones((mano.vertex_num, 1))
        inter_coord_valid = np.ones((mano.vertex_num))

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

        contact_data = dict(contact_h=contact_h)


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
    dataset_name = 'Hi4D'
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