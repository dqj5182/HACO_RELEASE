import os
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

from lib.core.config import cfg
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.mesh_utils import load_ply
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.human_models import mano


class HIC(Dataset):
    def __init__(self, transform, data_split):
        super(HIC, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'hic'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'HIC')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        # Load db
        with open(os.path.join(self.data_dir, 'HIC.json'), 'r') as f:
            self.db = json.load(f)

        # inter: 01, 02, 03, 04, 05, 06, 07 | single: 08, 09, 10, 11 | Single-HOI: 15, 19, 20, 21 | Inter-HOI: 16, 17, 18
        self.inter_seq_names = ['01', '02', '03', '04', '05', '06', '07']
        self.single_seq_names = ['08', '09', '10', '11']
        self.single_hoi_seq_names = ['15', '19', '20', '21']
        self.inter_hoi_seq_names = ['16', '17', '18']

        # Split train/test set (we only use inter seq)
        self.train_seq_names = ['01', '02', '03', '04', '05', '06'] # defined by me
        self.test_seq_names = ['07'] # defined by me

        if data_split == 'train':
            self.db['images'] = [img for img in self.db['images'] if img['seq_name'] in self.train_seq_names]
            self.db['annotations'] = [ann for ann in self.db['annotations'] if ann['image_id'] in {img['id'] for img in self.db['images']}]
        else:
            self.db['images'] = [img for img in self.db['images'] if img['seq_name'] in self.test_seq_names]
            self.db['annotations'] = [ann for ann in self.db['annotations'] if ann['image_id'] in {img['id'] for img in self.db['images']}]

        db_new = {'images': [], 'annotations': []}

        for index, each_db in enumerate(range(len(self.db['images']))):
            images_info = self.db['images'][index]
            annotations = self.db['annotations'][index]
            
            aid = annotations['id']
            image_id = annotations['image_id']
            seq_name = images_info['seq_name']
            file_name = images_info['file_name']
            img_w, img_h = images_info['width'], images_info['height']

            bbox = annotations['bbox']
            hand_type = annotations['hand_type']
            right_mano_path = annotations['right_mano_path']
            left_mano_path = annotations['left_mano_path']

            sample_id = image_id            
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)

            if contact_h.sum() == 0.:
                continue

            db_new['images'].append(self.db['images'][index])
            db_new['annotations'].append(self.db['annotations'][index])

        self.db = db_new


        self.cam_param = {'focal': [525.0, 525.0], 'princpt': [319.5, 239.5]} # this is fixed for HIC dataset


    def __len__(self):
        return len(self.db['images'])

    def __getitem__(self, index):
        images_info = self.db['images'][index]
        annotations = self.db['annotations'][index]
        
        aid = annotations['id']
        image_id = annotations['image_id']
        seq_name = images_info['seq_name']
        file_name = images_info['file_name']
        img_w, img_h = images_info['width'], images_info['height']

        bbox = annotations['bbox']
        hand_type = annotations['hand_type']
        right_mano_path = annotations['right_mano_path']
        left_mano_path = annotations['left_mano_path']

        sample_id = image_id

        # Load image
        orig_img_path = os.path.join(self.data_dir, file_name)
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand_r = annot_data['bbox_hand']
            cam_param = annot_data['cam_param']
        else:
             # We do not utilize joint_img as these are noisy
            joint_img_r, joint_img_l = np.array(annotations['joint_2d'])[:14], np.array(annotations['joint_2d'])[14:]
            joint_valid_r, joint_valid_l = np.array(annotations['joint_valid'])[:14], np.array(annotations['joint_valid'])[14:]

            # Load MANO meshes
            if right_mano_path is not None:
                mano_mesh_cam_r = load_ply(os.path.join(self.data_dir, right_mano_path))
                mano_joint_cam_r = np.dot(mano.joint_regressor, mano_mesh_cam_r)
            else:
                mano_mesh_cam_r = np.zeros((mano.vertex_num, 3), dtype=np.float32)
                mano_joint_cam_r = np.dot(mano.joint_regressor, mano_mesh_cam_r)
            if left_mano_path is not None:
                mano_mesh_cam_l = load_ply(os.path.join(self.data_dir, left_mano_path))
                mano_joint_cam_l = np.dot(mano.joint_regressor, mano_mesh_cam_l)
            else:
                mano_mesh_cam_l = np.zeros((mano.vertex_num, 3), dtype=np.float32)
                mano_joint_cam_l = np.dot(mano.joint_regressor, mano_mesh_cam_l)

            mesh_hand_right = trimesh.Trimesh(mano_mesh_cam_r, mano.layer['right'].faces)
            mesh_hand_left = trimesh.Trimesh(mano_mesh_cam_l, mano.layer['left'].faces)

            mano_mesh_img_r = cam2pixel(mano_mesh_cam_r, self.cam_param['focal'], self.cam_param['princpt'])[:, :2]
            mano_mesh_img_l = cam2pixel(mano_mesh_cam_l, self.cam_param['focal'], self.cam_param['princpt'])[:, :2]
            
            mano_joint_img_r = cam2pixel(mano_joint_cam_r, self.cam_param['focal'], self.cam_param['princpt'])[:, :2]
            mano_joint_img_l = cam2pixel(mano_joint_cam_l, self.cam_param['focal'], self.cam_param['princpt'])[:, :2]

            mano_joint_valid_r, mano_joint_valid_l = np.ones(mano.orig_joint_num), np.ones(mano.orig_joint_num)

            # Extract bbox with MANO joints as regular joints have many invalid joints
            bbox_hand_l = get_bbox(mano_joint_img_l, np.ones(len(mano_joint_img_l)), expansion_factor=cfg.DATASET.hand_bbox_expand_ratio)
            bbox_hand_r = get_bbox(mano_joint_img_r, np.ones(len(mano_joint_img_r)), expansion_factor=cfg.DATASET.hand_bbox_expand_ratio)
            bbox_hand_ih = get_bbox(np.concatenate((mano_joint_img_l, mano_joint_img_r), axis=0), np.ones(len(mano_joint_img_l)+len(mano_joint_img_r)), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)
            
            if False:
                annot_data = dict(sample_id=sample_id, mano_param={}, cam_param=self.cam_param, joint_cam=mano_joint_cam_r, joint_img=mano_joint_img_r, joint_valid=mano_joint_valid_r, obj_cam=mano_joint_cam_r, obj_img=mano_joint_img_r, bbox_hand=bbox_hand_r, bbox_obj=bbox_hand_l, bbox_ho=bbox_hand_ih, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)


        ######################################## PROCESS BBOX ########################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, _ = augmentation_contact(orig_img.copy(), bbox_hand_r, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
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
        ######################################## PROCESS BBOX ########################################


        if not self.use_preprocessed_data or (self.data_split != 'train'):
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_right, mesh_hand_left, cfg.MODEL.c_thres_ih)
            contact_data = dict(contact_h=contact_h)
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)


        if self.data_split == 'train':
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=str(sample_id), mano_valid=mano_valid)
        else:
            input_data = dict(image=img)
            targets_data = dict(contact_data=contact_data)
            meta_info = dict(sample_id=str(sample_id), orig_img=orig_img, mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)




if __name__ == "__main__":
    dataset_name = 'HIC'
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