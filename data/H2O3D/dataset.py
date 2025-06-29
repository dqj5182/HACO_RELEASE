import os
import os.path as osp
import numpy as np
import torch
import cv2
import pickle
import trimesh
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.transforms import cam2pixel
from lib.utils.mesh_utils import read_obj
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.mano_utils import change_flat_hand_mean
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, process_human_model_output_orig
from lib.utils.train_utils import get_contact_difficulty_sample_id


def swap_coord_sys(arr):
    if not isinstance(arr, np.ndarray):
        arr =  np.array(arr)
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return arr.dot(coordChangMat.T)



def swap_param_sys(hand_pose, hand_shape, hand_trans, side='right'):
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    cam_rot = cv2.Rodrigues(coordChangMat)[0].squeeze()
    cam_trans = np.zeros((3,))

    from lib.utils.mano_utils import ready_arguments
    mano_root='data/base_data/human_models/mano'

    if side == 'right':
        mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
    elif side == 'left':
        mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
    smpl_data = ready_arguments(mano_path)
    J = smpl_data['J'][0:1, :].T

    RAbsMat = cv2.Rodrigues(cam_rot)[0].dot(cv2.Rodrigues(hand_pose[:3])[0])
    RAbsRod = cv2.Rodrigues(RAbsMat)[0][:, 0]
    hand_trans = cv2.Rodrigues(cam_rot)[0].dot(J + np.expand_dims(np.copy(hand_trans), 0).T) + np.expand_dims(cam_trans / 1000, 0).T - J

    hand_pose[:3] = RAbsRod

    return hand_pose, np.array(hand_trans.r)[:, 0].astype(np.float32)


def get_sample_id(split, index):
    aid = split[index].replace('/', '-')
    image_id = aid
    seq_name = split[index].split('/')[0]
    file_name = split[index].split('/')[1]
    file_path = split[index]

    sample_id = image_id
    return sample_id


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
        self.root_path = root_path = osp.join('data', 'H2O3D')
        self.data_dir = os.path.join(self.root_path, 'data')

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 1
        else:
            sampling_ratio = 1

        # Load db
        with open(os.path.join(self.data_dir, f'{self.data_split_name}.txt'), 'r') as f:
            db = f.readlines()
            self.db = [line.strip() for line in db][::sampling_ratio]
        self.split = self.db

        self.use_preprocessed_data = True
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.contact_data_path, exist_ok=True)


        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id = get_sample_id(self.split, split_idx)
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
        aid = self.split[index].replace('/', '-')
        image_id = aid
        seq_name = self.split[index].split('/')[0]
        file_name = self.split[index].split('/')[1]
        file_path = self.split[index]

        sample_id = image_id    

        orig_img = load_img(os.path.join(self.data_dir, self.data_split_name, seq_name, 'rgb', f'{file_name}.jpg'))
        orig_img_shape = orig_img.shape[:2]
        img_h, img_w = orig_img_shape
        
        ################################## PROCESS JOINTS & CORNERS ##################################
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
        
        mano_valid = np.ones((1), dtype=np.float32)
        ################################## PROCESS JOINTS & CORNERS ##################################

        
        if not self.use_preprocessed_data or (self.data_split != 'train'):
            ###################################### READ ANNOTATION #######################################
            hand_pose_r, hand_trans_r = meta_data['rightHandPose'], meta_data['rightHandTrans']
            hand_pose_l, hand_trans_l = meta_data['leftHandPose'], meta_data['leftHandTrans']
            hand_shape = meta_data['handBeta']

            # Coordiante swap for H2O3D dataset
            hand_pose_r, hand_trans_r = swap_param_sys(hand_pose_r, hand_shape, hand_trans_r, side='right')
            hand_pose_l, hand_trans_l = swap_param_sys(hand_pose_l, hand_shape, hand_trans_l, side='left')

            # flat_hand_mean=True -> flat_hand_mean=False (only use when the dataset is based on flat_hand_mean=True)
            hand_pose_r = change_flat_hand_mean(hand_pose_r, remove=True, side='right')
            hand_pose_l = change_flat_hand_mean(hand_pose_l, remove=True, side='left')

            mano_param_r = {'pose': hand_pose_r, 'shape': hand_shape, 'trans': hand_trans_r, 'hand_type': 'right'}
            mano_param_l = {'pose': hand_pose_l, 'shape': hand_shape, 'trans': hand_trans_l, 'hand_type': 'left'}

            obj_rot, obj_trans = meta_data['objRot'], meta_data['objTrans']
            obj_cam = meta_data['objCorners3D']
            obj_name, obj_id = meta_data['objName'], meta_data['objLabel']

            obj_cam = swap_coord_sys(obj_cam)*1000 # swap_coord_sys is ok for this scenario
            obj_img = cam2pixel(obj_cam, cam_param['focal'], cam_param['princpt'])[:, :2]
            bbox_obj = get_bbox(obj_img, np.ones(len(obj_img)), expansion_factor=1.5)
            ###################################### READ ANNOTATION #######################################

            mano_mesh_cam_r, mano_joint_cam_r, mano_pose_r, mano_shape_r, mano_trans_r = process_human_model_output_orig(mano_param_r, cam_param)
            mano_mesh_cam_l, mano_joint_cam_l, mano_pose_l, mano_shape_l, mano_trans_l = process_human_model_output_orig(mano_param_l, cam_param)

            mano_mesh_img_r = cam2pixel(mano_mesh_cam_r, cam_param['focal'], cam_param['princpt'])[:, :2]
            mano_mesh_img_l = cam2pixel(mano_mesh_cam_l, cam_param['focal'], cam_param['princpt'])[:, :2]

            mesh_hand_r = trimesh.Trimesh(mano_mesh_cam_r, mano.layer['right'].faces)
            mesh_hand_l = trimesh.Trimesh(mano_mesh_cam_l, mano.layer['left'].faces)


            ########################## GROUND-TRUTH OBJECT MESH ###########################
            obj_rest_mesh_path = os.path.join(self.root_path, 'YCB_object_models', 'models', obj_name, 'textured_simple.obj') # H2O3D and HO3D share object models
            mesh_obj = read_obj(obj_rest_mesh_path)
            obj_verts = mesh_obj.v
            obj_faces = mesh_obj.f
            obj_verts = np.matmul(obj_verts, cv2.Rodrigues(obj_rot)[0].T) + obj_trans

            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            obj_verts = torch.tensor(obj_verts)[None, ...].float() @ coord_change_mat.T
            obj_verts = obj_verts[0].detach().cpu().numpy()

            obj_transform = np.eye(4)
            obj_transform[:3, :3] = cv2.Rodrigues(obj_rot)[0]
            obj_transform[:3, 3] = obj_trans

            obj_verts = obj_verts*1000

            mesh_obj = trimesh.Trimesh(obj_verts/1000, obj_faces)
            ########################## GROUND-TRUTH OBJECT MESH ###########################

            mesh_other = mesh_hand_l + mesh_obj

    
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_r, mesh_other, cfg.MODEL.c_thres)
            contact_data = dict(contact_h=contact_h)
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
            contact_h = contact_data['contact_h']



        ######################################## PROCESS BBOX ########################################
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
        ######################################## PROCESS BBOX ########################################


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
    dataset_name = 'H2O3D'
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
            if data == []:
                continue
            sample_id = data['meta_info']['sample_id'][0]
            contact_h = data['targets_data']['contact_data']['contact_h'][0].tolist()
            contact_h = np.array(contact_h, dtype=int)

            contact_data_save_path = os.path.join(contact_data_save_root_path, f'{sample_id}.npy')
            np.save(contact_data_save_path, contact_h)
    else:
        raise NotImplementedError