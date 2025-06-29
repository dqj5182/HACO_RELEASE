import os
import os.path as osp
import numpy as np
import cv2
import pickle
import trimesh
from tqdm import tqdm
from pycocotools.coco import COCO

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


def swap_param_sys(hand_pose, hand_shape, hand_trans):
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    cam_rot = cv2.Rodrigues(coordChangMat)[0].squeeze()
    cam_trans = np.zeros((3,))

    from lib.utils.mano_utils import ready_arguments
    mano_root='data/base_data/human_models/mano'
    side='right'

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


def get_sample_id(db, split, index):
    image_id = split[index]
    ann_ids = db.getAnnIds(imgIds=[image_id])
    ann = db.loadAnns(ann_ids)[0]
    img = db.loadImgs(image_id)[0]
    sample_id = str(image_id)
    return sample_id



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
        self.root_path = root_path = osp.join('data', 'HO3D')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.annot_dir = os.path.join(self.root_path, 'annotations')

        self.joints_name = ('wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index4', 'middle4', 'ring4', 'pinky4')
        self.mano_joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')

        self.db = COCO(osp.join(self.annot_dir, f"HO3D_{self.data_split_name}_data.json"))
        self.split = self.db.getImgIds()

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)


        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id = get_sample_id(self.db, self.split, split_idx)
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
        image_id = self.split[index]
        ann_ids = self.db.getAnnIds(imgIds=[image_id])
        ann = self.db.loadAnns(ann_ids)[0]
        img = self.db.loadImgs(image_id)[0]
        orig_img_path = osp.join(self.data_dir, self.data_split_name, img['file_name'])
        meta_path = osp.join(self.data_dir, self.data_split_name, img['file_name'].replace('/rgb/', '/meta/').replace('.png', '.pkl'))
        img_shape = (img['height'], img['width'])

        sample_id = str(image_id)

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)

        if self.data_split != 'train':
            return []

        if not self.use_preprocessed_data:
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)

            cam_k = meta_data['camMat']
            cam_param = {'focal': [cam_k[0][0], cam_k[0][0]], 'princpt': [cam_k[0][2], cam_k[1][2]]}

            hand_pose, hand_trans, hand_shape = meta_data['handPose'], meta_data['handTrans'], meta_data['handBeta']
            # Coordiante swap for HO3D dataset (HO3D has parameters flipped)
            hand_pose, hand_trans = swap_param_sys(hand_pose, hand_shape, hand_trans)
            # flat_hand_mean=True -> flat_hand_mean=False (only use when the dataset is based on flat_hand_mean=True)
            hand_pose = change_flat_hand_mean(hand_pose, remove=True)
            mano_param = {'pose': hand_pose, 'shape': hand_shape, 'trans': hand_trans[:, None], 'hand_type': 'right'}

            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape, mano_trans = process_human_model_output_orig(mano_param, cam_param)
            mano_mesh_img = cam2pixel(mano_mesh_cam, cam_param['focal'], cam_param['princpt'])[:, :2]
            
            mesh_hand = trimesh.Trimesh(mano_mesh_cam, mano.watertight_face['right'])

            obj_rot, obj_trans = meta_data['objRot'], meta_data['objTrans']
            obj_name, obj_id = meta_data['objName'], meta_data['objLabel']


            ########################## GROUND-TRUTH OBJECT MESH ###########################
            obj_rest_mesh_path = os.path.join('data', 'HO3D', 'YCB_object_models', 'models', obj_name, 'textured_simple.obj') # H2O3D and HO3D share object models
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


            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand, mesh_obj, cfg.MODEL.c_thres)
            contact_data = dict(contact_h=contact_h)
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
            contact_h = contact_data['contact_h']


        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_ho = annot_data['bbox_ho']
            cam_param = annot_data['cam_param']
        else:
            ################################## PROCESS JOINTS & CORNERS ##################################
            joint_cam = meta_data['handJoints3D']
            hand_joint_valid = np.ones(len(joint_cam))

            obj_cam = meta_data['objCorners3D']

            joint_cam = swap_coord_sys(joint_cam) # swap_coord_sys is ok for this scenario
            obj_cam = swap_coord_sys(obj_cam)*1000 # swap_coord_sys is ok for this scenario

            joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])[:, :2] # this is only for bbox extraction
            obj_img = cam2pixel(obj_cam, cam_param['focal'], cam_param['princpt'])[:, :2]

            bbox_hand = get_bbox(joint_img[hand_joint_valid==1, :], np.ones(len(joint_img[hand_joint_valid==1, :])), expansion_factor=cfg.DATASET.hand_bbox_expand_ratio)
            bbox_obj = get_bbox(obj_img, np.ones(len(obj_img)), expansion_factor=cfg.DATASET.obj_bbox_expand_ratio)
            bbox_ho = get_bbox(np.concatenate((joint_img[hand_joint_valid==1, :], obj_img), axis=0), np.ones(len(joint_img[hand_joint_valid==1, :])+len(obj_img)), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)
            ################################## PROCESS JOINTS & CORNERS ##################################

            if False:
                annot_data = dict(sample_id=str(sample_id), mano_param=mano_param, cam_param=cam_param, joint_cam=joint_cam, joint_img=joint_img, joint_valid=hand_joint_valid, obj_cam=obj_cam, obj_img=obj_img, bbox_hand=bbox_hand, bbox_obj=bbox_obj, bbox_ho=bbox_ho)
                np.savez(annot_data_path, **annot_data)


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
    dataset_name = 'HO3D'
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