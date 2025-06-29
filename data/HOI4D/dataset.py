import os
import os.path as osp
import re
import numpy as np
import cv2
import json
import pickle
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rt

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
from lib.utils.func_utils import load_img, get_bbox, natural_keys
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


obj_cls_mapping = [
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]
rigid = [
    'Bowl', 'Bottle', 'Chair', 'Mug', 'ToyCar', 'Knife', 'Kettle',
]


def read_rtd(anno):
    """From HOI4D"""
    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), trans, dim



def extract_id_objs_dict(data):
    pairs = {}
    
    # Recursive function to extract id-objs pairs
    def recurse(node):
        if isinstance(node, dict):
            if "id" in node and "objs" in node:
                pairs[node["id"]] = node["objs"][0]  # Assuming 'objs' always has one element
            if "children" in node:
                for child in node["children"]:
                    recurse(child)
        elif isinstance(node, list):
            for item in node:
                recurse(item)
    
    recurse(data)
    return pairs


def get_sample_id(split, index):
    image_id = split[index]
    sample_id = str(image_id)
    return sample_id


class HOI4D(Dataset):
    def __init__(self, transform, data_split):
        super(HOI4D, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'hoi4d'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'HOI4D')
        self.data_dir = os.path.join(self.root_path, 'data')

        datalist_path = os.path.join(self.data_dir, 'datalists', f'{data_split}_all.txt')
        with open(datalist_path, 'r') as f:
            lines = f.readlines()  # Each line becomes an element in the list
            datalist = [line.strip() for line in lines]  # Remove extra newlines or spaces

        db = []
        for line in datalist:
            # datadict[line] = os.listdir()
            seq_path = os.path.join(self.data_dir, 'HOI4D_color', 'HOI4D_release', line, 'align_rgb')
            sample_list = [item for item in os.listdir(seq_path) if '.mp4' not in item]
            sample_list.sort(key=natural_keys)

            for sample in sample_list:
                sample_id = line.replace('/', '-') + '-' + sample.replace('.jpg', '')
                db.append(sample_id)

        # Filter non-valid samples
        self.db = []

        for image_id in db:
            seq_name, image_name = re.match(r'(.+?)-(\d+)$', image_id).groups()
            seq_name = seq_name.replace('-', '/')

            obj_class_name = seq_name.split('/')[2]
            obj_cat = obj_cls_mapping[int(obj_class_name[1:])]
            
            if obj_cat not in rigid:
                continue

            hand_pose_r_data_path = os.path.join(self.data_dir, 'HOI4D_Handpose', 'handpose', 'refinehandpose_right', seq_name, f'{int(image_name)}.pickle')
            if not os.path.exists(hand_pose_r_data_path):
                continue
            
            objpose_path1 = os.path.join(self.data_dir, 'HOI4D_annotations', 'HOI4D_annotations', seq_name, 'objpose', f'{int(image_name)}.json')
            objpose_path2 = os.path.join(self.data_dir, 'HOI4D_annotations', 'HOI4D_annotations', seq_name, 'objpose', f'{image_name}.json')
            
            if (not os.path.exists(objpose_path1)) and (not os.path.exists(objpose_path2)):
                continue

            self.db.append(image_id)

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 5
        else:
            sampling_ratio = 1

        self.db = self.db[::sampling_ratio]
        self.split = self.db

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)


        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(self.contact_data_path, contact_means_path)

            new_split = [key for key in sample_id_difficulty_list]
            self.split = new_split


    def __len__(self):
        return len(self.split)


    def __getitem__(self, index):
        image_id = self.split[index]
        seq_name, image_name = re.match(r'(.+?)-(\d+)$', image_id).groups()
        seq_name = seq_name.replace('-', '/')

        camera_name = seq_name.split('/')[0]
        person_name = seq_name.split('/')[1]
        obj_class_name = seq_name.split('/')[2]
        obj_instance_name = seq_name.split('/')[3]
        room_name = seq_name.split('/')[4]
        room_layout_name = seq_name.split('/')[5]
        task_name = seq_name.split('/')[6]

        obj_cat = obj_cls_mapping[int(obj_class_name[1:])]
        obj_id = int(obj_instance_name[1:])

        orig_img_path = os.path.join(self.data_dir, 'HOI4D_color', 'HOI4D_release', seq_name, 'align_rgb', f'{image_name}.jpg')

        sample_id = str(image_id)

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        contact_data_path = os.path.join(self.contact_data_path, f'{sample_id}.npy')

        if os.path.exists(annot_data_path) and os.path.exists(contact_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand = annot_data['bbox_ho']

            contact_h = np.load(contact_data_path).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
        else:
            # Camera
            camera_path = os.path.join(self.data_dir, 'HOI4D_cameras', 'camera_params', camera_name, 'intrin.npy')
            cam_intr = np.load(camera_path)
            cam_param = {'focal': [cam_intr[0][0], cam_intr[1][1]], 'princpt': [cam_intr[0][2], cam_intr[1][2]]}

            # Hand
            hand_pose_r_data_path = os.path.join(self.data_dir, 'HOI4D_Handpose', 'handpose', 'refinehandpose_right', seq_name, f'{int(image_name)}.pickle')
            with open(hand_pose_r_data_path, 'rb') as f:
                hand_pose_r_data = pickle.load(f)

            hand_pose, hand_trans, hand_shape = hand_pose_r_data['poseCoeff'], hand_pose_r_data['trans'], hand_pose_r_data['beta']
            # flat_hand_mean=True -> flat_hand_mean=False (only use when the dataset is based on flat_hand_mean=True)
            hand_pose = change_flat_hand_mean(hand_pose, remove=True)
            mano_param = {'pose': hand_pose, 'shape': hand_shape, 'trans': hand_trans[:, None], 'hand_type': 'right'}

            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape, mano_trans = process_human_model_output_orig(mano_param, cam_param)
            mano_mesh_img = cam2pixel(mano_mesh_cam, cam_param['focal'], cam_param['princpt'])[:, :2]
            bbox_hand = get_bbox(mano_mesh_img, np.ones(len(mano_mesh_img)), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)

            hand_mesh = trimesh.Trimesh(mano_mesh_cam, mano.watertight_face['right'])

            # Object
            try:
                objpose_path = os.path.join(self.data_dir, 'HOI4D_annotations', 'HOI4D_annotations', seq_name, 'objpose', f'{int(image_name)}.json')
                with open(objpose_path, 'r') as f:
                    objpose_data = json.load(f)
            except:
                objpose_path = os.path.join(self.data_dir, 'HOI4D_annotations', 'HOI4D_annotations', seq_name, 'objpose', f'{image_name}.json')
                with open(objpose_path, 'r') as f:
                    objpose_data = json.load(f)

            obj_rot, obj_trans, obj_dim = read_rtd(objpose_data['dataList'][0])

            obj_mesh = None

            obj_cat = obj_cls_mapping[int(obj_class_name[1:])]
            obj_id = int(obj_instance_name[1:])

            obj_cat_id = f'{obj_cat}_{obj_id}'

            obj_art_type = 'rigid'
            obj_dir = osp.join(self.data_dir, f'HOI4D_CAD_models/{obj_art_type}/{obj_cat}/{obj_id:03d}.obj')
            obj_mesh = trimesh.load(obj_dir)

            R, _ = cv2.Rodrigues(obj_rot)
            obj_mesh.vertices = (R @ obj_mesh.vertices.T).T + obj_trans


            if True:
                annot_data = dict(sample_id=sample_id, bbox_ho=bbox_hand)
                np.savez(annot_data_path, **annot_data)

            # Contact
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(hand_mesh, obj_mesh, cfg.MODEL.c_thres)
            contact_data = dict(contact_h=contact_h)

            if True:
                np.save(contact_data_path, contact_h)


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
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
    dataset_name = 'HOI4D'
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