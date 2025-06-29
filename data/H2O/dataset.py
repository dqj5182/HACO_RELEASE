import os
import os.path as osp
import numpy as np
import cv2
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
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation
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



obj_class_list = ['background', 'book', 'espresso', 'lotion', 'spray', 'milk', 'cocoa', 'chips', 'cappuccino']
verb_label_list = ['background', 'grab', 'place', 'open', 'close', 'pour', 'take out', 'put in', 'apply', 'read', 'spray', 'squeeze']



def get_sample_id(split, index):
    aid = split[index]
    subject_name = aid.split('/')[0] + '_ego' # we only use ego split for H2O dataset
    seq_name = aid.split('/')[1]
    obj_id = aid.split('/')[2]
    cam_name = aid.split('/')[3]
    img_name = aid.split('/')[5]
    img_id = img_name.split('.png')[0]

    sample_id = f'{subject_name}-{seq_name}-{obj_id}-{cam_name}-{img_id}'
    return sample_id


class H2O(Dataset):
    def __init__(self, transform, data_split):
        super(H2O, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'h2o'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'H2O')
        self.data_dir = os.path.join(self.root_path, 'data')

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 1
        else:
            sampling_ratio = 1

        # DB
        db_path = os.path.join(self.data_dir, 'label_split', f'pose_{data_split}.txt')
        db = [line.strip() for line in open(db_path).readlines()][::sampling_ratio]
        new_db = []

        for aid in db:
            subject_name = aid.split('/')[0] + '_ego' # we only use ego split for H2O dataset
            seq_name = aid.split('/')[1]
            obj_id = aid.split('/')[2]
            cam_name = aid.split('/')[3]
            img_name = aid.split('/')[5]
            img_id = img_name.split('.png')[0]

            # Verb label
            verb_label_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'verb_label', f'{img_id}.txt')
            with open(verb_label_path) as f:
                verb_label = verb_label_list[int(f.read().strip())] # good: ['place', 'squeeze', 'close'], bad: ['take out']

            bad_contact_verb_label_list = ['take out', 'put in', 'close', 'apply', 'open'] # good: ['place', 'squeeze', 'read', 'spray', 'pour', 'grab']
            if verb_label in bad_contact_verb_label_list:
                continue
            new_db.append(aid)

        self.db = new_db
        self.split = self.db

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
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

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list if key in [*sample_id_to_split_id]]
            self.split = new_split

    def __len__(self):
        return len(self.split)


    def __getitem__(self, index):
        aid = self.split[index]
        subject_name = aid.split('/')[0] + '_ego' # we only use ego split for H2O dataset
        seq_name = aid.split('/')[1]
        obj_id = aid.split('/')[2]
        cam_name = aid.split('/')[3]
        img_name = aid.split('/')[5]
        img_id = img_name.split('.png')[0]

        sample_id = f'{subject_name}-{seq_name}-{obj_id}-{cam_name}-{img_id}'

        orig_img_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'rgb', f'{img_id}.png')

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand_r = annot_data['bbox_hand_r']
            mano_valid = annot_data['mano_valid']

            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
        else:
            # Verb label
            verb_label_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'verb_label', f'{img_id}.txt')
            with open(verb_label_path) as f:
                verb_label = verb_label_list[int(f.read().strip())] # good: ['place', 'squeeze', 'close'], bad: ['take out']

            # Hand
            hand_pose_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'hand_pose_mano', f'{img_id}.txt')
            with open(hand_pose_path) as f:
                hand_params = np.array(f.read().strip().split(), dtype=np.float32)
            
            hand_param_l_valid = hand_params[0] == 1.
            hand_trans_l = hand_params[1:4]
            hand_pose_l = hand_params[4:52]
            hand_shape_l = hand_params[52:62]

            hand_param_r_valid = hand_params[62] == 1.
            hand_trans_r = hand_params[63:66]
            hand_pose_r = hand_params[66:114]
            hand_shape_r = hand_params[114:124]

            if hand_param_l_valid and hand_param_r_valid:
                mano_valid = np.ones((1), dtype=np.float32)
            else:
                mano_valid = np.zeros((1), dtype=np.float32)

            # flat_hand_mean=True -> flat_hand_mean=False (only use when the dataset is based on flat_hand_mean=True)
            hand_pose_l = change_flat_hand_mean(hand_pose_l, remove=True, side='left')
            hand_pose_r = change_flat_hand_mean(hand_pose_r, remove=True, side='right')

            mano_param_l = {'pose': hand_pose_l, 'shape': hand_shape_l, 'trans': hand_trans_l[:, None], 'hand_type': 'left'}
            mano_param_r = {'pose': hand_pose_r, 'shape': hand_shape_r, 'trans': hand_trans_r[:, None], 'hand_type': 'right'}
            cam_intr_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'cam_intrinsics.txt')
            with open(cam_intr_path) as f:
                cam_intr = np.array(f.read().strip().split(), dtype=np.float32)
            cam_param = {'focal': [cam_intr[0], cam_intr[1]], 'princpt': [cam_intr[2], cam_intr[3]]}

            mano_mesh_cam_l, mano_joint_cam_l, mano_pose_l, mano_shape_l, mano_trans_l = process_human_model_output_orig(mano_param_l, cam_param)
            mano_mesh_cam_r, mano_joint_cam_r, mano_pose_r, mano_shape_r, mano_trans_r = process_human_model_output_orig(mano_param_r, cam_param)
            mano_mesh_img_l = cam2pixel(mano_mesh_cam_l, cam_param['focal'], cam_param['princpt'])[:, :2]
            mano_mesh_img_r = cam2pixel(mano_mesh_cam_r, cam_param['focal'], cam_param['princpt'])[:, :2]

            # Bounding box
            bbox_ho = get_bbox(np.concatenate((mano_mesh_img_l, mano_mesh_img_r), axis=0), np.ones(len(mano_mesh_img_l)+len(mano_mesh_img_r)), expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)
            bbox_hand_l = get_bbox(mano_mesh_img_l, np.ones(len(mano_mesh_img_l)), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio)
            bbox_hand_r = get_bbox(mano_mesh_img_r, np.ones(len(mano_mesh_img_r)), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio)

            mesh_hand_l = trimesh.Trimesh(mano_mesh_cam_l, mano.layer['left'].faces)
            mesh_hand_r = trimesh.Trimesh(mano_mesh_cam_r, mano.layer['right'].faces)

            # Object
            obj_pose_rt_path = os.path.join(self.data_dir, subject_name, seq_name, obj_id, cam_name, 'obj_pose_rt', f'{img_id}.txt')
            with open(obj_pose_rt_path) as f:
                obj_pose_rt = np.array(f.read().strip().split(), dtype=np.float32)
                obj_class_id, obj_transform = obj_pose_rt[0], obj_pose_rt[1:].reshape(4, 4)

            obj_class_name = obj_class_list[int(obj_class_id)]
            if obj_class_name != 'spray':
                obj_rest_mesh_path = os.path.join(self.data_dir, 'object', obj_class_name, f'{obj_class_name}.obj')
            else:
                obj_rest_mesh_path = os.path.join(self.data_dir, 'object', obj_class_name, f'lotion_{obj_class_name}.obj')
            mesh_obj = trimesh.load(obj_rest_mesh_path)
            mesh_obj.vertices = apply_homogeneous_transformation(torch.from_numpy(mesh_obj.vertices).float(), torch.from_numpy(obj_transform))

            mesh_others = mesh_hand_l + mesh_obj

            if False:
                annot_data = dict(sample_id=sample_id, bbox_ho=bbox_ho, bbox_hand_l=bbox_hand_l, bbox_hand_r=bbox_hand_r, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)

            # Contact
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_r, mesh_others, cfg.MODEL.c_thres)
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
    dataset_name = 'H2O'
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