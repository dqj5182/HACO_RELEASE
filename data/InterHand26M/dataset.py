import os
import os.path as osp
import numpy as np
import json
import pickle
import zlib
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
from lib.utils.func_utils import load_img, get_bbox, process_bbox
from lib.utils.transforms import world2cam, cam2pixel
from lib.utils.preprocessing import augmentation_contact, process_human_model_output_orig
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(db, db_anns_keys, index):
    aid = db_anns_keys[index]
    ann = db.anns[aid]
    image_id = ann['image_id']
    img = db.loadImgs(image_id)[0]
    
    capture_id = img['capture']
    seq_name = img['seq_name']
    cam = img['camera']
    frame_idx = img['frame_idx']
    img_width, img_height = img['width'], img['height']

    sample_id = image_id
    return sample_id


class InterHand26M(Dataset):
    def __init__(self, transform, data_split, trans_test='gt'):
        super(InterHand26M, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'interhand26m'

        self.data_split = data_split # train, test, val
        self.root_path = root_path = osp.join('data', 'InterHand26M')
        self.data_dir = os.path.join(self.root_path, 'data')
        self.img_path = os.path.join(self.root_path, 'images')
        self.annot_dir = annot_dir = os.path.join(self.root_path, 'annotations')

        # Major data path
        annot_data_path = os.path.join(annot_dir, data_split, 'InterHand2.6M_' + data_split + '_data.json')
        annot_camera_path = os.path.join(annot_dir, data_split, 'InterHand2.6M_' + data_split + '_camera.json')
        annot_joint_3d_path = os.path.join(annot_dir, data_split, 'InterHand2.6M_' + data_split + '_joint_3d.json')
        annot_mano_params_path = os.path.join(annot_dir, data_split, 'InterHand2.6M_' + data_split + '_MANO_NeuralAnnot.json')

        # RootNet data path
        if data_split == 'val':
            self.rootnet_output_path = os.path.join(self.root_path, 'rootnet_output/rootnet_interhand2.6m_output_val.json')
        else:
            self.rootnet_output_path = os.path.join(self.root_path, 'rootnet_output/rootnet_interhand2.6m_output_test.json')

        self.joint_num = 21 # single hand
        self.joints_name = ('R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1', 'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1', 'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1', 'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1', 'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1', 'R_Wrist', 'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1', 'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1', 'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1', 'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1', 'L_Wrist')
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}

        # Load camera
        with open(annot_camera_path) as f:
            self.cameras = json.load(f)

        # Load 3D joints
        with open(annot_joint_3d_path) as f:
            self.joints = json.load(f)

        # Load MANO params
        with open(annot_mano_params_path) as f:
            self.mano_params = json.load(f)

        # Load rootnet
        if (data_split == 'val' or data_split == 'test') and trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            self.rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                self.rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        # COCO Data — Optimized Caching
        cache_path = os.path.join(self.annot_dir, data_split, f'InterHand2.6M_{data_split}_data.pkl')

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.db_data = pickle.loads(zlib.decompress(f.read()))

            # Minimal Indexing for Faster Access
            self.db = COCO()
            self.db.anns = {ann['id']: ann for ann in self.db_data['annotations']}
            self.db.imgs = {img['id']: img for img in self.db_data['images']}
        else:
            self.db = COCO(annot_data_path)
            self.db_data = {
                'annotations': self.db.dataset['annotations'],
                'images': self.db.dataset['images']
            }

            with open(cache_path, 'wb') as f:
                f.write(zlib.compress(pickle.dumps(self.db_data)))

        self.db_anns_keys = list(self.db.anns.keys())

        # Optimized Filtering Using `set()` for Fast Membership Checking
        if self.use_preprocessed_data and self.data_split == 'train':
            contact_files = set(os.listdir(self.contact_data_path))
            # self.db_anns_keys = [key for key in self.db_anns_keys if f'{key}.npy'] # WARNING: when saving contact
            self.db_anns_keys = [key for key in self.db_anns_keys if f'{key}.npy' in contact_files] # after saving contact


        if data_split == 'train':
            sampling_ratio = 1 # WARNING: 10 when saving contact, 1 after saving contact
        else:
            sampling_ratio = 1
        self.db_anns_keys = self.db_anns_keys[::sampling_ratio]

        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_db_anns_id = {}
            for db_anns_idx in range(len(self.db_anns_keys)):
                each_sample_id = get_sample_id(self.db, self.db_anns_keys, db_anns_idx)
                if each_sample_id in sample_id_to_db_anns_id:
                    import pdb; pdb.set_trace()
                    raise KeyError(f"Key {each_sample_id} already exists in the dictionary.")
                else:
                    sample_id_to_db_anns_id[each_sample_id] = self.db_anns_keys[db_anns_idx]

            contact_means_path = os.path.join(f'data/base_data/contact_data/{dataset_name}/contact_means_{dataset_name}.npy')
            sample_id_difficulty_list = get_contact_difficulty_sample_id(self.contact_data_path, contact_means_path)

            new_db = [int(sample_id_to_db_anns_id[int(key)]) for key in sample_id_difficulty_list]
            self.db_anns_keys = new_db

    def __len__(self):
        return len(self.db_anns_keys)

    def __getitem__(self, index):
        aid = self.db_anns_keys[index]
        ann = self.db.anns[aid]
        image_id = ann['image_id']
        img = self.db.loadImgs(image_id)[0]
        
        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']
        img_width, img_height = img['width'], img['height']

        sample_id = image_id

        # Load image
        orig_img_path = os.path.join(self.img_path, self.data_split, img['file_name'])
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
            # apply camera extrinsic
            cam_param = self.cameras[str(capture_id)]
            t, R = np.array(cam_param['campos'][str(cam)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t

            focal, princpt = torch.FloatTensor(cam_param['focal'][str(cam)]).cuda()[None,:], torch.FloatTensor(cam_param['princpt'][str(cam)]).cuda()[None,:]
            cam_param = {'R': R, 't': t/1000, 'focal': focal[0].tolist(), 'princpt': princpt[0].tolist()}

            ###################################### READ ANNOTATION #######################################
            # MANO params
            if str(frame_idx) not in [*self.mano_params[str(capture_id)]]: # skip missing annotations
                return []

            # Hand meshes
            mano_param = self.mano_params[str(capture_id)][str(frame_idx)]
            if (mano_param['right'] is None) or (mano_param['left'] is None): # two hands data is needed
                return []

            mano_param_l = {'pose': mano_param['left']['pose'].copy(), 'shape': mano_param['left']['shape'].copy(), 'trans': mano_param['left']['trans'].copy(), 'hand_type': 'left'}
            mano_param_r = {'pose': mano_param['right']['pose'].copy(), 'shape': mano_param['right']['shape'].copy(), 'trans': mano_param['right']['trans'].copy(), 'hand_type': 'right'}

            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            ###################################### READ ANNOTATION #######################################

            ################################## PROCESS JOINTS & CORNERS ##################################
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)

            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

            joint_world = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid.reshape(-1,1)==0, (1,3))] = 1. # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal[0].detach().cpu().numpy(), princpt[0].detach().cpu().numpy())

            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

            joint_cam_r, joint_img_r, hand_joint_valid_r = joint_cam[self.joint_type['right']], joint_img[self.joint_type['right']], joint_valid[self.joint_type['right']]
            joint_cam_l, joint_img_l, hand_joint_valid_l = joint_cam[self.joint_type['left']], joint_img[self.joint_type['left']], joint_valid[self.joint_type['left']]
            hand_valid_r, hand_valid_l = hand_joint_valid_r.sum() > 0, hand_joint_valid_l.sum() > 0
            ################################## PROCESS JOINTS & CORNERS ##################################

            if hand_joint_valid_r.sum() == 0 or hand_joint_valid_l.sum() == 0: # skip via hand joint valid
                return []

            bbox_hand = get_bbox(np.concatenate((joint_img_r[hand_joint_valid_r==1, :], joint_img_l[hand_joint_valid_l==1, :]), axis=0), np.ones(len(joint_img_r[hand_joint_valid_r==1, :])+len(joint_img_l[hand_joint_valid_l==1, :])), expansion_factor=2.0) # expand 1.3 as joints are very tight bbox
            bbox_hand_r = get_bbox(joint_img_r[hand_joint_valid_r==1, :], np.ones(len(joint_img_r[hand_joint_valid_r==1, :])), expansion_factor=2.0) # expand 1.3 as joints are very tight bbox
            bbox_hand_l = get_bbox(joint_img_l[hand_joint_valid_l==1, :], np.ones(len(joint_img_l[hand_joint_valid_l==1, :])), expansion_factor=2.0) # expand 1.3 as joints are very tight bbox

            ######################################## PROCESS BBOX ########################################
            img_l, img2bb_trans_l, bb2img_trans_l, rot_l, do_flip_l, _ = augmentation_contact(orig_img.copy(), bbox_hand_l, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
            ######################################## PROCESS BBOX ########################################

            if False:
                annot_data = dict(sample_id=sample_id, mano_param=mano_param_r, cam_param=cam_param, joint_cam=joint_cam_r, joint_img=joint_img_r, joint_valid=hand_joint_valid_r, obj_cam=joint_cam_l, obj_img=joint_img_l, bbox_hand=bbox_hand_r, bbox_obj=bbox_hand_l, bbox_ho=bbox_hand, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)


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




        if not self.use_preprocessed_data or (self.data_split != 'train'):
            if (self.data_split == 'val' or self.data_split == 'test') and trans_test == 'rootnet':
                bbox = np.array(self.rootnet_result[str(aid)]['bbox'],dtype=np.float32)
            else:
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, cfg.MODEL.input_img_shape, (img_height, img_width))

            
            #################################### PROCESS MANO JOINTS ####################################
            mano_mesh_cam_left, mano_joint_cam_left, mano_pose_left, mano_shape_left, mano_trans_left = process_human_model_output_orig(mano_param_l, cam_param)
            mano_mesh_cam_right, mano_joint_cam_right, mano_pose_right, mano_shape_right, mano_trans_right = process_human_model_output_orig(mano_param_r, cam_param)

            hand_mesh_left = trimesh.Trimesh(mano_mesh_cam_left, mano.layer['left'].faces)
            hand_mesh_right = trimesh.Trimesh(mano_mesh_cam_right, mano.layer['right'].faces)
            #################################### PROCESS MANO JOINTS ####################################



            
            ########################### GROUND-TRUTH HAND MESH ############################
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(hand_mesh_right, hand_mesh_left, cfg.MODEL.c_thres_ih)
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
            meta_info = dict(sample_id=str(sample_id), mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)





if __name__ == "__main__":
    dataset_name = 'InterHand26M'
    data_split = 'train' # This dataset only has train set
    trans_test = 'gt' # ['gt', 'rootnet']
    task = 'debug'

    transform = transforms.ToTensor()

    dataset = eval(dataset_name)(transform, data_split, trans_test)
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