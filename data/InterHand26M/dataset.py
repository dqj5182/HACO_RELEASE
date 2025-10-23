import os
import json
import zlib
import pickle
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact
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



# Main dataloader code for InterHand26M dataset
class InterHand26M(Dataset):
    def __init__(self, transform, data_split, trans_test='gt'):
        super(InterHand26M, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'interhand26m'

        self.data_split = data_split # train, test, val
        self.root_path = root_path = os.path.join('data', 'InterHand26M')
        self.img_path = os.path.join(self.root_path, 'images')
        self.annot_dir = annot_dir = os.path.join(self.root_path, 'annotations')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # Major data path
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

        # COCO Data — Optimized Caching
        cache_path = os.path.join(self.annot_dir, data_split, f'InterHand2.6M_{data_split}_data.pkl')
        with open(cache_path, 'rb') as f:
            self.db_data = pickle.loads(zlib.decompress(f.read()))

        # Minimal Indexing for Faster Access
        self.db = COCO()
        self.db.anns = {ann['id']: ann for ann in self.db_data['annotations']}
        self.db.imgs = {img['id']: img for img in self.db_data['images']}

        # Organize db id based on split
        db_split_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            self.db_anns_keys = json.load(f)
            self.db_anns_keys = [int(item) for item in self.db_anns_keys]

        # Sort contact by difficulty (Balanced contact sampling)
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
            sample_id_difficulty_list = get_contact_difficulty_sample_id(sample_id_to_db_anns_id, self.contact_data_path, contact_means_path)

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


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_hand_r = annot_data['bbox_hand']
        cam_param = annot_data['cam_param']

        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################
        

        ############################### PROCESS CROP AND AUGMENTATION ###############################
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
        ############################### PROCESS CROP AND AUGMENTATION ###############################


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=str(sample_id), mano_valid=mano_valid)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)