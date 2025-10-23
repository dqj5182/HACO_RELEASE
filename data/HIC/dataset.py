import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact



# Main dataloader code for HIC dataset
class HIC(Dataset):
    def __init__(self, transform, data_split):
        super(HIC, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'hic'

        self.data_split = data_split
        self.root_path = root_path = os.path.join('data', 'HIC')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')

        # inter: 01, 02, 03, 04, 05, 06, 07 | single: 08, 09, 10, 11 | Single-HOI: 15, 19, 20, 21 | Inter-HOI: 16, 17, 18
        self.inter_seq_names = ['01', '02', '03', '04', '05', '06', '07']
        self.single_seq_names = ['08', '09', '10', '11']
        self.single_hoi_seq_names = ['15', '19', '20', '21']
        self.inter_hoi_seq_names = ['16', '17', '18']

        # Split train/test set (we only use inter seq)
        self.train_seq_names = ['01', '02', '03', '04', '05', '06']
        self.test_seq_names = ['07']

        self.cam_param = {'focal': [525.0, 525.0], 'princpt': [319.5, 239.5]} # this is fixed for HIC dataset

        # Organize db id based on split
        db_split_path = os.path.join(self.root_path, 'splits', f'{self.data_split}.json')
        with open(db_split_path, 'r') as f:
            self.db = json.load(f)


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


        ################################## LOAD ANNOTATION DATA #####################################
        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')

        annot_data = np.load(annot_data_path, allow_pickle=True)
        bbox_hand_r = annot_data['bbox_hand']
        cam_param = annot_data['cam_param']

        contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
        contact_data = dict(contact_h=contact_h)
        ################################## LOAD ANNOTATION DATA #####################################

        
        ############################### PROCESS CROP AND AUGMENTATION ################################
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
        ############################### PROCESS CROP AND AUGMENTATION ################################


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=str(sample_id), mano_valid=mano_valid)


        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)