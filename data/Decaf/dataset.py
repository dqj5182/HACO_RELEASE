import os
import cv2
import glob
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.func_utils import load_img
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(split, index):
    aid = split[index]
    seq_name = aid.split('_')[0]
    cam_name = aid.split('_')[1]
    img_name = aid.split('_')[2]
    sample_id = aid
    return sample_id


class Decaf(Dataset):
    def __init__(self, transform, data_split):
        super(Decaf, self).__init__()
        self.__dict__.update(locals())

        self.transfrom = transform
        dataset_name = 'decaf'

        self.data_split = data_split
        self.root_path = root_path = 'data/Decaf'
        self.data_dir = os.path.join(self.root_path, 'data')

        self.mano_joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')
        
        seq_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        cam_list = ['084', '100', '102', '108', '110', '111', '121', '122']

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 5
        else:
            sampling_ratio = 1

        # Make db
        contact_rh_data = {seq: np.load(os.path.join(self.data_dir, self.data_split, 'contacts', seq, 'contacts_rh.npy')) for seq in seq_list}
        
        bb_rh_data = {
            f"{seq}_{cam}": np.load(os.path.join(self.data_dir, self.data_split, 'right_hand_bbs', seq, f'{cam}.npy'))
            for seq in seq_list
            for cam in cam_list
        }


        self.db_contact = {}
        self.db_bb = {}
        for file in glob.glob("data/Decaf/data/train/params/*/params.pkl"):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
            seq = file.split('/')[-2]
            
            if seq not in contact_rh_data:
                continue  # Skip sequences that have no contact data
            
            valid_indices = np.where(contact_rh_data[seq].sum(axis=1) > 0)[0]
            
            for idx in valid_indices:
                img = list(data.keys())[idx]
                for cam in cam_list:
                    orig_img_path = os.path.join(self.data_dir, self.data_split, 'images', seq, cam, f'{img}.jpg')
                    if not os.path.exists(orig_img_path):
                        continue
                    self.db_contact[f"{seq}_{cam}_{img}"] = contact_rh_data[seq][idx]
                    self.db_bb[f"{seq}_{cam}_{img}"] = bb_rh_data[f"{seq}_{cam}"][idx]

        # Apply sampling ratio
        self.db_contact = {key: self.db_contact[key] for key in list(self.db_contact.keys())[::sampling_ratio]}
        self.split = [*self.db_contact]

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

            new_split = [sample_id_to_split_id[key] for key in sample_id_difficulty_list if key in [*sample_id_to_split_id]]
            self.split = new_split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        aid = self.split[index]
        seq_name = aid.split('_')[0]
        cam_name = aid.split('_')[1]
        img_name = aid.split('_')[2]
        sample_id = aid

        orig_img_path = os.path.join(self.data_dir, self.data_split, 'images', seq_name, cam_name, f'{img_name}.jpg')
        
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        contact_rh = self.db_contact[sample_id].astype(np.float32)
        bbox_rh = self.db_bb[sample_id].tolist() # GT bbox is in [x_min, y_min, x_max, y_max]
        # Change from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
        bbox_rh = np.array([bbox_rh[0], bbox_rh[1], bbox_rh[2]-bbox_rh[0], bbox_rh[3]-bbox_rh[1]])

        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_rh, self.data_split, enforce_flip=False)
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
        
        mano_valid = np.ones((1), dtype=np.float32)


        contact_data = dict(contact_h=contact_rh)
        

        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)



if __name__ == "__main__":
    dataset_name = 'Decaf'
    data_split = 'train'
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