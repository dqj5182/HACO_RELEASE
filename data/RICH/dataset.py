import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import glob
import pickle
import trimesh
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
import xml.etree.ElementTree as ET
from smplx.lbs import transform_mat

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.transforms import cam2pixel, apply_homogeneous_transformation_np
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(db, index):
    aid = db[index]
    seq_name = aid.split('/')[-3]
    seq_loc_name = seq_name.split('_')[0]
    annot_name = seq_name.split('_')[1]
    cam_name = aid.split('/')[-2]
    cam_id = int(cam_name.split('cam_')[-1])
    img_name = aid.split('/')[-1].split('.jpeg')[0] # we used jpg version
    img_annot_name = img_name.split('_')[0]
    sample_id = f'{seq_name}-{cam_name}-{img_name}'
    return sample_id


def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
    k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2


class CalibratedCamera(nn.Module):

    def __init__(self, calib_path='', rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None, 
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(CalibratedCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.calib_path = calib_path
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        import os.path as osp
        if not osp.exists(calib_path):
            raise FileNotFoundError('Could''t find {}.'.format(calib_path))
        else:
            focal_length_x, focal_length_y, center, rotation, translation, cam_center, _, _ \
                    = extract_cam_param_xml(xml_path=calib_path, dtype=dtype)
        
        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],               
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],                
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        rotation = rotation.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=False)
    
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = translation.view(3, -1).repeat(batch_size, 1, 1).squeeze(dim=-1)
        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter('translation', translation)
        
        cam_center = nn.Parameter(cam_center, requires_grad=False)
        self.register_parameter('cam_center', cam_center)

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points



def extract_camera_intrinsics(xml_file_path):
    """
    Extracts the camera intrinsics matrix from an XML file.

    Args:
        xml_file_path (str): Path to the XML file.

    Returns:
        np.ndarray: 3x3 camera intrinsics matrix.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract the Intrinsics matrix
    intrinsics_data = root.find("./Intrinsics/data").text.strip().split()
    intrinsics_matrix = np.array(intrinsics_data, dtype=float).reshape(3, 3)

    return intrinsics_matrix



class RICH(Dataset):
    def __init__(self, transform, data_split):
        super(RICH, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'rich'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'RICH')
        self.data_dir = os.path.join(self.root_path, 'data')

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        # SMPL, SMPLX, MANO conversion mappings
        smpl_to_smplx_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smpl_to_smplx.pkl')
        smplx_mano_mapping_path = os.path.join('data', 'base_data', 'conversions', 'smplx_to_mano.pkl')

        with open(smpl_to_smplx_mapping_path, 'rb') as f:
            self.smpl_to_smplx_mapping = pickle.load(f)

        with open(smplx_mano_mapping_path, 'rb') as f:
            self.smplx_to_mano_mapping = pickle.load(f)
            self.smplx_to_mano_mapping_l = self.smplx_to_mano_mapping["left_hand"]
            self.smplx_to_mano_mapping_r = self.smplx_to_mano_mapping["right_hand"]

        db = glob.glob(f"{self.data_dir}/images_jpg_subset/{data_split}/*/*/*.jpeg", recursive=True)
        self.db = []
        for each_db in db:
            seq_name = each_db.split('/')[-3]
            seq_loc_name = seq_name.split('_')[0]
            annot_name = seq_name.split('_')[1]
            cam_name = each_db.split('/')[-2]
            cam_id = int(cam_name.split('cam_')[-1])
            img_name = each_db.split('/')[-1].split('.jpeg')[0] # we used jpg version
            img_annot_name = img_name.split('_')[0]
            camera_path = os.path.join(self.data_dir, 'scan_calibration', seq_loc_name, 'calibration', f'{cam_id:03d}.xml')
            contact_mesh_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.obj')
            contact_mesh_data_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.pkl')
            if not os.path.exists(camera_path):
                continue
            if not os.path.exists(contact_mesh_path):
                continue
            if not os.path.exists(contact_mesh_data_path):
                continue

            # Skip contact_h if no contact at all for evaluation
            if data_split == 'test':
                sample_id = f'{seq_name}-{cam_name}-{img_name}'
                contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
                if contact_h.sum() == 0.:
                    continue
            self.db.append(each_db)

        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_db_id = {}
            for db_idx in range(len(self.db)):
                each_sample_id = get_sample_id(self.db, db_idx)
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
        seq_name = aid.split('/')[-3]
        seq_loc_name = seq_name.split('_')[0]
        annot_name = seq_name.split('_')[1]
        cam_name = aid.split('/')[-2]
        cam_id = int(cam_name.split('cam_')[-1])
        img_name = aid.split('/')[-1].split('.jpeg')[0] # we used jpg version
        img_annot_name = img_name.split('_')[0]
        sample_id = f'{seq_name}-{cam_name}-{img_name}'

        orig_img_path = os.path.join(self.data_dir, 'images_jpg_subset', self.data_split, seq_name, cam_name, f'{img_name}.jpeg')

        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)

        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand_r = annot_data['bbox_ho']
            cam_param = annot_data['cam_param']

            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
        else:
            # Camera
            camera_path = os.path.join(self.data_dir, 'scan_calibration', seq_loc_name, 'calibration', f'{cam_id:03d}.xml')
            cam = CalibratedCamera(calib_path=camera_path)
            cam_param = {'focal': [cam.focal_length_x.item(), cam.focal_length_y.item()], 'princpt': [cam.center[0][0].item(), cam.center[0][1].item()]}


            # Load contact
            contact_mesh_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.obj')
            contact_mesh_data_path = os.path.join(self.data_dir, 'hsc', self.data_split, seq_name, img_annot_name, f'{annot_name}.pkl')

            contact_mesh = trimesh.load(contact_mesh_path) # SMPLX
            contact_mesh_data = np.load(contact_mesh_data_path, allow_pickle=True) # ['contact', 'closest_triangles_id', 's2m_dist_id'] -> SMPL format
            
            contact_smplx = np.matmul(self.smpl_to_smplx_mapping['matrix'], contact_mesh_data['contact'])
            contact_h = contact_smplx[self.smplx_to_mano_mapping_r]
            contact_data = dict(contact_h=contact_h)

            # Extrinsic transformation
            ext_camera_transform = transform_mat(cam.rotation, cam.translation.unsqueeze(dim=-1))[0].detach().cpu().numpy()
            contact_mesh.vertices = apply_homogeneous_transformation_np(contact_mesh.vertices, ext_camera_transform)


            # Hand
            mesh_hand_r_cam = contact_mesh.vertices[self.smplx_to_mano_mapping_r]
            mesh_hand_r_img = cam2pixel(mesh_hand_r_cam, cam_param['focal'], cam_param['princpt'])
            bbox_hand_r = get_bbox(mesh_hand_r_img, np.ones(len(mesh_hand_r_img)), expansion_factor=cfg.DATASET.hand_scene_bbox_expand_ratio)

            if True:
                annot_data = dict(sample_id=sample_id, mano_param={}, cam_param=cam_param, joint_cam=mesh_hand_r_cam, joint_img=mesh_hand_r_img, joint_valid=np.array([]), obj_cam=np.array([]), obj_img=np.array([]), bbox_hand=np.array([]), bbox_obj=np.array([]), bbox_ho=bbox_hand_r, mano_valid=mano_valid)
                np.savez(annot_data_path, **annot_data)


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand_r, self.data_split, enforce_flip=False)
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
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid, orig_img=orig_img)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)
        





if __name__ == "__main__":
    dataset_name = 'RICH'
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
            contact_data_save_path = os.path.join(contact_data_save_root_path, f'{sample_id}.npy')

            if os.path.exists(contact_data_save_path):
                continue
            contact_h = data['targets_data']['contact_data']['contact_h'][0].tolist()
            contact_h = np.array(contact_h, dtype=int)

            np.save(contact_data_save_path, contact_h)
    else:
        raise NotImplementedError