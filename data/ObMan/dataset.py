import os
import cv2
import torch
import numpy as np
import json
import trimesh
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
import point_cloud_utils as pcu

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.mesh_utils import fast_load_obj
from lib.utils.mano_utils import change_flat_hand_mean
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, pca_to_axis_angle
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact
from lib.utils.train_utils import get_contact_difficulty_sample_id


def get_sample_id(db, split, index):
    aid = split[index]
    ann = db.anns[aid]
    img_data = db.loadImgs(ann['image_id'])[0]
    sample_id = img_data['file_name']
    return sample_id, img_data



class ObMan(Dataset):
    def __init__(self, transform, data_split):
        super(ObMan, self).__init__()
        self.__dict__.update(locals())
        
        self.transfrom = transform
        dataset_name = 'obman'

        if data_split == 'train':
            data_split_name = 'train_87k'
        elif data_split == 'test':
            data_split_name = 'test_6k'
        else:
            raise NotImplementedError
        
        self.data_split = data_split
        self.root_path = root_path = 'data/ObMan'

        self.stage_split = data_split_name.split('_')[0]
        self.data_dir = os.path.join(self.root_path, 'data')
        self.split_dir = os.path.join(self.root_path, 'splits')
        self.annot_dir = os.path.join(self.root_path, 'annotations')
        self.obj_model_dir = os.path.join(self.root_path, 'object_models')
        self.watertight_obj_model_dir = os.path.join(self.root_path, 'object_models', 'watertight_meshes')

        with open(os.path.join(self.split_dir , f'{data_split_name}.json'), 'r') as f:
            self.split = json.load(f)

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 5
        else:
            sampling_ratio = 1

        self.split = [int(idx) for idx in self.split] #[::sampling_ratio]

        self.anno_file = os.path.join(self.annot_dir, f'{dataset_name}_{data_split}.json')
        self.img_source = os.path.join(self.data_dir, data_split, 'rgb')
        self.seg_source = os.path.join(self.data_dir, data_split, 'segm')
        self.input_img_shape = cfg.MODEL.input_img_shape # should be (256, 256)

        self.cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
                                  [0., 0., 1.]]).astype(np.float32)
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)

        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            }
                        }
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')

        self.db = COCO(self.anno_file)

        self.start_point = 0
        self.end_point = len(self.split)
        self.length = self.end_point - self.start_point

        self.use_preprocessed_data = True
        self.use_preprocessed_watertight_mesh = True
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.contact_data_path, exist_ok=True)



        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id, _ = get_sample_id(self.db, self.split, split_idx)
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
        sample_id, img_data = get_sample_id(self.db, self.split, index)

        # Base path
        img_path = os.path.join(self.img_source, img_data['file_name'] + '.jpg')
        seg_path = os.path.join(self.seg_source, img_data['file_name'] + '.png')

        # Full image
        orig_img = load_img(img_path)
        orig_img_shape = orig_img.shape[:2]

        mano_valid = np.ones((1), dtype=np.float32)


        if not self.use_preprocessed_data or (self.data_split != 'train'):
            # Load official meta data
            with open(f'{self.data_dir}/{self.stage_split}/meta/{sample_id}.pkl', 'rb') as f:
                meta_data = pickle.load(f)

            obj_rest_mesh_path = meta_data['obj_path'].replace('/sequoia/data2/dataset/shapenet', self.obj_model_dir)
            obj_instance = obj_rest_mesh_path.split('/')[4] + '-' + obj_rest_mesh_path.split('/')[5]

            # Camera intrinsic parameter
            cam_param = {'focal': [self.cam_intr[0][0], self.cam_intr[1][1]], 'princpt': [self.cam_intr[0][2], self.cam_intr[1][2]]} # focal: [fx, fy], princpt: [cx, cy]

            # MANO params
            hand_poses = np.zeros(48)
            mano_pose = torch.zeros((1, 48))
            mano_pca = torch.from_numpy(meta_data['pca_pose']).float()
            mano_pose[:, 3:] = mano_pca
            hand_pose = mano_pose[0]

            hand_shape, hand_trans = meta_data['shape'], meta_data['trans'] # [10], [3]

            # Hand joints
            joint_cam = meta_data['coords_3d']
            joint_cam = self.cam_extr[:3, :3].dot(joint_cam.transpose()).transpose()
            root_joint = joint_cam[0]
            joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])[:, :2]
            joint_valid = np.ones(len(joint_cam))

            # Convert use_pca=True to use_pca=False (this is for unified MANO setting for HACO)
            hand_pose = pca_to_axis_angle(hand_pose[None])[0].detach().cpu().numpy()

            # Convert flat_hand_mean=True to flat_hand_mean=False (this is for unified MANO setting)
            hand_pose = change_flat_hand_mean(hand_pose, remove=True)

            mano_param = {'pose': hand_pose, 'shape': hand_shape, 'trans': hand_trans[:, None], 'hand_type': 'right'}

            # Object rot, trans
            obj_transform = meta_data['affine_transform']

            # This inherits ObMan to get accurate MANO mesh (ObMan does not have accurate MANO params but accurate MANO mesh)
            mano_mesh_cam = meta_data['verts_3d']
            mano_mesh_cam = self.cam_extr[:3, :3].dot(mano_mesh_cam.transpose()).transpose()
            mano_mesh_img = cam2pixel(mano_mesh_cam, cam_param['focal'], cam_param['princpt'])[:, :2]
            mesh_hand = trimesh.Trimesh(mano_mesh_cam, mano.layer['right'].faces)
            mano_joint_cam = np.dot(mano.joint_regressor, mano_mesh_cam)
            mano_joint_img = cam2pixel(mano_joint_cam, cam_param['focal'], cam_param['princpt'])[:, :2]

            # Segmentation
            hand_obj_seg = cv2.imread(seg_path)
            hand_seg = ((hand_obj_seg == 22).astype(float) + (hand_obj_seg == 24).astype(float))
            obj_seg = (hand_obj_seg == 100).astype(float)

            hand_full_seg = hand_seg[:, :, 1] == 1.
            hand_occ_seg = hand_seg[:, :, 0] == 1.

            obj_full_seg = obj_seg[:, :, 2] == 1.
            obj_occ_seg = obj_seg[:, :, 0] == 1.

            # Get hand mesh for depth rendering
            mesh_hand_watertight = trimesh.Trimesh(mano_mesh_cam, mano.watertight_face['right'])

            # Get object mesh
            watertight_obj_model_dir = os.path.join(self.watertight_obj_model_dir, obj_instance)

            if self.use_preprocessed_watertight_mesh and os.path.exists(watertight_obj_model_dir):
                mesh_obj_watertight = trimesh.load(os.path.join(watertight_obj_model_dir, 'model_normalized.obj'))
                
                # post-process
                trimesh.repair.fix_normals(mesh_obj_watertight)
                trimesh.repair.fix_inversion(mesh_obj_watertight)
                trimesh.repair.fill_holes(mesh_obj_watertight)

                obj_rest_mesh = mesh_obj_watertight # TODO: THIS IS NOT CORRECT FOR TEST SCENARIO WHERE VERTICES SHOULD BE DIRECTLY FROM GT OBJECT MESH. MANAGE THIS WITH IF STATEMENT
            else:
                print('Making watertight!!!!!')
                with open(obj_rest_mesh_path, 'r') as m_f:
                    obj_rest_mesh = fast_load_obj(m_f)[0]
                    obj_rest_mesh = trimesh.Trimesh(obj_rest_mesh['vertices'], obj_rest_mesh['faces'].astype(np.int16))
                # Make mesh_obj watertight: Takes quite a long time
                print('Building new watertight mesh!!!!')
                resolution = 50_000
                obj_rest_mesh.vertices, obj_rest_mesh.faces = pcu.make_mesh_watertight(obj_rest_mesh.vertices, obj_rest_mesh.faces, resolution)
                save_watertight_obj_model_dir = os.path.join(self.watertight_obj_model_dir, obj_instance)

                if not os.path.exists(os.path.join(save_watertight_obj_model_dir, 'model_normalized.obj')):
                    os.makedirs(save_watertight_obj_model_dir, exist_ok=True)
                    _ = obj_rest_mesh.export(os.path.join(save_watertight_obj_model_dir, 'model_normalized.obj'))

            obj_verts = np.array(obj_rest_mesh.vertices)
            obj_faces = np.array(obj_rest_mesh.faces)

            hom_obj_verts = np.concatenate([obj_verts, np.ones([obj_verts.shape[0], 1])], axis=1)
            obj_verts = obj_transform.dot(hom_obj_verts.T).T[:, :3]
            obj_verts = self.cam_extr[:3, :3].dot(obj_verts.transpose()).transpose()
            obj_verts = np.array(obj_verts).astype(np.float32)

            mesh_obj = trimesh.Trimesh(obj_verts, obj_faces)

            # Contact data
            if mesh_obj.is_watertight: # There are many meshes in ShapeNet that is non-watertight. We only care about watertight cases                    
                contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(mesh_hand_watertight, mesh_obj, cfg.MODEL.c_thres)
            else:
                contact_h, obj_coord_c, contact_valid, inter_coord_valid = np.zeros(mano.vertex_num, dtype=int), np.zeros((mano.vertex_num, 3), dtype=int), np.zeros((mano.vertex_num, 1), dtype=int), np.zeros(mano.vertex_num, dtype=int)
            contact_data = dict(contact_h=contact_h)


            # Load contact data
            contact_h, inter_coord_valid = contact_data['contact_h'], contact_data['inter_coord_valid']
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
            contact_h = contact_data['contact_h']



        ############################### PROCESS CROP AND AUGMENTATION ################################
        # Crop image
        bbox_ho = np.array([0, 0, 256, 256])
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_ho, self.data_split, enforce_flip=False)
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
    dataset_name = 'ObMan'
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