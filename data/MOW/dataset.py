import os
import cv2
import json
import trimesh
import numpy as np
import point_cloud_utils as pcu

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from lib.core.config import cfg
from lib.utils.preprocessing import augmentation_contact, process_human_model_output_orig, mask2bbox
from lib.utils.func_utils import load_img
from lib.utils.mesh_utils import center_vertices, load_obj_nr
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.human_models import mano



class MOW(Dataset):
    def __init__(self, transform, data_split):
        super(MOW, self).__init__()
        self.__dict__.update(locals())

        self.transfrom = transform
        dataset_name = 'mow'

        self.data_split = data_split
        self.root_path = root_path = 'data/MOW'

        self.data_dir = os.path.join(self.root_path, 'data')
        self.split_dir = os.path.join(self.root_path, 'splits') # This inherits IHOI
        self.watertight_obj_model_dir = os.path.join(self.data_dir, 'watertight_models')
        os.makedirs(self.watertight_obj_model_dir, exist_ok=True)

        with open(os.path.join(self.data_dir, 'poses.json'), 'r') as f:
            annos = json.load(f)

        self.db = {}
        for anno in annos:
            self.db[anno['image_id']] = anno
        del annos

        self.split = {'train': np.load('data/MOW/splits/mow_train.npy').tolist(), 'test': np.load('data/MOW/splits/mow_test.npy').tolist()}
        self.length = len(self.split[data_split])

        self.use_preprocessed_data = True
        self.use_preprocessed_watertight_mesh = True
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.contact_data_path, exist_ok=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample_id = self.split[self.data_split][index]
        ann = self.db[sample_id]
        image_id = ann['image_id']

        img_path = os.path.join(self.data_dir, 'images', f'{image_id}.jpg')
        orig_img = load_img(img_path)

        mask_ho_path = os.path.join(self.data_dir, 'masks/both', f'{image_id}.jpg')
        mask_ho = (cv2.imread(mask_ho_path) > 128)[:, :, 0]
        bbox_ho = mask2bbox(mask_ho, expansion_factor=cfg.DATASET.ho_bbox_expand_ratio)


        ############################### PROCESS CROP AND AUGMENTATION ################################
        # Crop image
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


        mano_valid = np.ones((1), dtype=np.float32)


        if not self.use_preprocessed_data:
            hand_t = ann['hand_t']
            hand_pose = ann['hand_pose']
            hand_R = ann['hand_R']
            hand_s = ann['hand_s']
            hand_trans = ann['trans']

            obj_instance = ann['obj_url'].split('/')[-1].split('.obj')[0]
            obj_rest_mesh_path = os.path.join(self.data_dir, 'models', f'{obj_instance}.obj')
            obj_R = np.array(ann['R']).reshape(3, 3)
            obj_t = np.array(ann['t']).reshape((1, 3))
            obj_s = np.array(ann['s'], dtype=np.float32)
            obj_name = ann['obj_name']

            mano_param = {'pose': np.array(hand_pose), 'shape': np.zeros(1), 'trans': np.array(hand_trans), 'hand_type': 'right'}
            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape, mano_trans = process_human_model_output_orig(mano_param, {}) # mano_mesh_cam is exactly same with output.vertices in official MOW
            
            mano_mesh_cam = (mano_mesh_cam @ np.array(hand_R).reshape(3, 3))
            mano_mesh_cam += np.array(hand_t)[:, None].transpose(1, 0)
            mano_mesh_cam *= np.array(hand_s) # mano_mesh_cam is exactly same with hand.vertices in official MOW 
            hand_mesh = trimesh.Trimesh(mano_mesh_cam, mano.watertight_face['right'])
           
            obj_rest_verts, obj_rest_faces = load_obj_nr(obj_rest_mesh_path)
            obj_rest_verts, obj_rest_faces = obj_rest_verts.detach().cpu().numpy(), obj_rest_faces.detach().cpu().numpy()
            obj_rest_mesh = trimesh.Trimesh(obj_rest_verts, obj_rest_faces)

            # Make object mesh watertight
            watertight_obj_model_path = os.path.join(self.watertight_obj_model_dir, f'{obj_instance}.obj')

            if self.use_preprocessed_watertight_mesh and os.path.exists(watertight_obj_model_path):
                mesh_obj_watertight = trimesh.load(watertight_obj_model_path)
                
                # post-process
                trimesh.repair.fix_normals(mesh_obj_watertight)
                trimesh.repair.fix_inversion(mesh_obj_watertight)
                trimesh.repair.fill_holes(mesh_obj_watertight)

                obj_rest_mesh = mesh_obj_watertight
            else:
                print('Building new watertight mesh!!!!')
                resolution = 50_000
                obj_rest_mesh.vertices, obj_rest_mesh.faces = pcu.make_mesh_watertight(obj_rest_mesh.vertices, obj_rest_mesh.faces, resolution)
                if not os.path.exists(watertight_obj_model_path):
                    _ = obj_rest_mesh.export(watertight_obj_model_path)

            obj_rest_verts, obj_rest_faces = center_vertices(obj_rest_mesh.vertices, obj_rest_mesh.faces)
            obj_verts = np.dot(obj_rest_verts, obj_R)
            obj_verts += obj_t
            obj_verts *= obj_s
            obj_mesh = trimesh.Trimesh(obj_verts, obj_rest_faces)

            # Contact data
            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(hand_mesh, obj_mesh, cfg.MODEL.c_thres_in_the_wild)
            contact_h = contact_h.astype(np.float32)
            contact_data = dict(contact_h=contact_h)

            if True:
                np.save(os.path.join(self.contact_data_path, f'{sample_id}.npy'), contact_h)
        else:
            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)


        input_data = dict(image=img)
        targets_data = dict(contact_data=contact_data)
        meta_info = dict(sample_id=sample_id, orig_img=orig_img, mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)