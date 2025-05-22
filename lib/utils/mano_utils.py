'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model.

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location),
  or a dictionary object.

'''
import os
import cv2
import torch
import numpy as np
import pickle
import chumpy as ch
from chumpy.ch import MatVecMult


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
             for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel()
                           for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s), ))


def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in [
            'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
            'betas', 'J'
    ]:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert (posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(pose_map_res)
    else:
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd_add = dd['posedirs'].dot(pose_map_res)
        dd['v_posed'] = dd['v_template'] + dd_add

    return dd



def get_mano_pca_basis(ncomps=45, use_pca=True, side='right', mano_root='data/base_data/human_models/mano'):
    if use_pca:
        ncomps = ncomps
    else:
        ncomps = 45

    if side == 'right':
        mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
    elif side == 'left':
        mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
    smpl_data = ready_arguments(mano_path)
    hands_components = smpl_data['hands_components']
    selected_components = hands_components[:ncomps]
    th_selected_comps = selected_components

    return torch.tensor(th_selected_comps, dtype=torch.float32)



def change_flat_hand_mean(hand_pose, remove=True, side='right', mano_root='data/base_data/human_models/mano'):
    if side == 'right':
        mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
    elif side == 'left':
        mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
    smpl_data = ready_arguments(mano_path)

    # Get hand mean
    hands_mean = smpl_data['hands_mean']
    hands_mean = hands_mean.copy() # hands_mean: (45)

    if remove:
        hand_pose[3:] = hand_pose[3:] - hands_mean
    else:
        hand_pose[3:] = hand_pose[3:] + hands_mean
    return hand_pose