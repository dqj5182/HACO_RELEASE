import numpy as np

import torch

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.core.loss import VCBLoss, RegLoss, SmoothRegLoss

V_regressor_336 = torch.tensor(np.load(cfg.MODEL.V_regressor_336_path), dtype=torch.float32)
V_regressor_84 = torch.tensor(np.load(cfg.MODEL.V_regressor_84_path), dtype=torch.float32)
J_regressor = torch.tensor(mano.joint_regressor, dtype=torch.float32)

# Loss function
vcb_mesh_loss = VCBLoss(v_type='mesh')
vcb_mesh_336_loss = VCBLoss(v_type='mesh_336')
vcb_mesh_84_loss = VCBLoss(v_type='mesh_84')
vcb_joint_loss = VCBLoss(v_type='joint')
reg_loss = RegLoss()
smooth_reg_loss = SmoothRegLoss()


def compute_loss(preds, targets, epoch):
    total_loss = 0

    batch_size = len(preds['contact_out'])
    contact_means = np.load(cfg.MODEL.contact_means_path)
    contact_means = torch.tensor(contact_means)[None].repeat(batch_size, 1)
    regularization_loss = reg_loss(preds['contact_out'], contact_means)
    smooth_regularization_loss = smooth_reg_loss(preds['contact_out'], torch.tensor(mano.layer['right'].faces.astype(np.int32)))

    # Calculate loss
    contact_h_mesh = targets['contact_data']['contact_h']
    contact_h_336 = 1 * (torch.mm(contact_h_mesh, V_regressor_336.T) > 0)
    contact_h_84 = 1 * (torch.mm(contact_h_mesh, V_regressor_84.T) > 0)
    contact_h_joint = 1 * (torch.mm(contact_h_mesh, J_regressor.T) > 0)

    contact_mesh_loss = vcb_mesh_loss(preds['contact_out'], contact_h_mesh, epoch)
    contact_336_loss = vcb_mesh_336_loss(preds['contact_336_out'], contact_h_336, epoch)
    contact_84_loss = vcb_mesh_84_loss(preds['contact_84_out'], contact_h_84, epoch)
    contact_joint_loss = vcb_joint_loss(preds['contact_joint_out'], contact_h_joint, epoch)
    contact_loss = contact_mesh_loss + contact_336_loss + contact_84_loss + contact_joint_loss + 0.1 * regularization_loss + smooth_regularization_loss

    total_loss = contact_loss

    loss_dict = dict(total_loss=total_loss, 
                    contact_mesh_loss=contact_mesh_loss, 
                    contact_336_loss=contact_336_loss, 
                    contact_84_loss=contact_84_loss, 
                    contact_joint_loss=contact_joint_loss, 
                    regularization_loss=regularization_loss, 
                    smooth_regularization_loss=smooth_regularization_loss
                    )
    return total_loss, loss_dict