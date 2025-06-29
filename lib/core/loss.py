import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import cfg
from lib.utils.human_models import mano


V_regressor_336 = torch.tensor(np.load(cfg.MODEL.V_regressor_336_path), dtype=torch.float32)
V_regressor_84 = torch.tensor(np.load(cfg.MODEL.V_regressor_84_path), dtype=torch.float32)
MANO_regressor = torch.tensor(mano.joint_regressor, dtype=torch.float32)


class VCBLoss(nn.Module):
    def __init__(self, beta=0.9999, v_type='mesh'):
        """
        Vertex-level class-Balanced Loss (VCB Loss)
        """
        super(VCBLoss, self).__init__()
        self.beta = beta
        self.v_type = v_type

        # Get samples per class for all datasets
        all_contact_samples = torch.from_numpy(np.load(cfg.MODEL.contact_data_path)).to(torch.int32)
        self.samples_per_cls = self.get_samples_per_cls(all_contact_samples, cfg.TRAIN.epoch)

        # BCE Loss without internal sigmoid
        self.bce_loss = nn.BCELoss(reduction='none')

    def get_samples_per_cls(self, all_contact_samples, total_epoch=10, total_sample_num=1000): # total_sample_num is heuristic
        # Efficient count of 0s and 1s per vertex
        count_v_0 = (all_contact_samples == 0).sum(dim=0)
        count_v_1 = (all_contact_samples == 1).sum(dim=0)
        all_v_contact_counts = torch.stack([count_v_0, count_v_1], dim=1).float()

        # Efficient count of 0s and 1s across all vertices
        count_0 = (all_contact_samples == 0).sum()
        count_1 = (all_contact_samples == 1).sum()
        all_contact_counts = torch.tensor([count_0, count_1]).float()

        # Normalize per-vertex class counts and scale
        denom_v = all_v_contact_counts.sum(dim=1, keepdim=True).clamp(min=1e-6)
        samples_per_cls_v = all_v_contact_counts / denom_v * total_sample_num

        # Normalize all-vertices class counts and scale
        denom = all_contact_counts.sum().clamp(min=1e-6)
        samples_per_cls = all_contact_counts / denom * total_sample_num

        # Weighted sum of VCB and CB loss
        samples_per_cls_v_dict = {}
        v_weight_max = 0.3
        for epoch in range(total_epoch):
            v_weight = v_weight_max * (epoch / (total_epoch - 1))
            all_weight = 1.0 - v_weight
            samples_per_cls_v_epoch = v_weight * samples_per_cls_v.clone() + all_weight * samples_per_cls.clone() # Broadcast global [2] stats to [V, 2] for blending
            samples_per_cls_v_epoch = samples_per_cls_v_epoch.clamp(min=20.0)
            samples_per_cls_v_epoch = samples_per_cls_v_epoch / samples_per_cls_v_epoch.sum(dim=1, keepdim=True) * total_sample_num # re-normalize clamped values
            samples_per_cls_v_dict[epoch] = samples_per_cls_v_epoch

        # Apply vertex/joint regressors based on type
        if self.v_type == 'mesh':
            for epoch_key in samples_per_cls_v_dict:
                samples_per_cls_v_dict[epoch_key] = samples_per_cls_v_dict[epoch_key]
        elif self.v_type == 'mesh_336':
            for epoch_key in samples_per_cls_v_dict:
                samples_per_cls_v_dict[epoch_key] = torch.matmul(V_regressor_336, samples_per_cls_v_dict[epoch_key])
        elif self.v_type == 'mesh_84':
            for epoch_key in samples_per_cls_v_dict:
                samples_per_cls_v_dict[epoch_key] = torch.matmul(V_regressor_84, samples_per_cls_v_dict[epoch_key])
        elif self.v_type == 'joint':
            for epoch_key in samples_per_cls_v_dict:
                samples_per_cls_v_dict[epoch_key] = torch.matmul(MANO_regressor, samples_per_cls_v_dict[epoch_key])
        else:
            raise NotImplementedError

        return samples_per_cls_v_dict

    def compute_class_weight(self, samples_per_cls, beta):
        """
        samples_per_cls: Tensor of shape [V, 2]  (class 0 count, class 1 count for each vertex)
        returns: class_weights of shape [V, 2]
        """
        effective_num = 1.0 - torch.pow(beta, samples_per_cls)      # [V, 2]
        class_weights = (1.0 - beta) / effective_num                # [V, 2]
        
        # Normalize per vertex (across classes) so sum over dim=1 is 1
        class_weights = class_weights / class_weights.sum(dim=1, keepdim=True)  # [V, 2]

        return class_weights  # [V, 2]
    
    def expand_class_weight(self, class_weights, pred, gt):
        w = class_weights[None, :, :] # class_weights: [num_verts, 2], w: [1, num_vertices, 2]
        g = gt.long()[..., None] # g: [batch, num_vertices, 1]
        class_weights = torch.gather(w.expand(gt.size(0), -1, -1), 2, g)[..., 0] # gt, pred: [batch_size, num_vertices], class_weights: [batch, num_vertices]
        class_weights_expanded = class_weights.to(pred.device) # gt, pred: [batch_size, num_vertices], self.class_weights: [2], class_weights: [batch_size, num_vertices]
        return class_weights_expanded

    def forward(self, pred, gt, epoch, valid=None): # pred: [batch, 778], gt: [batch, 778]
        gt = gt.to(pred.device).float()

        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0.0, device=pred.device)
            pred, gt = pred[valid], gt[valid]

        pred = pred.sigmoid()

        # Compute class-balanced weights
        if epoch not in self.samples_per_cls:
            epoch = max(self.samples_per_cls.keys())  # fallback to last epoch
        class_weights = self.compute_class_weight(self.samples_per_cls[epoch], self.beta)

        # Compute BCE with class-balanced weighting
        class_weights_expanded = self.expand_class_weight(class_weights.to(pred.device), pred, gt)
        vcb_loss = self.bce_loss(pred, gt) * class_weights_expanded  # Manual weighting, class_weights: [batch_size, num_vertices]

        # Penalize too confident positive predictions
        fp_reg_loss = ((gt == 0).float() * pred).mean()

        return vcb_loss.mean() + 0.5 * fp_reg_loss


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt_means, valid=None):
        gt_means = gt_means.to(pred.device).float()
        batch_size = gt_means.shape[0]
        
        if valid is not None:
            if not valid.any().item():
                return torch.tensor(0, device=pred.device).float()
            pred, gt_means = pred[valid], gt_means[valid]

        pred = pred.sigmoid()

        return self.criterion(pred, gt_means)


class SmoothRegLoss(nn.Module):
    def __init__(self):
        super(SmoothRegLoss, self).__init__()

    def build_adjacency_matrix(self, num_verts, faces):
        """ Constructs a sparse adjacency matrix for memory efficiency. """
        device = faces.device
        row_idx = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2],
                             faces[:, 1], faces[:, 2], faces[:, 0]])
        col_idx = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 0], faces[:, 1], faces[:, 2]])

        indices = torch.stack([row_idx, col_idx], dim=0)
        values = torch.ones(indices.shape[1], device=device)

        adjacency = torch.sparse_coo_tensor(indices, values,
                                            (num_verts, num_verts), device=device)
        
        # Add self-loops for improved stability
        self_loops = torch.eye(num_verts, device=device)
        adjacency += self_loops.to_sparse()

        return adjacency

    def forward(self, pred, faces):
        batch_size, num_verts = pred.shape
        pred_contact = pred.sigmoid()
        pred_non_contact = 1.0 - pred_contact  # Inverted for non-contact

        faces = faces.to(pred.device).long()

        # Build sparse adjacency matrix
        adjacency = self.build_adjacency_matrix(num_verts, faces)

        # Step 1: Propagate Contact and Non-Contact Points Separately
        propagated_contact = torch.stack([
            torch.sparse.mm(adjacency, pred_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)

        propagated_non_contact = torch.stack([
            torch.sparse.mm(adjacency, pred_non_contact[b].unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size)
        ], dim=0)

        # Step 2: Normalization for Stability
        normalized_propagated_contact = propagated_contact / (
            propagated_contact.max(dim=1, keepdim=True)[0] + 1e-6
        )

        normalized_propagated_non_contact = propagated_non_contact / (
            propagated_non_contact.max(dim=1, keepdim=True)[0] + 1e-6
        )

        # Step 3: Balanced Isolation Scores
        isolation_score_contact = torch.abs(pred_contact - normalized_propagated_contact)
        isolation_score_non_contact = torch.abs(pred_non_contact - normalized_propagated_non_contact)

        # Step 4: Approximate Total Cluster Sizes
        approx_total_contact_size = propagated_contact.sum(dim=1)
        approx_total_non_contact_size = propagated_non_contact.sum(dim=1)

        # Step 5: Final Loss — Weighted Combination for Balanced Penalty
        isolated_cluster_score = isolation_score_contact.sum(dim=1) + isolation_score_non_contact.sum(dim=1)
        
        # Adaptive Normalization
        norm_factor = (approx_total_contact_size + approx_total_non_contact_size) + 1e-3

        penalty_ratio = isolated_cluster_score / norm_factor
        loss_isolation = torch.log1p(penalty_ratio).mean()

        return loss_isolation