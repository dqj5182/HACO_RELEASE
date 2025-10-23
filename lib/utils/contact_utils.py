import gc
import torch
import numpy as np
from trimesh.proximity import ProximityQuery

from lib.utils.human_models import mano


def get_ho_contact_and_offset(mesh_hand, mesh_obj, c_thres):
    # Make sure that meshes are watertight and do not comntain inverted faces
    # Typically canonical space meshes are more stable

    pq = ProximityQuery(mesh_obj)
    obj_coord_c, dist, obj_coord_c_idx = pq.on_surface(mesh_hand.vertices.astype(np.float32))

    is_contact_h = (dist < c_thres)
    contact_h = (1. * is_contact_h).astype(np.float32)

    contact_valid = np.ones((mano.vertex_num, 1))
    inter_coord_valid = np.ones((mano.vertex_num))

    # Explicit cleanup
    del pq
    gc.collect()

    return np.array(contact_h), np.array(obj_coord_c), contact_valid, inter_coord_valid


def get_contact_thres(backbone_type='hamer'):
    # We select contact threshold that has best balance between precision and recall.
    if backbone_type == 'hamer': # seed 314
        return 0.52
    elif backbone_type == 'vit-l-16': # seed 327
        return 0.52
    elif backbone_type == 'vit-b-16': # seed 327
        return 0.5
    elif backbone_type == 'vit-s-16': # seed 314
        return 0.6
    elif backbone_type == 'handoccnet': # seed 314
        return 0.6
    elif backbone_type == 'hrnet-w48': # seed 314
        return 0.58
    elif backbone_type == 'hrnet-w32': # seed 314
        return 0.57
    elif backbone_type == 'resnet-152': # seed 314
        return 0.54
    elif backbone_type == 'resnet-101': # seed 314
        return 0.52
    elif backbone_type == 'resnet-50': # seed 314
        return 0.55
    elif backbone_type == 'resnet-34': # seed 314
        return 0.55
    elif backbone_type == 'resnet-18': # seed 314
        return 0.62
    else:
        raise NotImplementedError