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
    if backbone_type == 'hamer':
        return 0.5
    elif backbone_type == 'vit-l-16':
        return 0.55
    elif backbone_type == 'vit-b-16':
        return 0.5
    elif backbone_type == 'vit-s-16':
        return 0.5
    elif backbone_type == 'handoccnet':
        return 0.95
    elif backbone_type == 'hrnet-w48':
        return 0.5
    elif backbone_type == 'hrnet-w32':
        return 0.5
    elif backbone_type == 'resnet-152':
        return 0.55
    elif backbone_type == 'resnet-101':
        return 0.5
    elif backbone_type == 'resnet-50':
        return 0.5
    elif backbone_type == 'resnet-34':
        return 0.5
    elif backbone_type == 'resnet-18':
        return 0.5
    else:
        raise NotImplementedError