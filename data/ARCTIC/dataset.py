import os
import os.path as osp
import numpy as np
import cv2
import json
import trimesh
from tqdm import tqdm
from easydict import EasyDict
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import sys
sys.path.append(os.getcwd())

from lib.core.config import cfg
from lib.utils.human_models import mano
from lib.utils.transforms import cam2pixel
from lib.utils.func_utils import load_img, get_bbox
from lib.utils.contact_utils import get_ho_contact_and_offset
from lib.utils.preprocessing import augmentation_contact, process_human_model_output_orig
from lib.utils.train_utils import get_contact_difficulty_sample_id


def swap_coord_sys(arr):
    if not isinstance(arr, np.ndarray):
        arr =  np.array(arr)
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return arr.dot(coordChangMat.T)


def swap_param_sys(hand_pose, hand_shape, hand_trans):
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    cam_rot = cv2.Rodrigues(coordChangMat)[0].squeeze()
    cam_trans = np.zeros((3,))

    from lib.utils.mano_utils import ready_arguments
    mano_root='data/base_data/human_models/mano'
    side='right'

    if side == 'right':
        mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
    elif side == 'left':
        mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
    smpl_data = ready_arguments(mano_path)
    J = smpl_data['J'][0:1, :].T

    RAbsMat = cv2.Rodrigues(cam_rot)[0].dot(cv2.Rodrigues(hand_pose[:3])[0])
    RAbsRod = cv2.Rodrigues(RAbsMat)[0][:, 0]
    hand_trans = cv2.Rodrigues(cam_rot)[0].dot(J + np.expand_dims(np.copy(hand_trans), 0).T) + np.expand_dims(cam_trans / 1000, 0).T - J

    hand_pose[:3] = RAbsRod

    return hand_pose, np.array(hand_trans.r)[:, 0].astype(np.float32)


def construct_obj(object_model_p):
    # load vtemplate
    mesh_p = os.path.join(object_model_p, "mesh.obj")
    parts_p = os.path.join(object_model_p, f"parts.json")
    json_p = os.path.join(object_model_p, "object_params.json")
    obj_name = os.path.basename(object_model_p)

    top_sub_p = f"data/ARCTIC/data/meta/object_vtemplates/{obj_name}/top_keypoints_300.json"
    bottom_sub_p = top_sub_p.replace("top_", "bottom_")
    with open(top_sub_p, "r") as f:
        sub_top = np.array(json.load(f)["keypoints"])

    with open(bottom_sub_p, "r") as f:
        sub_bottom = np.array(json.load(f)["keypoints"])
    sub_v = np.concatenate((sub_top, sub_bottom), axis=0)

    with open(parts_p, "r") as f:
        parts = np.array(json.load(f), dtype=bool)

    assert os.path.exists(mesh_p), f"Not found: {mesh_p}"

    mesh = trimesh.exchange.load.load_mesh(mesh_p, process=False)
    mesh_v = mesh.vertices

    mesh_f = torch.LongTensor(mesh.faces)
    vidx = np.argmin(cdist(sub_v, mesh_v, metric="euclidean"), axis=1)
    parts_sub = parts[vidx]

    vsk = object_model_p.split("/")[-1]

    with open(json_p, "r") as f:
        params = json.load(f)
        rest = EasyDict()
        rest.top = np.array(params["mocap_top"])
        rest.bottom = np.array(params["mocap_bottom"])
        bbox_top = np.array(params["bbox_top"])
        bbox_bottom = np.array(params["bbox_bottom"])
        kp_top = np.array(params["keypoints_top"])
        kp_bottom = np.array(params["keypoints_bottom"])

    np.random.seed(1)

    obj = EasyDict()
    obj.name = vsk
    obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
    obj.v = torch.FloatTensor(mesh_v)
    obj.v_sub = torch.FloatTensor(sub_v)
    obj.f = torch.LongTensor(mesh_f)
    obj.parts = torch.LongTensor(parts)
    obj.parts_sub = torch.LongTensor(parts_sub)

    with open("data/ARCTIC/data/meta/object_meta.json", "r") as f:
        object_meta = json.load(f)
    obj.diameter = torch.FloatTensor(np.array(object_meta[obj.obj_name]["diameter"]))
    obj.bbox_top = torch.FloatTensor(bbox_top)
    obj.bbox_bottom = torch.FloatTensor(bbox_bottom)
    obj.kp_top = torch.FloatTensor(kp_top)
    obj.kp_bottom = torch.FloatTensor(kp_bottom)
    obj.mocap_top = torch.FloatTensor(np.array(params["mocap_top"]))
    obj.mocap_bottom = torch.FloatTensor(np.array(params["mocap_bottom"]))
    return obj


def pad_tensor_list(v_list: list):
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)
    return padded_tensor, v_len_list


def construct_obj_tensors(object_names):
    obj_list = []
    for k in object_names:
        object_model_p = f"data/ARCTIC/data/meta/object_vtemplates/%s" % (k)
        obj = construct_obj(object_model_p)
        obj_list.append(obj)

    bbox_top_list = []
    bbox_bottom_list = []
    mocap_top_list = []
    mocap_bottom_list = []
    kp_top_list = []
    kp_bottom_list = []
    v_list = []
    v_sub_list = []
    f_list = []
    parts_list = []
    parts_sub_list = []
    diameter_list = []
    for obj in obj_list:
        v_list.append(obj.v)
        v_sub_list.append(obj.v_sub)
        f_list.append(obj.f)

        # root_list.append(obj.root)
        bbox_top_list.append(obj.bbox_top)
        bbox_bottom_list.append(obj.bbox_bottom)
        kp_top_list.append(obj.kp_top)
        kp_bottom_list.append(obj.kp_bottom)
        mocap_top_list.append(obj.mocap_top / 1000)
        mocap_bottom_list.append(obj.mocap_bottom / 1000)
        parts_list.append(obj.parts + 1)
        parts_sub_list.append(obj.parts_sub + 1)
        diameter_list.append(obj.diameter)

    v_list, v_len_list = pad_tensor_list(v_list)
    p_list, p_len_list = pad_tensor_list(parts_list)
    ps_list = torch.stack(parts_sub_list, dim=0)
    assert (p_len_list - v_len_list).sum() == 0

    max_len = v_len_list.max()
    mask = torch.zeros(len(obj_list), max_len)
    for idx, vlen in enumerate(v_len_list):
        mask[idx, :vlen] = 1.0

    v_sub_list = torch.stack(v_sub_list, dim=0)
    diameter_list = torch.stack(diameter_list, dim=0)

    f_list, f_len_list = pad_tensor_list(f_list)

    bbox_top_list = torch.stack(bbox_top_list, dim=0)
    bbox_bottom_list = torch.stack(bbox_bottom_list, dim=0)
    kp_top_list = torch.stack(kp_top_list, dim=0)
    kp_bottom_list = torch.stack(kp_bottom_list, dim=0)

    obj_tensors = {}
    obj_tensors["names"] = object_names
    obj_tensors["parts_ids"] = p_list
    obj_tensors["parts_sub_ids"] = ps_list

    obj_tensors["v"] = v_list.float() / 1000
    obj_tensors["v_sub"] = v_sub_list.float() / 1000
    obj_tensors["v_len"] = v_len_list
    obj_tensors["f"] = f_list
    obj_tensors["f_len"] = f_len_list
    obj_tensors["diameter"] = diameter_list.float()

    obj_tensors["mask"] = mask
    obj_tensors["bbox_top"] = bbox_top_list.float() / 1000
    obj_tensors["bbox_bottom"] = bbox_bottom_list.float() / 1000
    obj_tensors["kp_top"] = kp_top_list.float() / 1000
    obj_tensors["kp_bottom"] = kp_bottom_list.float() / 1000
    obj_tensors["mocap_top"] = mocap_top_list
    obj_tensors["mocap_bottom"] = mocap_bottom_list
    obj_tensors["z_axis"] = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)
    return obj_tensors





def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing


OBJECTS = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",
]



class xdict(dict):
    """
    A subclass of Python's built-in dict class, which provides additional methods for manipulating and operating on dictionaries.
    """

    def __init__(self, mydict=None):
        """
        Constructor for the xdict class. Creates a new xdict object and optionally initializes it with key-value pairs from the provided dictionary mydict. If mydict is not provided, an empty xdict is created.
        """
        if mydict is None:
            return

        for k, v in mydict.items():
            super().__setitem__(k, v)

    def subset(self, keys):
        """
        Returns a new xdict object containing only the key-value pairs with keys in the provided list 'keys'.
        """
        out_dict = {}
        for k in keys:
            out_dict[k] = self[k]
        return xdict(out_dict)

    def __setitem__(self, key, val):
        """
        Overrides the dict.__setitem__ method to raise an assertion error if a key already exists.
        """
        assert key not in self.keys(), f"Key already exists {key}"
        super().__setitem__(key, val)

    def search(self, keyword, replace_to=None):
        """
        Returns a new xdict object containing only the key-value pairs with keys that contain the provided keyword.
        """
        out_dict = {}
        for k in self.keys():
            if keyword in k:
                if replace_to is None:
                    out_dict[k] = self[k]
                else:
                    out_dict[k.replace(keyword, replace_to)] = self[k]
        return xdict(out_dict)

    def rm(self, keyword, keep_list=[], verbose=False):
        """
        Returns a new xdict object with keys that contain keyword removed. Keys in keep_list are excluded from the removal.
        """
        out_dict = {}
        for k in self.keys():
            if keyword not in k or k in keep_list:
                out_dict[k] = self[k]
            else:
                if verbose:
                    print(f"Removing: {k}")
        return xdict(out_dict)

    def overwrite(self, k, v):
        """
        The original assignment operation of Python dict
        """
        super().__setitem__(k, v)

    def merge(self, dict2):
        """
        Same as dict.update(), but raises an assertion error if there are duplicate keys between the two dictionaries.

        Args:
            dict2 (dict or xdict): The dictionary or xdict instance to merge with.

        Raises:
            AssertionError: If dict2 is not a dictionary or xdict instance.
            AssertionError: If there are duplicate keys between the two instances.
        """
        assert isinstance(dict2, (dict, xdict))
        mykeys = set(self.keys())
        intersect = mykeys.intersection(set(dict2.keys()))
        assert len(intersect) == 0, f"Merge failed: duplicate keys ({intersect})"
        self.update(dict2)

    def mul(self, scalar):
        """
        Multiplies each value (could be tensor, np.array, list) in the xdict instance by the provided scalar.

        Args:
            scalar (float): The scalar to multiply the values by.

        Raises:
            AssertionError: If scalar is not a float.
        """
        if isinstance(scalar, int):
            scalar = 1.0 * scalar
        assert isinstance(scalar, float)
        out_dict = {}
        for k in self.keys():
            if isinstance(self[k], list):
                out_dict[k] = [v * scalar for v in self[k]]
            else:
                out_dict[k] = self[k] * scalar
        return xdict(out_dict)

    def prefix(self, text):
        """
        Adds a prefix to each key in the xdict instance.

        Args:
            text (str): The prefix to add.

        Returns:
            xdict: The xdict instance with the added prefix.
        """
        out_dict = {}
        for k in self.keys():
            out_dict[text + k] = self[k]
        return xdict(out_dict)

    def replace_keys(self, str_src, str_tar):
        """
        Replaces a substring in all keys of the xdict instance.

        Args:
            str_src (str): The substring to replace.
            str_tar (str): The replacement string.

        Returns:
            xdict: The xdict instance with the replaced keys.
        """
        out_dict = {}
        for k in self.keys():
            old_key = k
            new_key = old_key.replace(str_src, str_tar)
            out_dict[new_key] = self[k]
        return xdict(out_dict)

    def postfix(self, text):
        """
        Adds a postfix to each key in the xdict instance.

        Args:
            text (str): The postfix to add.

        Returns:
            xdict: The xdict instance with the added postfix.
        """
        out_dict = {}
        for k in self.keys():
            out_dict[k + text] = self[k]
        return xdict(out_dict)

    def sorted_keys(self):
        """
        Returns a sorted list of the keys in the xdict instance.

        Returns:
            list: A sorted list of keys in the xdict instance.
        """
        return sorted(list(self.keys()))

    def to(self, dev):
        """
        Moves the xdict instance to a specific device.

        Args:
            dev (torch.device): The device to move the instance to.

        Returns:
            xdict: The xdict instance moved to the specified device.
        """
        if dev is None:
            return self
        raw_dict = dict(self)
        return xdict(thing.thing2dev(raw_dict, dev))

    def to_torch(self):
        """
        Converts elements in the xdict to Torch tensors and returns a new xdict.

        Returns:
        xdict: A new xdict with Torch tensors as values.
        """
        return xdict(thing.thing2torch(self))

    def to_np(self):
        """
        Converts elements in the xdict to numpy arrays and returns a new xdict.

        Returns:
        xdict: A new xdict with numpy arrays as values.
        """
        return xdict(thing.thing2np(self))

    def tolist(self):
        """
        Converts elements in the xdict to Python lists and returns a new xdict.

        Returns:
        xdict: A new xdict with Python lists as values.
        """
        return xdict(thing.thing2list(self))

    def print_stat(self):
        """
        Prints statistics for each item in the xdict.
        """
        for k, v in self.items():
            _print_stat(k, v)

    def detach(self):
        """
        Detaches all Torch tensors in the xdict from the computational graph and moves them to the CPU.
        Non-tensor objects are ignored.

        Returns:
        xdict: A new xdict with detached Torch tensors as values.
        """
        return xdict(thing.detach_thing(self))

    def has_invalid(self):
        """
        Checks if any of the Torch tensors in the xdict contain NaN or Inf values.

        Returns:
        bool: True if at least one tensor contains NaN or Inf values, False otherwise.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    print(f"{k} contains nan values")
                    return True
                if torch.isinf(v).any():
                    print(f"{k} contains inf values")
                    return True
        return False

    def apply(self, operation, criterion=None):
        """
        Applies an operation to the values in the xdict, based on an optional criterion.

        Args:
        operation (callable): A callable object that takes a single argument and returns a value.
        criterion (callable, optional): A callable object that takes two arguments (key and value) and returns a boolean.

        Returns:
        xdict: A new xdict with the same keys as the original, but with the values modified by the operation.
        """
        out = {}
        for k, v in self.items():
            if criterion is None or criterion(k, v):
                out[k] = operation(v)
        return xdict(out)

    def save(self, path, dev=None, verbose=True):
        """
        Saves the xdict to disk as a Torch tensor.

        Args:
        path (str): The path to save the xdict.
        dev (torch.device, optional): The device to use for saving the tensor (default is CPU).
        verbose (bool, optional): Whether to print a message indicating that the xdict has been saved (default is True).
        """
        if verbose:
            print(f"Saving to {path}")
        torch.save(self.to(dev), path)


def quaternion_raw_multiply(a, b):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_invert(quaternion):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


class ObjectTensors(nn.Module):
    def __init__(self):
        super(ObjectTensors, self).__init__()
        self.obj_tensors = thing2dev(construct_obj_tensors(OBJECTS), "cpu")
        self.dev = None

    def forward_7d_batch(
        self,
        angles: (None, torch.Tensor),
        global_orient: (None, torch.Tensor),
        transl: (None, torch.Tensor),
        query_names: list,
        fwd_template: bool,
    ):
        self._sanity_check(angles, global_orient, transl, query_names, fwd_template)

        # store output
        out = xdict()

        # meta info
        obj_idx = np.array(
            [self.obj_tensors["names"].index(name) for name in query_names]
        )
        out["diameter"] = self.obj_tensors["diameter"][obj_idx]
        out["f"] = self.obj_tensors["f"][obj_idx]
        out["f_len"] = self.obj_tensors["f_len"][obj_idx]
        out["v_len"] = self.obj_tensors["v_len"][obj_idx]

        max_len = out["v_len"].max()
        out["v"] = self.obj_tensors["v"][obj_idx][:, :max_len]
        out["mask"] = self.obj_tensors["mask"][obj_idx][:, :max_len]
        out["v_sub"] = self.obj_tensors["v_sub"][obj_idx]
        out["parts_ids"] = self.obj_tensors["parts_ids"][obj_idx][:, :max_len]
        out["parts_sub_ids"] = self.obj_tensors["parts_sub_ids"][obj_idx]

        if fwd_template:
            return out

        # articulation + global rotation
        quat_arti = axis_angle_to_quaternion(self.obj_tensors["z_axis"] * angles)
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        # mm
        # collect entities to be transformed
        tf_dict = xdict()
        tf_dict["v_top"] = out["v"].clone()
        tf_dict["v_sub_top"] = out["v_sub"].clone()
        tf_dict["v_bottom"] = out["v"].clone()
        tf_dict["v_sub_bottom"] = out["v_sub"].clone()
        tf_dict["bbox_top"] = self.obj_tensors["bbox_top"][obj_idx]
        tf_dict["bbox_bottom"] = self.obj_tensors["bbox_bottom"][obj_idx]
        tf_dict["kp_top"] = self.obj_tensors["kp_top"][obj_idx]
        tf_dict["kp_bottom"] = self.obj_tensors["kp_bottom"][obj_idx]

        # articulate top parts
        for key, val in tf_dict.items():
            if "top" in key:
                val_rot = quaternion_apply(quat_arti[:, None, :], val)
                tf_dict.overwrite(key, val_rot)

        # global rotation for all
        for key, val in tf_dict.items():
            val_rot = quaternion_apply(quat_global[:, None, :], val)
            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict.overwrite(key, val_rot)

        # prep output
        top_idx = out["parts_ids"] == 1
        v_tensor = tf_dict["v_bottom"].clone()
        v_tensor[top_idx, :] = tf_dict["v_top"][top_idx, :]

        top_idx = out["parts_sub_ids"] == 1
        v_sub_tensor = tf_dict["v_sub_bottom"].clone()
        v_sub_tensor[top_idx, :] = tf_dict["v_sub_top"][top_idx, :]

        bbox = torch.cat((tf_dict["bbox_top"], tf_dict["bbox_bottom"]), dim=1)
        kp3d = torch.cat((tf_dict["kp_top"], tf_dict["kp_bottom"]), dim=1)

        out.overwrite("v", v_tensor)
        out.overwrite("v_sub", v_sub_tensor)
        out.overwrite("bbox3d", bbox)
        out.overwrite("kp3d", kp3d)
        return out

    def forward(self, angles, global_orient, transl, query_names):
        out = self.forward_7d_batch(
            angles, global_orient, transl, query_names, fwd_template=False
        )
        return out

    def forward_template(self, query_names):
        out = self.forward_7d_batch(
            angles=None,
            global_orient=None,
            transl=None,
            query_names=query_names,
            fwd_template=True,
        )
        return out

    def to(self, dev):
        self.obj_tensors = thing2dev(self.obj_tensors, dev)
        self.dev = dev

    def _sanity_check(self, angles, global_orient, transl, query_names, fwd_template):
        # sanity check
        if not fwd_template:
            # assume transl is in meter
            if transl is not None:
                transl = transl * 1000  # mm

            batch_size = angles.shape[0]
            assert angles.shape == (batch_size, 1)
            assert global_orient.shape == (batch_size, 3)
            if transl is not None:
                assert isinstance(transl, torch.Tensor)
                assert transl.shape == (batch_size, 3)
            assert len(query_names) == batch_size


def transform_kp2d(kp2d, bbox):
    # bbox: (cx, cy, scale) in the original image space
    # scale is normalized
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    cx, cy, scale = bbox
    s = 200 * scale  # to px
    cap_dim = 1000  # px
    factor = cap_dim / (1.5 * s)
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - 1.5 / 2 * s
    kp2d_cropped[:, 1] -= cy - 1.5 / 2 * s
    kp2d_cropped[:, 0] *= factor
    kp2d_cropped[:, 1] *= factor
    return kp2d_cropped


def get_sample_id(db, split, index):
    index = split[index]
    aid = db['imgnames'][index].split('./arctic_data/data/images/')[-1]
    subject_name = aid.split('/')[0]
    seq_name = aid.split('/')[1]
    obj_name, action_name = seq_name.split('_')[0], seq_name.split('_')[1]
    cam_name = aid.split('/')[2]
    img_name = aid.split('/')[3]
    img_id = img_name.split('.jpg')[0]

    sample_id = f'{subject_name}-{seq_name}-{cam_name}-{img_id}'
    return sample_id


class ARCTIC(Dataset):
    def __init__(self, transform, data_split):
        super(ARCTIC, self).__init__()
        self.__dict__.update(locals())

        self.transform = transform
        dataset_name = 'arctic'

        self.data_split = data_split
        self.root_path = root_path = osp.join('data', 'ARCTIC')
        self.data_dir = os.path.join(self.root_path, 'data')

        # Do sampling as the data for train set is large
        if data_split == 'train':
            sampling_ratio = 1
        else:
            sampling_ratio = 1

        # DB (protocol 1 + protocol 2)
        db_p1_path = os.path.join(self.data_dir, f'splits/p1_{self.data_split}.npy')
        db_p2_path = os.path.join(self.data_dir, f'splits/p2_{self.data_split}.npy')

        db_p1 = np.load(db_p1_path, allow_pickle=True).item() # keys: ['data_dict', 'imgnames'] | allocentric
        db_p2 = np.load(db_p2_path, allow_pickle=True).item() # keys: ['data_dict', 'imgnames'] | egocentric

        self.db = db_p2
        self.split = list(range(0, len(self.db['imgnames']), sampling_ratio))

        # Object layer
        self.object_layer = ObjectTensors()

        self.use_preprocessed_data = True
        self.annot_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'annot_data')
        self.contact_data_path = os.path.join(root_path, 'preprocessed_data', data_split, 'contact_data')
        os.makedirs(self.annot_data_path, exist_ok=True)
        os.makedirs(self.contact_data_path, exist_ok=True)

        # Sort contact by difficulty
        if self.data_split == 'train' and cfg.MODEL.balanced_sampling:
            sample_id_to_split_id = {}
            for split_idx in range(len(self.split)):
                each_sample_id = get_sample_id(self.db, self.split, split_idx)
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
        index = self.split[index]
        aid = self.db['imgnames'][index].split('./arctic_data/data/images/')[-1]
        subject_name = aid.split('/')[0]
        seq_name = aid.split('/')[1]
        obj_name, action_name = seq_name.split('_')[0], seq_name.split('_')[1]
        cam_name = aid.split('/')[2]
        img_name = aid.split('/')[3]
        img_id = img_name.split('.jpg')[0]

        sample_id = f'{subject_name}-{seq_name}-{cam_name}-{img_id}'

        orig_img_path = os.path.join(self.data_dir, 'cropped_images', subject_name, seq_name, cam_name, f'{img_id}.jpg')
        
        orig_img = load_img(orig_img_path)
        img_shape = orig_img.shape[:2]
        img_h, img_w = img_shape

        mano_valid = np.ones((1), dtype=np.float32)


        annot_data_path = os.path.join(self.annot_data_path, f'{sample_id}.npz')
        if os.path.exists(annot_data_path) and (self.data_split == 'train'):
            annot_data = np.load(annot_data_path, allow_pickle=True)
            bbox_hand_r = annot_data['bbox_hand_r']

            contact_h = np.load(os.path.join(self.contact_data_path, f'{sample_id}.npy')).astype(np.float32)
            contact_data = dict(contact_h=contact_h)
        else:
            data_dict = self.db['data_dict'][f'{subject_name}/{seq_name}']

            bbox_full = data_dict['bbox'][int(img_id)][int(cam_name)]
            joint_img_l = data_dict['2d']['joints.left'][int(img_id)][int(cam_name)]
            joint_img_r = data_dict['2d']['joints.right'][int(img_id)][int(cam_name)]

            if int(cam_name) == 0:
                ego_image_scale = 0.3

                bbox_full, joint_img_l, joint_img_r = ego_image_scale * bbox_full, ego_image_scale * joint_img_l, ego_image_scale* joint_img_r
            else:
                joint_img_l = transform_kp2d(joint_img_l, bbox_full)
                joint_img_r = transform_kp2d(joint_img_r, bbox_full)

            bbox_hand_l = get_bbox(joint_img_l, np.ones(len(joint_img_l)), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio)
            bbox_hand_r = get_bbox(joint_img_r, np.ones(len(joint_img_r)), expansion_factor=cfg.DATASET.ho_big_bbox_expand_ratio)

            cam_intr = data_dict['params']['K_ego'][int(img_id)]
            cam_param = {'focal': [cam_intr[0][0], cam_intr[1][1]], 'princpt': [cam_intr[0][2], cam_intr[1][2]]}

            # Hand
            hand_pose_l = np.concatenate((data_dict['params']['rot_l'][int(img_id)], data_dict['params']['pose_l'][int(img_id)]))
            hand_shape_l = data_dict['params']['shape_l'][int(img_id)]
            hand_trans_l = data_dict['params']['trans_l'][int(img_id)]

            hand_pose_r = np.concatenate((data_dict['params']['rot_r'][int(img_id)], data_dict['params']['pose_r'][int(img_id)]))
            hand_shape_r = data_dict['params']['shape_r'][int(img_id)]
            hand_trans_r = data_dict['params']['trans_r'][int(img_id)]

            mano_param_l = {'pose': hand_pose_l, 'shape': hand_shape_l, 'trans': hand_trans_l[:, None], 'hand_type': 'left'}
            mano_param_r = {'pose': hand_pose_r, 'shape': hand_shape_r, 'trans': hand_trans_r[:, None], 'hand_type': 'right'}
            
            mano_mesh_cam_l, mano_joint_cam_l, mano_pose_l, mano_shape_l, mano_trans_l = process_human_model_output_orig(mano_param_l, cam_param)
            mano_mesh_cam_r, mano_joint_cam_r, mano_pose_r, mano_shape_r, mano_trans_r = process_human_model_output_orig(mano_param_r, cam_param)
            
            mano_mesh_img_l = cam2pixel(mano_mesh_cam_l, cam_param['focal'], cam_param['princpt'])[:, :2]
            mano_mesh_img_r = cam2pixel(mano_mesh_cam_r, cam_param['focal'], cam_param['princpt'])[:, :2]
            
            hand_mesh_l = trimesh.Trimesh(mano_mesh_cam_l, mano.layer['left'].faces)
            hand_mesh_r = trimesh.Trimesh(mano_mesh_cam_r, mano.layer['right'].faces)

            # Object
            obj_arti = torch.Tensor([data_dict['params']['obj_arti'][int(img_id)]]).view(-1, 1)
            obj_rot, obj_trans = torch.Tensor(data_dict['params']['obj_rot'][int(img_id)])[None], torch.Tensor(data_dict['params']['obj_trans'][int(img_id)])[None] / 1000

            with torch.no_grad():
                obj_mesh_data = self.object_layer.forward_7d_batch(obj_arti, obj_rot, obj_trans, [obj_name], fwd_template=False)
                obj_mesh = trimesh.Trimesh(obj_mesh_data['v'][0], obj_mesh_data['f'][0])


            if False:
                annot_data = dict(sample_id=sample_id, bbox_hand_l=bbox_hand_l, bbox_hand_r=bbox_hand_r)
                np.savez(annot_data_path, **annot_data)

            # Contact
            mesh_others = hand_mesh_l + obj_mesh


            contact_h, obj_coord_c, contact_valid, inter_coord_valid = get_ho_contact_and_offset(hand_mesh_r, mesh_others, cfg.MODEL.c_thres)
            contact_data = dict(contact_h=contact_h)


        ############################### PROCESS CROP AND AUGMENTATION ################################
        img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale = augmentation_contact(orig_img.copy(), bbox_hand_r, self.data_split, enforce_flip=False) # TODO: CHNAGE THIS FOR TRAINING
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
            meta_info = dict(sample_id=sample_id, mano_valid=mano_valid)

        return dict(input_data=input_data, targets_data=targets_data, meta_info=meta_info)




if __name__ == "__main__":
    dataset_name = 'ARCTIC'
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
            contact_h = data['targets_data']['contact_data']['contact_h'][0].tolist()
            contact_h = np.array(contact_h, dtype=int)

            contact_data_save_path = os.path.join(contact_data_save_root_path, f'{sample_id}.npy')
            np.save(contact_data_save_path, contact_h)
    else:
        raise NotImplementedError