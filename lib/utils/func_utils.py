import cv2
import torch
import numpy as np


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB': img = img[:,:,::-1]
    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = (xmax-xmin)*expansion_factor;
    xmin = x_center - 0.5*width
    xmax = x_center + 0.5*width
    
    y_center = (ymin+ymax)/2.; height = (ymax-ymin)*expansion_factor;
    ymin = y_center - 0.5*height
    ymax = y_center + 0.5*height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, target_shape, original_img_shape):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = target_shape[1]/target_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox


def pca_to_axis_angle(pca_pose):
    """
    Converts the PCA pose representation from ManoLayer (use_pca=True)
    to full axis-angle pose (use_pca=False).
    
    Args:
    - pca_pose: The PCA components (batch_size x num_pca_comps).
    
    Returns:
    - full_pose: The full 48D axis-angle pose (batch_size x 48).
    """
    # Ensure pca_pose is a torch tensor
    if isinstance(pca_pose, np.ndarray):
        pca_pose = torch.tensor(pca_pose, dtype=torch.float32)

    global_rotation, hand_pose = pca_pose[:, :3], pca_pose[:, 3:]  # This should be a placeholder, adjust as needed.
    
    # Multiply the PCA components by the PCA basis to get the hand pose (45D)
    mano_th_selected_comps = get_mano_pca_basis(ncomps=45, use_pca=True, side='right', mano_root='data/base_data/human_models/mano')
    hand_pose = torch.mm(hand_pose, mano_th_selected_comps)
    
    # Add the mean hand pose to the result (broadcasting over the batch dimension)
    full_hand_pose = hand_pose
    
    # Concatenate the global rotation with the full hand pose
    full_pose = torch.cat([global_rotation, full_hand_pose], dim=1)  # Shape: (batch_size, 48)
    
    return full_pose


import re
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Load config
import yaml
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg