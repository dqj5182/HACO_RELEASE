import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

from lib.core.config import cfg
from lib.utils.human_models import mano


def get_aug_config_contact():
    # Augmentation intensity factors
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    trans_factor = 0.1 # Translation range (recommended 0.1 to 0.2)
    noise_std = 0.02 # Gaussian noise strength
    motion_blur_prob = 0.15 # Probability of applying motion blur
    extreme_crop_prob = 0.1 # Probability for extreme cropping
    extreme_crop_lvl = 0.3 # Crop intensity (recommended 0.2 to 0.4)
    low_res_prob = 0.05 # Probability for applying low resolution
    low_res_scale_range = (0.15, 0.5) # Range for low-res scaling

    # Scaling augmentation
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0

    # Rotation augmentation
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0

    # Color augmentation
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up)
    ])

    # Flipping augmentation
    do_flip = random.random() <= 0.5

    # Translation augmentation
    tx = np.clip(np.random.randn(), -1.0, 1.0) * trans_factor
    ty = np.clip(np.random.randn(), -1.0, 1.0) * trans_factor

    # Extreme cropping augmentation
    do_extreme_crop = random.random() <= extreme_crop_prob

    # Noise augmentation (returns standard deviation for Gaussian noise injection)
    add_noise = random.random() <= 0.3  # 30% chance of adding noise
    noise_std = noise_std if add_noise else 0.0

    # Motion blur augmentation
    apply_motion_blur = random.random() <= motion_blur_prob
    motion_blur_kernel_size = random.choice([3, 5, 7]) if apply_motion_blur else 0

    # Low-resolution augmentation
    apply_low_res = random.random() <= low_res_prob
    low_res_scale = random.uniform(*low_res_scale_range) if apply_low_res else 1.0

    return {
        'scale': scale,
        'rot': rot,
        'color_scale': color_scale,
        'do_flip': do_flip,
        'tx': tx,
        'ty': ty,
        'do_extreme_crop': do_extreme_crop,
        'extreme_crop_lvl': extreme_crop_lvl if do_extreme_crop else 0,
        'noise_std': noise_std,
        'motion_blur_kernel_size': motion_blur_kernel_size,
        'low_res_scale': low_res_scale # Added low-res scale parameter
    }


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image_contact(cvimg, bbox, scale, rot, do_flip, out_shape, tx=0.0, ty=0.0, bkg_color='black'):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if bkg_color == 'white':
        borderMode=cv2.BORDER_CONSTANT
        borderValue=(255, 255, 255)
    else:
        borderMode=cv2.BORDER_CONSTANT
        borderValue=(0, 0, 0)

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    # Add translation offset
    bb_c_x += tx * img_width
    bb_c_y += ty * img_height

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 
                                    out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=borderValue)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 
                                        out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def augmentation_contact(img, bbox, data_split, enforce_flip=None, bkg_color='black'):
    if data_split == 'train':
        aug_params = get_aug_config_contact()
    else:
        aug_params = {
            'scale': 1.0,
            'rot': 0.0,
            'color_scale': np.array([1, 1, 1]),
            'do_flip': False,
            'tx': 0.0,
            'ty': 0.0,
            'do_extreme_crop': False,
            'extreme_crop_lvl': 0.0,
            'noise_std': 0.0,
            'motion_blur_kernel_size': 0,
            'low_res_scale': 1.0  # No low-res in non-training mode
        }
    
    # Enforce flip if specified
    if enforce_flip is not None:
        aug_params['do_flip'] = enforce_flip

    # Apply geometric augmentations (scaling, rotation, flipping)
    img, trans, inv_trans = generate_patch_image_contact(
        img, bbox, aug_params['scale'], aug_params['rot'], 
        aug_params['do_flip'], cfg.MODEL.input_img_shape, 
        aug_params['tx'], aug_params['ty'], bkg_color
    )

    # Apply low-resolution augmentation
    if aug_params['low_res_scale'] < 1.0:  # Only apply if scaling down
        img = apply_low_res(img, aug_params['low_res_scale'])

    # Apply color augmentation
    img = np.clip(img * aug_params['color_scale'][None, None, :], 0, 255)

    # Apply extreme cropping
    if aug_params['do_extreme_crop']:
        img = apply_extreme_crop(img, aug_params['extreme_crop_lvl'])

    # Apply noise augmentation
    if aug_params['noise_std'] > 0:
        img = add_gaussian_noise(img, aug_params['noise_std'])

    # Apply motion blur augmentation
    if aug_params['motion_blur_kernel_size'] > 0:
        img = apply_motion_blur(img, aug_params['motion_blur_kernel_size'])

    return img, trans, inv_trans, aug_params['rot'], aug_params['do_flip'], aug_params['color_scale']


def apply_extreme_crop(img, crop_lvl):
    """Extreme cropping: Aggressively crop the image."""
    h, w = img.shape[:2]
    crop_size = max(1, int(min(h, w) * (1 - crop_lvl)))  # Prevent zero-size crops
    start_x = random.randint(0, max(0, w - crop_size))
    start_y = random.randint(0, max(0, h - crop_size))
    cropped_img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Preserve aspect ratio during resizing
    return cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(img, noise_std):
    """Add Gaussian noise to the image with proper scaling for data type."""
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    
    if img.dtype == np.uint8:
        noisy_img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
    elif img.dtype == np.float32:
        noisy_img = np.clip(img + noise, 0.0, 1.0).astype(np.float32)
    elif img.dtype == np.float64:
        noisy_img = np.clip(img + noise, 0.0, 1.0).astype(np.float64)
    else:
        raise TypeError("Unsupported image dtype. Expected uint8 or float32.")
        
    return noisy_img


def apply_motion_blur(img, kernel_size):
    """Apply motion blur to the image with a random direction."""
    kernel = np.zeros((kernel_size, kernel_size))
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])

    if direction == 'horizontal':
        kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    elif direction == 'vertical':
        kernel[:, (kernel_size - 1) // 2] = np.ones(kernel_size)
    elif direction == 'diagonal':
        np.fill_diagonal(kernel, 1)
    
    kernel /= kernel_size  # Normalize the kernel
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_low_res(img, scale_factor=0.25):
    """Simulate low-resolution effect by downsampling and upsampling."""
    if not (0 < scale_factor < 1):
        raise ValueError("scale_factor should be between 0 and 1.")

    h, w = img.shape[:2]

    # Calculate target dimensions for downsampling
    downsampled_size = (max(1, int(w * scale_factor)), max(1, int(h * scale_factor)))

    # Downsample using INTER_AREA for better quality in aggressive downsampling
    low_res_img = cv2.resize(img, downsampled_size, interpolation=cv2.INTER_AREA)

    # Upsample using INTER_NEAREST for strong pixelation effect
    return cv2.resize(low_res_img, (w, h), interpolation=cv2.INTER_NEAREST).astype(img.dtype)


def process_human_model_output_orig(human_model_param, cam_param):
    pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
    hand_type = human_model_param['hand_type']
    trans = human_model_param['trans']
    pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1,-1) # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation 
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        root_pose = pose[mano.orig_root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)
    
    # get root joint coordinate
    root_pose = pose[mano.orig_root_joint_idx].view(1,3)
    hand_pose = torch.cat((pose[:mano.orig_root_joint_idx,:], pose[mano.orig_root_joint_idx+1:,:])).view(1,-1)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices[0].numpy()
    joint_coord = np.dot(mano.joint_regressor, mesh_coord)
    
    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
        root_coord = joint_coord[mano.root_joint_idx,None,:]
        joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
        mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    
    joint_cam_orig = joint_coord.copy()
    mesh_cam_orig = mesh_coord.copy()
    pose_orig, shape_orig, trans_orig = torch.cat((root_pose, hand_pose), dim=-1)[0].detach().cpu().numpy(), shape[0].detach().cpu().numpy(), trans[0].detach().cpu().numpy()

    return mesh_cam_orig, joint_cam_orig, pose_orig, shape_orig, trans_orig


def mask2bbox(mask, expansion_factor=1.0):
    # Find non-zero elements (object pixels)
    coords = np.argwhere(mask)
    
    # Extract bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Compute width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Expand bounding box
    if expansion_factor > 0:
        x_min = max(0, int(x_min - width * expansion_factor / 2))
        y_min = max(0, int(y_min - height * expansion_factor / 2))
        x_max = min(mask.shape[1] - 1, int(x_max + width * expansion_factor / 2))
        y_max = min(mask.shape[0] - 1, int(y_max + height * expansion_factor / 2))

        # Recalculate width and height after expansion
        width = x_max - x_min + 1
        height = y_max - y_min + 1

    return (x_min, y_min, width, height)