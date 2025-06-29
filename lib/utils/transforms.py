import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / (cam_coord[:,2] + 1e-5) * f[0] + c[0]
    y = cam_coord[:,1] / (cam_coord[:,2] + 1e-5) * f[1] + c[1]
    z = cam_coord[:,2] + 1e-5
    return np.stack((x,y,z),1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def apply_homogeneous_transformation(vertices, transform_matrix):
    # Convert vertices to homogeneous coordinates (add a column of ones)
    num_verts = vertices.shape[0]
    verts_homogeneous = torch.cat([vertices, torch.ones((num_verts, 1), dtype=vertices.dtype, device=vertices.device)], dim=1)  # Shape (num_verts, 4)

    # Apply the homogeneous transformation
    transformed_homogeneous = torch.matmul(transform_matrix, verts_homogeneous.T).T  # Shape (num_verts, 4)

    # Convert back to Cartesian coordinates (divide by the homogeneous component)
    transformed_vertices = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, None]  # Shape (num_verts, 3)

    return transformed_vertices


def apply_homogeneous_transformation_np(vertices, transform_matrix):
    # Convert vertices to homogeneous coordinates (add a column of ones)
    num_verts = vertices.shape[0]
    verts_homogeneous = np.concatenate([vertices, np.ones((num_verts, 1), dtype=vertices.dtype)], axis=1)  # Shape (num_verts, 4)

    # Apply the homogeneous transformation
    transformed_homogeneous = np.dot(transform_matrix, verts_homogeneous.T).T  # Shape (num_verts, 4)

    # Convert back to Cartesian coordinates (divide by the homogeneous component)
    transformed_vertices = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, None]  # Shape (num_verts, 3)

    return transformed_vertices


# Revert MANO global rotation and translation
def inv_mano_global_orient(mano_verts, mano_root, mano_global_orient, mano_trans):
    """
    Reverts the global orientation and translation applied to MANO vertices
    (i.e., transforms them from the global coordinate space back to a local space).
    
    Args:
        mano_verts (Tensor): shape (num_verts, 3), the MANO vertices.
        mano_joints (Tensor): shape (num_joints, 3), the MANO joint positions.
        mano_global_orient (Tensor): shape (3,), global orientation in axis-angle format.
        mano_trans (Tensor): shape (3,), global translation.

    Returns:
        vertices_transformed (Tensor): shape (num_verts, 3), the locally transformed vertices.
        transform_matrix (Tensor): shape (4, 4), the homogeneous transformation matrix
                                   that undoes the global transform.
        transform_matrix_inv (Tensor): shape (4, 4), the inverse of transform_matrix
                                       (i.e., the forward transform).
    """
    device = mano_verts.device

    # 1) Convert global orientation (axis-angle) -> rotation matrix
    R = axis_angle_to_rotation_matrix(mano_global_orient)   # shape (3, 3)

    # 2) Invert rotation matrix
    #    (for an orthonormal rotation, inverse is transpose)
    R_inv = invert_rotation_matrix(R)

    # 3) Identify the 'root' for the transform
    #    Typically 'Wrist' in MANO
    wrist_position = mano_root
    adjust_root    = wrist_position

    # 4) Build the matrix that undoes global transform (global -> local)
    transform_matrix = torch.eye(4, device=device)
    transform_matrix[:3, :3] = R_inv
    transform_matrix[:3, 3]  = (
        -torch.matmul(R_inv, adjust_root) 
        - mano_trans 
        + wrist_position
    )

    # 5) Apply transform_matrix to vertices
    verts_hom = torch.cat(
        [mano_verts, torch.ones((mano_verts.shape[0], 1), device=device)],
        dim=1
    )
    vertices_transformed = (transform_matrix @ verts_hom.T).T[:, :3]

    # 6) Manually invert transform_matrix without torch.linalg.inv
    #    
    #    If T = [[A, b],
    #            [0, 1]],
    #    then T^-1 = [[A^-1, -A^-1 b],
    #                 [0,    1   ]].
    #
    #    Here, A = R_inv, so A^-1 = R (the original rotation),
    #    b = transform_matrix[:3, 3].
    #
    #    So T^-1[:3, :3] = R
    #       T^-1[:3, 3]  = -R @ b
    #
    transform_matrix_inv = torch.eye(4, device=device)
    transform_matrix_inv[:3, :3] = R  # because R is (R_inv)^-1
    transform_matrix_inv[:3, 3]  = -R @ transform_matrix[:3, 3]

    return vertices_transformed, transform_matrix, transform_matrix_inv