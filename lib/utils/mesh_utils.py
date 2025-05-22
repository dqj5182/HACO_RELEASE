import torch
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement



def center_vertices(vertices, faces, flip_y=True): # This is for MOW dataset
    """Centroid-align vertices."""
    vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces



def load_obj_nr(filename_obj, normalization=True, texture_size=4, load_texture=False, # load_obj function from neural_renderer (https://github.com/daniilidis-group/neural_renderer) and MOW (https://github.com/ZheC/MOW)
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32))

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)) - 1

    # load textures
    textures = None
    if load_texture:
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures = load_textures(filename_obj, filename_mtl, texture_size,
                                         texture_wrapping=texture_wrapping,
                                         use_bilinear=use_bilinear)
        if textures is None:
            raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces