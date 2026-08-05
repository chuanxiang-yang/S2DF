import torch
import numpy as np
import math
from PIL import Image
import skimage
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
import skimage.transform
import json
import os
import re
from tqdm import tqdm
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree
import errno
import urllib.request
from point_sample import *
import open3d as o3d
from octree import *

def get_mgrid(sidelen, dim=2, centered=True, include_end=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if include_end:
        denom = [s-1 for s in sidelen]
    else:
        denom = sidelen

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / denom[0]
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / denom[1]
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / denom[0]
        pixel_coords[..., 1] = pixel_coords[..., 1] / denom[1]
        pixel_coords[..., 2] = pixel_coords[..., 2] / denom[2]
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    if centered:
        pixel_coords -= 0.5

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class MeshSDF(Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path,octree_depth=9, mode='train', device='cpu'):
        self.file_path = file_path
        self.filename = self.file_path.split('/')[-1]
        self.sample_mode = ['near', 'near', 'near', 'trace', 'trace']
        self.device = device

        self.mesh = o3d.io.read_triangle_mesh(self.file_path)
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.uint32)
        self.vertices = self.normalize(self.vertices)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.vertices, self.triangles)
        self.bbox = np.array([np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)]).transpose()

        if mode == 'test':
            return

        self.octree = ArrayOctree(max_nodes=100000000, root_half=1.1, max_depth=octree_depth)
        build_octree_array(
            self.octree,
            self.scene,
            threshold=0.01,
        )
        storage = collect_morton_parent_nodes_vectorized(self.octree)
        self.queryer = MortonQueryStructure(
            storage["meta"],
            storage["parent_nodes"],
            storage["leaf_nodes"],
            device=device
        )
        self.sample()

    def sample(self):
        self.sampled_points = point_sample(torch.from_numpy(self.vertices), torch.from_numpy(self.triangles.astype(np.int32)), self.sample_mode, 100000*20)
        points_octree_value = self.queryer.query_points_value(self.sampled_points.to(self.device)).detach().cpu()
        # mask = (points_octree_value != -11) & (points_octree_value != -13)
        mask = points_octree_value > -8
        self.sampled_points = self.sampled_points[mask]
        self.sampled_sdf = self.scene.compute_signed_distance(self.sampled_points.numpy()).numpy()

    def sample2(self):
        self.sample_mode2 = ['near', 'near', 'near', 'near', 'near']
        self.sampled_points2 = point_sample(torch.from_numpy(self.vertices), torch.from_numpy(self.triangles.astype(np.int32)),
                                            self.sample_mode2, 10000*2000)


    def normalize(self,points):
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale
        return points


    def __getitem__(self, index):
        points_idx = np.random.randint(low=0, high=self.sampled_points.shape[0], size=(100000),dtype=np.int32)
        sampled_points = self.sampled_points[points_idx]
        sampled_sdf = self.sampled_sdf[points_idx]

        return {'coords': sampled_points}, {'sdf': sampled_sdf}

    def __len__(self):
        return 10000