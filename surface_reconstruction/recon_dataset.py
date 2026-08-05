# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import os.path
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from abc import ABC, abstractmethod
import torch
import open3d as o3d


class ReconDataset(data.Dataset):
    # A class to generate synthetic examples of basic shapes.
    # Generates clean and noisy point clouds sampled  + samples on a grid with their distance to the surface (not used in DiGS paper)
    def __init__(self, file_path, n_points=15000, n_samples=10000):
        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples

        self.o3d_point_cloud = o3d.io.read_point_cloud(self.file_path)

        # extract center and scale points and normals
        self.points, self.mnfld_n = self.get_mnfld_points()
        self.kd_tree = spatial.KDTree(self.points)
        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()

    def get_mnfld_points(self):
        # Returns points on the manifold
        points = np.asarray(self.o3d_point_cloud.points, dtype=np.float32)
        normals = np.asarray(self.o3d_point_cloud.normals, dtype=np.float32)
        # center and scale point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale
        return points, normals

    def __getitem__(self, index):
        manifold_idxes_permutation = np.random.permutation(self.points.shape[0])
        mnfld_idx = manifold_idxes_permutation[:self.n_points]
        manifold_points = self.points[mnfld_idx]  # (n_points, 3)

        nonmnfld_points = np.random.uniform(-1.1, 1.1,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points, 3)

        near_points = (manifold_points + 0.01 * np.random.randn(manifold_points.shape[0],
                                                                manifold_points.shape[1])).astype(np.float32)

        return {'points': manifold_points, 'nonmnfld_points': nonmnfld_points, 'near_points': near_points}

    def __len__(self):
        return self.n_samples
