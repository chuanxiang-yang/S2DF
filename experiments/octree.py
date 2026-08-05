import time

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple

import torch


class ArrayOctree:
    def __init__(self, max_nodes=10000000, root_half=1.0, max_depth=9):
        self.max_nodes = max_nodes
        self.root_half = root_half
        self.max_depth = max_depth

        self.centers = np.zeros((max_nodes, 3), dtype=np.float32)
        self.depth = np.zeros((max_nodes,), dtype=np.int8)
        self.is_leaf = np.zeros((max_nodes,), dtype=bool)
        self.values = np.full((max_nodes,), -5, dtype=np.int8)
        self.children = np.full((max_nodes, 8), -1, dtype=np.int32)
        self.node_count = 0

    def new_node(self, center, depth):
        idx = self.node_count
        self.node_count += 1
        self.centers[idx] = center
        self.depth[idx] = depth
        self.is_leaf[idx] = False
        self.children[idx] = -1
        return idx

    def batch_subdivide(self, idxs):
        N_split = len(idxs)
        if N_split == 0:
            return np.array([], dtype=np.int32)

        parent_centers = self.centers[idxs]  # (M,3)
        parent_depths = self.depth[idxs]  # (M,)

        OFFSETS = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], dtype=np.float32) * 0.5

        parent_halfs = self.root_half / (2 ** parent_depths.astype(np.float32))  # (M,)

        child_centers = parent_centers[:, None, :] + OFFSETS[None, :, :] * parent_halfs[:, None, None]
        child_centers = child_centers.reshape(-1, 3)  # (M*8,3)

        child_depths = np.repeat(parent_depths + 1, 8)  # (M*8,)

        start_idx = self.node_count
        child_ids = np.arange(start_idx, start_idx + N_split * 8, dtype=np.int32)

        self.centers[child_ids] = child_centers
        self.depth[child_ids] = child_depths
        self.is_leaf[child_ids] = False

        self.children[idxs[:, None], np.arange(8)] = child_ids.reshape(N_split, 8)
        self.node_count += N_split * 8
        return child_ids

    def query_points(self, points):

        N = points.shape[0]
        result = np.full(N, -1, dtype=np.int32)

        node_idx = np.zeros(N, dtype=np.int32)

        finished = np.zeros(N, dtype=bool)

        while True:
            active = ~finished
            if not np.any(active):
                break

            cur_nodes = node_idx[active]

            is_leaf = self.is_leaf[cur_nodes]

            leaf_global_mask = active.copy()
            leaf_global_mask[active] = is_leaf

            result[leaf_global_mask] = cur_nodes[is_leaf]
            finished[leaf_global_mask] = True

            down_global_mask = active.copy()
            down_global_mask[active] = ~is_leaf

            if not np.any(down_global_mask):
                continue

            pts = points[down_global_mask]
            nodes = node_idx[down_global_mask]
            centers = self.centers[nodes]

            offset = (pts > centers).astype(int)
            child_index = offset[:, 0] * 4 + offset[:, 1] * 2 + offset[:, 2]

            next_nodes = self.children[nodes, child_index]

            no_child = (next_nodes == -1)
            if np.any(no_child):
                no_child_mask = down_global_mask.copy()
                no_child_mask[down_global_mask] = no_child
                result[no_child_mask] = nodes[no_child]
                finished[no_child_mask] = True

            can_down = ~no_child
            if np.any(can_down):
                can_down_mask = down_global_mask.copy()
                can_down_mask[down_global_mask] = can_down
                node_idx[can_down_mask] = next_nodes[can_down]

        return result


def dist_fn(points, scene):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    return scene.compute_signed_distance(points).numpy()


def build_octree_array(octree, scene, threshold, nearest_distance_batch_fn=dist_fn, assign_value_fn=None):
    root = octree.new_node(center=np.array([0, 0, 0], dtype=np.float32), depth=0)
    max_depth = octree.max_depth
    current = np.array([root], dtype=np.int32)

    while len(current) > 0:
        centers = octree.centers[current]
        depths = octree.depth[current]
        half_diagonals = octree.root_half / (2 ** depths.astype(np.int32)) * (3 ** 0.5)

        signed_ds = nearest_distance_batch_fn(centers, scene)
        ds = np.abs(signed_ds)

        split_mask = (depths < max_depth) & (ds <= threshold + half_diagonals)

        leaf_mask = ~split_mask
        leaf_nodes = current[leaf_mask]
        octree.is_leaf[leaf_nodes] = True

        far_nodes = current[ds > threshold + half_diagonals]
        octree.values[far_nodes] = np.sign(signed_ds[ds > threshold + half_diagonals]) - 12

        split_nodes = current[split_mask]
        next_nodes = octree.batch_subdivide(split_nodes)
        current = np.array(next_nodes, dtype=np.int32)

    octree.centers = octree.centers[:octree.node_count]
    octree.depth = octree.depth[:octree.node_count]
    octree.is_leaf = octree.is_leaf[:octree.node_count]
    octree.values = octree.values[:octree.node_count]
    octree.children = octree.children[:octree.node_count]


# ------------------------
# Morton helpers (10bit per axis, depth<=10)
# ------------------------
def _split_by_2_10bit(n):
    n &= 0x3ff
    n = (n | (n << 16)) & 0x30000ff
    n = (n | (n << 8)) & 0x300f00f
    n = (n | (n << 4)) & 0x30c30c3
    n = (n | (n << 2)) & 0x9249249
    return n


def morton3D_encode(x, y, z):
    sx = _split_by_2_10bit(x)
    sy = _split_by_2_10bit(y)
    sz = _split_by_2_10bit(z)
    return (sx | (sy << 1) | (sz << 2)) & 0x3fffffff


def _compact1by2_10bit(n):
    n &= 0x9249249
    n = (n ^ (n >> 2)) & 0x30c30c3
    n = (n ^ (n >> 4)) & 0x300f00f
    n = (n ^ (n >> 8)) & 0x30000ff
    n = (n ^ (n >> 16)) & 0x3ff
    return n


def morton3D_decode(code):
    x = _compact1by2_10bit(code >> 0)
    y = _compact1by2_10bit(code >> 1)
    z = _compact1by2_10bit(code >> 2)
    return x, y, z


def collect_morton_parent_nodes_vectorized(octree):
    root_center = np.asarray(octree.centers[0], dtype=np.float32)
    root_half_size = float(octree.root_half)
    root_min = root_center - root_half_size
    root_size = root_half_size * 2.0

    N = len(octree.centers)


    non_leaf_nodes = np.where(~octree.is_leaf)[0]

    children_array = octree.children[non_leaf_nodes]  # (M,8)

    is_child_leaf = octree.is_leaf[children_array]
    parent_mask = np.all(is_child_leaf, axis=1)

    parent_idx = non_leaf_nodes[parent_mask]
    parent_children = octree.children[parent_idx]
    parent_values = octree.values[parent_children]

    collected_as_child = np.zeros(N, dtype=bool)
    collected_as_child[parent_children.flatten()] = True

    parent_depths = octree.depth[parent_idx]
    n = 1 << parent_depths.astype(np.int32)
    rel = (octree.centers[parent_idx] - root_min) / (root_size / n[:, None])
    ixiyiz = np.floor(rel + 1e-8).astype(np.int64)
    ixiyiz = np.clip(ixiyiz, 0, n[:, None] - 1)
    mortons = morton3D_encode(ixiyiz[:, 0], ixiyiz[:, 1], ixiyiz[:, 2])

    parent_nodes_list = [
        {'morton': int(m), 'depth': int(d), 'values': list(v)}
        for m, d, v in zip(mortons, parent_depths, parent_values)
    ]

    leaf_nodes = np.where(octree.is_leaf & ~collected_as_child)[0]
    leaf_depths = octree.depth[leaf_nodes]
    n_leaf = 1 << leaf_depths.astype(np.int32)
    rel_leaf = (octree.centers[leaf_nodes] - root_min) / (root_size / n_leaf[:, None])
    ixiyiz_leaf = np.floor(rel_leaf + 1e-8).astype(np.int64)
    ixiyiz_leaf = np.clip(ixiyiz_leaf, 0, n_leaf[:, None] - 1)
    mortons_leaf = morton3D_encode(ixiyiz_leaf[:, 0], ixiyiz_leaf[:, 1], ixiyiz_leaf[:, 2])

    leaf_values = octree.values[leaf_nodes]
    leaf_nodes_list = [
        {'morton': int(m), 'depth': int(d), 'value': int(v)}
        for m, d, v in zip(mortons_leaf, leaf_depths, leaf_values)
    ]

    return {
        'meta': {
            'root_center': root_center.tolist(),
            'root_half_size': root_half_size,
            'max_depth': int(octree.max_depth)
        },
        'parent_nodes': parent_nodes_list,
        'leaf_nodes': leaf_nodes_list
    }

def save_morton_parent_npz(storage: Dict, filename: str):
    parent_nodes_list = storage['parent_nodes']
    parent_mortons = np.array([n['morton'] for n in parent_nodes_list], dtype=np.uint32)
    parent_depths = np.array([n['depth'] for n in parent_nodes_list], dtype=np.int8)
    parent_values = np.array([n['values'] for n in parent_nodes_list], dtype=np.int8)

    leaf_nodes_list = storage['leaf_nodes']
    leaf_mortons = np.array([n['morton'] for n in leaf_nodes_list], dtype=np.uint32)
    leaf_depths = np.array([n['depth'] for n in leaf_nodes_list], dtype=np.int8)
    leaf_values = np.array([n['value'] for n in leaf_nodes_list], dtype=np.int8)

    np.savez_compressed(filename,
                        root_center=np.array(storage['meta']['root_center'], dtype=np.float64),
                        root_half_size=np.float64(storage['meta']['root_half_size']),
                        max_depth=np.int32(storage['meta']['max_depth']),
                        parent_mortons=parent_mortons,
                        parent_depths=parent_depths,
                        parent_values=parent_values,
                        leaf_mortons=leaf_mortons,
                        leaf_depths=leaf_depths,
                        leaf_values=leaf_values,
                        )

def load_morton_parent_npz(filename: str) -> Dict:
    data = np.load(filename)
    parent_nodes = [{'morton': int(m), 'depth': int(d), 'values': list(v)}
                    for m, d, v in zip(data['parent_mortons'], data['parent_depths'], data['parent_values'])]

    leaf_nodes = [{'morton': int(m), 'depth': int(d), 'value': v}
                  for m, d, v in zip(data['leaf_mortons'], data['leaf_depths'], data['leaf_values'])]
    meta = {'root_center': data['root_center'].tolist(),
            'root_half_size': float(data['root_half_size']),
            'max_depth': int(data['max_depth'])}
    return {'meta': meta, 'parent_nodes': parent_nodes, 'leaf_nodes': leaf_nodes}


class MortonQueryStructure:
    def __init__(self, meta, parent_nodes, leaf_nodes, device='cpu'):
        self.root_center = np.array(meta["root_center"], dtype=np.float64)
        self.root_half_size = float(meta["root_half_size"])
        self.max_depth = int(meta["max_depth"])

        self.root_min = self.root_center - self.root_half_size
        self.root_size = self.root_half_size * 2.0
        self.device = device

        self.parent_nodes = parent_nodes
        self.leaf_nodes = leaf_nodes

        self.num = 1 << self.max_depth

        self.value = np.zeros((self.num, self.num, self.num), dtype=np.int8)
        total = 0
        i = np.arange(8, dtype=np.int32)
        OFFSETS = np.stack([
            (i >> 2) & 1,
            (i >> 1) & 1,
            i & 1
        ], axis=1)  # (8,3)

        depths = np.array([n['depth'] for n in parent_nodes])  # (N,)
        mortons = np.array([n['morton'] for n in parent_nodes])  # (N,)
        values_list = np.array([n['values'] for n in parent_nodes])  # (N,8)

        pxs, pys, pzs = morton3D_decode(mortons)
        ps = np.stack([pxs, pys, pzs],axis=-1)
        child_xyz = (ps[:, None, :] << 1) | OFFSETS[None, :, :]
        child_xyz = child_xyz.reshape(-1,3)

        child_depths = depths[:, None] + 1  # (N,1)
        child_depths = np.broadcast_to(child_depths, (len(depths), 8)).reshape(-1)

        child_values = values_list.reshape(-1)
        max_depth_mask = child_depths==self.max_depth
        max_depth_child_xyz = child_xyz[max_depth_mask]

        self.value[max_depth_child_xyz[:, 0], max_depth_child_xyz[:, 1], max_depth_child_xyz[:, 2]] = child_values[max_depth_mask]
        total += max_depth_mask.sum()

        child_depths_dif = self.max_depth - child_depths[~max_depth_mask]
        scales = 1 << child_depths_dif
        child_xyz = child_xyz[~max_depth_mask]
        child_values = child_values[~max_depth_mask]

        child_xyz_start = child_xyz*scales[:,None]
        child_xyz_end = child_xyz_start + scales[:,None]

        for i in range(len(child_values)):
            value = child_values[i]
            child_ix_start = child_xyz_start[i, 0]
            child_iy_start = child_xyz_start[i, 1]
            child_iz_start = child_xyz_start[i, 2]

            child_ix_end = child_xyz_end[i, 0]
            child_iy_end = child_xyz_end[i, 1]
            child_iz_end = child_xyz_end[i, 2]

            self.value[child_ix_start:child_ix_end, child_iy_start:child_iy_end, child_iz_start:child_iz_end] = value

        depths = np.array([n['depth'] for n in leaf_nodes])  # (N,)
        depths_dif = self.max_depth - depths
        scales = 1 << depths_dif

        mortons = np.array([n['morton'] for n in leaf_nodes])  # (N,)
        values = np.array([n['value'] for n in leaf_nodes])  # (N,8)

        pxs, pys, pzs = morton3D_decode(mortons)
        ps = np.stack([pxs, pys, pzs],axis=-1)

        ps_start = ps*scales[:,None]
        ps_end = ps_start + scales[:,None]

        for i in range(len(leaf_nodes)):
            value = values[i]
            x_start = ps_start[i, 0]
            y_start = ps_start[i, 1]
            z_start = ps_start[i, 2]

            x_end = ps_end[i, 0]
            y_end = ps_end[i, 1]
            z_end = ps_end[i, 2]
            self.value[x_start:x_end, y_start:y_end, z_start:z_end] = value

        self.value = torch.from_numpy(self.value).to(device)
        self.root_center = torch.from_numpy(self.root_center).to(device)
        self.root_min = torch.from_numpy(self.root_min).to(device)


    def query_points_value(self, points):
        points = points.to(self.device)
        d = self.max_depth
        n_cells = 1 << d
        cell_size = self.root_size / n_cells
        rel = (points - self.root_min[None, :]) / cell_size
        idx = torch.floor(rel).to(torch.long)
        idx = idx.clip(0, n_cells - 1)

        return self.value[idx[:, 0], idx[:, 1], idx[:, 2]]



