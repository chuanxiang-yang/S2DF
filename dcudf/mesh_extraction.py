# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
import math
from scipy.sparse import coo_matrix
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
import os

from tqdm import tqdm
import scipy.spatial as spatial

from dcudf.VectorAdam import VectorAdam
import warnings
warnings.filterwarnings('ignore')


def threshold_MC(ndf, threshold,spacing,xyz,scale,translate):
    try:
        vertices, triangles,_,_ = measure.marching_cubes(
                            ndf, threshold,spacing=spacing)
        vertices = vertices + np.array([xyz[0][0], xyz[1][0], xyz[2][0]])
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
    except ValueError:
        print("threshold too high")
        mesh = None
    return mesh


def laplacian_calculation(mesh, equal_weight=True):
    """
    edit from trimesh function to return tensor
    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neightbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors
    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(np.ndarray)

    # stack neighbors to 1D arrays
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        # the distance from verticesex to neighbors
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        # normalize group and stack into single array
        data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def laplacian_step(laplacian_op,samples):
    laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3]) - samples[:, 0:3]
    return laplacian_v


def get_abc(vertices, faces):
    fvs = vertices[faces]
    sub_a = fvs[:, 0, :] - fvs[:, 1, :]
    sub_b = fvs[:, 1, :] - fvs[:, 2, :]
    sub_c = fvs[:, 0, :] - fvs[:, 2, :]
    sub_a = torch.linalg.norm(sub_a, dim=1)
    sub_b = torch.linalg.norm(sub_b, dim=1)
    sub_c = torch.linalg.norm(sub_c, dim=1)
    return sub_a, sub_b, sub_c


def calculate_s(vertices, faces):
    sub_a, sub_b, sub_c = get_abc(vertices,faces)
    p = (sub_a + sub_b + sub_c)/2

    s = p*(p-sub_a)*(p-sub_b)*(p-sub_c)
    s[s<1e-30]=1e-30

    sqrts = torch.sqrt(s)
    return sqrts


def get_mid(vertices, faces):
    fvs = vertices[faces]
    re = torch.mean(fvs,dim=1)
    return re


class dcudf:
    """DCUDF mesh extraction

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            query_func:       differentiable function, input tensor[batch_size, 3] and output tensor[batch_size]
            resolution:
            threshold:
            max_iter:
            normal_step:      end of step one
            laplacian_weight:
            bound_min:
            bound_max:
            is_cut:           if model is a open model, set to True to cut double cover
            region_rate:      region of seed and sink in mini-cut
            max_batch:        higher batch_size will have quicker speed. If you GPU memory is not enough, decrease it.
            learning_rate:
            warm_up_end:
            report_freq:      report loss every {report_freq}

        """
    def __init__(self,query_func,resolution,threshold,kd_tree,bbox=np.array([[-1, 1], [-1, 1], [-1, 1]]),device=None,translate=[0., 0., 0.], scale=1,
                 max_iter=400, normal_step=300,laplacian_weight=2000.0, bound_min=None,bound_max=None,
                 is_cut = True, region_rate=20,
                 max_batch=100000, learning_rate=0.0005, warm_up_end=25,
                 report_freq=1, watertight_separate=False):
        self.u = None
        self.mesh = None
        self.device = torch.device('cuda')
        self.kd_tree = kd_tree

        # Evaluating parameters
        self.bbox = bbox
        self.device = device
        self.translate = translate
        self.scale = scale
        self.max_iter = max_iter
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.normal_step = normal_step
        self.laplacian_weight = laplacian_weight
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.threshold = threshold

        if bound_min is None:
            bound_min = torch.tensor([-1+self.threshold, -1+self.threshold, -1+self.threshold], dtype=torch.float32)
        if bound_max is None:
            bound_max = torch.tensor([1-self.threshold, 1-self.threshold, 1-self.threshold], dtype=torch.float32)
        if isinstance(bound_min, list):
            bound_min = torch.tensor(bound_min, dtype=torch.float32)
        if isinstance(bound_max, list):
            bound_max = torch.tensor(bound_max, dtype=torch.float32)
        if isinstance(bound_min, np.ndarray):
            bound_min = torch.from_numpy(bound_min).float()
        if isinstance(bound_max, np.ndarray):
            bound_max = torch.from_numpy(bound_max).float()
        self.bound_min = bound_min - self.threshold
        self.bound_max = bound_max + self.threshold

        self.is_cut = is_cut
        self.region_rate = region_rate

        self.cut_time = None
        self.extract_time = None

        self.watertight_separate = watertight_separate
        if self.watertight_separate == 1:
            self.watertight_separate=True
        else:
            self.watertight_separate= False

        self.optimizer = None

        self.query_func = query_func

    def optimize(self):
        query_func = self.query_func

        # u = self.extract_fields()
        grid_dict = self.get_3d_grid()
        cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
        pnts = grid_dict["grid_points"]
        z = []
        pc = []
        for point in tqdm(torch.split(pnts, 100000, dim=0)):
            if (torch.cuda.is_available()):
                point = point.cuda(device=self.device)
            pred = self.query_func(point.type(torch.float32)).detach()
            pred[pred <= 0] = 1e-10
            pred = pred.sqrt()
            pred = pred.cpu().numpy()
            z.append(pred)
            pc.append(point.detach().cpu().numpy())
        z = np.concatenate(z, axis=0).reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                                              grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2]).astype(np.float64)

        self.mesh = threshold_MC(z, self.threshold, spacing=(cell_width, cell_width, cell_width),xyz=grid_dict['xyz'],scale=self.scale,translate=self.translate)
        cc = trimesh.graph.connected_components(self.mesh.face_adjacency, min_len=0)
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.faces)
        faces_output = np.asarray(self.mesh.faces)
        faces_mask_output = np.ones(faces.shape[0]).astype(bool)

        #
        for c in cc:
            c_faces_index = np.asarray(c)
            c_faces = faces[c_faces_index]
            c_faces = c_faces.flatten()
            c_vertices = vertices[c_faces]
            dist, _ = self.kd_tree.query(c_vertices, k=1, workers=-1)
            if dist.min() > self.threshold * 1.2:
                faces_mask_output[c_faces_index] = False
        faces_output = faces_output[faces_mask_output]
        self.mesh = trimesh.Trimesh(vertices, faces_output)
        self.mesh.update_faces(self.mesh.nondegenerate_faces())
        # init points
        xyz = torch.from_numpy(self.mesh.vertices.astype(np.float32)).to(self.device)
        xyz.requires_grad = True
        # set optimizer to xyz
        self.optimizer = VectorAdam([xyz])
        # init laplacian operation
        laplacian_op = laplacian_calculation(self.mesh).to(self.device)
        vertex_faces = np.asarray(self.mesh.vertex_faces)

        face_mask = np.ones_like(vertex_faces).astype(bool)
        face_mask[vertex_faces==-1] = False
        print('Optimizing...')
        for it in range(self.max_iter):
            if it == self.normal_step:
                points = xyz.detach().cpu().numpy()
                normal_mesh = trimesh.Trimesh(vertices=points, faces=self.mesh.faces, process=False)
                normals = torch.FloatTensor(normal_mesh.face_normals).to(self.device)
                origin_points = get_mid(xyz,self.mesh.faces).detach().clone()

            self.update_learning_rate(it)

            epoch_loss = 0
            self.optimizer.zero_grad()
            num_samples = xyz.shape[0]
            head = 0
            while head < num_samples:
                sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                df[df <= 0] = 1e-8
                df = df.sqrt()
                df_loss = df.mean()

                loss = df_loss
                if it <= self.normal_step:
                    s_value = calculate_s(xyz, self.mesh.faces)
                    face_weight = s_value[vertex_faces[head: min(head + self.max_batch, num_samples)]]

                    face_weight[~face_mask[head: min(head + self.max_batch, num_samples)]] = 0
                    face_weight = torch.sum(face_weight, dim=1)

                    face_weight = torch.sqrt(face_weight.detach())
                    face_weight = face_weight.max() / face_weight

                    lap_v = laplacian_step(laplacian_op, xyz)
                    lap_v = torch.mul(lap_v, lap_v)
                    lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
                    laplacian_loss = face_weight * torch.sum(lap_v, dim=1)
                    laplacian_loss = self.laplacian_weight * laplacian_loss.mean()
                    loss = loss + laplacian_loss

                epoch_loss += loss.data
                loss.backward(retain_graph=True)
                head += self.max_batch

            mid_num_samples = len(self.mesh.faces)
            mid_head = 0
            while mid_head < mid_num_samples:
                mid_points = get_mid(xyz, self.mesh.faces)
                sub_mid_points = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])]
                mid_df = query_func(sub_mid_points)
                mid_df[mid_df <= 0] = 1e-8
                mid_df = mid_df.sqrt()
                mid_df_loss = mid_df.mean()
                loss = mid_df_loss
                if it > self.normal_step:
                    offset = mid_points[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])] - origin_points[
                                                                                                         mid_head: min(
                                                                                                             mid_head + self.max_batch,
                                                                                                             mid_points.shape[
                                                                                                                 0])]
                    normal_loss = torch.norm(
                        torch.cross(offset, normals[mid_head: min(mid_head + self.max_batch, mid_points.shape[0])], dim=-1),
                        dim=-1)
                    normal_loss = 0.5* normal_loss.mean()
                    loss +=  normal_loss
                epoch_loss += loss.data
                loss.backward(retain_graph=True)
                mid_head += self.max_batch
            self.optimizer.step()

        final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=self.mesh.faces, process=False)
        cc = trimesh.graph.connected_components(final_mesh.face_adjacency, min_len=0)
        if self.is_cut == 1:
            from dcudf.mesh_cut import mesh_cut
            s = time.time()
            final_mesh_cuple = mesh_cut(final_mesh,region_rate = self.region_rate)
            t = time.time()
            self.cut_time = t-s
            if final_mesh_cuple is not None:
                final_mesh_1 = final_mesh_cuple[0]
                final_mesh_2 = final_mesh_cuple[1]

                if len(final_mesh_1.vertices)>len(final_mesh_2.vertices):
                    final_mesh = final_mesh_1
                else:
                    final_mesh = final_mesh_2
            else:
                print("It seems that model is too complex, cutting failed. Or just rerunning to try again.")
        elif self.watertight_separate:
            mesh = self.watertight_postprocess(final_mesh)
            if not mesh == None:
                final_mesh = mesh
        else:
            pass
        return self.mesh,final_mesh

    def extract_fields(self):

        N = 32
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)

        u = np.zeros([self.resolution, self.resolution, self.resolution], dtype=np.float32)
        # with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(self.device)
                    val = self.query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    val[val < 0] = 0
                    val = np.sqrt(val)
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        self.u = u
        return u

    def get_3d_grid(self,eps=0.1,dtype=np.float16):
        # generate points on a uniform grid within  a given range
        # reimplemented from SAL : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
        # and IGR : https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py

        shortest_axis = np.argmin(self.bbox[:, 1] - self.bbox[:, 0])
        if (shortest_axis == 0):
            x = np.linspace(self.bbox[0, 0] - eps, self.bbox[0, 1] + eps, self.resolution)

            length = np.max(x) - np.min(x)
            y = np.arange(self.bbox[1, 0] - eps, self.bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))

            z = np.arange(self.bbox[2, 0] - eps, self.bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        elif (shortest_axis == 1):
            y = np.linspace(self.bbox[1, 0] - eps, self.bbox[1, 1] + eps, self.resolution)
            length = np.max(y) - np.min(y)
            x = np.arange(self.bbox[0, 0] - eps, self.bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))

            z = np.arange(self.bbox[2, 0] - eps, self.bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        elif (shortest_axis == 2):
            z = np.linspace(self.bbox[2, 0] - eps, self.bbox[2, 1] + eps, self.resolution)
            length = np.max(z) - np.min(z)

            x = np.arange(self.bbox[0, 0] - eps, self.bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
            y = np.arange(self.bbox[1, 0] - eps, self.bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

        xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype))  #
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16)
        return {"grid_points": grid_points,
                "shortest_axis_length": length,
                "xyz": [x, y, z],
                "shortest_axis_index": shortest_axis}
    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        if iter_step>=200:
            lr *= 0.1
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def watertight_postprocess(self, mesh):
        meshes = mesh.split()
        mask = np.zeros(len(meshes))
        from pytorch3d.ops import box3d_overlap
        boxes = []
        for m in meshes:
            boxes.append(trimesh.bounds.corners(m.bounding_box.bounds))
        boxes = torch.FloatTensor(boxes)

        intersection_vol, iou_3d = box3d_overlap(boxes, boxes, eps=1e-12)
        for i in range((iou_3d.shape[0])):
            for j in range(i+1, (iou_3d.shape[0])):
                if iou_3d[i][j]>0.9:
                    mask[i] = 1
        re_mesh = None
        for i in range(len(mask)):
            if mask[i] == 1:
                if re_mesh == None:
                    re_mesh=meshes[i]
                else:
                    re_mesh = re_mesh + meshes[i]
        return re_mesh


