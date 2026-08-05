# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch

def per_face_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    """
    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    normals = torch.linalg.cross(vec_a, vec_b)
    return normals

def area_weighted_distribution(
        V: torch.Tensor,
        F: torch.Tensor,
        normals: torch.Tensor = None):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    """

    if normals is None:
        normals = per_face_normals(V, F)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10

    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))

def random_face(
    V : torch.Tensor,
    F : torch.Tensor,
    num_samples : int,
    distrib=None):
    """Return an area weighted random sample of faces and their normals from the mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): num of samples to return
        distrib: distribution to use. By default, area-weighted distribution is used.
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    normals = per_face_normals(V, F)

    idx = distrib.sample([num_samples])

    return F[idx], normals[idx]

def sample_surface(
        V: torch.Tensor,
        F: torch.Tensor,
        num_samples: int,
        distrib=None):
    """Sample points and their normals on mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    # Select faces & sample their surface
    fidx, normals = random_face(V, F, num_samples, distrib)
    f = V[fidx]

    u = torch.sqrt(torch.rand(num_samples)).to(V.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(V.device).unsqueeze(-1)

    samples = (1 - u) * f[:, 0, :] + (u * (1 - v)) * f[:, 1, :] + u * v * f[:, 2, :]

    return samples, normals


def sample_near_surface(
    V : torch.Tensor,
    F : torch.Tensor,
    num_samples: int,
    variance : float = 0.01,
    distrib=None):
    """Sample points near the mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)
    samples = sample_surface(V, F, num_samples, distrib)[0]
    samples += torch.randn_like(samples) * variance
    return samples


def sample_uniform(num_samples: int):
    """Sample uniformly in [-1,1] bounding volume.

    Args:
        num_samples(int) : number of points to sample
    """
    return torch.rand(num_samples, 3) * 2.2 - 1.1

def point_sample(
    V : torch.Tensor,
    F : torch.Tensor,
    techniques : list,
    num_samples : int):
    """Sample points from a mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique
    """

    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(V, F)

    samples = []
    for technique in techniques:
        if technique =='trace':
            samples.append(sample_surface(V, F, num_samples, distrib=distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(V, F, num_samples, distrib=distrib))
        elif technique == 'rand':
            samples.append(sample_uniform(num_samples).to(V.device))
    samples = torch.cat(samples, dim=0)
    return samples

