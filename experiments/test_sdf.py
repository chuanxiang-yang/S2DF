import sys
import os

from octree import load_morton_parent_npz, MortonQueryStructure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
import configargparse
import dataio
import utils
import training
import loss_functions
import modules
from functools import partial
import logging

logging.getLogger("fvcore").setLevel(logging.ERROR)
torch.set_num_threads(8)

p = configargparse.ArgumentParser()

# config file, output directories
p.add('-c', '--config', required=False, is_config_file=True,
      help='Path to config file.')
p.add_argument('--logging_root', type=str, default='../logs',
               help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='subdirectory in logging_root for checkpoints, summaries')

# general training
p.add_argument('--model_type', type=str, default='mfn',
               help='options: mfn, siren, ff, finer, wire, nerfpe')
p.add_argument('--hidden_size', type=int, default=128,
               help='size of hidden layer')
p.add_argument('--hidden_layers', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
p.add_argument('--num_steps', type=int, default=20000,
               help='number of training steps')
p.add_argument('--ckpt_step', type=int, default=0,
               help='step at which to resume training')
p.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
p.add_argument('--seed', default=None,
               help='random seed for experiment reproducibility')

# mfn options
p.add_argument('--multiscale', action='store_true', default=False,
               help='use multiscale')
p.add_argument('--max_freq', type=int, default=512,
               help='The network-equivalent sample rate used to represent the signal.'
                    + 'Should be at least twice the Nyquist frequency.')
p.add_argument('--input_scales', nargs='*', type=float, default=None,
               help='fraction of resolution growth at each layer')
p.add_argument('--output_layers', nargs='*', type=int, default=None,
               help='layer indices to output, beginning at 1')

# mlp options
p.add_argument('--w0', default=30, type=int,
               help='omega_0 parameter for siren')
p.add_argument('--pe_scale', default=5, type=float,
               help='positional encoding scale')
#
p.add_argument('--fbs', type=float, default=None, help='')

# sdf model and sampling
p.add_argument('--num_pts_on', type=int, default=10000,
               help='number of on-surface points to sample')
p.add_argument('--coarse_scale', type=float, default=1e-1,
               help='laplacian scale factor for coarse samples')
p.add_argument('--fine_scale', type=float, default=1e-3,
               help='laplacian scale factor for fine samples')
p.add_argument('--coarse_weight', type=float, default=1e-2,
               help='weight to apply to coarse loss samples')

# data i/o
p.add_argument('--shape', type=str, default='bunny',
               help='name of point cloud shape in xyz format')
p.add_argument('--mesh_path', type=str,
               default='../data/armadillo.xyz',
               help='path for input point cloud')
p.add_argument('--num_workers', default=0, type=int,
               help='number of workers')

# tensorboard summary
p.add_argument('--steps_til_ckpt', type=int, default=50000,
               help='epoch frequency to save a checkpoint')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='epoch frequency to update tensorboard summary')

opt = p.parse_args()
device = 'cuda:' + str(opt.gpu)


def init_dataloader(opt):

    sdf_dataset = dataio.MeshSDF(opt.mesh_path,mode='test')

    dataloader = DataLoader(sdf_dataset, shuffle=True,
                            batch_size=1, pin_memory=True,
                            num_workers=opt.num_workers)

    return sdf_dataset,dataloader


def init_model(opt):
    ''' return appropriate model given experiment configs '''

    model_ = modules.CoordinateNet

    model = model_(nl='sine',
                   in_features=3,
                   out_features=1,
                   num_hidden_layers=opt.hidden_layers,
                   hidden_features=opt.hidden_size,
                   w0=opt.w0,
                   is_sdf=True)
    model.to(device)
    return model



if __name__ == '__main__':
    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    opt.filename = opt.mesh_path.split('/')[-1]

    opt.root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(opt.root_path)

    log_filename = os.path.join(opt.root_path,'out_test.log')
    log_file = open(log_filename, 'w')
    if opt.seed:
        torch.manual_seed(int(opt.seed))
        np.random.seed(int(opt.seed))

    sdf_dataset,dataloader = init_dataloader(opt)

    model = init_model(opt)
    dict = torch.load(os.path.join(opt.root_path, 'checkpoints', 'model_final.pth'), weights_only=True, map_location=device)
    octree = load_morton_parent_npz(os.path.join(opt.root_path, 'checkpoints', 'octree.npz'))
    model.load_state_dict(dict)
    octree_queryer = MortonQueryStructure(
        octree["meta"],
        octree["parent_nodes"],
        octree["leaf_nodes"],
        device = device
    )
    cp, scale, bbox = sdf_dataset.cp, sdf_dataset.scale, sdf_dataset.bbox

    mesh = utils.implicit2mesh(model, 256, translate=-cp, scale=1 / scale, device=device,octree_queryer=octree_queryer,log_file=log_file)
    os.makedirs(opt.root_path,exist_ok=True)
    output_ply_filepath = os.path.join(opt.root_path, opt.filename.replace('obj', 'ply'))
    mesh.export(output_ply_filepath, vertex_normal=True)
    print('***' * 50)

