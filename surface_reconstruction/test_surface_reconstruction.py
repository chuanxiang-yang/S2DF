import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import trimesh
import numpy as np
import argparse
from pyhocon import ConfigFactory
import models.Net as Net
import recon_dataset as dataset
import surface_recon_args
import scipy.spatial as spatial

from dcudf.mesh_extraction import dcudf
def remove_far(mesh, points,dis_trunc=0.05):
    # gt_pts: trimesh
    # mesh: trimesh
    kd_tree = spatial.KDTree(points)
    distances, vertex_ids = kd_tree.query(mesh.vertices,p=2, distance_upper_bound=dis_trunc)
    faces_remaining = []
    faces = mesh.faces
    vertices = mesh.vertices
    for i in range(faces.shape[0]):
        if get_aver(distances, faces[i]) < dis_trunc:
            faces_remaining.append(faces[i])
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned
def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0

if __name__ == '__main__':
    args = surface_recon_args.get_test_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda:" + str(args.gpu_idx) if (torch.cuda.is_available()) else "cpu")

    index = args.index
    model_dir = os.path.join(args.logdir, 'trained_models')

    outdir = os.path.join(os.path.dirname(args.logdir), 'result_meshes')
    outdir_remove_far = os.path.join(os.path.dirname(args.logdir), 'result_meshes_remove_far')


    file_path = os.path.join(args.dataset_path, args.file_name)
    test_set = dataset.ReconDataset(file_path, 100000, n_samples=1)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    Net = Net.Network(in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim,
                      nl=args.nl,
                      decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                      init_type=args.init_type, neuron_type=args.neuron_type)

    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    for epoch in args.epoch_n:
        trained_model_filename = os.path.join(model_dir, 'model_%d.pth' % (epoch))
        Net.load_state_dict(torch.load(trained_model_filename, map_location=device))
        Net.to(device)
        latent = None
        print("Converting implicit to mesh for file {}".format(args.file_name))
        points, cp, scale, bbox, kd_tree = test_set.points,test_set.cp, test_set.scale, test_set.bbox, test_set.kd_tree
        object_bbox_min =bbox.T[0]
        object_bbox_max =bbox.T[1]

        evaluator = dcudf(lambda pts: Net.decoder(pts) / 1000, conf.get_int('evaluate.resolution'),
                          conf.get_float('evaluate.threshold'),kd_tree,
                          bbox = bbox,device=device,translate=-cp, scale=1/scale,
                          max_iter=conf.get_int("evaluate.max_iter"), normal_step=conf.get_int("evaluate.normal_step"),
                          laplacian_weight=conf.get_int("evaluate.laplacian_weight"),
                          bound_min=object_bbox_min, bound_max=object_bbox_max,
                          is_cut=conf.get_int("evaluate.is_cut"), region_rate=conf.get_int("evaluate.region_rate"),
                          max_batch=conf.get_int("evaluate.max_batch"),
                          learning_rate=conf.get_float("evaluate.learning_rate"),
                          warm_up_end=conf.get_int("evaluate.warm_up_end"),
                          report_freq=conf.get_int("evaluate.report_freq"),
                          watertight_separate=conf.get_int("evaluate.watertight_separate"))
        _,mesh = evaluator.optimize()
        mesh.apply_scale(scale)
        mesh.apply_translation(cp)
        os.makedirs(outdir, exist_ok=True)
        mesh.export(
            outdir + '/' + '{}_{}.ply'.format(args.file_name, str(conf.get_float('evaluate.threshold'))))

        #From CAP-UDF
        if conf.get_float('evaluate.far') > 0:
            os.makedirs(outdir_remove_far, exist_ok=True)
            points = points * scale + cp
            mesh = remove_far(mesh,points,conf.get_float('evaluate.far'))
            mesh.export(outdir_remove_far + '/' + '{}_{}.ply'.format(args.file_name[0:-4], str(conf.get_float('evaluate.threshold'))))