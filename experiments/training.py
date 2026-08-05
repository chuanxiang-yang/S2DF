from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch
import utils
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import forward_models
from functools import partial
import shutil

from octree import collect_morton_parent_nodes_vectorized, save_morton_parent_npz


def train(model, train_dataloader, steps, lr, steps_til_summary,
          steps_til_checkpoint, model_dir, loss_fn, summary_fn,
          prefix_model_dir='', val_dataloader=None, double_precision=False,
          clip_grad=False, use_lbfgs=False, loss_schedules=None, params=None,
          ckpt_step=0, use_lr_scheduler=False, is_wire=False,sdf_dataset=None,log_file=None,device='cuda:0'):

    is_amsgrad = True
    if is_wire:
        is_amsgrad = False
    if params is None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=is_amsgrad)
    else:
        optim = torch.optim.Adam(lr=lr, params=params, amsgrad=is_amsgrad)

    scheduler = None
    # if use_lr_scheduler:
    #     optim.param_groups[0]['lr'] = 1
    #     log_scheduler = partial(lr_log_schedule, num_steps=steps, nw=1, lr0=lr, lrn=1e-4)
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=log_scheduler)

    scheduler = lr_scheduler.MultiStepLR(optim,
                                         milestones=[steps * 0.7, steps * 0.8, steps * 0.9],
                                         gamma=0.25)

    if os.path.exists(model_dir):
        pass
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)

    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    # e.g. epochs=1k, len(train_dataloader)=25
    train_generator = iter(train_dataloader)

    train_losses = []
    for step in range(steps):

        if not step % steps_til_checkpoint and step:
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir,
                       'model_step_%04d.pth' % (step + ckpt_step)))
            np.savetxt(os.path.join(checkpoints_dir,
                       'train_losses_step_%04d.txt' % (step + ckpt_step)),
                       np.array(train_losses))
        try:
            model_input, gt = next(train_generator)
        except StopIteration:
            train_generator = iter(train_dataloader)
            model_input, gt = next(train_generator)


        model_input = dict2cuda(model_input,device)
        gt = dict2cuda(gt,device)

        model_output = model(model_input)
        loss_dict = loss_fn(model_output, gt)
        train_loss = loss_dict["total_loss"]

        if not step % steps_til_summary:
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dir,
                                    'model_current.pth'))
        optim.zero_grad(set_to_none=True)
        train_loss.backward()
        if clip_grad:
            if isinstance(clip_grad, bool):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optim.step()

        if scheduler is not None:
            scheduler.step()

        if not step % 100:
            lr = torch.tensor(optim.param_groups[0]['lr'])

            utils.log_string("total_loss: {}, lr={:.3e}".format(loss_dict["total_loss"].item(), lr), log_file)
            loss_list = loss_dict['losses']
            loss_str = ', '.join([f'loss{i + 1} : {loss.item():.3e}' for i, loss in enumerate(loss_list)])
            utils.log_string(
                '[{:4d}/{} ({:.0f}%)], {}'.format(
                    step, steps, 100. * step / steps, loss_str
                ),
                log_file
            )
            utils.log_string('', log_file)
        if not step % 1000:
            sdf_dataset.sample()
    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))

    sdf_dataset.sample2()
    utils.cal_layer_cube(model, pnts=sdf_dataset.sampled_points2, device=device, octree=sdf_dataset.octree, threshold=0.00015)
    storage = collect_morton_parent_nodes_vectorized(sdf_dataset.octree)
    save_morton_parent_npz(storage, os.path.join(checkpoints_dir,'octree.npz'))


def dict2cuda(a_dict,device='cuda:0'):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.to(device)})
        elif isinstance(value, dict):
            tmp.update({key: dict2cuda(value,device)})
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.to(device) for v in value]})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cpu() for v in value]})
        else:
            tmp.update({key: value})
    return tmp




def lr_log_schedule(it, num_steps=1e6, nw=2500, lr0=1e-3, lrn=5e-6, lambdaw=0.01):
    return (lambdaw + (1 - lambdaw) * np.sin(np.pi/2 * np.clip(it/nw, 0, 1))) \
        * np.exp((1 - it/num_steps) * np.log(lr0) + (it/num_steps) * np.log(lrn))





def sample_pdf(model_inputs, model_outputs, offset=5e-3,
               idx=-1):
    ''' hierarchical sampling code for neural radiance fields '''

    z_vals = model_inputs['t']
    bins = .5*(z_vals[..., 1:, :] + z_vals[..., :-1, :]).squeeze()
    bins = bins.clone().detach().requires_grad_(True)

    if 'combined' in model_outputs:
        if isinstance(model_outputs['combined']['model_out']['output'], list):
            pred_sigma = model_outputs['combined']['model_out']['output'][idx][..., -1:]
            t_intervals = model_outputs['combined']['model_in']['t_intervals']
        else:
            pred_sigma = model_outputs['combined']['model_out']['output'][..., -1:]
            t_intervals = model_outputs['combined']['model_in']['t_intervals']
    else:
        pred_sigma = model_outputs['sigma']['model_out']['output']
        t_intervals = model_outputs['sigma']['model_in']['t_intervals']

    if isinstance(pred_sigma, list):
        pred_sigma = pred_sigma[idx]

    pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)[..., :-1, 0]

    # blur weights
    pred_weights = torch.cat((pred_weights, pred_weights[..., -1:]), dim=-1)
    weights_max = torch.maximum(pred_weights[..., :-1], pred_weights[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
    pred_weights = weights_blur + offset

    pdf = pred_weights / torch.sum(pred_weights, dim=-1, keepdim=True)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1).squeeze()  # batch_pixels, num_bins=samples_per_ray-1)
    cdf = cdf.detach()
    num_samples = pred_sigma.shape[-2]
    u = torch.rand(list(cdf.shape[:-1])+[num_samples], device=pred_weights.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds), inds-1)
    above = torch.min((cdf.shape[-1]-1)*torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), -1)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0])/denom
    t_vals = (bins_g[..., 0] + t*(bins_g[..., 1]-bins_g[..., 0])).unsqueeze(-1)
    t_vals, _ = torch.sort(t_vals, dim=-2)

    ray_dirs = model_inputs['ray_directions']
    ray_orgs = model_inputs['ray_origins']

    t_vals = t_vals[..., 0]
    t_intervals = t_vals[..., 1:] - t_vals[..., :-1]
    t_intervals = torch.cat((t_intervals, 1e10*torch.ones_like(t_intervals[:, 0:1])), dim=-1)
    t_intervals = (t_intervals * ray_dirs.norm(p=2, dim=-1))[..., None]
    t_vals = t_vals[..., None]

    if ray_dirs.ndim == 4:
        t_vals = t_vals[None, ...]

    model_inputs.update({'t': t_vals})
    model_inputs.update({'ray_samples': ray_orgs + ray_dirs * t_vals})
    model_inputs.update({'t_intervals': t_intervals})

    return model_inputs
