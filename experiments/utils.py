import os
import time

import torch
import trimesh
import numpy as np
from fvcore.nn import FlopCountAnalysis
from skimage import measure
from tqdm import tqdm


import matplotlib.pyplot as plt
import torch
import os
import sys
import training
import numpy as np
import dataio
from torchvision.utils import make_grid
import skimage.metrics
import skimage.transform
import torch.nn
from tqdm import tqdm
import forward_models
import modules
import skimage.io


def to_numpy(x):
    return x.detach().cpu().numpy()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_simple_1D_function_summary(dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train'):

    model_input, gt = dataset[0]
    model_input = {k: v.unsqueeze(0) for k, v in model_input.items()}
    model_input = training.dict2cuda(model_input)
    model_output = model(model_input)

    pred_func = to_numpy(model_output['model_out']['output'].squeeze())   # B, Samples, DimOut
    coords = to_numpy(gt['coords'].squeeze())       # B, Samples, DimIn

    val_coords = coords
    val_pred_func = pred_func
    val_gt_func = to_numpy(gt['func'].squeeze())                    # B, Samples, DimOut

    idx = model_input['idx'].cpu().long().detach().numpy().squeeze()
    train_coords = coords[idx]
    train_pred_func = pred_func[idx]

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(val_coords, val_gt_func, label='GT', linewidth=2)
    plt.plot(val_coords, val_pred_func, label='Val')
    plt.plot(train_coords, train_pred_func, '.', label='Train', markersize=8)

    plt.ylim([-1, 1])
    plt.legend()

    plt.subplot(212)
    oversample = 32

    freqs = np.fft.fftshift(np.fft.fftfreq(oversample * len(val_coords), 1/len(val_coords)))
    val_gt_func = np.pad(val_gt_func, ((0, (oversample-1)*len(val_gt_func))))
    val_pred_func = np.pad(val_pred_func, ((0, (oversample-1)*len(val_pred_func))))

    plt.plot(freqs, np.fft.fftshift(np.abs(np.fft.fft(val_gt_func))), label='GT', linewidth=2)
    plt.plot(freqs, np.fft.fftshift(np.abs(np.fft.fft(val_pred_func))), label='Val')
    plt.xlim(-32, 32)

    plt.tight_layout()
    writer.add_figure(prefix + '/gt_vs_pred', fig, global_step=total_steps)


def write_multiscale_image_summary(image_resolution, train_dataset, model, model_input, gt,
                                   model_output, writer, total_steps, prefix='train_',
                                   use_resized=False, val_dataset=None, write_images=False):

    if use_resized:
        gt_imgs = [dataio.lin2img(gt_img, image_resolution) for gt_img in gt['img']]
        gt_img = gt_imgs[-1]
    else:
        gt_img = dataio.lin2img(gt['img'], image_resolution)
        gt_imgs = [gt_img, ]

    pred_img = [dataio.lin2img(out, image_resolution).clamp(0, 1) for out in model_output['model_out']['output']]

    if isinstance(model, modules.MultiscaleBACON):
        write_psnr(pred_img[-1], gt_img, writer, total_steps, prefix+'img_')

        output_vs_gt = torch.cat((*gt_imgs, *pred_img), dim=0)

        all_imgs = gt_imgs + pred_img

        spectrums = [torch.fft.fftshift(torch.fft.fft2(img)) for img in all_imgs]

        smax = torch.max(spectrums[0].real)
        spectrums = [spectrum / smax * 50 for spectrum in spectrums]

        spectrums = [(torch.clamp(abs(torch.norm(spectrum, dim=1, keepdim=True)), 0, 1))**(1/2) for spectrum in spectrums]
        spectrums = [spectrum.repeat(1, 3, 1, 1) for spectrum in spectrums]
        output_vs_gt = torch.cat((output_vs_gt, *spectrums), dim=0)

        if write_images:
            logdir = os.path.join(writer.log_dir, 'imgs')
            cond_mkdir(logdir)
            for idx, im in enumerate(all_imgs):
                im = im.squeeze(0)
                im = im.permute(1, 2, 0)
                im = (im * 255).detach().cpu().numpy().astype(np.uint8)
                skimage.io.imsave(os.path.join(logdir, f'im_{idx}_{total_steps:04d}.png'), im)
            for idx, im in enumerate(spectrums):
                im = im.squeeze(0)
                im = im.permute(1, 2, 0)
                im = (im * 255).detach().cpu().numpy().astype(np.uint8)
                skimage.io.imsave(os.path.join(logdir, f'spectrum_{idx}_{total_steps:04d}.png'), im)

        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False,
                         normalize=False, nrow=output_vs_gt.shape[0]//2), global_step=total_steps)

    else:
        if use_resized:
            output_vs_gt = torch.cat((*gt_imgs, *pred_img), dim=0)
            writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False,
                             normalize=False, nrow=output_vs_gt.shape[0]), global_step=total_steps)
            write_psnr(pred_img[-1], gt_imgs[-1], writer, total_steps, prefix+'img_')

        else:
            output_vs_gt = torch.cat((gt_img, *pred_img), dim=0)
            writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False,
                             normalize=False, nrow=output_vs_gt.shape[0]), global_step=total_steps)
            write_psnr(pred_img[-1], gt_img, writer, total_steps, prefix+'img_')

    image_resolution = [2*r for r in image_resolution]
    model_input, gt = val_dataset[0]
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cuda()})
        else:
            tmp.update({key: value})
    model_input = tmp

    gt = training.dict2cuda(gt)

    with torch.no_grad():
        model_output = model(model_input)

    if isinstance(gt['img'], list):
        gt_img = gt['img'][-1][None, ...]
    else:
        gt_img = gt['img'][None, ...]

    gt_img = dataio.lin2img(gt_img, image_resolution)
    pred_img = [dataio.lin2img(out, image_resolution).clamp(0, 1) for out in model_output['model_out']['output']]

    output_vs_gt = torch.cat((gt_img, *pred_img), dim=-1)
    writer.add_image(prefix + 'val_gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    write_psnr(pred_img[-1], gt_img, writer, total_steps, 'val_img_')


def write_image_summary(image_resolution, train_dataset, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_',
                        val_dataset=None):

    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out']['output'], image_resolution)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    write_psnr(pred_img, gt_img, writer, total_steps, prefix+'img_')

    # validation samples
    if val_dataset is None:
        return

    image_resolution = [2*r for r in image_resolution]
    model_input, gt = val_dataset[0]
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cuda()})
        else:
            tmp.update({key: value})
    model_input = tmp
    gt = {key: value[None, ...].cuda() for key, value in gt.items()}

    with torch.no_grad():
        model_output = model(model_input)

    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out']['output'], image_resolution)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'val_gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    write_psnr(pred_img, gt_img, writer, total_steps, 'val_img_')


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        # p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        # trgt = (trgt / 2.) + 0.5

        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)


def write_multiscale_radiance_summary(models, train_dataloader, val_dataloader, loss_fn, optims,
                                      meta, gt, writer, total_steps,
                                      chunk_size_eval, num_views_to_disp_at_training,
                                      hierarchical_sampling=False):
    print('Running validation and logging...')

    chunk_size = chunk_size_eval

    '''' Log training set '''
    # sample rays across the whole image
    train_dataloader.dataset.toggle_logging_sampling()

    in_dict, meta_dict, gt_dict = next(iter(train_dataloader))
    in_dict = subsample_dict(in_dict, num_views_to_disp_at_training)

    # show progress
    samples_per_view = train_dataloader.dataset.samples_per_view
    num_chunks = num_views_to_disp_at_training * samples_per_view // chunk_size
    pbar = tqdm(total=4*len(models)*int(num_chunks))

    # Here, the number of images we get depend on the batch_size which is likely not going to be 1
    # so, be aware that we are processing multiple images
    with torch.no_grad():
        out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                    for key, model in models.items()}

        if hierarchical_sampling:
            # num_samples = 128
            in_dict = training.sample_pdf(in_dict, out_dict, idx=0)
            out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                        for key, model in models.items()}

    # Plot the sampling
    fig_sampling = plot_samples(out_dict)
    writer.add_figure('samples', fig_sampling, global_step=total_steps)

    if isinstance(gt_dict['pixel_samples'], list):
        gt_view = [v[0:num_views_to_disp_at_training, :, :, :3].detach().cpu() for v in gt_dict['pixel_samples']]
        view_shape = gt_view[0].shape
    else:
        gt_view = gt_dict['pixel_samples'][0:num_views_to_disp_at_training, :, :, :3].detach().cpu()  # Views,H,W,C
        view_shape = gt_view.shape

    pred_sigmas = [pred[..., -1:] for pred in out_dict['combined']['model_out']['output']]
    pred_rgbs = [pred[..., :-1] for pred in out_dict['combined']['model_out']['output']]

    if isinstance(out_dict['combined']['model_in']['t_intervals'], list):
        t_intervals = [t_interval for t_interval in out_dict['combined']['model_in']['t_intervals']]
    else:
        t_intervals = out_dict['combined']['model_in']['t_intervals']

    pred_views = []
    for idx, (sigma, rgb) in enumerate(zip(pred_sigmas, pred_rgbs)):

        if isinstance(t_intervals, list):
            t_interval = t_intervals[idx]
        else:
            t_interval = t_intervals

        pred_weights = forward_models.compute_transmittance_weights(sigma, t_interval)
        pred_pixels = forward_models.compute_tomo_radiance(pred_weights, rgb)

        # log the images
        pred_view = pred_pixels.view(view_shape).detach().cpu()  # Views,H,W,C
        pred_view = torch.clamp(pred_view, 0, 1)
        pred_views.append(pred_view.permute(0, 3, 1, 2))

    if isinstance(gt_view, list):
        gt_view = [v.permute(0, 3, 1, 2) for v in gt_view]
        train_psnr = peak_signal_noise_ratio(gt_view[-1][0], pred_views[-1][0])
    else:
        gt_view = gt_view.permute(0, 3, 1, 2)[0]
        train_psnr = peak_signal_noise_ratio(gt_view[0], pred_views[-1][0])

    writer.add_scalar("train: PSNR", train_psnr, global_step=total_steps)

    pred_view = make_grid(torch.cat(pred_views, dim=0), scale_each=False, normalize=False)

    if isinstance(gt_view, list):
        gt_view = make_grid(torch.cat(gt_view, dim=0), scale_each=False, normalize=False)

    if pred_view.shape[1] < 512:
        scale = 512 // pred_view.shape[1]
        pred_view = torch.nn.functional.interpolate(pred_view.unsqueeze(0), scale_factor=scale, mode='nearest')
        pred_view = pred_view.squeeze(0)
        gt_view = torch.nn.functional.interpolate(gt_view.unsqueeze(0), scale_factor=scale, mode='nearest')
        gt_view = gt_view.squeeze(0)

    writer.add_image("train: Pred", pred_view, global_step=total_steps)
    writer.add_image("train: GT", gt_view, global_step=total_steps)

    # reset sampling back to defaults
    train_dataloader.dataset.toggle_logging_sampling()

    # Free by hand to be sure
    del in_dict, meta_dict, gt_dict
    del pred_view, pred_pixels, pred_weights,

    ''' Log Validation images '''
    val_dataloader.dataset.toggle_logging_sampling()

    in_dict, meta_dict, gt_dict = next(iter(val_dataloader))

    with torch.no_grad():
        out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                    for key, model in models.items()}

        if hierarchical_sampling:
            pred_z_weights = []
            for i in range(4):
                # keep track of these to log the depth
                if 'combined' in out_dict:
                    sigma_tmp = out_dict['combined']['model_out']['output'][i][..., -1:]
                    t_interval_tmp = out_dict['combined']['model_in']['t_intervals']
                else:
                    sigma_tmp = out_dict['sigma']['model_out']['output'][i][..., -1:]
                    t_interval_tmp = out_dict['sigma']['model_in']['t_intervals']

                pred_z_weights.append(forward_models.compute_transmittance_weights(sigma_tmp, t_interval_tmp))

            in_dict = training.sample_pdf(in_dict, out_dict, idx=0)
            out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                        for key, model in models.items()}

    gt_dict_reshaped = gt_dict.copy()
    gt_dict_reshaped['pixel_samples'] = [g.view(1, -1, 3) for g in gt_dict_reshaped['pixel_samples']]
    losses = loss_fn(out_dict, gt_dict_reshaped)
    for loss_name, loss in losses.items():
        single_loss = loss.mean()
        writer.add_scalar('val_' + loss_name, single_loss, total_steps)

    if isinstance(gt_dict['pixel_samples'], list):
        gt_view = [v[0, :, :, :3].detach().cpu() for v in gt_dict['pixel_samples']]
        view_shape = gt_view[0].shape
    else:
        gt_view = gt_dict['pixel_samples'][0, :, :, :3].detach().cpu()  # Views,H,W,C
        view_shape = gt_view.shape

    pred_views = []
    pred_disps = []

    pred_sigmas = [pred[..., -1:] for pred in out_dict['combined']['model_out']['output']]
    pred_rgbs = [pred[..., :-1] for pred in out_dict['combined']['model_out']['output']]
    if isinstance(out_dict['combined']['model_in']['t_intervals'], list):
        t_intervals = [t_interval for t_interval in out_dict['combined']['model_in']['t_intervals']]
    else:
        t_intervals = out_dict['combined']['model_in']['t_intervals']

    for idx, (sigma, rgb) in enumerate(zip(pred_sigmas, pred_rgbs)):

        if isinstance(t_intervals, list):
            t_interval = t_intervals[idx]
        else:
            t_interval = t_intervals

        pred_weights = forward_models.compute_transmittance_weights(sigma, t_interval)
        pred_pixels = forward_models.compute_tomo_radiance(pred_weights, rgb)

        if not hierarchical_sampling:
            pred_z_weights = pred_weights

        pred_depth = forward_models.compute_tomo_depth(pred_z_weights[idx], meta_dict['zs'])
        pred_disp = forward_models.compute_disp_from_depth(pred_depth, pred_z_weights[idx])

        pred_view = pred_pixels.view(view_shape).detach().cpu().permute(2, 0, 1)
        pred_view = torch.clamp(pred_view, 0, 1)
        pred_disp_view = pred_disp.view(*view_shape[:2], 1).detach().cpu().permute(2, 0, 1)

        pred_views.append(pred_view[None, ...])
        pred_disps.append(pred_disp_view[None, ...])

    if isinstance(gt_view, list):
        gt_view = [v.permute(2, 0, 1)[None, ...] for v in gt_view]
        val_psnr = peak_signal_noise_ratio(gt_view[-1][0], pred_views[-1])
    else:
        gt_view = gt_view.permute(2, 0, 1)
        val_psnr = peak_signal_noise_ratio(gt_view, pred_views[-1])

    writer.add_scalar("val: PSNR", val_psnr, global_step=total_steps)

    pred_view = make_grid(torch.cat(pred_views, dim=0), scale_each=False, normalize=False)

    if isinstance(gt_view, list):
        gt_view = make_grid(torch.cat(gt_view, dim=0), scale_each=False, normalize=False)

    pred_disp_view = make_grid(torch.cat(pred_disps, dim=0), scale_each=False, normalize=False)

    if gt_view.shape[1] < 512:
        scale = 512 // gt_view.shape[1]
        pred_view = torch.nn.functional.interpolate(pred_view.unsqueeze(0), scale_factor=scale, mode='nearest')
        pred_disp_view = torch.nn.functional.interpolate(pred_disp_view.unsqueeze(0), scale_factor=scale, mode='nearest')
        pred_view = pred_view.squeeze(0)
        pred_disp_view = pred_disp_view.squeeze(0)
        gt_view = torch.nn.functional.interpolate(gt_view.unsqueeze(0), scale_factor=scale, mode='nearest')
        gt_view = gt_view.squeeze(0)

    writer.add_image("val: GT", gt_view, global_step=total_steps)
    writer.add_image("val: Pred", pred_view, global_step=total_steps)
    writer.add_image("val: Pred disp", pred_disp_view, global_step=total_steps)

    val_dataloader.dataset.toggle_logging_sampling()

    pbar.close()


def write_radiance_summary(models, train_dataloader, val_dataloader, loss_fn, optims,
                           meta, gt, writer, total_steps,
                           chunk_size_eval, num_views_to_disp_at_training,
                           hierarchical_sampling=False):
    print('Running validation and logging...')

    chunk_size = chunk_size_eval

    '''' Log training set '''
    # sample rays across the whole image
    train_dataloader.dataset.toggle_logging_sampling()

    in_dict, meta_dict, gt_dict = next(iter(train_dataloader))
    in_dict = subsample_dict(in_dict, num_views_to_disp_at_training)

    # show progress
    samples_per_view = train_dataloader.dataset.samples_per_view
    num_chunks = num_views_to_disp_at_training * samples_per_view // chunk_size
    pbar = tqdm(total=2*len(models)*int(num_chunks))

    # Here, the number of images we get depend on the batch_size which is likely not going to be 1
    # so, be aware that we are processing multiple images
    with torch.no_grad():
        out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                    for key, model in models.items()}

    if hierarchical_sampling:
        num_samples = 128
        in_dict = training.sample_pdf(in_dict, out_dict, num_samples)

        out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                    for key, model in models.items()}

    # Plot the sampling
    fig_sampling = plot_samples(out_dict)
    writer.add_figure('samples', fig_sampling, global_step=total_steps)
    gt_view = gt_dict['pixel_samples'][0:num_views_to_disp_at_training, :, :, :3].detach().cpu()

    pred_sigma = out_dict['combined']['model_out']['output'][..., -1:]
    pred_rgb = out_dict['combined']['model_out']['output'][..., :-1]
    t_intervals = out_dict['combined']['model_in']['t_intervals']

    pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)
    pred_pixels = forward_models.compute_tomo_radiance(pred_weights, pred_rgb)

    # log the images
    pred_view = pred_pixels.view(gt_view.shape).detach().cpu()  # Views,H,W,C
    pred_view = torch.clamp(pred_view, 0, 1)
    train_psnr = peak_signal_noise_ratio(gt_view[0], pred_view[0])
    writer.add_scalar("train: PSNR", train_psnr, global_step=total_steps)

    # add videos takes B,T,C,H,W and we simply use it here to tile images T=1
    writer.add_video("train: GT", gt_view.permute(0, 3, 1, 2)[:, None, :, :, :], global_step=total_steps)
    writer.add_video("train: Pred", pred_view.permute(0, 3, 1, 2)[:, None, :, :, :], global_step=total_steps)

    # reset sampling back to defaults
    train_dataloader.dataset.toggle_logging_sampling()

    # Free by hand to be sure
    del in_dict, meta_dict, gt_dict
    del pred_view, pred_pixels, pred_weights,

    ''' Log Validation images '''
    num_samples = 1
    val_dataloader.dataset.toggle_logging_sampling()

    for n in range(num_samples):  # we run a for loop of num_samples instead of a batch to use less cuda mem
        in_dict, meta_dict, gt_dict = next(iter(val_dataloader))

        with torch.no_grad():
            out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                        for key, model in models.items()}

        if hierarchical_sampling:
            num_samples = 128
            in_dict = training.sample_pdf(in_dict, out_dict, num_samples)

            out_dict = {key: process_batch_in_chunks(in_dict, model, chunk_size, progress=pbar)
                        for key, model in models.items()}

        gt_dict_reshaped = gt_dict.copy()
        gt_dict_reshaped['pixel_samples'] = gt_dict_reshaped['pixel_samples'].view(1, -1, 3)
        losses = loss_fn(out_dict, gt_dict_reshaped)

        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

        pred_sigma = out_dict['combined']['model_out']['output'][..., -1:]
        pred_rgb = out_dict['combined']['model_out']['output'][..., :-1]
        t_intervals = out_dict['combined']['model_in']['t_intervals']

        pred_weights = forward_models.compute_transmittance_weights(pred_sigma, t_intervals)
        pred_pixels = forward_models.compute_tomo_radiance(pred_weights, pred_rgb)

        pred_depth = forward_models.compute_tomo_depth(pred_weights, meta_dict['zs'])
        pred_disp = forward_models.compute_disp_from_depth(pred_depth, pred_weights)
        gt_view = gt_dict['pixel_samples'][0, :, :, :3].detach().cpu()

        pred_view = pred_pixels.view(gt_view.shape).detach().cpu().permute(2, 0, 1)
        pred_view = torch.clamp(pred_view, 0, 1)
        pred_disp_view = pred_disp.view(gt_view[:, :, 0:1].shape).detach().cpu().permute(2, 0, 1)
        gt_view = gt_view.permute(2, 0, 1)

        val_psnr = peak_signal_noise_ratio(gt_view, pred_view)
        writer.add_scalar("val: PSNR", val_psnr, global_step=total_steps)

        # nearest neighbor upsample image for easier viewing
        if gt_view.shape[1] < 512:
            scale = 512 // gt_view.shape[1]
            gt_view = torch.nn.functional.interpolate(gt_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            pred_view = torch.nn.functional.interpolate(pred_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            pred_disp_view = torch.nn.functional.interpolate(pred_disp_view.unsqueeze(0), scale_factor=scale, mode='nearest')
            gt_view = gt_view.squeeze(0)
            pred_view = pred_view.squeeze(0)
            pred_disp_view = pred_disp_view.squeeze(0)

        writer.add_image("val: GT", gt_view, global_step=total_steps)
        writer.add_image("val: Pred", pred_view, global_step=total_steps)
        writer.add_image("val: Pred disp", pred_disp_view, global_step=total_steps)

    val_dataloader.dataset.toggle_logging_sampling()

    # close progress bar
    pbar.close()


def make_contour_plot(array_2d, mode='log', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)
    else:
        fig = plt.gcf()

    if(mode == 'log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode == 'lin'):
        num_levels = 10
        levels = np.linspace(-.5, .5, num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    fig.colorbar(CS, ax=ax)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_slice(model, writer, total_steps, prefix='train_', is_multi=False):
    ''' write slices of sdf in each plane '''

    slice_coords_2d = dataio.get_mgrid(512)

    with torch.no_grad():

        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]),
                                    slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}
        yz_model_out = model(yz_slice_model_input)

        sdf_values = yz_model_out['model_out']
        all_sdf_values = sdf_values if is_multi else [sdf_values, ]

        fig, axs = plt.subplots(1, len(all_sdf_values), figsize=(2.75*len(all_sdf_values), 2.75), dpi=100)
        for idx, sdf_values in enumerate(all_sdf_values):
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            ax = axs if not isinstance(axs, np.ndarray) else axs[idx]
            make_contour_plot(sdf_values, ax=ax)

        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:, :1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,  -1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        all_sdf_values = sdf_values if is_multi else [sdf_values, ]

        fig, axs = plt.subplots(1, len(all_sdf_values), figsize=(2.75*len(all_sdf_values), 2.75), dpi=100)
        for idx, sdf_values in enumerate(all_sdf_values):
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            ax = axs if not isinstance(axs, np.ndarray) else axs[idx]
            make_contour_plot(sdf_values, ax=ax)

        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:, :2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        all_sdf_values = sdf_values if is_multi else [sdf_values, ]

        fig, axs = plt.subplots(1, len(all_sdf_values), figsize=(2.75*len(all_sdf_values), 2.75), dpi=100)
        for idx, sdf_values in enumerate(all_sdf_values):
            sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
            ax = axs if not isinstance(axs, np.ndarray) else axs[idx]
            make_contour_plot(sdf_values, ax=ax)

        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)
        plt.close('all')


def write_multiscale_sdf_summary(model, model_input, gt, model_output, writer,
                                 total_steps, prefix='train_'):
    write_sdf_slice(model, writer, total_steps, prefix, is_multi=True)


def write_sdf_summary(model, model_input, gt, model_output, writer,
                      total_steps, prefix='train_'):
    write_sdf_slice(model, writer, total_steps, prefix, is_multi=False)


def plot_samples(out_dict, num_rays_to_visu=10, xlim=(0, 6)):
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = plt.subplot(2, 1, 1)
    plt.title('sigma ray samples')

    if 'combined' in out_dict:
        if isinstance(out_dict['combined']['model_in']['t_intervals'], list):
            t_intervals = out_dict['combined']['model_in']['t_intervals'][-1][0, ..., 0]
        else:
            t_intervals = out_dict['combined']['model_in']['t_intervals'][0, ..., 0]
    else:
        t_intervals = out_dict['sigma']['model_in']['t_intervals'][0, ..., 0]

    t_transformed = torch.cumsum(t_intervals, dim=-1).cpu().detach()

    num_rays = t_transformed.shape[0]
    ts = t_transformed[num_rays//2:num_rays//2+num_rays_to_visu, :-1]
    num_samples = ts.shape[1]
    idcs = torch.arange(0, num_rays_to_visu).reshape(-1, 1).repeat(1, num_samples).float()
    idcs2 = torch.arange(0, num_samples).repeat(num_rays_to_visu).float()
    plt.scatter(ts.reshape(-1), idcs.reshape(-1), marker='|', c=idcs2.reshape(-1)/num_samples, cmap='prism')
    ax.set_ylabel('ray idx')
    ax.set_xlabel('sample position')
    ax.set_yticklabels([])
    plt.xlim(xlim)

    ax = plt.subplot(2, 1, 2)
    plt.title('rgb ray samples')
    t_transformed = torch.cumsum(t_intervals, dim=-1).cpu().detach()  # we could have t but it requires many more changes
    num_rays = t_transformed.shape[0]
    ts = t_transformed[num_rays//2:num_rays//2+num_rays_to_visu, :-1]
    num_samples = ts.shape[1]
    idcs = torch.arange(0, num_rays_to_visu).reshape(-1, 1).repeat(1, num_samples).float()
    idcs2 = torch.arange(0, num_samples).repeat(num_rays_to_visu).float()
    plt.scatter(ts.reshape(-1), idcs.reshape(-1), marker='|', c=idcs2.reshape(-1)/num_samples, cmap='prism')
    ax.set_ylabel('ray idx')
    ax.set_xlabel('sample position')
    ax.set_yticklabels([])
    plt.xlim(xlim)

    return fig


def process_batch_in_chunks(in_dict, model, max_chunk_size=1024, progress=None):
    in_chunked = []
    for key in in_dict:
        num_views, num_rays, num_samples_per_rays, num_dims = in_dict[key].shape
        chunks = torch.split(in_dict[key].view(-1, num_samples_per_rays, num_dims), max_chunk_size)
        in_chunked.append(chunks)

    list_chunked_batched_in = \
        [{k: v for k, v in zip(in_dict.keys(), curr_chunks)} for curr_chunks in zip(*in_chunked)]
    del in_chunked

    list_chunked_batched_out_out = {}
    list_chunked_batched_out_in = {}

    for chunk_batched_in in list_chunked_batched_in:
        chunk_batched_in = {k: v.cuda() for k, v in chunk_batched_in.items()}
        tmp = model(chunk_batched_in)
        tmp = training.dict2cpu(tmp)

        for key in tmp['model_out']:
            if tmp['model_out'][key] is None:
                continue

            if isinstance(tmp['model_out'][key], list):
                for idx, elem in enumerate(tmp['model_out'][key]):
                    out_ = elem.detach().clone().requires_grad_(False)
                    list_chunked_batched_out_out.setdefault(key, [[] for _ in range(len(tmp['model_out'][key]))])[idx].append(out_)
            else:
                out_ = tmp['model_out'][key].detach().clone().requires_grad_(False)
                list_chunked_batched_out_out.setdefault(key, []).append(out_)

        for key in tmp['model_in']:
            if tmp['model_in'][key] is None:
                continue

            in_ = tmp['model_in'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_in.setdefault(key, []).append(in_)

        del tmp, chunk_batched_in

        if progress is not None:
            progress.update(1)

    # Reassemble the output chunks in a batch
    batched_out = {}
    shape_out = list([num_views, num_rays, num_samples_per_rays, num_dims])
    for key in list_chunked_batched_out_out:
        if isinstance(list_chunked_batched_out_out[key][0], list):
            batched_out_lin = []
            for idx, li in enumerate(list_chunked_batched_out_out[key]):
                b = torch.cat(li, dim=0)
                shape_out[-1] = b.shape[-1]
                shape_out[-2] = -1
                batched_out_lin.append(b.reshape(shape_out))
            batched_out[key] = batched_out_lin
        else:
            batched_out_lin = torch.cat(list_chunked_batched_out_out[key], dim=0)
            shape_out[-1] = batched_out_lin.shape[-1]
            shape_out[-2] = -1
            batched_out[key] = batched_out_lin.reshape(shape_out)

    batched_in = {}
    shape_in = list([num_views, num_rays, num_samples_per_rays, num_dims])
    for key in list_chunked_batched_out_in:
        batched_in_lin = torch.cat(list_chunked_batched_out_in[key], dim=0)
        shape_in[-1] = batched_in_lin.shape[-1]
        shape_in[-2] = -1
        batched_in[key] = batched_in_lin.reshape(shape_in)

    # print(f"batched_out={batched_out.shape}")
    return {'model_in': batched_in, 'model_out': batched_out}


def peak_signal_noise_ratio(gt, pred):
    ''' Calculate PSNR using GT and predicted image (assumes valid values between 0 and 1 '''
    pred = torch.clamp(pred, 0, 1)
    return 10 * torch.log10(1 / torch.mean((gt - pred)**2))


def subsample_dict(in_dict, num_views):
    return {key: value[0:num_views, ...] for key, value in in_dict.items()}


class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def log_string(out_str, log_file):
    out_str = str(out_str)
    # helper function to log a string to file and print it
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)

def get_3d_grid(resolution=100, bbox=1.1 * np.array([[-1, 1], [-1, 1], [-1, 1]]), device=None, eps=0.0,
                dtype=np.float32):
    # generate points on a uniform grid within  a given range
    # reimplemented from SAL : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    # and IGR : https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py

    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])
    if (shortest_axis == 0):
        x = np.linspace(bbox[0, 0] - eps, bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(bbox[1, 0] - eps, bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype))  #
    # grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16),
    #                                          device=device)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}



def cal_layer_cube(model, pnts, device, octree, threshold):
    node_idxs = octree.query_points(pnts.numpy())
    points_octree_value = octree.values[node_idxs]

    mask = points_octree_value > -8

    pnts = pnts[mask]
    node_idxs = node_idxs[mask]
    unique_idxs, inverse = np.unique(node_idxs, return_inverse=True)
    pnts.requires_grad_()
    N = pnts.shape[0]
    z = torch.zeros((10, N, 1)) - 1

    j = 0
    model_input = {}

    for point in tqdm(torch.split(pnts, 300000, dim=0)):
        point = point.to(device).type(torch.float32)
        model_input['coords'] = point

        pred_list = model(model_input)['model_out']
        pred = None
        index_start = j * 300000
        index_end = index_start + 300000
        for (i, tem) in enumerate(pred_list):
            if pred == None:
                pred = tem
            else:
                pred = pred + tem
            z[i, index_start:index_end] = pred.detach().cpu()
        j += 1
    z = z[0:i + 1]

    layer_cube = np.ones((N, 1))
    for i in range(z.shape[0]):
        if i == 0:
            continue
        else:
            mask_1 = torch.abs(z[-1] - z[i - 1]) >= threshold
            mask_2 = torch.sign(z[i - 1]) != torch.sign(z[-1])
            mask = (mask_1.squeeze()) | mask_2.squeeze()
            layer_cube[mask] = i + 1

    max_vals = octree.values[unique_idxs]
    np.maximum.at(max_vals, inverse, layer_cube.flatten())
    octree.values[unique_idxs] = max_vals
    octree.values[octree.values == -5] = np.round(max_vals.mean())


def implicit2mesh(model, grid_res, translate=[0., 0., 0.], scale=1, device=None, octree_queryer=None, log_file=None):
    model.eval()
    bbox = np.array([[-1, 1], [-1, 1], [-1, 1]]) * 1.1
    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    print('Finished getting grid_dict')
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"].to(device)

    N = pnts.shape[0]
    t0 = time.perf_counter()

    points_octree_value = octree_queryer.query_points_value(pnts)
    all_z = torch.zeros(N, device=device)


    mask_0 = points_octree_value < -9

    mask_1 = ~mask_0

    pnts = pnts[mask_1]
    layer_cube = points_octree_value[mask_1]

    t1 = time.perf_counter()
    log_string('octree: {}'.format(t1 - t0),log_file)

    t1 = time.perf_counter()

    z_pnts = []
    model_input = {}
    for sub_pnts,sub_layer_cube in zip(torch.split(pnts, 300000, dim=0),torch.split(layer_cube, 300000, dim=0)):
        model_input['coords'] = sub_pnts
        preds = model(model_input, layers=sub_layer_cube)['model_out']
        N = sub_pnts.size(0)
        real_pred = torch.zeros(N, 1, device=device)
        for i, pred in enumerate(preds):
            real_pred[(sub_layer_cube >= (i + 1))] += pred
        z_pnts.append(real_pred.detach().cpu().numpy())
    t2 = time.perf_counter()
    log_string('net: {}'.format(t2 - t1),log_file)

    all_z[mask_0] = points_octree_value[mask_0].to(torch.float32) + 12
    all_z = all_z.detach().cpu().numpy()

    z_pnts = np.concatenate(z_pnts, axis=0)
    all_z[mask_1.cpu().numpy()] = z_pnts.flatten()
    all_z = all_z.reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                  grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2])

    verts, faces, normals, values = measure.marching_cubes(volume=all_z, level=0.0,
                                                           spacing=(cell_width, cell_width, cell_width))

    verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
    verts = verts * (1 / scale) - translate
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)

    return mesh
