# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils

def neumann_loss(mnfld_grad):
    neumann_term = (mnfld_grad.norm(2, dim=2)).abs().mean()
    return neumann_term

class Loss(nn.Module):
    def __init__(self, weights=[3e3, 1e2, 5e1, 1e2]):
        super().__init__()
        self.weights = weights

    def forward(self, output_pred, mnfld_points, nonmnfld_points,near_points):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        manifold_pred = output_pred["manifold_pnts_pred"]
        near_points_pred = output_pred['near_points_pred']

        mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        near_points_grad = utils.gradient(near_points, near_points_pred)

        # hessian_term
        mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
        mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])

        near_points_dx = utils.gradient(near_points, near_points_grad[:, :, 0])
        near_points_dy = utils.gradient(near_points, near_points_grad[:, :, 1])

        mnfld_det =torch.tensor([0],device=mnfld_points.device)
        near_points_det =torch.tensor([0],device=mnfld_points.device)

        if dims == 2:
            a = 2
            #manifold
            mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1) - a * torch.eye(2,device=mnfld_points.device)
            mnfld_det = torch.det(mnfld_hessian_term).abs().mean()
            #near_points
            near_points_hessian_term = torch.stack((near_points_dx, near_points_dy), dim=-1) - a * torch.eye(2,
                                                                                                             device=mnfld_points.device)
            near_points_det = torch.det(near_points_hessian_term).abs().mean()

            nonmnfld_points_pred = output_pred['nonmanifold_pnts_pred']
            nonmnfld_points_grad = utils.gradient(nonmnfld_points, nonmnfld_points_pred)
            nonmnfld_points_dx = utils.gradient(nonmnfld_points, nonmnfld_points_grad[:, :, 0])
            nonmnfld_points_dy = utils.gradient(nonmnfld_points, nonmnfld_points_grad[:, :, 1])
            nonmnfld_points_hessian_term = torch.stack((nonmnfld_points_dx, nonmnfld_points_dy), dim=-1) - a * torch.eye(2,
                                                                                                             device=mnfld_points.device)
            nonmnfld_points_det = torch.det(nonmnfld_points_hessian_term).abs().mean()

        if dims == 3:
            a = 2000
            mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
            near_points_dz = utils.gradient(near_points, near_points_grad[:, :, 2])
            #
            # # manifold
            mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy,mnfld_dz), dim=-1)  - a * torch.eye(3,device=mnfld_points.device)
            mnfld_det = torch.det(mnfld_hessian_term).abs().mean()

            # near_points
            near_points_hessian_term = torch.stack((near_points_dx, near_points_dy,near_points_dz), dim=-1) - a * torch.eye(3, device=mnfld_points.device)
            near_points_det = torch.det(near_points_hessian_term).abs().mean()

        ma_term = mnfld_det + near_points_det

        neumann_term = neumann_loss(mnfld_grad)

        # manifold_pred_mean = manifold_pred.mean()
        dirichlet_term = torch.abs(manifold_pred).mean()
        inter_term = torch.exp(-5e2 * torch.abs(near_points_pred)).mean()
        if dims == 2:
            inter_term = torch.exp(-1e2 * torch.abs(nonmnfld_points_pred)).mean()
        #########################################
        # Losses
        #########################################
        loss = self.weights[0]*dirichlet_term + self.weights[1] * inter_term + self.weights[2]*neumann_term + \
               self.weights[3] * ma_term

        return {"loss": loss, 'dirichlet_term': dirichlet_term, 'inter_term': inter_term,
                'neumann_term': neumann_term, 'ma_term':ma_term,
                }
