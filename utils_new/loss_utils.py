# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from utils_new.tool_utils import muLaw_tonemap


class RGBLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ssim_weight = configs["lambda_ssim"]
        self.mu = configs["mu"]
        self.use_tonemap = configs["use_tonemap"]
        self.color_loss_type = configs["color_loss_type"]
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(configs["device"])

    def forward(self, rgb_raw, gt_rgb_raw):
        if self.use_tonemap:
            rgb = muLaw_tonemap(torch.clamp(rgb_raw, 0.0, 1.0), self.mu)
            gt_rgb = muLaw_tonemap(gt_rgb_raw, self.mu)
        else:
            rgb = torch.clamp(rgb_raw, 0.0, 1.0)
            gt_rgb = gt_rgb_raw

        if self.color_loss_type == "l1":
            color_loss = torch.abs(rgb - gt_rgb).view(rgb.shape[0], -1).mean(1)
        elif self.color_loss_type == "l2":
            color_loss = (torch.abs(rgb - gt_rgb) ** 2).view(rgb.shape[0], -1).mean(1)
        else:
            raise NotImplementedError
        # ssim_loss = 1.0 - self.ssim(rgb.permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2))
        ssim_loss = 1.0 - fused_ssim(
            rgb.permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2), padding="valid"
        )
        return (
            1.0 - self.ssim_weight
        ) * color_loss.mean() + self.ssim_weight * ssim_loss, [
            color_loss[i].item() for i in range(len(color_loss))
        ]


class DepthLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.depth_weight = configs["lambda_depth"]

    def forward(self, depth, gt_depth):
        if self.depth_weight > 0:
            valid_depth = (gt_depth > 0).float()
            l1_loss = torch.abs(depth - gt_depth) * valid_depth

            return l1_loss.sum() / valid_depth.sum() * self.depth_weight
        else:
            return 0


class NormalLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.normal_weight = configs["lambda_normal"]

    def forward(self, normals, normals_from_depth, alphas):
        if self.normal_weight > 0:
            normals = normals.permute((0, 3, 1, 2))
            normals_from_depth *= alphas.detach()
            normals_from_depth = normals_from_depth.permute((0, 3, 1, 2))
            normal_error = 1 - torch.abs((normals * normals_from_depth).sum(dim=1))
            return self.normal_weight * normal_error.mean()
        else:
            return 0


class DistortionLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.distortion_weight = configs["lambda_distortion"]

    def forward(self, distortion):
        if self.distortion_weight > 0:
            return self.distortion_weight * distortion.mean()
        else:
            return 0


class NormalRegularizationLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.normal_reg_weight = configs["lambda_normal_reg"]
        self.normal_reg_threshold = configs["normal_reg_threshold"]
        self.grad_cov = torch.zeros((3, 3, 3, 3)).float().cuda()
        for i in range(3):
            self.grad_cov[i, i, :, :] = (
                torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().cuda()
                / 8.0
            )

    def forward(self, normals, colors):
        if self.normal_reg_weight > 0:
            color_grad = F.conv2d(colors, self.grad_cov, padding=1)
            color_grad_norm = color_grad.pow(2).sum(dim=1).sqrt()

            normals_dx = 1 - (normals[:, :, :-1, :-1] * normals[:, :, 1:, :-1]).sum(
                dim=1
            )
            normals_dy = 1 - (normals[:, :, :-1, :-1] * normals[:, :, :-1, 1:]).sum(
                dim=1
            )
            normals_grad_norm = normals_dx + normals_dy

            color_mask = (color_grad_norm < self.normal_reg_threshold).float().detach()

            return self.normal_reg_weight * (
                (normals_grad_norm * color_mask[:, :-1, :-1]).mean()
            ), color_mask

        else:
            return 0, None


class GaussianLoss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.gloss_weight = configs["lambda_g"]
        self.g_loss_type = configs["g_loss_type"]
        self.min_scale_ratio = configs["min_scale_ratio"]
        self.device = configs["device"]

    def forward(self, gaussians, level=-1):
        if self.gloss_weight == 0.0:
            return torch.tensor(0.0).to(self.device)

        if self.g_loss_type == "iso":
            scaling = gaussians.get_scaling(level)
            g_loss = torch.mean((scaling - scaling.mean(dim=1, keepdim=True)) ** 2)
        elif self.g_loss_type == "min_constraint":
            scaling = gaussians.get_scaling(level)
            max_scaling = torch.max(scaling, dim=1, keepdim=True)[0]
            min_scaling = torch.min(scaling, dim=1, keepdim=True)[0]
            g_loss = torch.mean(
                (
                    torch.clamp(min_scaling * self.min_scale_ratio, max=max_scaling)
                    - max_scaling
                )
                ** 2
            )

        return self.gloss_weight * g_loss
