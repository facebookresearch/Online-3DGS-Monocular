# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import math
from errno import EEXIST
from os import makedirs, path
from typing import NamedTuple

import cv2
import imageio
import numpy as np
import torch
import yaml
# from PIL import Image  # Removed unused import


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def loadExr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def saveExr(path, img):
    if len(img.shape) == 3 and img.shape[2] == 2:
        zeros = np.zeros((img.shape[0], img.shape[1], 1))
        img = np.concatenate([img, zeros], axis=2)

    img = img.astype(np.float32)

    imageio.imwrite(path, img)


def saveTensorAsPNG(tensor, path):
    if tensor.shape[0] <= 3:
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        img = tensor.detach().cpu().numpy()
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def saveTensorAsEXR(tensor, path):
    if tensor.shape[0] <= 3:
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    else:
        img = tensor.detach().cpu().numpy().astype(np.float32)
    saveExr(path, img)


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def rgb_to_sh_np(rgb: np.ndarray) -> np.ndarray:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def load_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


# return in bgr
def convert_depth_to_jetimg(depth):
    # depth = depth / 4.0
    # depth = depth ** 0.5
    depth = np.clip(depth, 0.0, 1.0)
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    return depth


def convert_err_to_jetimg(err, scale=1.0):
    err = err * scale
    err = np.clip(err, 0.0, 1.0)
    err = (err * 255).astype(np.uint8)
    err = cv2.applyColorMap(err, cv2.COLORMAP_JET)

    return err


def gammaCorrection(img, gamma=2.2):
    invGamma = 1.0 / gamma

    try:
        return np.power(np.clip(img, 0.0, 1.0), invGamma)
    except:
        return torch.pow(torch.clamp(img, 0.0, 1.0), invGamma)


def invGammaCorrection(img, gamma=2.2):
    try:
        return np.power(img, gamma)
    except:
        return torch.pow(img, gamma)


def adjustExposure(img, exposure=1.0):
    if exposure == 1.0:
        return img

    deGammaImg = invGammaCorrection(img, gamma=2.2)
    adjustedImg = deGammaImg * exposure
    return gammaCorrection(adjustedImg, gamma=2.2)


def adjustImg(img, exposure=1.0, gamma=1.0):
    if exposure == 1.0:
        adjustedImg = img
    else:
        deGammaImg = invGammaCorrection(img, gamma=2.2)
        adjustedImg = deGammaImg * exposure
        adjustedImg = gammaCorrection(adjustedImg, gamma=2.2)

    if gamma == 1.0:
        return adjustedImg
    else:
        return gammaCorrection(adjustedImg, gamma=gamma)


def adjustImg_muLaw_Numpy(img, exposure=1.0, gamma=1.0):
    if exposure == 1.0:
        adjustedImg = img
    else:
        deGammaImg = inverse_muLaw_tonemap_numpy(img)
        adjustedImg = deGammaImg * exposure
        adjustedImg = muLaw_tonemap_numpy(adjustedImg)

    if gamma == 1.0:
        return adjustedImg
    else:
        return muLaw_tonemap_numpy(adjustedImg)


def muLaw_tonemap(x, mu=8.0):
    return torch.log(1 + mu * x) / np.log(1 + mu)


def muLaw_tonemap_numpy(x, mu=8.0):
    return np.log(1 + mu * x) / np.log(1 + mu)


def inverse_muLaw_tonemap_numpy(x, mu=8.0):
    return (np.exp(x * np.log(1 + mu)) - 1) / mu


def interpolate_pose(pose1, pose2, alpha):
    pose1_cam2world = np.linalg.inv(pose1)
    pose2_cam2world = np.linalg.inv(pose2)

    pose1_pos = pose1_cam2world[:3, 3]
    pose2_pos = pose2_cam2world[:3, 3]

    pose1_rot = pose1_cam2world[:3, :3]
    pose2_rot = pose2_cam2world[:3, :3]

    diff_rot = np.matmul(np.linalg.inv(pose1_rot), pose2_rot)
    diff_rot_vec = cv2.Rodrigues(diff_rot)[0].reshape(3)

    interpolated_pos = pose1_pos + (pose2_pos - pose1_pos) * alpha
    interpolated_rot_diff = cv2.Rodrigues(diff_rot_vec * alpha)[0].reshape(3, 3)
    interpolated_rot = np.matmul(pose1_rot, interpolated_rot_diff)

    interpolated_pose = np.eye(4)
    interpolated_pose[:3, 3] = interpolated_pos
    interpolated_pose[:3, :3] = interpolated_rot

    interpolated_pose = np.linalg.inv(interpolated_pose).astype(np.float32)

    return interpolated_pose


def interpolate_cam_param(cam1, cam2, now_time):  # only pose and exposure_gain
    assert cam1["width"] == cam2["width"]
    assert cam1["height"] == cam2["height"]
    assert cam1["fx"] == cam2["fx"]
    assert cam1["fy"] == cam2["fy"]
    assert cam1["cx"] == cam2["cx"]
    assert cam1["cy"] == cam2["cy"]
    assert cam1["near"] == cam2["near"]
    assert cam1["far"] == cam2["far"]

    time1 = cam1["timestamp"]  # in seconds
    time2 = cam2["timestamp"]  # in seconds

    assert time1 < time2 and now_time >= time1 and now_time <= time2

    alpha = (now_time - time1) / (time2 - time1)

    pose1 = cam1["pose"]
    pose2 = cam2["pose"]

    interpolated_pose = interpolate_pose(pose1, pose2, alpha)

    exposure_gain_1 = cam1["exposure_gain"]
    exposure_gain_2 = cam2["exposure_gain"]

    interpolated_exposure_gain = (
        exposure_gain_1 + (exposure_gain_2 - exposure_gain_1) * alpha
    )
    interpolated_uid = cam1["uid"] + (cam2["uid"] - cam1["uid"]) * alpha

    interpolated_cam = {}
    interpolated_cam["width"] = cam1["width"]
    interpolated_cam["height"] = cam1["height"]
    interpolated_cam["fx"] = cam1["fx"]
    interpolated_cam["fy"] = cam1["fy"]
    interpolated_cam["cx"] = cam1["cx"]
    interpolated_cam["cy"] = cam1["cy"]
    interpolated_cam["pose"] = interpolated_pose
    interpolated_cam["exposure_gain"] = interpolated_exposure_gain
    interpolated_cam["timestamp"] = now_time
    interpolated_cam["near"] = cam1["near"]
    interpolated_cam["far"] = cam1["far"]
    interpolated_cam["uid"] = interpolated_uid

    return interpolated_cam


def downsample_img(img, aggragate="mean"):
    # img shape: B x H x W x C

    res = []
    for dx in range(2):
        for dy in range(2):
            res.append(img[:, dx::2, dy::2, :])

    if aggragate == "mean":
        return torch.mean(torch.stack(res), dim=0)
    elif aggragate == "max":
        return torch.max(torch.stack(res), dim=0)[0]
    else:
        raise NotImplementedError


def unproject_depth(depths, int_mat):
    device = depths.device
    pts_2d_idx = torch.arange(depths.shape[2] * depths.shape[3], device=device)
    pts_2d_x = pts_2d_idx % depths.shape[3] + 0.5
    pts_2d_y = pts_2d_idx // depths.shape[3] + 0.5

    local_coords = torch.stack(
        [pts_2d_x, pts_2d_y, torch.ones_like(pts_2d_x, device=device)], dim=1
    ).float()

    int_mat_inv_T = torch.inverse(int_mat.transpose(0, 1))

    local_coords = torch.mm(local_coords, int_mat_inv_T).view(
        1, local_coords.shape[0], 3
    ) * depths.view(depths.shape[0], -1, 1)

    local_coords = local_coords.view(
        depths.shape[0], depths.shape[2], depths.shape[3], 3
    )
    local_coords = local_coords.permute(0, 3, 1, 2)

    return local_coords


def convert_depth_to_normal(depths, int_mat):
    if len(depths.shape) == 2:
        depths = depths.view(1, 1, depths.shape[0], depths.shape[1])
    elif len(depths.shape) == 3:
        depths = depths.squeeze(0).squeeze(-1).unsqueeze(0).unsqueeze(0)

    local_coords = unproject_depth(depths, int_mat)

    normal_map = torch.zeros_like(local_coords, device=local_coords.device)

    dx = local_coords[:, :, 2:, 1:-1] - local_coords[:, :, :-2, 1:-1]
    dy = local_coords[:, :, 1:-1, 2:] - local_coords[:, :, 1:-1, :-2]

    normal_map[:, :, 1:-1, 1:-1] = torch.nn.functional.normalize(
        torch.cross(dx, dy, dim=1), dim=1
    )

    return normal_map


class ssim_map:
    def __init__(self, device, window_size=11, val_range=1):
        self.device = device
        self.sigma = 1.5

        self.L = val_range
        self.C1 = (0.01 * self.L) ** 2
        self.C2 = (0.03 * self.L) ** 2

        self.window = self.create_window(window_size, channel=1, sigma=self.sigma).to(
            self.device
        )
        self.padd = window_size // 2
        self.channel = 1

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel, sigma=1.5):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def cal_ssim_map(self, img1, img2):
        assert img1.shape[1] == self.channel

        padd = self.padd

        mu1 = torch.nn.functional.conv2d(
            torch.nn.functional.pad(img1, (padd, padd, padd, padd)),
            self.window,
            groups=self.channel,
        )
        mu2 = torch.nn.functional.conv2d(
            torch.nn.functional.pad(img2, (padd, padd, padd, padd)),
            self.window,
            groups=self.channel,
        )
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = (
            torch.nn.functional.conv2d(
                torch.nn.functional.pad(img1 * img1, (padd, padd, padd, padd)),
                self.window,
                groups=self.channel,
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                torch.nn.functional.pad(img2 * img2, (padd, padd, padd, padd)),
                self.window,
                groups=self.channel,
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                torch.nn.functional.pad(img1 * img2, (padd, padd, padd, padd)),
                self.window,
                groups=self.channel,
            )
            - mu1_mu2
        )
        v1 = 2.0 * sigma12 + self.C2
        v2 = sigma1_sq + sigma2_sq + self.C2
        # cs = v1 / v2  # contrast sensitivity (removed unused variable)
        ssim_map = ((2 * mu1_mu2 + self.C1) * v1) / ((mu1_sq + mu2_sq + self.C1) * v2)
        # Clip SSIM values to valid range
        ssim_map = torch.clamp(ssim_map, 0, 1)
        # Calculate SSIM difference map
        ssim_diff_map = 1 - ssim_map
        return ssim_diff_map
