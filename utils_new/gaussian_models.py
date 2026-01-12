# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from depth_cov.depth_cov_estimator import DepthCovEstimator
from gsplat.rendering import rasterization, rasterization_2dgs
from plyfile import PlyData, PlyElement
from torch import nn
from utils_new.camera_utils import Camera, unproject_pts_tensor
from utils_new.hash_utils import HashBlock
from utils_new.logging_utils import Log
from utils_new.tool_utils import BasicPointCloud, inverse_sigmoid, rgb_to_sh_np


def get_expon_lr_func(lr_init, lr_final, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


class Gaussians:
    def __init__(
        self,
        BS: int = 6,
        scene_scale: float = 1.0,
        init_config=None,
        max_sh_degree: int = 2,
    ):
        _xyz = torch.zeros((1, 3), dtype=torch.float32)
        _sh0 = torch.zeros((1, 1, 3), dtype=torch.float32)
        _shN = torch.zeros((1, (max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32)
        _scaling = torch.log(torch.ones((1, 3), dtype=torch.float32) * 1e-8)
        _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float()
        _opacity = inverse_sigmoid(
            0.1 * torch.ones((_xyz.shape[0],), dtype=torch.float32)
        )

        self.is_optimize = True
        self.max_sh_degree = max_sh_degree

        self.scene_scale = scene_scale
        self.BS = BS
        self.lr_step = 0

        self.scheduler_args = {}

        self.non_trainable_params = {
            "init_scales": torch.ones((1,), dtype=torch.float32) * 1e-8,
        }

        if init_config is not None:
            self.raw_lr = {
                "means": init_config["means_lr_init"],
                "scales": init_config["scales_lr_init"],
                "quats": init_config["quats_lr_init"],
            }
            self.scheduler_args["means"] = get_expon_lr_func(
                lr_init=init_config["means_lr_init"],
                lr_final=init_config["means_lr_final"],
                max_steps=init_config["lr_final_step"],
            )
            self.scheduler_args["scales"] = get_expon_lr_func(
                lr_init=init_config["scales_lr_init"],
                lr_final=init_config["scales_lr_final"],
                max_steps=init_config["lr_final_step"],
            )
            self.scheduler_args["quats"] = get_expon_lr_func(
                lr_init=init_config["quats_lr_init"],
                lr_final=init_config["quats_lr_final"],
                max_steps=init_config["lr_final_step"],
            )

            self.max_steps = init_config["lr_final_step"]
            params = [
                # name, value, lr
                (
                    "means",
                    torch.nn.Parameter(_xyz),
                    init_config["means_lr_init"] * scene_scale,
                ),
                ("scales", torch.nn.Parameter(_scaling), init_config["scales_lr_init"]),
                ("quats", torch.nn.Parameter(_rotation), init_config["quats_lr_init"]),
                (
                    "opacities",
                    torch.nn.Parameter(_opacity),
                    init_config["opacities_lr"],
                ),
                ("sh0", torch.nn.Parameter(_sh0), init_config["sh_lr"]),
                ("shN", torch.nn.Parameter(_shN), init_config["sh_lr"] / 20),
            ]
        else:
            self.raw_lr = {
                "means": 1.6e-4,
                "scales": 0.005,
                "quats": 0.001,
            }
            self.scheduler_args["means"] = get_expon_lr_func(
                lr_init=1.6e-4, lr_final=1.6e-4, max_steps=1
            )
            self.scheduler_args["scales"] = get_expon_lr_func(
                lr_init=0.005, lr_final=0.005, max_steps=1
            )
            self.scheduler_args["quats"] = get_expon_lr_func(
                lr_init=0.001, lr_final=0.001, max_steps=1
            )
            self.max_steps = 1
            params = [
                # name, value, lr
                ("means", torch.nn.Parameter(_xyz), 1.6e-4 * scene_scale),
                ("scales", torch.nn.Parameter(_scaling), 5e-3),
                ("quats", torch.nn.Parameter(_rotation), 1e-3),
                ("opacities", torch.nn.Parameter(_opacity), 5e-2),
                ("sh0", torch.nn.Parameter(_sh0), 2.5e-3),
                ("shN", torch.nn.Parameter(_shN), 2.5e-3 / 20),
            ]

        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in params})

        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, _, lr in params
        }

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.device = "cpu"

    @property
    def get_scaling(self):
        return self.scaling_activation(self.splats["scales"])

    @property
    def get_rotation(self):
        return self.splats[
            "quats"
        ]  # gsplats will normalize the quat inside the rasterizer

    @property
    def get_xyz(self):
        return self.splats["means"]

    @property
    def get_features(self):
        return torch.cat((self.splats["sh0"], self.splats["shN"]), dim=1)  # [N, K, 3]

    @property
    def get_opacity(self):
        return self.opacity_activation(self.splats["opacities"])

    @property
    def get_num(self):
        return int(self.splats["means"].shape[0])

    @property
    def get_init_scales(self):
        return self.non_trainable_params["init_scales"]

    @property
    def get_params(self):
        return self.splats

    def reset_optimizer(self, config, BS):
        self.max_steps = config["max_steps"]
        self.lr_step = 0

        self.scheduler_args = {}

        self.raw_lr = {
            "means": config["means_lr_init"],
            "scales": config["scales_lr_init"],
        }
        self.scheduler_args["means"] = get_expon_lr_func(
            lr_init=config["means_lr_init"],
            lr_final=config["means_lr_final"],
            max_steps=config["max_steps"],
        )
        self.scheduler_args["scales"] = get_expon_lr_func(
            lr_init=config["scales_lr_init"],
            lr_final=config["scales_lr_final"],
            max_steps=config["max_steps"],
        )

        params = [
            # name, value, lr
            ("means", config["means_lr_init"] * self.scene_scale),
            ("scales", config["scales_lr_init"]),
            ("quats", config["quats_lr"]),
            ("opacities", config["opacities_lr"]),
            ("sh0", config["sh_lr"]),
            ("shN", config["sh_lr"] / 20),
        ]

        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, lr in params
        }

    def update_opt_lr(self, new_lr, data_type):  # update lr in optimizer
        self.optimizers[data_type].param_groups[0]["lr"] = new_lr

    def update_lr(self, new_lr, data_type):  # update lr
        self.raw_lr[data_type] = new_lr
        scene_scale = self.scene_scale if data_type != "quats" else 1.0

        self.update_opt_lr(new_lr * scene_scale * math.sqrt(self.BS), data_type)

    def update_scene_scale(self, new_scale):
        self.scene_scale = new_scale

        self.update_opt_lr(
            self.raw_lr["means"] * new_scale * math.sqrt(self.BS), "means"
        )
        self.update_opt_lr(self.raw_lr["scales"] * math.sqrt(self.BS), "scales")

    def step_lr(self):
        self.lr_step += 1

        for data_type in self.scheduler_args:
            new_lr = self.scheduler_args[data_type](self.lr_step)
            self.update_lr(new_lr, data_type)

        return self.lr_step < self.max_steps

    def get_pos_lr(self):
        return self.optimizers["means"].param_groups[0]["lr"]

    def clean(self):
        self.splats = None
        self.optimizers = None
        self.is_optimize = False

    def to_device(self, device):
        self.splats = self.splats.to(device)
        for name in self.non_trainable_params:
            self.non_trainable_params[name] = self.non_trainable_params[name].to(device)
        self.device = device

    def disable_grad(self):
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False
        self.splats.requires_grad_(False)
        self.is_optimize = False

    def enable_grad(self):
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True
        self.splats.requires_grad_(True)
        self.is_optimize = True

    @torch.no_grad()
    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for name, optimizer in self.optimizers.items():
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1
            extension_tensor = new_params[name]  # newly added tensor
            p = group["params"][0]
            stored_state = optimizer.state.get(p, None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del optimizer.state[p]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    @torch.no_grad()
    def extend_gaussians(self, new_params):
        new_optimized_tensors = self.cat_tensors_to_optimizer(new_params)
        self.splats["means"] = new_optimized_tensors["means"]
        self.splats["scales"] = new_optimized_tensors["scales"]
        self.splats["quats"] = new_optimized_tensors["quats"]
        self.splats["opacities"] = new_optimized_tensors["opacities"]
        self.splats["sh0"] = new_optimized_tensors["sh0"]
        self.splats["shN"] = new_optimized_tensors["shN"]

        self.non_trainable_params["init_scales"] = torch.cat(
            [self.non_trainable_params["init_scales"], new_params["init_scales"]], dim=0
        )

    @torch.no_grad()
    def extend_gaussians_from_color_points(self, pts, colors, init_scale=1e-4):
        xyz = torch.from_numpy(np.asarray(pts)).float().to(self.device)
        sh0 = (
            torch.from_numpy(np.asarray(rgb_to_sh_np(colors)))
            .float()
            .reshape(-1, 1, 3)
            .to(self.device)
        )
        shN = torch.zeros(
            (sh0.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3),
            device=self.device,
            dtype=torch.float32,
        )
        opacity = self.inverse_opacity_activation(
            0.5 * torch.ones((xyz.shape[0],), device=self.device, dtype=torch.float32)
        )

        if isinstance(init_scale, float):
            init_scale_param = (
                torch.ones((xyz.shape[0],), device=self.device, dtype=torch.float32)
                * init_scale
            )
            scales = (
                torch.ones((xyz.shape[0], 3), device=self.device, dtype=torch.float32)
                * init_scale
            )
        elif isinstance(init_scale, np.ndarray):
            if init_scale.shape[1] == 1:
                init_scale_param = (
                    torch.from_numpy(init_scale).float().to(self.device).reshape(-1)
                )
                init_scale = np.repeat(init_scale, 3, axis=1)
            elif init_scale.shape[1] == 3:
                init_scale_param = (
                    torch.from_numpy(np.mean(init_scale, axis=1))
                    .float()
                    .to(self.device)
                    .reshape(-1)
                )
            else:
                raise NotImplementedError(
                    "init_scale should be either float or np.ndarray"
                )
            scales = torch.from_numpy(init_scale).float().to(self.device)
        else:
            raise NotImplementedError("init_scale should be either float or np.ndarray")

        quats = torch.zeros((xyz.shape[0], 4), device=self.device, dtype=torch.float32)
        quats[:, 0] = 1

        new_params = {
            "means": xyz,
            "scales": scales,
            "quats": quats,
            "opacities": opacity,
            "sh0": sh0,
            "shN": shN,
            "init_scales": torch.exp(
                init_scale_param
            ),  # actual scales are after exp op
        }

        self.extend_gaussians(new_params)

    @torch.no_grad()
    def prune_w_opacity(self, threshold):
        opacity = self.get_opacity

        valid_mask = opacity > threshold

        new_params = {}
        for name, optimizer in self.optimizers.items():
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1
            p = group["params"][0]
            stored_state = optimizer.state.get(p, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]

                del optimizer.state[p]
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid_mask].requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                new_params[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid_mask].requires_grad_(True)
                )
                new_params[name] = group["params"][0]

        for name in self.splats:
            self.splats[name] = new_params[name]

        for name in self.non_trainable_params:
            self.non_trainable_params[name] = self.non_trainable_params[name][
                valid_mask
            ]

    def update(self):
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

    @torch.no_grad()
    def load_from_ply(self, path, max_sh_degree=2):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])

        sh0 = np.zeros((xyz.shape[0], 1, 3))
        sh0[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        sh0[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        sh0[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        shN = np.zeros((xyz.shape[0], ((max_sh_degree + 1) ** 2 - 1) * 3))

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        for idx, attr_name in enumerate(extra_f_names):
            shN[:, idx] = np.asarray(plydata.elements[0][attr_name])
        shN = shN.reshape((shN.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)).transpose(
            0, 2, 1
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).float().to(self.device)
        sh0 = torch.from_numpy(sh0).float().to(self.device)
        shN = torch.from_numpy(shN).float().to(self.device)
        opacities = torch.from_numpy(opacities).float().to(self.device)
        scales = torch.from_numpy(scales).float().to(self.device)
        rots = torch.from_numpy(rots).float().to(self.device)
        init_scales = (
            torch.ones((xyz.shape[0],), dtype=torch.float32, device=self.device) * 1e-4
        )

        new_params = {
            "means": xyz,
            "scales": scales,
            "quats": rots,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
            "init_scales": torch.exp(init_scales),
        }

        self.extend_gaussians(new_params)


class GaussianModel:
    def __init__(self, configs):
        self.configs = configs

        self.active_sh_degree = configs["Model"]["sh_degree"]
        self.max_sh_degree = configs["Model"]["sh_degree"]

        self.gaussian_pos_schedule_steps = configs["Model"][
            "gaussian_pos_schedule_steps"
        ]

        self.use_anti_aliasing = configs["Model"]["use_anti_aliasing"]
        self.render_mode = configs["Model"]["render_mode"]
        self.device = configs["Model"]["device"]

        self.radius_clip = configs["Model"][
            "radius_clip"
        ]  # Do not render gaussians smaller than this radius
        self.init_scale_size = float(
            configs["Model"]["init_scale_size"]
        )  # Initial scale of the gaussians
        self.camera_scale_rescalar = (
            float(configs["Model"]["camera_scale_rescalar"])
            if "camera_scale_rescalar" in configs["Model"]
            else 1.0
        )  # Rescale the camera scale to avoid gaussians too small to render

        self.BS = (
            configs["Mapper"]["active_window_size"]
            + configs["Mapper"]["coarse_active_window_size"]
            + configs["Mapper"]["KFGraph"]["global_window_size"]
        )
        self.scene_scale = configs["Model"]["scene_scale"]
        self.init_gaussian_config = configs["Model"]["init_gaussian_config"]
        self.init_gaussian_config["lr_final_step"] = self.gaussian_pos_schedule_steps

        self.MAX_LEVEL = 4

        self.init_gaussian_groups()

        self.hash_block = HashBlock(configs["HashBlock"])

        self.densification_mode = configs["Model"]["densification_mode"]
        if "semi-dense_extra-pts" in self.densification_mode:
            self.extra_pts_num = configs["Model"]["extra_pts_num"]
            self.err_threshold = configs["Model"]["err_threshold"]
            self.semi_dense_err_threshold = configs["Model"]["semi_dense_err_threshold"]

            if "adaptive" in configs["Model"]["densification_mode"]:
                self.init_scale_offset = (
                    configs["Model"]["init_scale_offset"]
                    if "init_scale_offset" in configs["Model"]
                    else 0.0
                )

        self.opacity_prune_threshold = configs["Model"]["opacity_prune_threshold"]

        self.gaussian_type = configs["Model"]["gaussian_type"]

        Log("Gaussian Type: {}".format(self.gaussian_type), tag="GaussianModel")

        self.scene_exposure_gain = configs["Mapper"]["scene_exposure_gain"]

        self.gaussians_size_filter = (
            configs["Model"]["gaussians_size_filter"]
            if "gaussians_size_filter" in configs["Model"]
            else False
        )

        self.vignette_imgs = None

        if "DepthCovEstimator" in configs["Model"]:
            configs["Model"]["DepthCovEstimator"]["device"] = self.device
            self.depth_cov_estimator = DepthCovEstimator(
                configs["Model"]["DepthCovEstimator"]
            )

    def init_gaussian_groups(self):
        self.current_gaussian_group = {}
        self.active_gaussian_groups = {}
        self.gaussian_groups = []
        self.valid_groups = []

        for i in range(self.MAX_LEVEL):
            self.current_gaussian_group[i] = i
            self.active_gaussian_groups[i] = [i]
            self.gaussian_groups.append(
                Gaussians(
                    BS=self.BS,
                    scene_scale=self.scene_scale,
                    init_config=self.init_gaussian_config,
                    max_sh_degree=self.max_sh_degree,
                )
            )
            self.gaussian_groups[-1].to_device(self.device)
            self.valid_groups.append(i)

    @property
    def get_num_gaussians(self):
        total_num = 0
        for j in range(self.MAX_LEVEL):
            total_num += int(
                np.sum(
                    [
                        self.gaussian_groups[i].get_num
                        for i in self.active_gaussian_groups[j]
                    ]
                )
            )
        return total_num

    def get_xyz(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_xyz
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_xyz
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)

    def get_scaling(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_scaling
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_scaling
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)

    def get_rotation(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_rotation
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_rotation
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)

    def get_opacity(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_opacity
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_opacity
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)

    def get_features(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_features
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_features
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

    def get_num_active_groups(self, level):
        return len(self.active_gaussian_groups[level])

    def get_num_optimized_groups(self, level):
        return np.sum(
            [
                self.gaussian_groups[i].is_optimize
                for i in self.active_gaussian_groups[level]
            ]
        )

    def get_current_group_size(self):
        return self.gaussian_groups[self.current_gaussian_group].get_num

    def set_vignette_img(self, vignette_img):
        if vignette_img is not None:
            vignette_img = torch.from_numpy(vignette_img).float().to(self.device)
            self.vignette_imgs = [vignette_img.unsqueeze(0)]

            for _ in range(1, Camera.MAX_LEVEL):
                vignette_img = Camera.downsample(vignette_img, mode="color")
                self.vignette_imgs.append(vignette_img.unsqueeze(0))
        else:
            self.vignette_imgs = []

    def get_vignette_img(self, level=0):
        if self.vignette_imgs is None or len(self.vignette_imgs) == 0:
            return None
        return self.vignette_imgs[level]

    def render_3dgs(self, cam, level=0):
        # means = torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)
        # scales = torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)
        # rotations = torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)
        # opacities = torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)
        # shs = torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        # for i in range(level, self.MAX_LEVEL):
        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

            # if i - level <= 1:
            #     means.append(_means)
            #     scales.append(_scales)
            #     rotations.append(_rotations)
            #     opacities.append(_opacities)
            #     shs.append(_shs)
            # else:
            #     means.append(_means.detach())
            #     scales.append(_scales.detach())
            #     rotations.append(_rotations.detach())
            #     opacities.append(_opacities.detach())
            #     shs.append(_shs.detach())

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)

        rasterize_mode = "antialiased" if self.use_anti_aliasing else "classic"

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=cam.get_pose()[
                None, :, :
            ],  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=cam.get_int_mat(level)[None, ...],  # [1, 3, 3]
            width=cam.get_width(level),
            height=cam.get_height(level),
            rasterize_mode=rasterize_mode,
            near_plane=cam.near,
            far_plane=cam.far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
        )

        assert render_colors.shape[0] == 1, "batch size should be 1"

        colors = render_colors[0]
        if colors.shape[2] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[2] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        render_alphas = render_alphas[0]

        colors = colors * cam.exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "opacity": render_alphas,
            }

        return render_pkg

    def render_2dgs(self, cam, level=0):
        # means = torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)
        # scales = torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)
        # rotations = torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)
        # opacities = torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)
        # shs = torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        # for i in range(level, self.MAX_LEVEL):
        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

            # if i - level <= 1:
            #     means.append(_means)
            #     scales.append(_scales)
            #     rotations.append(_rotations)
            #     opacities.append(_opacities)
            #     shs.append(_shs)
            # else:
            #     means.append(_means.detach())
            #     scales.append(_scales.detach())
            #     rotations.append(_rotations.detach())
            #     opacities.append(_opacities.detach())
            #     shs.append(_shs.detach())

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=cam.get_pose()[
                None, :, :
            ],  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=cam.get_int_mat(level)[None, ...],  # [1, 3, 3]
            width=cam.get_width(level),
            height=cam.get_height(level),
            near_plane=cam.near,
            far_plane=cam.far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
            distloss=True,
        )

        assert render_colors.shape[0] == 1, "batch size should be 1"

        colors = render_colors[0]
        if colors.shape[2] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[2] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        render_alphas = render_alphas[0]

        colors = colors * cam.exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }

        return render_pkg

    def render(self, cam, level=0):
        if self.gaussian_type == "2dgs":
            return self.render_2dgs(cam, level)
        elif self.gaussian_type == "3dgs":
            return self.render_3dgs(cam, level)
        else:
            raise NotImplementedError

    def render_batch_3dgs(self, cams, random_bg=False, detach_gaussians=False, level=0):
        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)

        if detach_gaussians:
            means = means.detach()
            scales = scales.detach()
            rotations = rotations.detach()
            opacities = opacities.detach()
            shs = shs.detach()

        rasterize_mode = "antialiased" if self.use_anti_aliasing else "classic"

        poses = torch.stack([cam.get_pose() for cam in cams], dim=0)
        Ks = torch.stack([cam.get_int_mat(level) for cam in cams], dim=0)

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=poses,  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=Ks,  # [N, 3, 3]
            width=cams[0].get_width(level),
            height=cams[0].get_height(level),
            rasterize_mode=rasterize_mode,
            near_plane=cams[0].near,
            far_plane=cams[0].far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
        )

        colors = render_colors
        if colors.shape[3] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[3] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        if random_bg:
            bgc = torch.rand((colors.shape[0], 1, 1, 3)).float().to(colors.device)
            colors = colors + bgc * (1 - render_alphas)

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        exposure_gain = (
            torch.from_numpy(
                np.array([cam.exposure_gain for cam in cams]).reshape(-1, 1, 1, 1)
            )
            .float()
            .to(self.device)
        )
        colors = colors * exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "opacity": render_alphas,
            }

        return render_pkg

    def render_batch_2dgs(self, cams, random_bg=False, detach_gaussians=False, level=0):
        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)

        if detach_gaussians:
            means = means.detach()
            scales = scales.detach()
            rotations = rotations.detach()
            opacities = opacities.detach()
            shs = shs.detach()

        poses = torch.stack([cam.get_pose() for cam in cams], dim=0)
        Ks = torch.stack([cam.get_int_mat(level) for cam in cams], dim=0)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=poses,  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=Ks,  # [N, 3, 3]
            width=cams[0].get_width(level),
            height=cams[0].get_height(level),
            near_plane=cams[0].near,
            far_plane=cams[0].far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
            distloss=True,
        )

        colors = render_colors
        if colors.shape[3] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[3] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        if random_bg:
            bgc = torch.rand((colors.shape[0], 1, 1, 3)).float().to(colors.device)
            colors = colors + bgc * (1 - render_alphas)

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        exposure_gain = (
            torch.from_numpy(
                np.array([cam.exposure_gain for cam in cams]).reshape(-1, 1, 1, 1)
            )
            .float()
            .to(self.device)
        )
        colors = colors * exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }

        return render_pkg

    def render_batch(self, cams, random_bg=False, detach_gaussians=False, level=0):
        if self.gaussian_type == "2dgs":
            return self.render_batch_2dgs(
                cams,
                random_bg=random_bg,
                detach_gaussians=detach_gaussians,
                level=level,
            )
        elif self.gaussian_type == "3dgs":
            return self.render_batch_3dgs(
                cams,
                random_bg=random_bg,
                detach_gaussians=detach_gaussians,
                level=level,
            )
        else:
            raise NotImplementedError

    def reset_optimizer(self, config, BS):
        for i in self.active_gaussian_groups:
            self.gaussian_groups[i].reset_optimizer(config, BS)

    def step_all_lr(self):
        for i in self.active_gaussian_groups:
            self.gaussian_groups[i].step_lr()

    def get_avg_pos_lr(self):
        avg_pos_lr = 0
        for i in self.active_gaussian_groups:
            avg_pos_lr += self.gaussian_groups[i].get_pos_lr()
        return avg_pos_lr / len(self.active_gaussian_groups)

    def update(self):
        for i in self.active_gaussian_groups:
            if self.gaussian_groups[i].is_optimize:
                self.gaussian_groups[i].update()

    def create_new_group(self, level=0):
        if self.gaussian_pos_schedule_steps > 0:  # update position lr rate
            raise NotImplementedError

        self.gaussian_groups.append(
            Gaussians(
                BS=self.BS,
                scene_scale=self.scene_scale,
                init_config=self.init_gaussian_config,
                max_sh_degree=self.max_sh_degree,
            )
        )
        self.gaussian_groups[-1].to_device(self.device)
        self.current_gaussian_group[level] = len(self.gaussian_groups) - 1
        self.active_gaussian_groups[level].append(self.current_gaussian_group)
        self.valid_groups.append(self.current_gaussian_group)
        Log(
            "Creating new group: active groups: {}, totol groups: {}".format(
                len(self.active_gaussian_groups), len(self.gaussian_groups)
            ),
            tag="GaussianModel",
        )

    def deactivate_gaussian_group(self, group_idx, level):
        if group_idx in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].remove(group_idx)
            self.gaussian_groups[group_idx].to_device("cpu")
            return True
        else:
            return False

    def activate_gaussian_group(self, group_idx, level):
        if group_idx not in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].append(group_idx)
            self.gaussian_groups[group_idx].to_device(self.device)
            return True
        else:
            return False

    def remove_group(self, group_idx, level):
        if group_idx in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].remove(group_idx)

        if group_idx in self.valid_groups:
            self.valid_groups.remove(group_idx)
            self.gaussian_groups[group_idx].clean()

    def remove_optimization(self, group_idx):
        if self.gaussian_groups[group_idx].is_optimize:
            self.gaussian_groups[group_idx].disable_grad()
            return True
        else:
            return False

    def add_optimization(self, group_idx):
        if not self.gaussian_groups[group_idx].is_optimize:
            self.gaussian_groups[group_idx].enable_grad()
            return True
        else:
            return False

    @torch.no_grad()
    def add_new_gaussians(self, cam, create_new_group=False, render_pkg=None, level=0):
        # t1 = torch.cuda.Event(enable_timing=True)
        # t2 = torch.cuda.Event(enable_timing=True)

        cur_view_scale_size = None

        if (
            create_new_group
            and self.gaussian_groups[self.current_gaussian_group].get_num > 0
        ):
            self.create_new_group(level=level)

        if self.densification_mode == "adaptive_semi-dense_extra-pts":
            extra_pts_num = int(
                self.extra_pts_num // (2**level) ** 1.5
            )  # theoretically it should be 2.0, use 1.5 to generate more points

            sparse_depth = cam.get_sparse_depth(level)

            device = sparse_depth.device

            flatten_depth = sparse_depth.reshape(-1)
            sparse_pts_mask = flatten_depth > 0
            flatten_index = torch.arange(len(flatten_depth)).to(device)
            pts_2d = flatten_index[sparse_pts_mask]

            w = cam.get_width(level)
            pts_2d = torch.stack([pts_2d % w + 0.5, pts_2d // w + 0.5], axis=-1).float()
            pts_depth = flatten_depth[sparse_pts_mask]

            if len(pts_2d) > 500:
                # random_indexes = np.random.choice(len(pts_2d), 500, replace=False)
                if len(pts_2d) > 3000:
                    random_indexes = torch.randint(len(pts_2d), (500,)).to(device)
                else:
                    random_indexes = torch.randperm(len(pts_2d))[:500].to(device)
                selected_pts_2d = pts_2d[random_indexes]
                selected_pts_depth = pts_depth[random_indexes]
            else:
                selected_pts_2d = pts_2d
                selected_pts_depth = pts_depth

            if render_pkg is not None:
                diff = render_pkg["diff"].reshape(-1)
                valid_extra = torch.logical_and(
                    (diff >= self.err_threshold), torch.logical_not(sparse_pts_mask)
                )

                valid_semi_pts = diff[sparse_pts_mask] >= self.semi_dense_err_threshold
                pts_2d = pts_2d[valid_semi_pts]
                pts_depth = pts_depth[valid_semi_pts]  # remove low error regions

                extra_pts_2d = flatten_index[valid_extra]
                if len(extra_pts_2d) > 0 and extra_pts_num > 0:
                    # extra_pts_2d = np.random.choice(extra_pts_2d, min(extra_pts_num, len(extra_pts_2d)), replace=False)
                    # extra_pts_2d_index = torch.randperm(len(extra_pts_2d))[:min(extra_pts_num, len(extra_pts_2d))].to(device)
                    extra_pts_2d_index = torch.randint(
                        len(extra_pts_2d), (min(extra_pts_num, len(extra_pts_2d)),)
                    ).to(device)
                    extra_pts_2d = extra_pts_2d[extra_pts_2d_index]
                    extra_pts_2d = torch.stack(
                        [extra_pts_2d % w + 0.5, extra_pts_2d // w + 0.5], axis=-1
                    ).float()

                else:
                    extra_pts_2d = None
            else:
                # extra_pts_2d = np.random.choice(flatten_index, extra_pts_num, replace=False)
                extra_pts_2d_index = torch.randperm(len(flatten_index))[
                    :extra_pts_num
                ].to(device)
                extra_pts_2d = flatten_index[extra_pts_2d_index]
                extra_pts_2d = torch.stack(
                    [extra_pts_2d % w + 0.5, extra_pts_2d // w + 0.5], axis=-1
                ).float()

            if extra_pts_2d is not None:
                # est_depth, std_mask = self.depth_cov_estimator.query(cam.get_gt_image(level), selected_pts_depth, selected_pts_2d, extra_pts_2d)
                est_depth, std_mask = self.depth_cov_estimator.query_tensor(
                    cam.get_gt_image(level),
                    selected_pts_depth,
                    selected_pts_2d,
                    extra_pts_2d,
                )

                # Log("Valid additional pts: {}".format(np.sum(std_mask)), tag="GaussianModel")

                if torch.sum(std_mask) > 0:
                    extra_pts_depth = est_depth[std_mask]
                    extra_pts_2d = extra_pts_2d[std_mask]

                    # Uncomment this part if need to visualize feature points
                    # extra_pts_idx = extra_pts_2d[:, 1].int() * cam.get_width(level) + extra_pts_2d[:, 0].int()
                    # vis_feature_mask = sparse_depth.detach().cpu().numpy().reshape(-1)
                    # vis_feature_mask[vis_feature_mask > 0] = 1.0
                    # vis_feature_mask[extra_pts_idx.detach().cpu().numpy()] = 0.5
                    # cam.feature_mask = vis_feature_mask.reshape(sparse_depth.shape[0], sparse_depth.shape[1], 1)

                    pts_2d = torch.cat(
                        [pts_2d, extra_pts_2d], dim=0
                    )  # concat semi-dense pts with extra pts
                    pts_depth = torch.cat([pts_depth, extra_pts_depth], dim=0)

            # pts_3d = unproject_pts(pts_2d, pts_depth, cam.get_int_mat(level).cpu().numpy(), cam.get_raw_pose().detach().cpu().numpy())
            pts_3d = unproject_pts_tensor(
                pts_2d, pts_depth, cam.get_int_mat(level), cam.get_raw_pose().detach()
            )

            color_img = cam.get_gt_image(level)

            vig_img = self.get_vignette_img(level)

            if vig_img is not None:
                color_for_pts = (
                    color_img.to(self.device)
                    * self.scene_exposure_gain
                    / cam.exposure_gain
                    / vig_img
                )
                invalid_mask = vig_img == 0
                color_for_pts[invalid_mask] = 0.0
                color_for_pts = color_for_pts[0]
            else:
                color_for_pts = (
                    color_img.to(self.device)
                    * self.scene_exposure_gain
                    / cam.exposure_gain
                )

            # color_for_pts = cv2.blur(color_for_pts, (3, 3))
            color_for_pts = color_for_pts.reshape(-1, 3)

            pts_2d_index = (
                pts_2d[:, 1].int() * cam.get_width(level) + pts_2d[:, 0].int()
            )

            pts_color = color_for_pts[pts_2d_index]

            init_scale = (
                torch.log(
                    0.5 * pts_depth / ((cam.get_fx(level) + cam.get_fy(level)) / 2.0)
                ).reshape(-1, 1)
                + self.init_scale_offset
            )
            cur_view_scale_size = cam.get_view_size(level)

            pts_3d = pts_3d.cpu().numpy()
            pts_color = pts_color.cpu().numpy()
            init_scale = init_scale.cpu().numpy()

        else:
            raise NotImplementedError

        cur_view_scale_size *= self.camera_scale_rescalar

        before_filter_num = pts_3d.shape[0]
        occupied_mask = self.hash_block.getOccupy(
            pts_3d, pts_color, cur_view_scale_size
        )
        pts_3d = pts_3d[~occupied_mask]
        pts_color = pts_color[~occupied_mask]

        no_conflict_index = self.hash_block.get_no_conflict_index(
            pts_3d, cur_view_scale_size
        )
        pts_3d = pts_3d[no_conflict_index]
        pts_color = pts_color[no_conflict_index]

        if isinstance(init_scale, np.ndarray):
            init_scale = init_scale[~occupied_mask]
            init_scale = init_scale[no_conflict_index]

        self.hash_block.setOccupy(pts_3d, pts_color, cur_view_scale_size)
        Log(
            "Adding new gaussians (before/after): {} / {}".format(
                before_filter_num, pts_3d.shape[0]
            ),
            tag="GaussianModel",
        )

        self.gaussian_groups[
            self.current_gaussian_group[level]
        ].extend_gaussians_from_color_points(pts_3d, pts_color, init_scale)

    def add_new_gaussians_pts_only(self, cam):
        color_pts_depth = cam.get_color_pts_depth()

        if len(color_pts_depth) == 0:
            return

        assert color_pts_depth.shape[1] == 7

        pts_3d = color_pts_depth[:, :3]
        pts_color = color_pts_depth[:, 3:6]
        depth = color_pts_depth[:, 6]

        init_scale = (
            np.log(0.5 * depth / ((cam.get_fx(0) + cam.get_fy(0)) / 2.0)).reshape(-1, 1)
            + self.init_scale_offset
        )
        cur_view_scale_size = cam.get_view_size(0)

        cur_view_scale_size *= self.camera_scale_rescalar

        before_filter_num = pts_3d.shape[0]
        occupied_mask = self.hash_block.getOccupy(
            pts_3d, pts_color, cur_view_scale_size
        )
        pts_3d = pts_3d[~occupied_mask]
        pts_color = pts_color[~occupied_mask]

        no_conflict_index = self.hash_block.get_no_conflict_index(
            pts_3d, cur_view_scale_size
        )
        pts_3d = pts_3d[no_conflict_index]
        pts_color = pts_color[no_conflict_index]

        if isinstance(init_scale, np.ndarray):
            init_scale = init_scale[~occupied_mask]
            init_scale = init_scale[no_conflict_index]

        self.hash_block.setOccupy(pts_3d, pts_color, cur_view_scale_size)
        Log(
            "Adding new gaussians (before/after): {} / {}".format(
                before_filter_num, pts_3d.shape[0]
            ),
            tag="GaussianModel",
        )

        self.gaussian_groups[
            self.current_gaussian_group[0]
        ].extend_gaussians_from_color_points(pts_3d, pts_color, init_scale)

    # will merge group_idx_from to group_idx_to, and remove group_idx_from
    # by default it will merge to the first group
    # also change the pointer of current_gaussian_group if needed
    def merge_gaussian_group(self, group_idx_from, group_idx_to=0):
        raise NotImplementedError

    def prune_w_opacity(self):
        if self.opacity_prune_threshold > 0:
            before_num = self.get_num_gaussians
            for j in range(self.MAX_LEVEL):
                for i in self.active_gaussian_groups[j]:
                    self.gaussian_groups[i].prune_w_opacity(
                        self.opacity_prune_threshold
                    )
            after_num = self.get_num_gaussians
            Log(
                "Pruning done: before/after: {}/{}".format(before_num, after_num),
                tag="GaussianModel",
            )

    @torch.no_grad()
    def save_as_ply(self, path):
        xyz = (
            torch.cat(
                [self.gaussian_groups[i].splats["means"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        normals = np.zeros_like(xyz)

        f_dc = (
            torch.cat(
                [self.gaussian_groups[i].splats["sh0"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            torch.cat(
                [self.gaussian_groups[i].splats["shN"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = (
            torch.cat(
                [
                    self.gaussian_groups[i].splats["opacities"]
                    for i in self.valid_groups
                ],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        scales = (
            torch.cat(
                [self.gaussian_groups[i].splats["scales"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        rotations = (
            torch.cat(
                [self.gaussian_groups[i].splats["quats"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )

        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        for i in range(f_rest.shape[1]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotations.shape[1]):
            l.append("rot_{}".format(i))

        dtype_full = [(attribute, "f4") for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def load_from_ply(self, path):
        assert len(self.gaussian_groups) <= 4
        assert self.gaussian_groups[0].get_num <= 1

        self.gaussian_groups[0].load_from_ply(path, self.max_sh_degree)
        Log(
            "Loaded gaussians from ply",
            self.gaussian_groups[0].get_num,
            tag="GaussianModel",
        )
