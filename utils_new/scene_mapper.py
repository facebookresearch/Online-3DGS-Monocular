# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from os.path import join as pjoin

import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision.transforms import v2
from tqdm import tqdm
from utils_new.dataset import load_dataset
from utils_new.gaussian_models import GaussianModel

try:
    from utils_new.stream_loader import load_streamer
except:
    print("Warning: Stream loader not found. Please check the dependency.")
# from torchmetrics import StructuralSimilarityIndexMeasure  # Removed unused import

from utils_new.camera_utils import (
    # Camera_Optimizer,  # Removed unused import
    # in_frustum_region,  # Removed unused import
    SE3_Camera_Optimizer,
)
from utils_new.frame_checker import FrameChecker
from utils_new.kf_graph import cal_cams_covisibility, KFGraph
from utils_new.logging_utils import Log
from utils_new.loss_utils import (
    DepthLoss,
    DistortionLoss,
    GaussianLoss,
    NormalLoss,
    NormalRegularizationLoss,
    RGBLoss,
)
from utils_new.tool_utils import (
    convert_depth_to_normal,
    mkdir_p,
    saveTensorAsEXR,
    saveTensorAsPNG,
    ssim_map,
)


class SceneMapper(mp.Process):
    def __init__(self, configs):
        super(SceneMapper, self).__init__()
        self.configs = configs
        self.gaussians = GaussianModel(configs)
        self.cur_idx = 0
        self.cur_view = None

        self.last_record_time = torch.cuda.Event(enable_timing=True)

        self.last_record_time.record()

        self.initialization_frames = configs["Mapper"]["initialization_frames"]
        self.processed_frames = 0

        self.scene_exposure_gain = configs["Mapper"]["scene_exposure_gain"]

        # key frames methods
        # ------------------------- Legacy code -------------------------
        # self.kf_selection_method = configs["Mapper"]["kf_selection_method"]
        # self.last_kf_info = None
        # self.kf_interval = configs["Mapper"]["kf_interval"]
        # self.kf_overlap_ratio = configs["Mapper"]["kf_overlap_ratio"]
        # self.kf_cameras = []
        # self.num_additional_views = configs["Mapper"]["num_additional_views"]
        # ----------------------------------------------------------------
        self.kf_graph = KFGraph(configs["Mapper"]["KFGraph"])
        self.global_window_size = configs["Mapper"]["KFGraph"]["global_window_size"]

        self.last_add_gaussians = -1000000000

        self.cur_group_gaussian_frames = 0
        self.group_max_gaussian_frames = configs["Mapper"]["group_max_gaussian_frames"]

        self.device = configs["Mapper"]["device"]

        # Key frames and active window
        self.active_window_size = configs["Mapper"]["active_window_size"]
        self.active_window = []

        self.coarse_active_window_size = configs["Mapper"]["coarse_active_window_size"]
        self.coarse_active_window = []
        self.coarse_pool_size = configs["Mapper"]["coarse_pool_size"]
        self.coarse_pool = []
        self.coarse_level_interval = configs["Mapper"]["coarse_level_interval"]

        self.add_gaussians_interval = configs["Mapper"]["add_gaussians_interval"]

        self.prune_interval = configs["Mapper"]["prune_interval"]

        self.frame_checker = FrameChecker(configs["Mapper"]["FrameChecker"])

        self.use_multi_reso = configs["Mapper"]["use_multi_reso"]

        self.pin_kf_gpu = (
            configs["Mapper"]["pin_kf_gpu"]
            if "pin_kf_gpu" in configs["Mapper"]
            else False
        )  # whether pin key frames to gpu

        self.save_exr = self.configs["Results"]["save_exr"]

        mkdir_p(pjoin(self.configs["Results"]["save_dir"], "online", "point_cloud"))
        self.save_model_interval = configs["Mapper"]["save_model_interval"]

        # save optimization log info
        self.opt_log = {}
        # save tracked poses
        self.opt_log["tracked_poses"] = {}
        # save orig poses
        self.opt_log["poses_pair"] = {}
        # cam opt iterations
        self.opt_log["cam_opt_iterations"] = {}
        # gaussian opt iterations
        self.opt_log["gaussian_opt_iterations"] = {}
        # loss info
        self.opt_log["l1_err"] = {}
        # is key frame
        self.opt_log["is_key_frame"] = {}

        # Dataset
        self.use_dataset = configs["Mapper"]["use_dataset"]
        if self.use_dataset:
            configs["Dataset"]["scene_exposure_gain"] = self.scene_exposure_gain
            self.dataset = load_dataset(configs["Dataset"])
            vignette_img = self.dataset.dataset.get_vignette
            self.sample_cam = self.dataset.dataset.get_sample_cam()
            self.dataset = iter(self.dataset)
            self.gaussians.set_vignette_img(vignette_img)
        else:
            raise NotImplementedError(
                "Stream loader is not implemented for this mapper."
            )  # Songyin: Modify this if needed
            configs["Streamer"]["scene_exposure_gain"] = self.scene_exposure_gain
            self.dataset = load_streamer(configs["Streamer"])
            self.first_frame = True

        # Optimizer
        self.optimization_iters = configs["Mapper"]["optimization_iters"]
        self.intialization_iters = configs["Mapper"]["initialization_iters"]
        self.camera_optimizer = SE3_Camera_Optimizer(
            configs["Mapper"]["CameraOptimizer"]
        )
        self.pose_opt_steps = configs["Mapper"]["CameraOptimizer"]["pose_opt_steps"]
        self.pose_refine_init_steps = configs["Mapper"]["CameraOptimizer"][
            "pose_refine_init_steps"
        ]

        self.post_refinement_config = configs["Mapper"]["post_refinement"]

        self.use_random_bg = (
            configs["Mapper"]["use_random_bg"]
            if "use_random_bg" in configs["Mapper"]
            else False
        )

        # Loss function
        self.image_loss = RGBLoss(configs["Loss"])
        self.gaussian_loss = GaussianLoss(configs["Loss"])
        self.depth_loss = DepthLoss(configs["Loss"])
        self.normal_loss = NormalLoss(configs["Loss"])
        self.distortion_loss = DistortionLoss(configs["Loss"])
        self.normal_reg_loss = NormalRegularizationLoss(configs["Loss"])

        # ssim func
        # self.ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)
        self.ssim_func = ssim_map(self.device)
        self.gaussian_blurrer = v2.GaussianBlur(kernel_size=(5, 5), sigma=0.5)

        # some gaussian initial parameters
        self.err_threshold = configs["Model"]["err_threshold"]
        self.semi_dense_err_threshold = configs["Model"]["semi_dense_err_threshold"]

    # Optimize gaussians (do not densify or prune)
    def optimize(self, is_last_frame=False, is_key_frame=False, level=0, steps=-1):
        if len(self.active_window) == 0:
            return

        if self.gaussians.get_num_gaussians == 0:
            return

        color_mask = None

        # t1.record()
        optimized_local_window = []
        used_cams = []
        exist_idx = []
        cur_res_idx = 0
        for cam in self.active_window:
            if cam.cam_idx not in exist_idx:
                cam.to_device(self.device)
                optimized_local_window.append(cam)
                used_cams.append(cam)
                exist_idx.append(cam.cam_idx)
                if cam.cam_idx == self.cur_idx:
                    cur_res_idx = len(optimized_local_window) - 1
        for cam in self.coarse_active_window:
            if cam.cam_idx not in exist_idx:
                if not self.pin_kf_gpu:
                    cam.to_device(self.device)
                optimized_local_window.append(cam)
                used_cams.append(cam)
                exist_idx.append(cam.cam_idx)

        gt_imgs = torch.stack(
            [cam.get_gt_image(level) for cam in optimized_local_window]
        )
        gt_depths = torch.stack(
            [cam.get_sparse_depth(level) for cam in optimized_local_window]
        )

        # More budget for the for batch
        if self.initialization_frames == 0:
            optimization_steps = self.intialization_iters
            Log("Initialize map with 300 iterations", tag="SceneMapper")
        elif is_last_frame:
            optimization_steps = self.intialization_iters * 2
        else:
            optimization_steps = self.optimization_iters
        self.initialization_frames -= 1

        if steps >= 0:
            optimization_steps = steps

        for step in range(optimization_steps):
            # t3.record()

            # t1.record()
            global_windows = self.kf_graph.get_and_update_global_window()
            for cam in global_windows:
                if cam.cam_idx not in exist_idx:
                    if not self.pin_kf_gpu:
                        cam.to_device(self.device)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.get_gt_image(level) for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.get_sparse_depth(level) for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            if len(global_windows) > 0:
                iter_window = optimized_local_window + global_windows
                iter_gt_imgs = torch.cat([gt_imgs, global_gt_imgs], dim=0)
                iter_gt_depths = torch.cat([gt_depths, global_gt_depths], dim=0)
            else:
                iter_window = optimized_local_window
                iter_gt_imgs = gt_imgs
                iter_gt_depths = gt_depths

            if step < self.pose_opt_steps:
                # only opt pose for limited steps to avoid bad performance
                for cam in iter_window:
                    cam.set_opt_pose(True)

            # t1.record()
            batch_render_pkg = self.gaussians.render_batch(
                iter_window, self.use_random_bg, level=level
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            # t1.record()
            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            # t1.record()
            if self.normal_reg_loss.normal_reg_weight > 0.0:
                render_normal = convert_depth_to_normal(
                    batch_render_depth.view(
                        batch_render_depth.shape[0],
                        1,
                        batch_render_depth.shape[1],
                        batch_render_depth.shape[2],
                    ),
                    iter_window[0].get_int_mat(),
                )
            else:
                render_normal = None

            # t1.record()
            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            loss_gaussians = self.gaussian_loss(self.gaussians)

            loss_normal_reg, color_mask = self.normal_reg_loss(
                render_normal, iter_gt_imgs.permute(0, 3, 1, 2)
            )

            if (
                "normal" in batch_render_pkg
                and "normal_from_depth" in batch_render_pkg
                and "opacity" in batch_render_pkg
            ):
                loss_normal = self.normal_loss(
                    batch_render_pkg["normal"],
                    batch_render_pkg["normal_from_depth"],
                    batch_render_pkg["opacity"],
                )
            else:
                loss_normal = 0.0

            if "distortion" in batch_render_pkg:
                loss_distortion = self.distortion_loss(batch_render_pkg["distortion"])
            else:
                loss_distortion = 0.0

            loss = (
                loss_img
                + loss_gaussians
                + loss_depth
                + loss_normal
                + loss_distortion
                + loss_normal_reg
            )

            # t1.record()
            if len(global_windows) > 0:
                for cam, err in zip(
                    iter_window[-self.global_window_size :],
                    l1_values[-self.global_window_size :],
                ):
                    self.kf_graph.update_err(cam.cam_idx, err)

            loss.backward()

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

            if step < self.pose_opt_steps:
                self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            # t1.record()
            if step < self.pose_opt_steps:
                # only opt pose for limited steps to avoid bad performance
                for cam in iter_window:
                    cam.set_opt_pose(False)

            # save tracked poses
            for i, cam in enumerate(iter_window):
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.cam_idx not in self.opt_log["l1_err"]:
                        self.opt_log["l1_err"][cam.cam_idx] = {}
                    self.opt_log["l1_err"][cam.cam_idx][self.cur_view.cam_idx] = (
                        l1_values[i]
                    )

                    if cam.name not in self.opt_log["gaussian_opt_iterations"]:
                        self.opt_log["gaussian_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["gaussian_opt_iterations"][cam.name] += 1

                    if step < self.pose_opt_steps:
                        if cam.name not in self.opt_log["cam_opt_iterations"]:
                            self.opt_log["cam_opt_iterations"][cam.name] = 1
                        else:
                            self.opt_log["cam_opt_iterations"][cam.name] += 1

            if step == optimization_steps - 1:
                cur_view_render_pkg = {}
                cur_view_render_pkg["gt"] = iter_gt_imgs[cur_res_idx]
                cur_view_render_pkg["sparse_depth"] = optimized_local_window[
                    cur_res_idx
                ].get_sparse_depth(level)
                cur_view_render_pkg["render"] = batch_render_pkg["render"][cur_res_idx]
                cur_view_render_pkg["opacity"] = batch_render_pkg["opacity"][
                    cur_res_idx
                ]
                if "depth" in batch_render_pkg:
                    cur_view_render_pkg["depth"] = batch_render_pkg["depth"][
                        cur_res_idx
                    ]
                if "normal" in batch_render_pkg:
                    cur_view_render_pkg["normal"] = batch_render_pkg["normal"][
                        cur_res_idx
                    ]
                if "normal_from_depth" in batch_render_pkg:
                    cur_view_render_pkg["normal_from_depth"] = batch_render_pkg[
                        "normal_from_depth"
                    ][cur_res_idx]
                if "distortion" in batch_render_pkg:
                    cur_view_render_pkg["distortion"] = batch_render_pkg["distortion"][
                        cur_res_idx
                    ]

                if color_mask is not None:
                    cur_view_render_pkg["color_mask"] = color_mask[
                        cur_res_idx
                    ].unsqueeze(-1)

        # t1.record()
        if not self.pin_kf_gpu:
            for cam in used_cams:
                cam.to_device("cpu")

        if self.pin_kf_gpu and (not is_key_frame):  # do not keep non kfs in gpu
            self.cur_view.to_device("cpu")

        self.kf_graph.add_new_cam_for_global(optimized_local_window[cur_res_idx])

        return cur_view_render_pkg

    def get_frame(self):
        if self.use_dataset:
            cur_view = next(self.dataset, None)

            # if cur_view is not None:
            #     cur_view.to_device(self.device)
        else:
            cur_view = self.dataset.get_data()
            if self.first_frame:
                self.first_frame = False
                self.gaussians.set_vignette_img(self.dataset.get_vignette)
                self.sample_cam = self.dataset.get_sample_cam()

        return cur_view

    # Add new gaussians
    def densification(self, level=0):
        cur_level = level
        if self.initialization_frames <= 0:
            with torch.no_grad():
                # if not self.pin_kf_gpu:
                #     self.cur_view.to_device(self.device)

                self.cur_view.to_device(self.device)

                attemp_cur_render_pkg = self.gaussians.render(
                    self.cur_view, level=cur_level
                )

                rendered_img = (
                    attemp_cur_render_pkg["render"]
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    .mean(dim=1, keepdims=True)
                )
                gt_img = (
                    self.cur_view.get_gt_image(level=cur_level)
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    .mean(dim=1, keepdims=True)
                )

                rendered_img = torch.clamp(
                    self.gaussian_blurrer(rendered_img), 0.0, 1.0
                )
                gt_img = torch.clamp(self.gaussian_blurrer(gt_img), 0.0, 1.0)
                # _, diff_img = self.ssim_func(rendered_img, gt_img)
                # attemp_cur_render_pkg["diff"] = 1.0 - diff_img.squeeze(0).permute(1, 2, 0).mean(dim=-1).cpu()
                diff_img = self.ssim_func.cal_ssim_map(rendered_img, gt_img)
                attemp_cur_render_pkg["diff"] = (
                    diff_img.squeeze(0).permute(1, 2, 0).mean(dim=-1)
                )

                # Log("SSIM Diff: Max/Min: {:.4f}/{:.4f}".format(attemp_cur_render_pkg["diff"].max().item(), attemp_cur_render_pkg["diff"].min().item()), tag="SceneMapper")
                attemp_cur_render_pkg["depth"][
                    attemp_cur_render_pkg["opacity"] < 0.1
                ] = -1

                # if not self.pin_kf_gpu:
                #     self.cur_view.to_device("cpu")
        else:
            attemp_cur_render_pkg = None

        if (
            self.cur_view.cam_idx - self.last_add_gaussians
            > self.add_gaussians_interval
        ):
            self.last_add_gaussians = self.cur_view.cam_idx
            create_new_gaussians = (
                self.cur_group_gaussian_frames >= self.group_max_gaussian_frames
            )
            if create_new_gaussians:
                self.cur_group_gaussian_frames = 0
            self.gaussians.add_new_gaussians(
                self.cur_view,
                create_new_group=create_new_gaussians,
                render_pkg=attemp_cur_render_pkg,
                level=cur_level,
            )
            self.cur_group_gaussian_frames += 1
        elif self.initialization_frames >= 0:
            raise NotImplementedError("Not implemented")
            self.last_add_gaussians = self.cur_view.cam_idx
            self.gaussians.add_new_gaussians(
                self.cur_view, render_pkg=attemp_cur_render_pkg, level=cur_level
            )  # Add new gaussians for every frames before initialization
            self.cur_group_gaussian_frames += 1

        return attemp_cur_render_pkg

    # add new gaussians simpler
    def densification_pts_only(self):
        self.gaussians.add_new_gaussians_pts_only(self.cur_view)

    # Update keyframe cameras
    # also add new gaussians here if needed
    def update_kf(self):
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)

        t1.record()
        cur_view = self.get_frame()
        t2.record()
        torch.cuda.synchronize()
        get_frame_time = t1.elapsed_time(t2) / 1000.0
        t3.record()

        if cur_view is None:  # End of the sequence
            return True, False, 0, 0

        self.cur_view = cur_view

        # update the current index
        self.cur_idx = cur_view.cam_idx

        # update the number of processed frames
        self.processed_frames += 1

        # prune gaussians
        if self.prune_interval > 0 and self.processed_frames % self.prune_interval == 0:
            self.gaussians.prune_w_opacity()

        # Check frame's quality: (rotation speed, pts num)
        # is_good_frame = self.frame_checker.check(cur_view) or self.initialization_frames >= 0
        is_good_frame = self.frame_checker.check(cur_view)

        # Naively way to add key frames
        if is_good_frame:  # Do not add key frames if the rotation speed is too high
            is_key_frame = self.kf_graph.update_frame(cur_view)
        else:
            is_key_frame = False

        if self.pin_kf_gpu:
            self.cur_view.to_device(self.device)

        # Use all frames for training (but not all of them are key frames)
        if len(self.active_window) < self.active_window_size:
            self.active_window.append(cur_view)
        else:
            self.active_window.pop(0)
            self.active_window.append(cur_view)

        # Naively way to add coarse key frames
        if is_key_frame:
            if len(self.coarse_pool) <= self.coarse_active_window_size:
                self.coarse_active_window = self.coarse_pool
            else:
                covis = (
                    np.array(
                        [
                            cal_cams_covisibility(cur_view, cam)
                            for cam in self.coarse_pool
                        ]
                    )
                    + 1e-6
                )

                prob = covis / np.sum(covis)

                self.coarse_active_window = np.random.choice(
                    self.coarse_pool,
                    self.coarse_active_window_size,
                    replace=False,
                    p=prob,
                )

            if len(self.coarse_pool) < self.coarse_pool_size:
                self.coarse_pool.append(cur_view)
            elif self.coarse_pool_size != 0:
                self.coarse_pool.pop(0)
                self.coarse_pool.append(cur_view)

        # Only add key frame camera to camera optimizer
        if is_key_frame:
            self.camera_optimizer.add_cam(cur_view)

        Log(
            "Process frame: {} | total gaussians {} | kf size {}".format(
                cur_view.cam_idx,
                self.gaussians.get_num_gaussians,
                self.kf_graph.get_kf_num,
            ),
            tag="SceneMapper",
        )
        # Log("Active window {} | Coarse active window {}".format([cam.cam_idx for cam in self.active_window], [cam.cam_idx for cam in self.coarse_active_window]), tag="SceneMapper")

        t4.record()
        torch.cuda.synchronize()
        frame_process_time = t3.elapsed_time(t4) / 1000.0

        self.opt_log["is_key_frame"][cur_view.name] = is_key_frame

        return False, is_key_frame, get_frame_time, frame_process_time

    def camera_refinement(self):
        if len(self.active_window) == 0:
            return

        if self.gaussians.get_num_gaussians == 0:
            return

        optimized_local_window = []
        exist_idx = []
        # cur_res_idx is removed as it was assigned but never used
        for cam in self.active_window:
            if cam.cam_idx not in exist_idx:
                cam.to_device(self.device)
                optimized_local_window.append(cam)
                exist_idx.append(cam.cam_idx)
                # Removed assignment to cur_res_idx
        for cam in self.coarse_active_window:
            if cam.cam_idx not in exist_idx:
                cam.to_device(self.device)
                optimized_local_window.append(cam)
                exist_idx.append(cam.cam_idx)

        gt_imgs = torch.stack([cam.get_gt_image() for cam in optimized_local_window])
        gt_mask = (
            torch.sum(gt_imgs, dim=-1, keepdim=True) <= 0.0000001
        ).float()  # ignore invalid pixels in

        for cam in optimized_local_window:
            cam.set_opt_pose(True)

        for _ in range(self.pose_refine_init_steps):
            batch_render_pkg = self.gaussians.render_batch(
                optimized_local_window, self.use_random_bg, detach_gaussians=True
            )
            batch_render = batch_render_pkg["render"]

            iter_gt_imgs = gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, _ = self.image_loss(batch_render, iter_gt_imgs)

            loss = loss_img
            loss.backward()

            self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            # save tracked poses
            for cam in optimized_local_window:
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.name not in self.opt_log["cam_opt_iterations"]:
                        self.opt_log["cam_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["cam_opt_iterations"][cam.name] += 1

        for cam in optimized_local_window:
            cam.set_opt_pose(False)
        # Don't need to set them back to cpu, they will be used again in the map optimization
        # for cam in optimized_local_window:
        #     cam.to_device("cpu")

    def post_refinement(self):
        if self.kf_graph.global_widndow_size == 0:
            return

        self.gaussians.reset_optimizer(
            self.post_refinement_config, self.kf_graph.global_widndow_size
        )
        origin_kf_method = self.kf_graph.get_get_next_camera_method()
        self.kf_graph.set_get_next_camera_method("random")

        opt_cam = self.post_refinement_config["opt_cam"]

        used_cams = []
        exist_idx = []

        pbar = tqdm(range(self.post_refinement_config["max_steps"]))

        for step in pbar:
            global_windows = self.kf_graph.get_and_update_global_window()
            for cam in global_windows:
                if cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    if opt_cam:
                        cam.set_opt_pose(True)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.get_gt_image() for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.get_sparse_depth() for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            iter_window = global_windows
            iter_gt_imgs = global_gt_imgs
            iter_gt_depths = global_gt_depths

            batch_render_pkg = self.gaussians.render_batch(
                iter_window, self.use_random_bg
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            loss_gaussians = self.gaussian_loss(self.gaussians)

            if (
                "normal" in batch_render_pkg
                and "normal_from_depth" in batch_render_pkg
                and "opacity" in batch_render_pkg
            ):
                loss_normal = self.normal_loss(
                    batch_render_pkg["normal"],
                    batch_render_pkg["normal_from_depth"],
                    batch_render_pkg["opacity"],
                )
            else:
                loss_normal = 0.0

            if "distortion" in batch_render_pkg:
                loss_distortion = self.distortion_loss(batch_render_pkg["distortion"])
            else:
                loss_distortion = 0.0

            loss = (
                loss_img + loss_gaussians + loss_depth + loss_normal + loss_distortion
            )

            for cam, err in zip(
                iter_window[-self.global_window_size :],
                l1_values[-self.global_window_size :],
            ):
                self.kf_graph.update_err(cam.cam_idx, err)

            loss.backward()

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

            if opt_cam:
                self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            self.gaussians.step_all_lr()  # Update learning rate

            # save tracked poses
            for i, cam in enumerate(iter_window):
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.cam_idx not in self.opt_log["l1_err"]:
                        self.opt_log["l1_err"][cam.cam_idx] = {}
                    self.opt_log["l1_err"][cam.cam_idx][self.cur_view.cam_idx] = (
                        l1_values[i]
                    )

                    if cam.name not in self.opt_log["gaussian_opt_iterations"]:
                        self.opt_log["gaussian_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["gaussian_opt_iterations"][cam.name] += 1

                    if step < self.pose_opt_steps:
                        if cam.name not in self.opt_log["cam_opt_iterations"]:
                            self.opt_log["cam_opt_iterations"][cam.name] = 1
                        else:
                            self.opt_log["cam_opt_iterations"][cam.name] += 1

            pbar.set_description(
                "Loss: {:.4f} | Mean Lr: {:.7f}".format(
                    loss.item(), self.gaussians.get_avg_pos_lr()
                )
            )

        for cam in used_cams:
            if opt_cam:
                cam.set_opt_pose(False)
            cam.to_device("cpu")

        self.kf_graph.set_get_next_camera_method(origin_kf_method)

    def additional_optimization(self):
        used_cams = []
        exist_idx = []

        for step in range(5):
            global_windows = self.kf_graph.get_and_update_global_window()
            for cam in global_windows:
                if cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.gt_img for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.sparse_depth for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            iter_window = global_windows
            iter_gt_imgs = global_gt_imgs
            iter_gt_depths = global_gt_depths

            batch_render_pkg = self.gaussians.render_batch(
                iter_window, self.use_random_bg
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            # loss_gaussians = self.gaussian_loss(self.gaussians)

            est_depths, est_vars = self.depth_cov_estimator.estimate_depth(
                iter_gt_imgs.permute(0, 3, 1, 2),
                batch_render_depth.view(
                    batch_render_depth.shape[0],
                    1,
                    batch_render_depth.shape[1],
                    batch_render_depth.shape[2],
                ),
            )

            est_normal = convert_depth_to_normal(
                est_depths, global_windows[0].get_int_mat()
            )

            render_normal = convert_depth_to_normal(
                batch_render_depth.view(
                    batch_render_depth.shape[0],
                    1,
                    batch_render_depth.shape[1],
                    batch_render_depth.shape[2],
                ),
                global_windows[0].get_int_mat(),
            )

            print(est_depths.shape)
            loss_normal = (
                (1.0 - torch.abs(est_normal * render_normal).sum(1)) * (est_vars < 0.01)
            ).mean()
            loss_depth = (
                (est_depths.squeeze(-1) - batch_render_depth) ** 2 * (est_vars < 0.01)
            ).mean()

            if step == 0:
                cur_idx = self.cur_view.cam_idx

                saveTensorAsEXR(
                    est_depths[0],
                    pjoin("tmp/visualize_normals", "est_depth_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    est_vars[0],
                    pjoin("tmp/visualize_normals", "est_var_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    est_normal[0].permute(1, 2, 0),
                    pjoin("tmp/visualize_normals", "est_normal_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    render_normal[0].permute(1, 2, 0),
                    pjoin(
                        "tmp/visualize_normals", "render_normal_{}.exr".format(cur_idx)
                    ),
                )
                #     saveTensorAsEXR(iter_gt_imgs[0], pjoin("tmp/visualize_normals", "gt_{}.exr".format(cur_idx)))
                # saveTensorAsEXR(batch_render[0], pjoin("tmp/visualize_normals", "render_{}.exr".format(cur_idx)))
                saveTensorAsEXR(
                    batch_render_depth[0],
                    pjoin(
                        "tmp/visualize_normals", "render_depth_{}.exr".format(cur_idx)
                    ),
                )

            Log(
                "Additional optimization: loss_normal: {}, loss_depth: {}, loss_img: {}".format(
                    loss_normal, loss_depth, loss_img
                ),
                tag="SceneMapper",
            )

            loss = loss_img + loss_normal + loss_depth

            loss.backward()

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

        for cam in used_cams:
            cam.to_device("cpu")

    def run(self):  # Main process of scene mapper
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        now = torch.cuda.Event(enable_timing=True)

        infos = {
            "fps": 0,
            "Num gaussians": 0,
        }

        tracked_info = {
            "scenes": {
                "scene_exposure_gain": self.gaussians.scene_exposure_gain,
                "use_anti_aliasing": self.gaussians.use_anti_aliasing,
                "sh_degree": self.gaussians.max_sh_degree,
                "radius_clip": self.gaussians.radius_clip,
            },
            "cameras": [],
        }

        while True:
            # cur_view_render_pkg = None  # Removed unused variable

            is_last_frame, is_key_frame, get_frame_time, frame_process_time = (
                self.update_kf()
            )

            if not is_last_frame:
                self.opt_log["poses_pair"][self.cur_view.cam_idx] = (
                    self.cur_view.get_raw_pose().detach().cpu().numpy(),
                    self.cur_view.get_pose().detach().cpu().numpy(),
                    is_key_frame,
                )

            if self.use_multi_reso:
                LEVEL_VALUE = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            else:
                LEVEL_VALUE = 0

            start.record()
            if is_key_frame:
                self.optimize(
                    is_last_frame=is_last_frame,
                    is_key_frame=is_key_frame,
                    level=LEVEL_VALUE,
                )

            # uncomment this part if going to save online recon results
            # self.optimize(is_last_frame=is_last_frame, is_key_frame=is_key_frame, level=0, steps=1)

            end.record()
            torch.cuda.synchronize()
            map_time = start.elapsed_time(end) / 1000.0

            if is_last_frame:
                break

            # after initialization, we add gaussians after the optimization
            # if self.initialization_frames < 0:
            start.record()
            if is_key_frame:
                self.densification(level=LEVEL_VALUE)
            else:
                self.densification_pts_only()
            end.record()
            torch.cuda.synchronize()
            densification_time = start.elapsed_time(end) / 1000.0

            tracked_info["cameras"].append(
                {
                    "uid": self.cur_view.cam_idx,
                    "name": self.cur_view.name
                    if self.cur_view.name is not None
                    else "",
                    # "pose": self.cur_view.get_pose().detach().cpu().numpy().tolist(), # pose here is not the final pose
                    "raw_pose": self.cur_view.get_raw_pose()
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "fx": self.cur_view.fx,
                    "fy": self.cur_view.fy,
                    "cx": self.cur_view.cx,
                    "cy": self.cur_view.cy,
                    "width": self.cur_view.width,
                    "height": self.cur_view.height,
                    "exposure_gain": self.cur_view.exposure_gain,
                    "near": self.cur_view.near,
                    "far": self.cur_view.far,
                    "timestamp": time.time(),
                    "is_key_frame": is_key_frame,
                }
            )

            start.record()

            if self.processed_frames % self.save_model_interval == 0:
                self.gaussians.save_as_ply(
                    pjoin(
                        self.configs["Results"]["save_dir"],
                        "online",
                        "point_cloud",
                        f"{self.cur_idx:06d}.ply",
                    )
                )

            end.record()
            torch.cuda.synchronize()
            visualization_time = start.elapsed_time(end) / 1000.0
            # Log("Visualization time: ", visualization_time, "s", tag="SceneMapper")
            Log(
                "Time | Get: {:.2f} ms | Process: {:2f} ms | Optimization: {:.2f} ms | Densification: {:.2f} ms | Visualize: {:.2f} ms".format(
                    get_frame_time * 1000,
                    frame_process_time * 1000,
                    map_time * 1000,
                    densification_time * 1000,
                    visualization_time * 1000,
                ),
                tag="SceneMapper",
            )

            now.record()
            torch.cuda.synchronize()
            total_time = self.last_record_time.elapsed_time(now) / 1000.0
            self.last_record_time.record()

            infos["fps"] = 1.0 / total_time
            infos["Num gaussians"] = self.gaussians.get_num_gaussians
            infos["Exposure*Gain"] = self.cur_view.exposure_gain
            infos["Get frame time"] = get_frame_time * 1000
            infos["Frame process time"] = frame_process_time * 1000
            infos["Optimization time"] = map_time * 1000
            infos["Densification time"] = densification_time * 1000
            infos["Visualization time"] = visualization_time * 1000
            infos["GPU Memory Usage"] = "{:03f} GB".format(
                torch.cuda.memory_allocated(self.device) / (1024**3)
            )

        # Post Refinement
        start.record()
        self.post_refinement()
        end.record()
        torch.cuda.synchronize()
        post_refinement_time = start.elapsed_time(end) / 1000.0
        Log("Post Refinement time: ", post_refinement_time, "s", tag="SceneMapper")

        # Save the final model and other parameters
        self.gaussians.save_as_ply(
            pjoin(self.configs["Results"]["save_dir"], "point_cloud.ply")
        )

        for i in range(len(tracked_info["cameras"])):
            name = tracked_info["cameras"][i]["name"]
            if name is not None and name in self.opt_log["tracked_poses"]:
                tracked_info["cameras"][i]["pose"] = self.opt_log["tracked_poses"][
                    name
                ].tolist()

            if name is not None and name in self.opt_log["cam_opt_iterations"]:
                tracked_info["cameras"][i]["cam_opt_iterations"] = self.opt_log[
                    "cam_opt_iterations"
                ][name]

            if name is not None and name in self.opt_log["gaussian_opt_iterations"]:
                tracked_info["cameras"][i]["gaussian_opt_iterations"] = self.opt_log[
                    "gaussian_opt_iterations"
                ][name]

        json.dump(
            tracked_info,
            open(
                pjoin(self.configs["Results"]["save_dir"], "tracked_info.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )

        json.dump(
            self.opt_log["l1_err"],
            open(
                pjoin(self.configs["Results"]["save_dir"], "l1_err.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )

        if self.gaussians.get_vignette_img() is not None:
            saveTensorAsPNG(
                self.gaussians.get_vignette_img()[0],
                pjoin(self.configs["Results"]["save_dir"], "vignette.png"),
            )

        # meta info
        meta_info = {}
        meta_info["num_gaussians"] = self.gaussians.get_num_gaussians
        meta_info["num_keyframes"] = self.kf_graph.get_kf_num
        meta_info["num_processed_frames"] = self.processed_frames
        meta_info["kf_ids"] = self.kf_graph.get_kf_cam_idx
        return meta_info, self.opt_log
