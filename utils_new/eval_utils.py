# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from os.path import join as pjoin

import numpy as np

import open3d as o3d

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from utils_new.dataset import load_dataset

from utils_new.logging_utils import Log
from utils_new.tool_utils import (
    interpolate_pose,
    mkdir_p,
    saveTensorAsEXR,
    saveTensorAsPNG,
)


def eval_gaussians(gaussians, optimization_infos, configs):
    # tracked_poses = optimization_infos["tracked_poses"]
    poses_pairs = optimization_infos["poses_pair"]
    is_key_frames = optimization_infos["is_key_frame"]

    final_eval_poses = {}

    with torch.no_grad():
        if "Testset" in configs:
            save_mesh = (
                configs["Results"]["save_mesh"]
                if "save_mesh" in configs["Results"]
                else False
            )
            voxel_size = (
                configs["Results"]["voxel_size"]
                if "voxel_size" in configs["Results"]
                else 5.0 / 512.0
            )

            if save_mesh:
                depth_scale = 1.0
                depth_max = 10000.0
                device = o3d.core.Device("cpu:0")
                vbg = o3d.t.geometry.VoxelBlockGrid(
                    attr_names=("tsdf", "weight", "color"),
                    attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                    attr_channels=((1), (1), (3)),
                    voxel_size=voxel_size,
                    block_count=100000,
                    device=device,
                )

            configs["Testset"]["scene_exposure_gain"] = configs["Mapper"][
                "scene_exposure_gain"
            ]
            test_dataset = load_dataset(configs["Testset"])

            vignette_img = test_dataset.dataset.get_vignette
            gaussians.set_vignette_img(vignette_img)

            if vignette_img is not None:
                vignette_tensor = (
                    torch.from_numpy(vignette_img).permute(2, 0, 1).cuda().float()
                )
            else:
                vignette_tensor = None

            save_dir = pjoin(configs["Results"]["save_dir"], "eval")
            save_exr = configs["Results"]["save_exr"]
            render_imgs_dir = pjoin(save_dir, "renders")
            gt_imgs_dir = pjoin(save_dir, "gt")
            mkdir_p(render_imgs_dir)
            mkdir_p(gt_imgs_dir)

            cal_lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to("cuda:0")
            psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda:0")
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda:0")

            psnr_scores = []
            ssim_scores = []
            lpips_scores = []
            results = {}

            # last_orig_pose = None
            # last_actual_pose = None
            last_kf_ids = None
            last_kf_orig_pose = None
            last_kf_actual_pose = None

            for idx, cam in enumerate(tqdm(test_dataset)):
                if cam.name is not None:
                    if is_key_frames[cam.name]:
                        cam.update_pose_numpy(poses_pairs[cam.cam_idx][1])
                        last_kf_orig_pose = poses_pairs[cam.cam_idx][0]
                        last_kf_actual_pose = poses_pairs[cam.cam_idx][1]
                        last_kf_ids = idx
                    else:
                        j = cam.cam_idx + 1
                        while j in poses_pairs and (not poses_pairs[j][2]):
                            j += 1

                        if j not in poses_pairs:
                            assert j == len(test_dataset)
                            next_kf_actual_pose = last_kf_actual_pose
                            next_kf_orig_pose = last_kf_orig_pose
                        else:
                            next_kf_orig_pose = poses_pairs[j][0]
                            next_kf_actual_pose = poses_pairs[j][1]

                        if last_kf_ids is None:
                            interpolated_orig_pose = next_kf_orig_pose
                            interpolated_actual_pose = next_kf_actual_pose
                        else:
                            alpha = (idx - last_kf_ids) / (j - last_kf_ids)

                            interpolated_orig_pose = interpolate_pose(
                                last_kf_orig_pose, next_kf_orig_pose, alpha
                            )
                            interpolated_actual_pose = interpolate_pose(
                                last_kf_actual_pose, next_kf_actual_pose, alpha
                            )

                        cur_orig_pose = poses_pairs[cam.cam_idx][0]

                        pose = (
                            cur_orig_pose
                            @ np.linalg.inv(interpolated_orig_pose)
                            @ interpolated_actual_pose
                        )
                        cam.update_pose_numpy(pose)

                final_eval_poses[cam.name] = (
                    cam.get_pose().detach().cpu().numpy().tolist()
                )

                cam.to_device("cuda:0")

                render_pkg = gaussians.render(cam)
                render_img = render_pkg["render"]
                render_img = torch.clamp(render_img, min=0.0, max=1.0)

                # depth_img = render_pkg["depth"]

                gt_img = cam.get_gt_image()

                if save_exr:
                    saveTensorAsEXR(
                        render_img, pjoin(render_imgs_dir, "{:05d}.exr".format(idx))
                    )
                    # saveTensorAsEXR(depth_img, pjoin(render_imgs_dir, "{:05d}_depth.exr".format(idx)))
                saveTensorAsPNG(
                    render_img, pjoin(render_imgs_dir, "{:05d}.png".format(idx))
                )

                if configs["Results"]["save_gt"]:
                    saveTensorAsPNG(
                        gt_img, pjoin(gt_imgs_dir, "{:05d}.png".format(idx))
                    )

                render_img = render_img.permute(2, 0, 1)
                gt_img = gt_img.permute(2, 0, 1)

                gt_mask = (
                    torch.sum(gt_img, dim=0, keepdim=True) <= 0.0000001
                ).float()  # ignore invalid pixels in gt
                gt_img = gt_img * (1 - gt_mask) + render_img * gt_mask

                if vignette_tensor is not None:
                    gt_img = gt_img / vignette_tensor
                    render_img = render_img / vignette_tensor
                    gt_img[torch.isnan(gt_img)] = 0.0
                    render_img[torch.isnan(render_img)] = 0.0

                    render_img = torch.clamp(render_img, min=0.0, max=1.0)
                    gt_img = torch.clamp(gt_img, min=0.0, max=1.0)

                psnr_v = psnr(render_img.unsqueeze(0), gt_img.unsqueeze(0))
                ssim_v = ssim(render_img.unsqueeze(0), gt_img.unsqueeze(0))
                lpips_v = cal_lpips(render_img.unsqueeze(0), gt_img.unsqueeze(0))

                psnr_scores.append(psnr_v.item())
                ssim_scores.append(ssim_v.item())
                lpips_scores.append(lpips_v.item())
                results["{:05d}".format(idx)] = {
                    "psnr": psnr_v.item(),
                    "ssim": ssim_v.item(),
                    "lpips": lpips_v.item(),
                }

                # update mesh
                if save_mesh:
                    if is_key_frames[cam.name]:
                        render_depth = render_pkg["depth"]
                        depth = render_depth.detach().cpu().numpy()
                        color = render_img.permute(1, 2, 0).detach().cpu().numpy()

                        depth_o3d = np.ascontiguousarray(depth)
                        depth_o3d = o3d.t.geometry.Image(depth_o3d)
                        color_o3d = np.ascontiguousarray(color)
                        color_o3d = o3d.t.geometry.Image(color_o3d)
                        w2c_o3d = cam.get_pose().detach().cpu().numpy()

                        fx = cam.fx
                        fy = cam.fy
                        cx = cam.cx
                        cy = cam.cy
                        intrinsic = o3d.core.Tensor(
                            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                            o3d.core.Dtype.Float64,
                        )

                        w2c_o3d = o3d.core.Tensor(w2c_o3d, o3d.core.Dtype.Float64)
                        color_o3d = color_o3d.to(device)
                        depth_o3d = depth_o3d.to(device)

                        frustum_block_coords = vbg.compute_unique_block_coordinates(
                            depth_o3d, intrinsic, w2c_o3d, depth_scale, depth_max
                        )

                        vbg.integrate(
                            frustum_block_coords,
                            depth_o3d,
                            color_o3d,
                            intrinsic,
                            w2c_o3d,
                            depth_scale,
                            depth_max,
                        )

            results["mean"] = {
                "psnr": np.mean(psnr_scores),
                "ssim": np.mean(ssim_scores),
                "lpips": np.mean(lpips_scores),
            }

            Log(
                f'mean psnr: {results["mean"]["psnr"]}, ssim: {results["mean"]["ssim"]}, lpips: {results["mean"]["lpips"]}',
                tag="Eval",
            )

            json.dump(
                results,
                open(pjoin(save_dir, "final_result.json"), "w", encoding="utf-8"),
                indent=4,
            )

            json.dump(
                final_eval_poses,
                open(pjoin(save_dir, "final_eval_poses.json"), "w", encoding="utf-8"),
                indent=4,
            )

            # save mesh
            if save_mesh:
                print("Saving mesh...")
                mesh_out_file = os.path.join(save_dir, "mesh.ply")
                mesh = vbg.extract_triangle_mesh(weight_threshold=1.0)
                mesh = mesh.to_legacy()
                o3d.io.write_triangle_mesh(mesh_out_file, mesh)
                # o3d_mesh = volume.extract_triangle_mesh()
                # o3d_mesh = clean_mesh(o3d_mesh)
                # o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
                print("Mesh saved to", mesh_out_file)

            return results["mean"]
        else:
            results = {
                "mean": {
                    "psnr": "nan",
                    "ssim": "nan",
                    "lpips": "nan",
                }
            }
            return results["mean"]
