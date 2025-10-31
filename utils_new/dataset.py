# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from os.path import join as pjoin

import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader
from utils_new.camera_utils import Camera
from utils_new.logging_utils import Log
from utils_new.tool_utils import focal2fov, loadExr


def single_batch_collate_fn(batch):
    assert len(batch) == 1  # only one sample in a batch

    return batch[0]


class ArialParser:
    def __init__(
        self,
        input_folder,
        begin_cutoff=-1,
        end_cutoff=-1,
        stride=1,
        exclude_interval=-1,
        data_source="aria",
    ):
        self.input_folder = input_folder
        self.load_poses(
            self.input_folder,
            begin_cutoff,
            end_cutoff,
            stride,
            exclude_interval,
            data_source,
        )
        self.load_per_frame_point_cloud(self.input_folder, data_source)
        self.n_img = len(self.color_paths)

    def load_poses(
        self,
        datapath,
        begin_cutoff=-1,
        end_cutoff=-1,
        stride=1,
        exclude_interval=-1,
        data_source="aria",
    ):
        if data_source == "aria":
            f = open(pjoin(datapath, "trajectory.json"))
        elif data_source == "orb":
            f = open(pjoin(datapath, "trajectory_orb.json"))
        else:
            raise NotImplementedError
        data = json.load(f)

        infos = data["cameras"]

        if begin_cutoff > 0:
            infos = infos[begin_cutoff:]

        if end_cutoff > 0:
            infos = infos[:-end_cutoff]

        infos = infos[::stride]

        if exclude_interval > 0:
            infos = [infos[i] for i in range(len(infos)) if i % exclude_interval != 0]

        self.color_paths, self.poses, self.frames = [], [], []

        for info in infos:
            pose = np.array(info["T_camera_world"]).astype(np.float32)
            color_path = pjoin(datapath, "rectified", info["image"])

            self.color_paths.append(color_path)
            self.poses += [pose]

            frame = {"file_path": str(color_path), "transform_matrix": pose.tolist()}

            self.frames.append(frame)

        f.close()

    def load_per_frame_point_cloud(self, datapath, data_source="aria"):
        self.pts_list = None

        if data_source == "aria":
            pts_dir = pjoin(datapath, "point_cloud")
        elif data_source == "orb":
            pts_dir = pjoin(datapath, "orb_point_clouds")
        else:
            raise NotImplementedError

        if os.path.exists(pts_dir):
            self.pts_list = []
            for color_path in self.color_paths:
                idx = int(os.path.basename(color_path).split(".")[0].split("_")[-1])
                pts_path = os.path.join(pts_dir, "point_cloud_{}.txt".format(idx))
                self.pts_list.append(pts_path)


class ArialDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]

        # Camera intrinsic prameters
        calibration = config["Calibration"]
        self.near = calibration["near"]
        self.far = calibration["far"]
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.scene_exposure_gain = config["scene_exposure_gain"]
        #############################################################

        self.depth_type = config["depth_type"] if "depth_type" in config else "none"
        self.data_source = config["data_source"] if "data_source" in config else "aria"
        self.max_pts_num = config["max_pts_num"] if "max_pts_num" in config else -1

        Log("Data source: {}".format(self.data_source), tag="Dataset")

        begin_cutoff = -1 if "begin_cutoff" not in config else config["begin_cutoff"]
        end_cutoff = -1 if "end_cutoff" not in config else config["end_cutoff"]
        stride = 1 if "stride" not in config else config["stride"]
        exclude_interval = (
            -1 if "exclude_interval" not in config else config["exclude_interval"]
        )

        parser = ArialParser(
            self.dataset_path,
            begin_cutoff=begin_cutoff,
            end_cutoff=end_cutoff,
            stride=stride,
            exclude_interval=exclude_interval,
            data_source=self.data_source,
        )
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.poses = parser.poses
        self.pts_list = parser.pts_list

        if "vignette" in config and config["vignette"]:
            Log("Use vignette correction.", tag="Dataset")
            self.vignette = (
                np.array(Image.open(pjoin(self.dataset_path, "vignette.png"))) / 255.0
            )
        else:
            self.vignette = None

        self.use_vignette_type = config["use_vignette_type"]

        self.sample_cam = Camera.init_from_dataset(
            0,
            None,
            None,
            torch.from_numpy(np.eye(4)).float(),
            None,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.fovx,
            self.fovy,
            self.height,
            self.width,
            self.near,
            self.far,
            self.scene_exposure_gain,
            0,
        )

        Log("Dataset size: {}".format(self.num_imgs), tag="Dataset")

    def __len__(self):
        return self.num_imgs

    @property
    def get_vignette(self):
        if self.use_vignette_type == "post-render":
            return self.vignette
        else:
            return None

    def get_poses(self, interval):
        poses = []
        for p in self.poses[::interval]:
            poses.append(p)
        return poses

    def get_sample_cam(self):
        return self.sample_cam

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        image = np.array(Image.open(color_path))

        color_image_for_init_pts = image.copy() / 255.0
        if self.vignette is not None:
            color_image_for_init_pts = color_image_for_init_pts / self.vignette
            invalid_mask = self.vignette == 0
            color_image_for_init_pts[invalid_mask] = 0
            color_image_for_init_pts = cv2.blur(color_image_for_init_pts, (5, 5))

        if self.vignette is not None and self.use_vignette_type == "pre-render":
            image = image / self.vignette
        image = torch.from_numpy(image / 255.0).clamp(0.0, 1.0).float()

        pose = self.poses[idx]

        prev_pose = self.poses[idx - 1] if idx > 0 else None
        if prev_pose is None:
            rot_speed = 0
        else:
            cur_rot = np.linalg.inv(pose)[:3, :3]
            prev_rot = np.linalg.inv(prev_pose)[:3, :3]
            rot_diff = np.matmul(cur_rot, np.linalg.inv(prev_rot))
            rot_speed = (
                np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)) * 180 / np.pi
            )

        pose = torch.from_numpy(pose)

        point_cloud_file_path = self.pts_list[idx]
        xyz = []
        if os.path.exists(point_cloud_file_path.replace(".txt", ".npy")):
            xyz = np.load(point_cloud_file_path.replace(".txt", ".npy"))
        else:
            with open(point_cloud_file_path, "r") as f:
                buf = f.readlines()
                for line in buf:
                    xyz.append(list(map(float, line.split())))
                xyz = np.array(xyz)

        if self.max_pts_num > 0:
            np.random.shuffle(xyz)
            xyz = xyz[: self.max_pts_num]

        # depth image
        if self.depth_type == "none":
            depth = None
        elif self.depth_type == "3dgs":
            img_idx = int(os.path.basename(color_path).split(".")[0].split("_")[-1])
            depth = loadExr(
                pjoin(
                    self.dataset_path,
                    "est_depth_3dgs",
                    "{:05d}_depth.exr".format(img_idx),
                )
            )
        else:
            raise NotImplementedError

        name = os.path.basename(color_path)

        cam = Camera.init_from_dataset(
            idx,
            image,
            color_image_for_init_pts,
            pose,
            xyz,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.fovx,
            self.fovy,
            image.shape[0],
            image.shape[1],
            self.near,
            self.far,
            self.scene_exposure_gain,  # for loading from a dataset, we consider it as fixed exposure
            rot_speed,
            depth=depth,
            name=name,
        )

        return cam


class Arial_ViewSyn_Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        dataset_path = config["dataset_path"]

        # Camera intrinsic prameters
        calibration = config["Calibration"]
        self.near = calibration["near"]
        self.far = calibration["far"]
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        #############################################################

        begin_cutoff = -1 if "begin_cutoff" not in config else config["begin_cutoff"]
        end_cutoff = -1 if "end_cutoff" not in config else config["end_cutoff"]
        stride = 1 if "stride" not in config else config["stride"]

        self.scene_exposure_gain = config["scene_exposure_gain"]

        if "vignette" in config and config["vignette"]:
            Log("Use vignette correction.", tag="Dataset")
            self.vignette = (
                np.array(Image.open(pjoin(dataset_path, "vignette.png"))) / 255.0
            )
        else:
            self.vignette = None

        self.use_vignette_type = config["use_vignette_type"]

        parser = ArialParser(
            dataset_path,
            begin_cutoff=begin_cutoff,
            end_cutoff=end_cutoff,
            stride=stride,
        )

        original_poses = parser.poses

        key_frame_dist = config["key_frame_dist"]
        novel_view_num_per_interval = config["novel_view_num_per_interval"]
        perturb_radius = config["perturb_radius"]

        key_poses = [original_poses[0]]
        last_pos = np.linalg.inv(key_poses[-1])[:3, 3]
        for pose in original_poses:
            cur_pose = np.linalg.inv(pose)[:3, 3]
            if np.linalg.norm(cur_pose - last_pos) > key_frame_dist:
                key_poses.append(pose)
                last_pos = cur_pose

        self.poses = []
        for i in range(len(key_poses) - 1):
            self.poses.append(key_poses[i].astype(np.float32))

            cur_pose_cam2world = np.linalg.inv(key_poses[i])
            next_pose_cam2world = np.linalg.inv(key_poses[i + 1])

            cur_position = cur_pose_cam2world[:3, 3]
            next_position = next_pose_cam2world[:3, 3]

            cur_rot = cur_pose_cam2world[:3, :3]
            next_rot = next_pose_cam2world[:3, :3]

            diff_rot = np.matmul(np.linalg.inv(cur_rot), next_rot)
            diff_rot_vec = cv2.Rodrigues(diff_rot)[0].reshape(3)

            pos_z = (next_position - cur_position) / np.linalg.norm(
                next_position - cur_position
            )

            tmp_v = np.array([0, 0, 1])
            if np.dot(pos_z, tmp_v) > 0.99:
                tmp_v = np.array([0, 1, 0])
            pos_x = np.cross(pos_z, tmp_v)
            pos_x = pos_x / np.linalg.norm(pos_x)
            pos_y = np.cross(pos_z, pos_x)
            pos_x = np.cross(pos_y, pos_z)

            for j in range(novel_view_num_per_interval):
                alpha = j / novel_view_num_per_interval
                alpha = 0.0 if j == 0 else 1.0 / (1 + (alpha / (1 - alpha)) ** (-2))
                alpha_diff_rot_vec = alpha * diff_rot_vec
                alpha_diff_rot_mat = cv2.Rodrigues(alpha_diff_rot_vec)[0].reshape(3, 3)
                alpha_rot_mat = np.matmul(cur_rot, alpha_diff_rot_mat)

                cur_radius = perturb_radius * (1 - alpha) * alpha * 4
                angle = np.pi * 2 * alpha

                dx = cur_radius * np.cos(angle)
                dy = cur_radius * np.sin(angle)

                pos = (
                    (1 - alpha) * cur_position
                    + alpha * next_position
                    + dx * pos_x
                    + dy * pos_y
                )

                interp_pose = np.eye(4)
                interp_pose[:3, :3] = alpha_rot_mat
                interp_pose[:3, 3] = pos

                self.poses.append(
                    np.linalg.inv(interp_pose).astype(np.float32)
                )  # inverse to world2cam

        self.poses.append(key_poses[-1].astype(np.float32))
        self.num_poses = len(self.poses)

        Log("Dataset size: {}".format(self.num_poses), tag="Dataset")

    def __len__(self):
        return self.num_poses

    @property
    def get_vignette(self):
        if self.use_vignette_type == "post-render":
            return self.vignette
        else:
            return None

    def __getitem__(self, idx):
        pose = self.poses[idx]

        prev_pose = self.poses[idx - 1] if idx > 0 else None
        if prev_pose is None:
            rot_speed = 0
        else:
            cur_rot = np.linalg.inv(pose)[:3, :3]
            prev_rot = np.linalg.inv(prev_pose)[:3, :3]
            rot_diff = np.matmul(cur_rot, np.linalg.inv(prev_rot))
            rot_speed = (
                np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)) * 180 / np.pi
            )

        pose = torch.from_numpy(pose)

        cam = Camera.init_from_dataset(
            idx,
            None,
            None,
            pose,
            None,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.fovx,
            self.fovy,
            self.height,
            self.width,
            self.near,
            self.far,
            self.scene_exposure_gain,
            rot_speed,
        )

        return cam


def load_dataset(configs, ret_original_dataset=False):
    num_threads = configs["num_threads"]
    if configs["type"] == "aria":
        dataset = ArialDataset(configs)
    elif configs["type"] == "aria-viewsyn":
        dataset = Arial_ViewSyn_Dataset(configs)
    else:
        raise ValueError("Unknown dataset type")

    if ret_original_dataset:
        return dataset
    else:
        dataset_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_threads,
            collate_fn=single_batch_collate_fn,
        )

        return dataset_loader
