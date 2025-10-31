# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import time
from glob import glob
from os.path import join as pjoin

import numpy as np
import torch

from PIL import Image
from utils_new.camera_utils import Camera

from utils_new.logging_utils import Log
from utils_new.tool_utils import focal2fov


class Tracker:
    def __init__(self, config):
        self.dataset_path = config["dataset_path"]
        self.config = config

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
        ################################################################

        begin_cutoff = -1 if "begin_cutoff" not in config else config["begin_cutoff"]
        end_cutoff = -1 if "end_cutoff" not in config else config["end_cutoff"]
        stride = 1 if "stride" not in config else config["stride"]
        exclude_interval = (
            -1 if "exclude_interval" not in config else config["exclude_interval"]
        )

        suffix = config["suffix"] if "suffix" in config else ".png"

        if "img_folder_path" in config:
            img_folder_path = config["img_folder_path"]
        else:
            img_folder_path = "rectified"

        color_paths = glob(pjoin(self.dataset_path, img_folder_path, "*" + suffix))
        color_paths = sorted(color_paths)

        if begin_cutoff > 0:
            color_paths = color_paths[begin_cutoff:]
        if end_cutoff > 0:
            color_paths = color_paths[:-end_cutoff]
        color_paths = color_paths[::stride]
        if exclude_interval > 0:
            color_paths = [
                color_paths[i]
                for i in range(len(color_paths))
                if i % exclude_interval != 0
            ]

        self.color_paths = color_paths
        self.num_imgs = len(self.color_paths)

        if "track_folder" in config:
            track_folder = config["track_folder"]
            track_color_paths = glob(
                pjoin(self.dataset_path, track_folder, "*" + suffix)
            )
            track_color_paths = sorted(track_color_paths)

            if begin_cutoff > 0:
                track_color_paths = track_color_paths[begin_cutoff:]
            if end_cutoff > 0:
                track_color_paths = track_color_paths[:-end_cutoff]
            track_color_paths = track_color_paths[::stride]
            if exclude_interval > 0:
                track_color_paths = [
                    track_color_paths[i]
                    for i in range(len(track_color_paths))
                    if i % exclude_interval != 0
                ]
            self.track_color_paths = track_color_paths
        else:
            self.track_color_paths = None

        if "vignette" in config and config["vignette"]:
            Log("Use vignette correction.", tag="Dataset")
            self.vignette = (
                np.array(Image.open(pjoin(self.dataset_path, "vignette.png"))) / 255.0
            )
        else:
            self.vignette = None
        self.use_vignette = config["vignette"]
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

        self.cur_time = None
        self.prev_pose = None
        self.frame_time = 0
        self.frame_idx = 0
        self.fps = config["fps"] if "fps" in config else -1

        # Tracker
        self.tracker = None

        # Songyin: Can have two version, stable single process and faster multi-process
        self.q_buffer = mp.Manager().Queue(maxsize=50)
        self.get_data_process = mp.Process(target=self.get_single_data_mp)
        self.get_data_process.start()

    def get_single_data(self):
        if self.frame_idx >= self.num_imgs:
            return None

        last_time = self.cur_time
        self.cur_time = time.time()

        if self.fps > 0 and last_time is not None:
            sleep_time = 1.0 / self.fps - (self.cur_time - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.cur_time = time.time()

        if self.fps > 0:
            self.frame_time += 1 / self.fps
        else:
            self.frame_time = self.cur_time

        image = np.array(Image.open(self.color_paths[self.frame_idx]))

        if self.track_color_paths is not None:
            track_image = np.array(Image.open(self.track_color_paths[self.frame_idx]))
        else:
            track_image = image.copy()

        pose_3x4 = self.tracker.process_image_mono(track_image, self.frame_time)
        pts = np.array(self.tracker.get_current_points())

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = np.array(pose_3x4).reshape(3, 4)

        if self.prev_pose is None:
            rot_speed = 0
        else:
            cur_rot = np.linalg.inv(pose)[:3, :3]
            prev_rot = np.linalg.inv(self.prev_pose)[:3, :3]
            rot_diff = np.matmul(cur_rot, np.linalg.inv(prev_rot))
            rot_speed = (
                np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)) * 180 / np.pi
            )
        self.prev_pose = pose  # update prev pose

        image = image / 255.0
        image_for_init_pts = image.copy()
        if self.vignette is not None:
            image_for_init_pts = image_for_init_pts / self.vignette
            invalid_mask = self.vignette == 0
            image_for_init_pts[invalid_mask] = 0
        if self.vignette is not None and self.use_vignette_type == "pre-render":
            image = image / self.vignette
        image = torch.from_numpy(image).clamp(0.0, 1.0).float()

        pose = torch.from_numpy(pose).float()

        name = os.path.basename(self.color_paths[self.frame_idx])

        cam = Camera.init_from_dataset(
            self.frame_idx,
            image,
            image_for_init_pts,  # not used
            pose,
            pts,
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
            depth=None,
            name=name,
        )

        self.frame_idx += 1

        return cam

    def get_single_data_mp(self):
        # TODO: Initialize the tracker

        while True:
            data = self.get_single_data()
            self.q_buffer.put(data)
            if data is None:
                break

    def get_data(self):
        data = self.q_buffer.get()

        if data is None:
            self.get_data_process.join()

        return data

    @property
    def get_vignette(self):
        return self.vignette

    def get_sample_cam(self):
        return self.sample_cam


def load_streamer(config):
    raise NotImplementedError("Stream loader is not implemented for this mapper.")
