# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from utils_new.camera_utils import in_frustum_region


def cal_covisiblity(
    pts1, ext1, int1, max_depth1, pts2, ext2, int2, max_depth2, img_shape
):
    pts_from_1_in_2 = in_frustum_region(
        pts1, ext2, int2, img_shape, max_depth=max_depth2
    )
    pts_from_2_in_1 = in_frustum_region(
        pts2, ext1, int1, img_shape, max_depth=max_depth1
    )

    pts_from_1_in_2_num = np.sum(pts_from_1_in_2)
    pts_from_2_in_1_num = np.sum(pts_from_2_in_1)

    pts_only_in_1_num = np.sum(~pts_from_1_in_2)
    pts_only_in_2_num = np.sum(~pts_from_2_in_1)

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return 0.1

    if (
        (pts_from_1_in_2_num + pts_from_2_in_1_num) / 2
        + pts_only_in_1_num
        + pts_only_in_2_num
    ) == 0:
        return 0.1

    return ((pts_from_1_in_2_num + pts_from_2_in_1_num) / 2) / (
        (pts_from_1_in_2_num + pts_from_2_in_1_num) / 2
        + pts_only_in_1_num
        + pts_only_in_2_num
    )


def cal_cams_covisibility(cam1, cam2):
    img_shape = (cam1.height, cam1.width)

    covisibility = cal_covisiblity(
        cam1.get_pts(),
        cam1.get_raw_pose().detach().cpu().numpy(),
        cam1.K.cpu().numpy(),
        cam1.max_depth,
        cam2.get_pts(),
        cam2.get_raw_pose().detach().cpu().numpy(),
        cam2.K.cpu().numpy(),
        cam2.max_depth,
        img_shape,
    )

    return covisibility


class KFGraph:
    def __init__(self, configs):
        self.configs = configs

        self.last_kf_info = None

        self.kf_selection_method = configs["kf_selection_method"]
        self.kf_interval = configs["kf_interval"]
        self.kf_overlap_ratio = configs["kf_overlap_ratio"]
        self.kf_cameras = []

        self.kf_cameras_pos3D = None

        self.camIdx_to_kfIdx = {}

        self.idx_list = np.empty(0, dtype=np.int32)

        self.errors = np.empty(0, dtype=np.float32)

        self.global_widndow_size = configs["global_window_size"]

        self.cur_random_window = []  # store the index of the cameras in kf_cameras

        self.get_next_camera_method = configs["get_next_camera_method"]

        self.covisibility_dict = {}  # This one uses cam idx (which is different from the idx in kf_cameras)

        # hyper-parameters
        self.pos_clamp = configs["pos_clamp"]
        self.pos_sigma = configs["pos_sigma"]
        self.cov_clamp = configs["cov_clamp"]
        self.cov_eps = configs["cov_eps"]
        self.err_sigma = configs["err_sigma"]
        self.ris_size = configs["ris_size"]
        self.idx_diff_sigma = configs["idx_diff_sigma"]

    @property
    def get_kf_num(self):
        return len(self.kf_cameras)

    @property
    def get_kf_cam_idx(self):
        return [cam.cam_idx for cam in self.kf_cameras]

    def update_err(self, cam_idx, err):
        self.errors[self.camIdx_to_kfIdx[cam_idx]] = err

    def cal_covisibility_cams(self, cam1, cam2):
        if (
            cam1.cam_idx in self.covisibility_dict
            and cam2.cam_idx in self.covisibility_dict[cam1.cam_idx]
        ):
            return self.covisibility_dict[cam1.cam_idx][cam2.cam_idx]

        covisibility = cal_cams_covisibility(cam1, cam2)

        if (
            cam1.cam_idx in self.covisibility_dict
            and cam2.cam_idx in self.covisibility_dict
        ):
            self.covisibility_dict[cam1.cam_idx][cam2.cam_idx] = covisibility
            self.covisibility_dict[cam2.cam_idx][cam1.cam_idx] = covisibility

        return covisibility

    def init_global_window(self):
        assert len(self.kf_cameras) > 0
        for _ in range(self.global_widndow_size):
            self.cur_random_window.append(0)

    def next_idx_ris(self, cur_idx, cur_cam):
        # other_list = list(range(self.get_kf_num))
        # if cur_idx in other_list:
        #     other_list.remove(cur_idx)
        if cur_idx != -1:
            other_list = np.concatenate(
                [self.idx_list[:cur_idx], self.idx_list[cur_idx + 1 :]]
            )
        else:
            other_list = self.idx_list

        if len(other_list) == 0:
            return cur_idx

        # other_list = np.array(other_list)

        positions_diff = np.linalg.norm(
            cur_cam.pos3D[None, ...] - self.kf_cameras_pos3D, axis=1
        )
        positions_diff = positions_diff[other_list]

        positions_diff_prob = np.exp(
            -np.clip(positions_diff, self.pos_clamp, float("inf")) / self.pos_sigma
        )
        positions_diff_prob = positions_diff_prob / np.sum(positions_diff_prob)

        idx_of_other_list = np.arange(len(other_list))
        candidate_list_idx = np.random.choice(
            idx_of_other_list, self.ris_size, p=positions_diff_prob, replace=True
        )
        candidate_list = other_list[candidate_list_idx]
        candidate_list_original_prob = positions_diff_prob[candidate_list_idx]

        cov_ratio_list = [
            self.cal_covisibility_cams(cur_cam, self.kf_cameras[idx])
            for idx in candidate_list
        ]

        cov_ratio_list = np.array(cov_ratio_list)
        cov_ratio_list_prob = (
            np.clip(cov_ratio_list, 0.0, self.cov_clamp) + self.cov_eps
        )
        cov_ratio_list_prob = cov_ratio_list_prob / candidate_list_original_prob  # ris
        cov_ratio_list_prob = cov_ratio_list_prob / np.sum(cov_ratio_list_prob)

        next_idx = np.random.choice(candidate_list, p=cov_ratio_list_prob)

        return next_idx

    def next_idx_ris_idx_diff(self, cur_idx, cur_cam):
        # other_list = list(range(self.get_kf_num))
        # if cur_idx in other_list:
        #     other_list.remove(cur_idx)
        if cur_idx != -1:
            other_list = np.concatenate(
                [self.idx_list[:cur_idx], self.idx_list[cur_idx + 1 :]]
            )
        else:
            other_list = self.idx_list

        if len(other_list) == 0:
            return cur_idx

        idx_diff = self.idx_list[-1] - other_list
        idx_diff_prob = np.exp(self.idx_diff_sigma * (-idx_diff))
        idx_diff_prob = idx_diff_prob / np.sum(idx_diff_prob)

        # other_list = np.array(other_list)

        positions_diff = np.linalg.norm(
            cur_cam.pos3D[None, ...] - self.kf_cameras_pos3D, axis=1
        )
        positions_diff = positions_diff[other_list]

        positions_diff_prob = np.exp(
            -np.clip(positions_diff, self.pos_clamp, float("inf")) / self.pos_sigma
        )
        positions_diff_prob = positions_diff_prob / np.sum(positions_diff_prob)

        first_selection_prob = positions_diff_prob * idx_diff_prob
        first_selection_prob = first_selection_prob / np.sum(first_selection_prob)

        idx_of_other_list = np.arange(len(other_list))
        candidate_list_idx = np.random.choice(
            idx_of_other_list, self.ris_size, p=first_selection_prob, replace=True
        )
        candidate_list = other_list[candidate_list_idx]
        candidate_list_original_prob = first_selection_prob[candidate_list_idx]

        cov_ratio_list = [
            self.cal_covisibility_cams(cur_cam, self.kf_cameras[idx])
            for idx in candidate_list
        ]

        cov_ratio_list = np.array(cov_ratio_list)
        cov_ratio_list_prob = (
            np.clip(cov_ratio_list, 0.0, self.cov_clamp) + self.cov_eps
        )
        cov_ratio_list_prob = cov_ratio_list_prob / np.sum(cov_ratio_list_prob)

        idx_diff = self.idx_list[-1] - candidate_list
        idx_diff_prob = np.exp(self.idx_diff_sigma * (-idx_diff))
        idx_diff_prob = idx_diff_prob / np.sum(idx_diff_prob)

        final_selection_prob = (
            cov_ratio_list_prob * idx_diff_prob / candidate_list_original_prob
        )
        final_selection_prob = final_selection_prob / np.sum(final_selection_prob)

        next_idx = np.random.choice(candidate_list, p=final_selection_prob)

        return next_idx

    def next_idx_position(self, cur_idx, cur_cam):
        # other_list = list(range(self.get_kf_num))
        # if cur_idx in other_list:
        #     other_list.remove(cur_idx)
        if cur_idx != -1:
            other_list = np.concatenate(
                [self.idx_list[:cur_idx], self.idx_list[cur_idx + 1 :]]
            )
        else:
            other_list = self.idx_list

        if len(other_list) == 0:
            return cur_idx

        # other_list = np.array(other_list)

        positions_diff = np.linalg.norm(
            cur_cam.pos3D[None, ...] - self.kf_cameras_pos3D, axis=1
        )
        positions_diff = positions_diff[other_list]

        positions_diff_prob = np.exp(
            -np.clip(positions_diff, self.pos_clamp, float("inf")) / self.pos_sigma
        )
        positions_diff_prob = positions_diff_prob / np.sum(positions_diff_prob)

        return np.random.choice(other_list, p=positions_diff_prob)

    def next_idx_err(self, cur_idx, cur_cam):
        prob = np.exp(self.err_sigma * self.errors)
        prob = prob / np.sum(prob)

        next_idx = np.random.choice(self.idx_list, p=prob)

        return next_idx

    def next_idx_err_idx_diff(self, cur_idx, cur_cam):
        prob = np.exp(self.err_sigma * self.errors)
        idx_diff = self.idx_list[-1] - self.idx_list
        idx_diff_prob = np.exp(self.idx_diff_sigma * (-idx_diff))
        prob = prob * idx_diff_prob
        prob = prob / np.sum(prob)

        next_idx = np.random.choice(self.idx_list, p=prob)

        return next_idx

    def get_get_next_camera_method(self):
        return self.get_next_camera_method

    def set_get_next_camera_method(self, method):
        self.get_next_camera_method = method

    def get_next_idx(self, cur_idx, cur_cam):
        if self.get_next_camera_method == "random":
            return np.random.randint(self.get_kf_num)
        elif self.get_next_camera_method == "covisibility_ris":
            return self.next_idx_ris(cur_idx, cur_cam)
        elif self.get_next_camera_method == "covisibility_ris_idx_diff":
            return self.next_idx_ris_idx_diff(cur_idx, cur_cam)
        elif self.get_next_camera_method == "position":
            return self.next_idx_position(cur_idx, cur_cam)
        elif self.get_next_camera_method == "err":
            return self.next_idx_err(cur_idx, cur_cam)
        elif self.get_next_camera_method == "err_idx_diff":
            return self.next_idx_err_idx_diff(cur_idx, cur_cam)
        else:
            raise NotImplementedError

    def add_new_cam_for_global(
        self, cur_cam
    ):  # This cam won't be added but its next one is added
        next_idx = self.get_next_idx(-1, cur_cam)

        if len(self.cur_random_window) > 0:
            self.cur_random_window.pop(0)  # pop the first one
            self.cur_random_window.append(next_idx)

    def get_and_update_global_window(self):
        return_cams = [self.kf_cameras[idx] for idx in self.cur_random_window]
        self.update_global_window()
        return return_cams

    def get_global_window(self):
        return [self.kf_cameras[idx] for idx in self.cur_random_window]

    def update_global_window(self):
        self.cur_random_window = [
            self.get_next_idx(idx, self.kf_cameras[idx])
            for idx in self.cur_random_window
        ]

    def add_new_cam_to_kf(self, cur_view, covisibility=None):
        self.kf_cameras.append(cur_view)
        self.idx_list = np.concatenate([self.idx_list, [self.get_kf_num - 1]])
        self.camIdx_to_kfIdx[cur_view.cam_idx] = len(self.kf_cameras) - 1

        init_err = np.mean(self.errors) if len(self.errors) > 0 else 1.0
        self.errors = np.concatenate([self.errors, [init_err]])

        self.kf_cameras_pos3D = (
            cur_view.pos3D[None, ...]
            if self.kf_cameras_pos3D is None
            else np.concatenate(
                [self.kf_cameras_pos3D, cur_view.pos3D[None, ...]], axis=0
            )
        )

        if covisibility is None:
            self.covisibility_dict[cur_view.cam_idx] = {}
        else:
            self.covisibility_dict[cur_view.cam_idx] = {
                self.last_kf_info["cam_idx"]: covisibility
            }
            self.covisibility_dict[self.last_kf_info["cam_idx"]][cur_view.cam_idx] = (
                covisibility
            )

        self.last_kf_info = {
            "cam_idx": cur_view.cam_idx,
            "pose": cur_view.get_raw_pose().detach().cpu().numpy(),
            "K": cur_view.K.detach().cpu().numpy(),
            "pts": cur_view.get_pts(),
            "max_depth": cur_view.max_depth,
            "img_shape": (cur_view.height, cur_view.width),
        }

    def update_frame(self, cur_view):
        is_keyframe = False

        if self.kf_selection_method == "fixed":
            if (
                self.last_kf_info is None
                or cur_view.cam_idx - self.last_kf_info["cam_idx"] > self.kf_interval
            ):
                self.add_new_cam_to_kf(cur_view)
                is_keyframe = True

        elif self.kf_selection_method == "overlap":
            if self.last_kf_info is None:
                self.add_new_cam_to_kf(cur_view)
                is_keyframe = True
            else:
                cur_pose = cur_view.get_raw_pose().detach().cpu().numpy()
                cur_K = cur_view.K.detach().cpu().numpy()
                cur_img_shape = (cur_view.height, cur_view.width)
                cur_pts = cur_view.get_pts()
                cur_max_depth = cur_view.max_depth

                overlap_ratio = cal_covisiblity(
                    self.last_kf_info["pts"],
                    self.last_kf_info["pose"],
                    self.last_kf_info["K"],
                    self.last_kf_info["max_depth"],
                    cur_pts,
                    cur_pose,
                    cur_K,
                    cur_max_depth,
                    cur_img_shape,
                )

                if overlap_ratio < self.kf_overlap_ratio:
                    self.add_new_cam_to_kf(cur_view, overlap_ratio)
                    is_keyframe = True

        elif self.kf_selection_method == "overlap_mixed":
            if self.last_kf_info is None:
                self.add_new_cam_to_kf(cur_view)
                is_keyframe = True
            else:
                cur_pose = cur_view.get_raw_pose().detach().cpu().numpy()
                cur_K = cur_view.K.detach().cpu().numpy()
                cur_img_shape = (cur_view.height, cur_view.width)
                cur_pts = cur_view.get_pts()
                cur_max_depth = cur_view.max_depth

                overlap_ratio = cal_covisiblity(
                    self.last_kf_info["pts"],
                    self.last_kf_info["pose"],
                    self.last_kf_info["K"],
                    self.last_kf_info["max_depth"],
                    cur_pts,
                    cur_pose,
                    cur_K,
                    cur_max_depth,
                    cur_img_shape,
                )

                if (
                    overlap_ratio < self.kf_overlap_ratio
                    or cur_view.cam_idx - self.last_kf_info["cam_idx"]
                    > self.kf_interval
                ):
                    self.add_new_cam_to_kf(cur_view, overlap_ratio)
                    is_keyframe = True
        else:
            raise NotImplementedError

        if len(self.kf_cameras) == 1 and len(self.cur_random_window) == 0:
            self.init_global_window()

        return is_keyframe
