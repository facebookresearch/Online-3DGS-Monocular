# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
from lietorch import SE3
from torch import nn


# code from gsplats
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """

    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


# class Camera_Optimizer:
#     def __init__(self, configs):

#         self.use_camera_opt = configs["use_camera_opt"]
#         if self.use_camera_opt:
#             self.lr = configs["lr"]
#             self.weight_decay = configs["weight_decay"]
#             self.device = configs["device"]
#             self.optimizer = None
#             # self.optimizer = torch.optim.Adam([], lr=configs["lr"], weight_decay=configs["weight_decay"])

#     def build_opt(self, cams):
#         if self.use_camera_opt:
#             params = []
#             for cam in cams:
#                 params.append(cam.embeds)
#             self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)


#     def step(self):
#         if self.use_camera_opt:
#             self.optimizer.step()
#             self.optimizer.zero_grad(set_to_none=True)


class Camera_Optimizer:
    def __init__(self, configs):
        self.use_camera_opt = configs["use_camera_opt"]
        if self.use_camera_opt:
            self.max_num = (
                configs["max_num"] if "max_num" in configs else 8192
            )  # this number is supposed to be the power of 2
            self.use_hierarchy = (
                configs["use_hierarchy"] if "use_hierarchy" in configs else False
            )
            self.lr = configs["lr"]
            self.weight_decay = configs["weight_decay"]
            self.device = configs["device"]
            self.embeds = nn.Embedding(self.max_num * 2, 9).to(self.device)
            torch.nn.init.zeros_(self.embeds.weight)
            self.identity = (
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float().to(self.device)
            )
            self.cur_idx = 0
            self.total_num = 0
            self.optimizer = torch.optim.Adam(
                [self.embeds.weight], lr=self.lr, weight_decay=self.weight_decay
            )
            # self.optimizer = torch.optim.Adam([], lr=configs["lr"], weight_decay=configs["weight_decay"])

    def get_embeds(self, idx):
        raw_embeds = self.embeds.weight[idx]
        return torch.cat([raw_embeds[:3], self.identity + raw_embeds[3:]])

    def get_delta_transform(self, idx):
        embeds = self.get_embeds(idx)
        dx = embeds[:3]
        drot = embeds[3:]
        rot = rotation_6d_to_matrix(drot)
        extra_transform = torch.eye(4, device=self.device)
        extra_transform[:3, :3] = rot
        extra_transform[:3, 3] = dx
        return extra_transform

    def get_max_depth(self):
        if self.use_hierarchy:
            return int(np.ceil(np.log(self.total_num) / np.log(2))) + 1
        else:
            return 1

    def add_cam(self, cam):
        if self.use_camera_opt:
            cam.set_embeds_trainable(
                self, self.cur_idx + self.max_num
            )  # by default, embeds is not trainable
            self.cur_idx += 1
            self.total_num += 1
            assert self.total_num <= self.max_num

    def step(self):
        if self.use_camera_opt:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)


class SE3_Camera_Optimizer:
    def __init__(self, configs):
        self.use_camera_opt = configs["use_camera_opt"]
        if self.use_camera_opt:
            self.max_num = (
                configs["max_num"] if "max_num" in configs else 8192
            )  # this number is supposed to be the power of 2
            self.use_hierarchy = (
                configs["use_hierarchy"] if "use_hierarchy" in configs else False
            )
            self.lr = configs["lr"]
            self.weight_decay = configs["weight_decay"]
            self.device = configs["device"]
            self.embeds = nn.Embedding(self.max_num * 2, 7).to(self.device)
            torch.nn.init.zeros_(self.embeds.weight)
            with torch.no_grad():
                self.embeds.weight[:, -1] = (
                    1.0  # initialize the last element to be 1.0 for unit quaternion
                )
            self.cur_idx = 0
            self.total_num = 0
            self.optimizer = torch.optim.Adam(
                [self.embeds.weight], lr=self.lr, weight_decay=self.weight_decay
            )

    def get_delta_transform(self, idx):
        embeds = self.embeds.weight[idx]

        dx = embeds[:3]
        dquat = embeds[3:]
        dquat = dquat / torch.norm(dquat)

        delta_transform = SE3(torch.cat([dx, dquat]))

        return delta_transform

    def get_max_depth(self):
        if self.use_hierarchy:
            return int(np.ceil(np.log(self.total_num) / np.log(2))) + 1
        else:
            return 1

    def add_cam(self, cam):
        if self.use_camera_opt:
            cam.set_embeds_trainable(
                self, self.cur_idx + self.max_num
            )  # by default, embeds is not trainable
            self.cur_idx += 1
            self.total_num += 1
            assert self.total_num <= self.max_num

    def step(self):
        if self.use_camera_opt:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)


class Camera(nn.Module):
    # Legacy code
    # def __init__(
    #     self,
    #     name,
    #     uid,
    #     color,
    #     color_for_pts,
    #     pose,
    #     pts,
    #     fx,
    #     fy,
    #     cx,
    #     cy,
    #     fovx,
    #     fovy,
    #     image_height,
    #     image_width,
    #     near,
    #     far,
    #     exposure_gain,
    #     rot_speed,
    #     depth
    # ):
    #     super(Camera, self).__init__()
    #     self.cam_idx = uid
    #     if name is not None:
    #         self.name = name
    #     elif type(uid) is int:
    #         self.name = "cam_{:05d}".format(uid)
    #     elif type(uid) is float:
    #         self.name = "cam_{:.3f}".format(uid)
    #     else:
    #         raise NotImplementedError("uid type not supported")

    #     self.raw_pose = pose
    #     self.pose = pose
    #     self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
    #     # self.identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float()
    #     # self.embeds = torch.zeros(9, dtype=torch.float32)
    #     self.camera_opt = None
    #     self.cam_opt_idx = -1
    #     self.opt_pose = False

    #     self.gt_img = color

    #     self.fx = fx
    #     self.fy = fy
    #     self.cx = cx
    #     self.cy = cy
    #     self.FoVx = fovx
    #     self.FoVy = fovy
    #     self.height = image_height
    #     self.width = image_width

    #     self.near = near
    #     self.far = far

    #     self.exposure_gain = exposure_gain

    #     self.depth = depth

    #     self.feature_mask = None
    #     self.sparse_depth = None
    #     self.view_scale_size = 0.01

    #     self.est_depth = None
    #     self.est_normal = None

    #     if color_for_pts is not None and pts is not None:
    #         self.pts_color, self.pts_2d, self.max_depth, self.median_depth, self.feature_mask, self.sparse_depth = self.pts_culling_color(pts, color_for_pts)

    #         # self.feature_mask = cv2.dilate(self.feature_mask, np.ones((5, 5), np.float32), iterations=2)
    #         # self.feature_mask = cv2.GaussianBlur(self.feature_mask, (15, 15), 0)
    #         self.feature_mask = self.feature_mask.reshape(self.feature_mask.shape[0], self.feature_mask.shape[1], 1)
    #         self.feature_mask = torch.from_numpy(self.feature_mask).float()

    #         self.sparse_depth = torch.from_numpy(self.sparse_depth).float()
    #         self.view_scale_size = 0.5 * self.median_depth / ((self.fx + self.fy) / 2)

    #     self.rot_speed = rot_speed
    #     self.pos3D = np.linalg.inv(self.raw_pose.numpy())[:3, 3]

    MAX_LEVEL = 4

    def __init__(
        self,
        name,
        uid,
        color,
        color_for_pts,
        pose,
        pts,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        near,
        far,
        exposure_gain,
        rot_speed,
        depth,
    ):
        super(Camera, self).__init__()
        self.cam_idx = uid
        if name is not None:
            self.name = name
        elif type(uid) is int:
            self.name = "cam_{:05d}".format(uid)
        elif type(uid) is float:
            self.name = "cam_{:.3f}".format(uid)
        else:
            raise NotImplementedError("uid type not supported")

        self.raw_pose = pose
        self.pose = pose
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
        # self.identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float()
        # self.embeds = torch.zeros(9, dtype=torch.float32)
        self.camera_opt = None
        self.cam_opt_idx = -1
        self.opt_pose = False

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.height = image_height
        self.width = image_width
        self.opt_steps = 0

        assert (
            self.height % (2 ** (self.MAX_LEVEL - 1)) == 0
        ), f"height {self.height} is not divisible by 2^{self.MAX_LEVEL-1}"
        assert (
            self.width % (2 ** (self.MAX_LEVEL - 1)) == 0
        ), f"width {self.width} is not divisible by 2^{self.MAX_LEVEL-1}"

        self.near = near
        self.far = far

        self.exposure_gain = exposure_gain

        self.depth = depth

        self.feature_mask = None
        self.sparse_depths = []

        if pts is not None:
            (
                self.max_depth,
                self.median_depth,
                sparse_depth,
                self.pts,
                self.color_pts_depth,
            ) = self.project_pts(pts, color_for_pts)

            sparse_depth = torch.from_numpy(sparse_depth).float()
            self.sparse_depths.append(sparse_depth)
            for _level in range(1, self.MAX_LEVEL):
                sparse_depth = Camera.downsample(sparse_depth, mode="depth")
                self.sparse_depths.append(sparse_depth)

        if color is None:
            self.gt_imgs = []
        else:
            self.gt_imgs = [color]
            for _level in range(1, self.MAX_LEVEL):
                color = Camera.downsample(color, mode="color")
                self.gt_imgs.append(color)

        self.rot_speed = rot_speed
        self.pos3D = np.linalg.inv(self.raw_pose.numpy())[:3, 3]

    def to_device(self, device):
        self.pose = self.pose.to(device)
        self.raw_pose = self.raw_pose.to(device)
        self.K = self.K.to(device)
        # if self.gt_img is not None:
        #     self.gt_img = self.gt_img.to(device)
        # if self.sparse_depth is not None:
        #     self.sparse_depth = self.sparse_depth.to(device)

        if len(self.gt_imgs) > 0:
            self.gt_imgs = [img.to(device) for img in self.gt_imgs]

        if len(self.sparse_depths) > 0:
            self.sparse_depths = [depth.to(device) for depth in self.sparse_depths]

    def update_pose_numpy(self, pose, exposure_gain=None):
        self.pose = torch.from_numpy(pose).float()
        self.pos3D = np.linalg.inv(self.pose.numpy())[:3, 3]
        if exposure_gain is not None:
            self.exposure_gain = exposure_gain

    def get_color_pts_depth(self):
        return self.color_pts_depth

    def get_int_mat(self, level=0):
        if level == 0:
            return self.K
        else:
            new_K = self.K.clone()
            width_ratio = self.get_width_ratio(level)
            height_ratio = self.get_height_ratio(level)
            new_K[0, 0] *= width_ratio
            new_K[0, 2] *= width_ratio
            new_K[1, 1] *= height_ratio
            new_K[1, 2] *= height_ratio
            return new_K

    def inc_opt_steps(self):
        self.opt_steps += 1

    def get_opt_steps(self):
        return self.opt_steps

    def get_pts(self):
        return self.pts

    def get_view_size(self, level=0):
        cur_fx = self.get_fx(level)
        cur_fy = self.get_fy(level)

        return 0.5 * self.median_depth / ((cur_fx + cur_fy) / 2)

    def get_fx(self, level=0):
        if level == 0:
            return self.fx
        else:
            return self.fx * self.get_width_ratio(level)

    def get_fy(self, level=0):
        if level == 0:
            return self.fy
        else:
            return self.fy * self.get_height_ratio(level)

    def get_cx(self, level=0):
        if level == 0:
            return self.cx
        else:
            return self.cx * self.get_width_ratio(level)

    def get_cy(self, level=0):
        if level == 0:
            return self.cy
        else:
            return self.cy * self.get_height_ratio(level)

    def get_height(self, level=0):
        if level == 0:
            return self.height
        else:
            return int(self.height // 2**level)

    def get_width(self, level=0):
        if level == 0:
            return self.width
        else:
            return int(self.width // 2**level)

    def get_width_ratio(self, level=0):
        if level == 0:
            return 1.0
        else:
            raw_width = self.get_width(0)
            cur_width = self.get_width(level)
            return cur_width / raw_width

    def get_height_ratio(self, level=0):
        if level == 0:
            return 1.0
        else:
            raw_height = self.get_height(0)
            cur_height = self.get_height(level)
            return cur_height / raw_height

    @staticmethod
    def downsample(img, mode="color"):
        imgs = []
        for dx in range(2):
            for dy in range(2):
                imgs.append(img[dy::2, dx::2])

        imgs = torch.stack(imgs, dim=0)
        if mode == "depth":  # 4 x N x M
            mask = (imgs > 0).float()
            res = torch.sum(imgs * mask, dim=0) / torch.sum(mask, dim=0)
        elif mode == "color":  # 4 x N x M x 3
            mask = (torch.sum(imgs, dim=3, keepdim=True) > 0).float()  # 4 x N x M x 1
            res = torch.sum(imgs * mask, dim=0) / torch.sum(mask, dim=0)
        else:
            raise NotImplementedError("mode {} not supported".format(mode))

        res[torch.isnan(res)] = 0.0

        return res

    def get_gt_image(self, level=0):
        return self.gt_imgs[level]

    def get_sparse_depth(self, level=0):
        return self.sparse_depths[level]

    def get_raw_pose(self):
        return self.raw_pose

    def set_embeds_trainable(self, opt, idx):
        # self.embeds = self.embeds.to(device)
        # self.identity = self.identity.to(device)
        # self.embeds.requires_grad = True
        self.camera_opt = opt
        self.cam_opt_idx = idx

    def set_opt_pose(self, flag):
        self.opt_pose = flag

    def get_pose(self):
        if self.camera_opt is None or (not self.opt_pose):
            return self.pose.detach()
        else:
            ##########################3 first version #########################
            # embeds = self.camera_opt.get_embeds(self.cam_opt_idx)
            # dx = embeds[:3]
            # drot = embeds[3:]
            # rot = rotation_6d_to_matrix(drot)
            # extra_transform = torch.eye(4, device=self.raw_pose.device)
            # extra_transform[:3, :3] = rot
            # extra_transform[:3, 3] = dx
            # final_pose = torch.matmul(extra_transform, self.raw_pose)

            ############################# Second version #############################
            # final_pose = self.raw_pose
            # depth_layers = self.camera_opt.get_max_depth()
            # cur_idx = self.cam_opt_idx
            # for i in range(depth_layers):
            #     delta_transform = self.camera_opt.get_delta_transform(cur_idx)
            #     final_pose = torch.matmul(delta_transform, final_pose)
            #     cur_idx = cur_idx // 2

            # self.pose = final_pose
            # return final_pose

            ########################### Third version (Lietorch) #############################
            delta_transform = None
            depth_layers = self.camera_opt.get_max_depth()
            cur_idx = self.cam_opt_idx
            for _ in range(depth_layers):
                if delta_transform is None:
                    delta_transform = self.camera_opt.get_delta_transform(cur_idx)
                else:
                    delta_transform = (
                        self.camera_opt.get_delta_transform(cur_idx) * delta_transform
                    )
                cur_idx = cur_idx // 2

            self.pose = torch.matmul(delta_transform.matrix(), self.raw_pose)
            return self.pose

    # def set_embeds_trainable(self):
    #     self.embeds.requires_grad = True

    # def set_embeds_not_trainable(self):
    #     self.embeds.requires_grad = False

    # def get_pose(self):

    #     embeds = self.embeds
    #     dx = embeds[:3]
    #     drot = embeds[3:]
    #     rot = rotation_6d_to_matrix(drot + self.identity)

    #     if self.pose.device != dx.device:
    #         extra_transform = torch.eye(4, device=self.pose.device)
    #         extra_transform[:3, :3] = rot.to(self.pose.device)
    #         extra_transform[:3, 3] = dx.to(self.pose.device)
    #     else:
    #         extra_transform = torch.eye(4, device=self.pose.device)
    #         extra_transform[:3, :3] = rot
    #         extra_transform[:3, 3] = dx

    #     return torch.matmul(extra_transform, self.pose)

    @staticmethod
    def init_from_dataset(
        uid,
        color,
        color_for_pts,
        pose,
        pts,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        near,
        far,
        exposure_gain,
        rot_speed,
        depth=None,
        name=None,
    ):
        return Camera(
            name,
            uid,
            color,
            color_for_pts,
            pose,
            pts,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            image_height,
            image_width,
            near,
            far,
            exposure_gain,
            rot_speed,
            depth,
        )

    @staticmethod
    def init_from_dict(cam_params):
        return Camera(
            None,
            cam_params["uid"],
            None,
            None,
            torch.from_numpy(np.array(cam_params["pose"])).float(),
            None,
            cam_params["fx"],
            cam_params["fy"],
            cam_params["cx"],
            cam_params["cy"],
            None,
            None,
            cam_params["height"],
            cam_params["width"],
            cam_params["near"],
            cam_params["far"],
            cam_params["exposure_gain"],
            None,
            None,
        )

    def pts_culling_color(self, pts, color):
        if pts is None or pts.shape[0] == 0:
            return (
                np.empty((0, 6)),
                np.empty((0, 2)),
                float("inf"),
                1.0,
                np.zeros_like(color[..., 0]),
                np.zeros_like(color[..., 0]),
            )

        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts = np.concatenate((pts, ones), axis=1)

        ext_mat = self.raw_pose.numpy()
        int_mat = self.K.numpy()
        img_shape = color.shape[:2]

        pts_proj = np.matmul(pts, ext_mat.T)[:, :3]
        valid_mask1 = pts_proj[:, 2] > 0
        pts_screen = np.matmul(pts_proj, int_mat.T)
        pts_screen = pts_screen[:, :2] / pts_screen[:, 2:3]
        feature_mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.float32)

        valid_mask = (
            (pts_screen[:, 0] >= 0)
            & (pts_screen[:, 0] <= img_shape[1] - 1)
            & (pts_screen[:, 1] >= 0)
            & (pts_screen[:, 1] <= img_shape[0] - 1)
            & valid_mask1
        )

        sparse_depth = np.zeros(img_shape[0] * img_shape[1], dtype=np.float32)
        if np.sum(valid_mask) == 0:
            max_depth = float("inf")
            median_depth = 1.0  # hard coded
        else:
            max_depth = np.max(pts_proj[valid_mask, 2])
            median_depth = np.median(pts_proj[valid_mask, 2])

        valid_pts = pts[valid_mask]
        valid_pts_screen = pts_screen[valid_mask]
        flatten_valid_pts_screen = np.round(valid_pts_screen).astype(np.int32)
        flatten_valid_pts_screen = (
            flatten_valid_pts_screen[:, 1] * img_shape[1]
            + flatten_valid_pts_screen[:, 0]
        )

        flatten_color = color.reshape(-1, 3)[flatten_valid_pts_screen]
        valid_pts = valid_pts[:, :3]

        valid_color = np.sum(flatten_color, axis=1) > 0
        valid_pts = valid_pts[valid_color]
        flatten_color = flatten_color[valid_color]
        valid_pts_screen = valid_pts_screen[valid_color]
        flatten_valid_pts_screen = flatten_valid_pts_screen[valid_color]

        feature_mask[flatten_valid_pts_screen] = 1
        feature_mask = feature_mask.reshape(img_shape[0], img_shape[1])

        sparse_depth[flatten_valid_pts_screen] = pts_proj[valid_mask, 2][valid_color]
        sparse_depth = sparse_depth.reshape(img_shape[0], img_shape[1])

        raise NotImplementedError("Not implemented yet")
        # valid pts should be (N*4) here, it is a bug should be fixed

        return (
            np.concatenate([valid_pts, flatten_color], axis=1),
            valid_pts_screen,
            max_depth,
            median_depth,
            feature_mask,
            sparse_depth,
        )

    def project_pts(self, pts, color):
        if pts is None or pts.shape[0] == 0:
            return (
                float("inf"),
                1.0,
                np.zeros((self.height, self.width), dtype=np.float32),
                np.empty((0, 3)),
                np.empty((0, 7)),
            )

        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts = np.concatenate((pts, ones), axis=1)

        ext_mat = self.raw_pose.numpy()
        int_mat = self.K.numpy()
        img_shape = (self.height, self.width)

        pts_proj = np.matmul(pts, ext_mat.T)[:, :3]
        valid_mask1 = pts_proj[:, 2] > 0  # in front of camera
        pts_screen = np.matmul(pts_proj, int_mat.T)
        pts_screen = pts_screen[:, :2] / pts_screen[:, 2:3]
        # feature_mask = np.zeros(img_shape[0] * img_shape[1], dtype=np.float32)

        valid_mask = (
            (pts_screen[:, 0] >= 0)
            & (pts_screen[:, 0] <= img_shape[1] - 1)
            & (pts_screen[:, 1] >= 0)
            & (pts_screen[:, 1] <= img_shape[0] - 1)
            & valid_mask1
        )

        if np.sum(valid_mask) == 0:
            max_depth = float("inf")
            median_depth = 1.0  # hard coded
        else:
            max_depth = np.max(pts_proj[valid_mask, 2])
            median_depth = np.median(pts_proj[valid_mask, 2])

        valid_pts = pts[valid_mask]
        valid_pts_screen = pts_screen[valid_mask]
        valid_pts_screen = (valid_pts_screen).astype(np.int32)
        flatten_valid_pts_screen = (
            valid_pts_screen[:, 1] * img_shape[1] + valid_pts_screen[:, 0]
        )

        flatten_color = color.reshape(-1, 3)[flatten_valid_pts_screen]

        sparse_depth = np.zeros(img_shape[0] * img_shape[1], dtype=np.float32)
        sparse_depth[flatten_valid_pts_screen] = pts_proj[valid_mask, 2]
        sparse_depth = sparse_depth.reshape(img_shape[0], img_shape[1])

        return (
            max_depth,
            median_depth,
            sparse_depth,
            pts[valid_mask, :3],
            np.concatenate(
                [valid_pts[:, :3], flatten_color, pts_proj[valid_mask, 2:3]], axis=1
            ),
        )

    def print(self):
        print("Camera UID: {}".format(self.cam_idx))
        print("Camera pose: {}".format(self.pose))
        print("Camera K: {}".format(self.K))
        print("Cammera image size: {}x{}".format(self.image_width, self.image_height))


def in_frustum_region(pts, ext_mat, int_mat, img_shape, max_depth=float("inf")):
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts = np.concatenate((pts, ones), axis=1)

    pts_proj = np.matmul(pts, ext_mat.T)[:, :3]
    valid_mask1 = np.logical_and(pts_proj[:, 2] > 0, pts_proj[:, 2] < max_depth)
    pts_screen = np.matmul(pts_proj, int_mat.T)
    pts_screen = pts_screen / pts_screen[:, 2:3]

    valid_mask = (
        (pts_screen[:, 0] >= 0)
        & (pts_screen[:, 0] <= img_shape[1] - 1)
        & (pts_screen[:, 1] >= 0)
        & (pts_screen[:, 1] <= img_shape[0] - 1)
        & valid_mask1
    )

    return valid_mask


def unproject_pts(pts_2d, depths, int_mat, ext_mat):
    local_coords = np.concatenate([pts_2d, np.ones_like(pts_2d[:, :1])], axis=1)
    local_coords = np.matmul(local_coords, np.linalg.inv(int_mat.T)) * depths[:, None]
    world_coords = np.concatenate(
        [local_coords, np.ones_like(local_coords[:, :1])], axis=1
    )
    world_coords = np.matmul(world_coords, np.linalg.inv(ext_mat.T))

    return world_coords[:, :3]


def unproject_pts_tensor(pts_2d, depths, int_mat, ext_mat):
    local_coords = torch.cat(
        [pts_2d, torch.ones_like(pts_2d[:, :1], device=pts_2d.device)], dim=1
    )
    local_coords = torch.mm(local_coords, torch.linalg.inv(int_mat.T)) * depths[:, None]
    world_coords = torch.cat(
        [
            local_coords,
            torch.ones_like(local_coords[:, :1], device=local_coords.device),
        ],
        dim=1,
    )
    world_coords = torch.mm(world_coords, torch.linalg.inv(ext_mat.T))

    return world_coords[:, :3]
