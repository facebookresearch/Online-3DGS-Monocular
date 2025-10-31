# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class FrameChecker:
    def __init__(self, configs):
        self.configs = configs

        self.use_frame_checker = configs["use_frame_checker"]
        self.rot_speed_threshold = configs["rot_speed_threshold"]
        self.pts_num_threshold = configs["pts_num_threshold"]

    def check(self, cam):
        if self.use_frame_checker:
            if (
                cam.rot_speed > self.rot_speed_threshold
                or len(cam.get_pts()) < self.pts_num_threshold
            ):
                return False

        return True
