# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from os.path import join as pjoin

import torch
import torch.multiprocessing as mp
import yaml

from utils_new.eval_utils import eval_gaussians
from utils_new.logging_utils import Log

from utils_new.scene_mapper import SceneMapper
from utils_new.tool_utils import load_config, mkdir_p


class SLAM:
    def __init__(self, configs):
        self.scene_mapper = SceneMapper(configs)
        self.configs = configs

    def run(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        Log("Start reconstruction", tag="SLAM")
        scene_mapper_meta, optimization_infos = self.scene_mapper.run()
        Log("End reconstruction", tag="SLAM")

        end.record()

        torch.cuda.synchronize()
        online_recon_time = start.elapsed_time(end) / 1000.0  # in seconds
        Log("Total reconstruction time: ", online_recon_time, "s", tag="SLAM")
        Log("Total Frames: ", scene_mapper_meta["num_processed_frames"], tag="SLAM")
        Log("Total Gaussians: ", scene_mapper_meta["num_gaussians"], tag="SLAM")
        Log("Total keyframes: ", scene_mapper_meta["num_keyframes"], tag="SLAM")

        info = {}
        info["online_recon_time"] = online_recon_time
        info["num_processed_frames"] = scene_mapper_meta["num_processed_frames"]
        info["num_gaussians"] = scene_mapper_meta["num_gaussians"]
        info["num_keyframes"] = scene_mapper_meta["num_keyframes"]
        info["kf_ids"] = scene_mapper_meta["kf_ids"]

        torch.cuda.empty_cache()

        # Evaluate the reconstructed gaussians
        start.record()
        eval_res = eval_gaussians(
            self.scene_mapper.gaussians, optimization_infos, self.configs
        )
        end.record()
        torch.cuda.synchronize()
        eval_time = start.elapsed_time(end) / 1000.0  # in seconds

        info["eval_time"] = eval_time
        info["eval_res"] = eval_res

        json.dump(
            info,
            open(
                pjoin(self.configs["Results"]["save_dir"], "results.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )


def set_default_values(configs):
    if "scene_exposure_gain" not in configs["Mapper"]:
        configs["Mapper"]["scene_exposure_gain"] = 20.0


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default="")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = (
        config["Dataset"]["name"]
        if config["Mapper"]["use_dataset"]
        else config["Streamer"]["name"]
    )
    save_dir = os.path.join(
        config["Results"]["save_dir"],
        dataset_name,
        current_datetime + "_" + args.exp_name,
    )
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        documents = yaml.dump(config, file)

    Log("saving results in " + save_dir, tag="SLAM")

    set_default_values(config)

    slam = SLAM(config)

    slam.run()

    # All done
    Log("Done.", tag="SLAM")
