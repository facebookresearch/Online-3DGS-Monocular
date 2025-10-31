# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import rich

_log_styles = {
    "SceneMapper": "bold green",
    "SLAM": "bold magenta",
    "Novel-view-render": "bold magenta",
    "Eval": "bold red",
    "GaussianModel": "green1",
    "Debug": "bold cyan1",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="SceneMapper"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
