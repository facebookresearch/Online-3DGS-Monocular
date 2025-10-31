#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torchmetrics==1.4.1
python -m pip install gsplat
python -m pip install git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157
python -m pip install lpips==0.1.4 matplotlib==3.5.1 munch==4.0.0 nerfview==0.0.2 numpy==1.26.3 opencv-python==4.10.0.84 plyfile==1.1 scikit-image==0.24.0 scipy==1.13.1 six==1.16.0 tqdm==4.66.5 trimesh==4.4.9 viser==0.2.1 open3d

python -m pip install pip==23.1
python -m pip install pytorch-lightning==1.8.2
python -m pip install -e DepthCov-Modified --no-build-isolation

python -m pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git
