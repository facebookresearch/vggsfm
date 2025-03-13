# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Dict

from hydra.utils import instantiate

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from huggingface_hub import PyTorchModelHubMixin


class VGGSfM(nn.Module,
             PyTorchModelHubMixin,
             repo_url="https://github.com/facebookresearch/vggsfm",
             pipeline_tag="image-to-3d",
             license="cc-by-nc-sa-4.0",
             coders={
                DictConfig : (
                    lambda x: OmegaConf.to_container(x, resolve=True),  # Encoder: how to convert a `DictConfig` to a valid jsonable value?
                    lambda data: OmegaConf.create(data),  # Decoder: how to reconstruct a `DictConfig` from a dictionary?
                ),
            }
    ):
    def __init__(self, TRACK: DictConfig, CAMERA: DictConfig, TRIANGULAE: DictConfig, cfg: DictConfig = None):
        """
        Initializes a VGGSfM model

        TRACK, CAMERA, TRIANGULAE are the dicts to construct the model modules
        cfg is the whole hydra config
        """
        super().__init__()

        self.cfg = cfg

        # models.TrackerPredictor
        self.track_predictor = instantiate(TRACK, _recursive_=False, cfg=cfg)

        # models.CameraPredictor
        self.camera_predictor = instantiate(CAMERA, _recursive_=False, cfg=cfg)

        # models.Triangulator
        self.triangulator = instantiate(TRIANGULAE, _recursive_=False, cfg=cfg)
