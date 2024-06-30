# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Tuple, Union

from hydra.utils import instantiate


class VGGSfM(nn.Module):
    def __init__(self, TRACK: Dict, CAMERA: Dict, TRIANGULAE: Dict, cfg=None):
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
