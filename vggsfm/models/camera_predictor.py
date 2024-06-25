# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


from .modules import AttnBlock, CrossAttnBlock, Mlp, ResidualBlock

from .utils import get_2d_sincos_pos_embed, PoseEmbedding, pose_encoding_to_camera, camera_to_pose_encoding


logger = logging.getLogger(__name__)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class CameraPredictor(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_heads=8,
        mlp_ratio=4,
        z_dim: int = 768,
        down_size=336,
        att_depth=8,
        trunk_depth=4,
        backbone="dinov2b",
        pose_encoding_type="absT_quaR_OneFL",
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type

        if self.pose_encoding_type == "absT_quaR_OneFL":
            self.target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":
            self.target_dim = 9

        self.backbone = self.get_backbone(backbone)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.input_transform = Mlp(in_features=z_dim, out_features=hidden_size, drop=0)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim, n_harmonic_functions=(hidden_size // self.target_dim) // 2, append_input=False
        )

        self.pose_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))  # register

        self.pose_branch = Mlp(
            in_features=hidden_size, hidden_features=hidden_size * 2, out_features=hidden_size + self.target_dim, drop=0
        )

        self.ffeat_updater = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())

        self.self_att = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(self.att_depth)
            ]
        )

        self.cross_att = nn.ModuleList(
            [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(self.att_depth)]
        )

        self.trunk = nn.Sequential(
            *[
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention)
                for _ in range(trunk_depth)
            ]
        )

        self.gamma = 0.8

        nn.init.normal_(self.pose_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def forward(self, reshaped_image, preliminary_cameras=None, iters=4, batch_size=None, rgb_feat_init=None):
        """
        reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: cameras in opencv coordinate.
        """

        if rgb_feat_init is None:
            # Get the 2D image features
            rgb_feat, B, S, C = self.get_2D_image_features(reshaped_image, batch_size)
        else:
            rgb_feat = rgb_feat_init
            B, S, C = rgb_feat.shape

        if preliminary_cameras is not None:
            # Init the pred_pose_enc by preliminary_cameras
            pred_pose_enc = (
                camera_to_pose_encoding(preliminary_cameras, pose_encoding_type=self.pose_encoding_type)
                .reshape(B, S, -1)
                .to(rgb_feat.dtype)
            )
        else:
            # Or you can use random init for the poses
            pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat.device)

        rgb_feat_init = rgb_feat.clone()

        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()

            # Embed the camera parameters and add to rgb_feat
            pose_embed = self.embed_pose(pred_pose_enc)
            rgb_feat = rgb_feat + pose_embed

            # Run trunk transformers on rgb_feat
            rgb_feat = self.trunk(rgb_feat)

            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(rgb_feat)
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]

            rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat

            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc

            # Residual connection
            rgb_feat = (rgb_feat + rgb_feat_init) / 2

        # Pose encoding to Cameras
        pred_cameras = pose_encoding_to_camera(pred_pose_enc, pose_encoding_type=self.pose_encoding_type, to_OpenCV=True)
        pose_predictions = {
            "pred_pose_enc": pred_pose_enc,
            "pred_cameras": pred_cameras,
            "rgb_feat_init": rgb_feat_init,
        }

        return pose_predictions

    def get_backbone(self, backbone):
        """
        Load the backbone model.
        """
        if backbone == "dinov2s":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        elif backbone == "dinov2b":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def get_2D_image_features(self, reshaped_image, batch_size):
        # Get the 2D image features
        if reshaped_image.shape[-1] != self.down_size:
            reshaped_image = F.interpolate(
                reshaped_image, (self.down_size, self.down_size), mode="bilinear", align_corners=True
            )

        with torch.no_grad():
            reshaped_image = self._resnet_normalize_image(reshaped_image)
            rgb_feat = self.backbone(reshaped_image, is_training=True)
            # B x P x C
            rgb_feat = rgb_feat["x_norm_patchtokens"]

        rgb_feat = self.input_transform(rgb_feat)
        rgb_feat = self.norm(rgb_feat)

        rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=batch_size)

        B, S, P, C = rgb_feat.shape
        patch_num = int(math.sqrt(P))

        # add embedding of 2D spaces
        pos_embed = get_2d_sincos_pos_embed(C, grid_size=(patch_num, patch_num)).permute(0, 2, 3, 1)[None]
        pos_embed = pos_embed.reshape(1, 1, patch_num * patch_num, C).to(rgb_feat.device)

        rgb_feat = rgb_feat + pos_embed

        # register for pose
        pose_token = self.pose_token.expand(B, S, -1, -1)
        rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)

        B, S, P, C = rgb_feat.shape

        for idx in range(self.att_depth):
            # self attention
            rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S)
            rgb_feat = self.self_att[idx](rgb_feat)
            rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=B, s=S)

            feat_0 = rgb_feat[:, 0]
            feat_others = rgb_feat[:, 1:]

            # cross attention
            feat_others = rearrange(feat_others, "b m p c -> b (m p) c", m=S - 1, p=P)
            feat_others = self.cross_att[idx](feat_others, feat_0)

            feat_others = rearrange(feat_others, "b (m p) c -> b m p c", m=S - 1, p=P)
            rgb_feat = torch.cat([rgb_feat[:, 0:1], feat_others], dim=1)

        rgb_feat = rgb_feat[:, :, 0]

        return rgb_feat, B, S, C
