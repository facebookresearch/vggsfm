# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from util.embedding import TimeStepEmbedding, PoseEmbedding, TimeStepEmbeddingWoL
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding
from einops import rearrange, repeat

import numpy as np
import torch
import torch.nn as nn
from util.get_fundamental_matrix import get_essential_matrix
from util.metric import batched_all_pairs, closed_form_inverse
from hydra.utils import instantiate
from util.camera_transform import pose_encoding_to_camera
import util.model_utils as model_utils
from util.relative_pose import intri_from_size
from kornia.geometry.epipolar import sampson_epipolar_distance
import torch.nn.functional as F
import math

from .trackerV3.blocks import AttnBlock, CrossAttnBlock, Mlp, ResidualBlock
from .trackerV3.model_utils import get_2d_sincos_pos_embed


logger = logging.getLogger(__name__)


class CameraPredictor(nn.Module):
    def __init__(
        self,
        # TRANSFORMER: Dict,
        # CROSS: Dict,
        hidden_size=256,
        num_heads=8,
        mlp_ratio=4,
        target_dim: int = 8,
        z_dim: int = 384,
        down_size=224,
        att_depth=4,
        trunk_depth=4,
        backbone="dinov2",
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg


        self.pose_encoding_type = cfg.MODEL.pose_encoding_type
        if self.pose_encoding_type == "absT_quaR_OneFL":
            target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":
            target_dim = 9

        self.down_size = down_size
        self.target_dim = target_dim

        if backbone == "dinov2s":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        elif backbone == "dinov2b":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        else:
            raise NotImplementedError("backbone not implemented")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.camera_conv = cfg.camera_conv

        # wait, maybe better to avoid using concatenate pose embed
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim, n_harmonic_functions=(hidden_size // target_dim) // 2, append_input=False
        )

        self.att_depth = att_depth
        self.input_transform = Mlp(in_features=z_dim, out_features=hidden_size, drop=0)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.self_att = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention, cfg=cfg)
                for _ in range(self.att_depth)
            ]
        )

        self.cross_att = nn.ModuleList(
            [
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, cfg=cfg)
                for _ in range(self.att_depth)
            ]
        )

        self.gamma = 0.8

        if self.cfg.camera_pool == "max":
            self.pool_fn = F.adaptive_max_pool2d
        elif self.cfg.camera_pool == "mean":
            self.pool_fn = F.adaptive_avg_pool2d
        else:
            import pdb

            pdb.set_trace()
            print("check your pooling method")

        if self.cfg.concat_pose:
            trunk_dim = hidden_size + self.target_dim
        else:
            trunk_dim = hidden_size

        if self.camera_conv:
            self.conv1 = self._make_conv_layer(hidden_size, hidden_size, stride=2)
            self.conv2 = self._make_conv_layer(hidden_size, hidden_size, stride=2)
            self.conv3 = self._make_conv_layer(hidden_size, hidden_size, stride=2)
            self.conv4 = self._make_conv_layer(hidden_size, hidden_size, stride=2)

            self.trunk = nn.Sequential(
                *[
                    AttnBlock(trunk_dim, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention, cfg=cfg)
                    for _ in range(trunk_depth)
                ]
            )
        else:
            self.pose_token = nn.Parameter(torch.zeros(1, 1, 1, trunk_dim))
            nn.init.normal_(self.pose_token, std=1e-6)

        if self.cfg.also_trunk:
            self.trunk = nn.Sequential(
                *[
                    AttnBlock(trunk_dim, num_heads, mlp_ratio=mlp_ratio, attn_class=nn.MultiheadAttention, cfg=cfg)
                    for _ in range(trunk_depth)
                ]
            )

        _RESNET_MEAN = [0.485, 0.456, 0.406]
        _RESNET_STD = [0.229, 0.224, 0.225]
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

        if self.camera_conv:
            self.pose_branch = Mlp(in_features=trunk_dim, out_features=target_dim, drop=0)
        else:
            self.pose_branch = Mlp(
                in_features=trunk_dim, hidden_features=trunk_dim * 2, out_features=trunk_dim + target_dim, drop=0
            )

            self.ffeat_updater = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())

    def _make_conv_layer(self, in_planes, out_planes, stride=1, norm_fn="instance"):
        layer1 = ResidualBlock(in_planes, out_planes, norm_fn, stride=stride)
        layer2 = ResidualBlock(out_planes, out_planes, norm_fn, stride=1)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def forward(
        self, rgbs, pose_pre=None, iters=4, batch_size=None, gt_cameras=None, crop_params=None, rgb_feat_init=None
    ):
        # TODO: implement iterations
        # pose_8p

        # resize rgbs to the target size, because we usually don't need very high dimension
        if rgb_feat_init is None:
            if rgbs.shape[-1] != self.down_size:
                rgbs = F.interpolate(rgbs, (self.down_size, self.down_size), mode="bilinear", align_corners=True)

            with torch.no_grad():
                # B x (P**2) x C
                # note we need to transform to imagenet mean and std
                rgbs = self._resnet_normalize_image(rgbs)
                rgb_feat = self.backbone(rgbs, is_training=True)

            rgb_feat = rgb_feat["x_norm_patchtokens"]

            rgb_feat = self.input_transform(rgb_feat)
            rgb_feat = self.norm(rgb_feat)

            rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=batch_size)

            B, S, P, C = rgb_feat.shape
            psize = int(math.sqrt(P))

            # add embedding of sequence (actually just one-shot to indicate the first frame. others are equivalent)
            # add embedding of 2D spaces
            pos_embed = get_2d_sincos_pos_embed(C, grid_size=(psize, psize)).permute(0, 2, 3, 1)[None]
            pos_embed = pos_embed.reshape(1, 1, psize * psize, C).to(rgb_feat.device)

            rgb_feat = rgb_feat + pos_embed

            if not self.camera_conv:
                pose_token = self.pose_token.expand(B, S, -1, -1)
                rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)

            B, S, P, C = rgb_feat.shape

            for idx in range(self.att_depth):
                if self.cfg.cam_selfpatch:
                    rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S)
                    rgb_feat = self.self_att[idx](rgb_feat)
                    rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=B, s=S)
                else:
                    rgb_feat = rearrange(rgb_feat, "b s p c -> b (s p) c", s=S, p=P)
                    rgb_feat = self.self_att[idx](rgb_feat)
                    rgb_feat = rearrange(rgb_feat, "b (s p) c -> b s p c", s=S, p=P)

                feat_0 = rgb_feat[:, 0]
                feat_others = rgb_feat[:, 1:]

                feat_others = rearrange(feat_others, "b m p c -> b (m p) c", m=S - 1, p=P)

                feat_others = self.cross_att[idx](feat_others, feat_0)

                feat_others = rearrange(feat_others, "b (m p) c -> b m p c", m=S - 1, p=P)
                rgb_feat = torch.cat([rgb_feat[:, 0:1], feat_others], dim=1)

            if self.camera_conv:
                rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S).permute(0, 2, 1)
                rgb_feat = rgb_feat.reshape(B * S, C, psize, psize)

                # Max pool or Avg pool?
                # Do we really need so many layers?
                rgb_feat = self.conv1(rgb_feat)
                rgb_feat = self.conv2(rgb_feat)
                rgb_feat = self.conv3(rgb_feat)
                rgb_feat = self.conv4(rgb_feat)

                rgb_feat = self.pool_fn(rgb_feat, (1, 1)).squeeze(-1).squeeze(-1)

                rgb_feat = rearrange(rgb_feat, "(b s) c -> b s c", b=B, s=S)
            else:
                rgb_feat = rgb_feat[:, :, 0]
        else:
            rgb_feat = rgb_feat_init
            B, S, C = rgb_feat.shape

        if pose_pre is not None:
            pred_pose_enc = (
                camera_to_pose_encoding(pose_pre, pose_encoding_type=self.pose_encoding_type)
                .reshape(B, S, -1)
                .to(rgb_feat.dtype)
            )
        else:
            # init pred_pose here
            pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat.device)

        if gt_cameras is not None:
            gt_pose_enc = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

        loss_pose = 0

        rgb_feat_init = rgb_feat.clone()

        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()
            pose_embed = self.embed_pose(pred_pose_enc)

            rgb_feat = rgb_feat + pose_embed

            # if only using embed instead of concat, can we force the model not to cheat?
            if self.cfg.concat_pose:
                rgb_feat = torch.cat([rgb_feat, pred_pose_enc], dim=-1)

            if self.camera_conv:
                rgb_feat = self.trunk(rgb_feat) + rgb_feat
                delta_pred_pose_enc = self.pose_branch(rgb_feat)
            elif self.cfg.also_trunk:
                rgb_feat = self.trunk(rgb_feat)  # + rgb_feat
                delta = self.pose_branch(rgb_feat)

                delta_pred_pose_enc = delta[..., : self.target_dim]
                delta_feat = delta[..., self.target_dim :]
                rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat
            else:
                delta = self.pose_branch(rgb_feat)

                delta_pred_pose_enc = delta[..., : self.target_dim]
                delta_feat = delta[..., self.target_dim :]
                rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat

            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc

            if gt_cameras is not None:
                loss_now = (pred_pose_enc.reshape(B * S, self.target_dim) - gt_pose_enc).abs()
                loss_weight = self.gamma ** (iters - 1 - iter_num)

                loss_pose = loss_pose + loss_now.mean() * loss_weight

            # feature residual
            if self.cfg.concat_pose:
                rgb_feat = rgb_feat[:, :, : -self.target_dim]

            rgb_feat = (rgb_feat + rgb_feat_init) / 2

        loss_pose = 10 * (loss_pose / iters)

        # output format
        pred_cameras = pose_encoding_to_camera(pred_pose_enc, pose_encoding_type=self.pose_encoding_type)
        pose_predictions = {
            "pred_pose_enc": pred_pose_enc,
            "pred_cameras": pred_cameras,
            "loss_pose": loss_pose,
            "rgb_feat_init": rgb_feat_init,
        }


        return pose_predictions