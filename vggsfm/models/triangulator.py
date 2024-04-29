# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

import logging
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from einops import rearrange, repeat

import kornia
import pycolmap
from torch.cuda.amp import autocast

from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


# #####################
from ..two_view_geo.utils import inlier_by_fundamental
from ..utils.triangulation import (
    create_intri_matrix,
    triangulate_by_pair,
    init_BA,
    refine_pose,
    triangulate_tracks,
    global_BA,
    iterative_global_BA,
)

from ..utils.utils import transform_camera_relative_to_first


class Triangulator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        """
        The module for triangulation and BA adjustment 
        
        NOTE In VGGSfM v1.1, we remove the learnable parameters of Triangulator
        """
        self.cfg = cfg

    def forward(
        self,
        pred_cameras,
        pred_tracks,
        pred_vis,
        images,
        preliminary_dict,
        pred_score=None,
        fmat_thres=0.5,
        BA_iters=3,
        max_reproj_error=4,
        init_tri_angle_thres=16,
        min_valid_track_length=3,
        return_in_pt3d=True,
    ):
        """
        Conduct triangulation and bundle adjustment given
        the input pred_cameras, pred_tracks, pred_vis, and pred_score

        We use the pred_cameras from camera_predictor but it can be any init.
        Please note pred_tracks are defined in pixels.
        """

        device = pred_tracks.device

        with autocast(dtype=torch.float32):
            B, S, _, H, W = images.shape
            _, _, N, _ = pred_tracks.shape

            assert B == 1  # The released implementation now only supports batch=1 during inference

            image_size = torch.tensor([W, H], dtype=pred_tracks.dtype, device=device)

            # OPTIONAL: make cameras relative to the first camera
            pred_cameras = transform_camera_relative_to_first(pred_cameras, B * S)

            # Convert the PyTorch3D cameras predicted by camera_predictor
            # to the convention of OpenCV
            # TODO: do not use PyTorch3D in VGGSfM v2

            # extrinsics: B x S x 3 x 4
            # intrinsics: B x S x 3 x 3
            # focal_length, principal_point : B x S x 2
            extrinsics, intrinsics, focal_length, principal_point = pt3d_camera_to_opencv_EFP(
                pred_cameras, image_size, B, S
            )

            # Normalize points by intrinsics
            tracks_normalized = (pred_tracks - principal_point.unsqueeze(-2)) / focal_length.unsqueeze(-2)

            # Get the fmat and the inliers
            fmat_preliminary = preliminary_dict["fmat"]
            inlier_fmat = inlier_by_fundamental(fmat_preliminary, pred_tracks, max_error=fmat_thres)

            # Visibility inlier
            inlier_vis = pred_vis > 0.05  # TODO: avoid hardcoded
            inlier_vis = inlier_vis[:, 1:]

            # Intersection of inlier_fmat and inlier_vis
            inlier = torch.logical_and(inlier_fmat, inlier_vis)

            # For initialization
            # we first triangulate a point cloud for each pair of query-reference images,
            # i.e., we have S-1 3D point clouds
            # points_3d_pair: B*(S-1) x N x 3
            points_3d_pair, cheirality_mask_pair, triangle_value_pair = triangulate_by_pair(
                extrinsics, tracks_normalized
            )

            # Check which point cloud can provide sufficient inliers
            # that pass the triangulation angle and cheirality check
            # Pick the highest inlier one as the initial point cloud
            trial_count = 0
            while trial_count < 5:
                # If no success, relax the constraint
                # try at most 5 times
                triangle_mask = triangle_value_pair >= init_tri_angle_thres
                inlier_total = torch.logical_and(inlier, cheirality_mask_pair)
                inlier_total = torch.logical_and(inlier_total, triangle_mask)
                inlier_num_per_frame = inlier_total.sum(dim=-1)

                max_num_inlier = inlier_num_per_frame.max()
                max_num_inlier_ratio = max_num_inlier / N

                # We accept a pair only when the numer of inliers and the ratio
                # is higher than a thres
                if (max_num_inlier >= 100) and (max_num_inlier_ratio >= 0.25):
                    break

                if init_tri_angle_thres < 2:
                    break

                init_tri_angle_thres = init_tri_angle_thres // 2
                trial_count += 1

            # Remove B dim
            # To simplify the code, now we only support B==1 during inference
            extrinsics = extrinsics[0]
            intrinsics = intrinsics[0]
            pred_tracks = pred_tracks[0]
            inlier = inlier[0]
            inlier_total = inlier_total[0]
            pred_vis = pred_vis[0]
            pred_score = pred_score[0]

            # Conduct BA on the init point cloud and init pair
            points3D_init, extrinsics, intrinsics, track_init_mask, reconstruction, init_idx = init_BA(
                extrinsics, intrinsics, pred_tracks, points_3d_pair, inlier_total, image_size
            )

            # Given we have a well-conditioned point cloud,
            # we can optimize all the cameras by absolute pose refinement as in
            # https://github.com/colmap/colmap/blob/4ced4a5bc72fca93a2ffaea2c7e193bc62537416/src/colmap/estimators/pose.cc#L207
            # Basically it is a bundle adjustment without optmizing 3D points
            # It is fine even this step fails
            extrinsics, intrinsics, valid_intri_mask = refine_pose(
                extrinsics, intrinsics, inlier, points3D_init, pred_tracks, track_init_mask, image_size, init_idx
            )

            # Well if an frame has an invalid intri after optimization
            # e.g., minus focal length
            # we assume its correspondences are wrong, and ignore all of them in the following BA
            # this usually happens when the frames are highly symmetric or "doppelgangers”
            pred_vis[~valid_intri_mask] = 0

            # TODO: well we may have some bugs here, do we?
            # TODO: keep the optimized 3D points

            principal_point_refined = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
            focal_length_refined = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
            tracks_normalized_refined = (pred_tracks - principal_point_refined) / focal_length_refined

            # Conduct triangulation to all the frames
            # We adopt LORANSAC here again
            best_triangulated_points, best_inlier_num, best_inlier_mask = triangulate_tracks(
                extrinsics, tracks_normalized_refined, track_vis=pred_vis, track_score=pred_score, max_ransac_iters=128
            )

            valid_tracks = best_inlier_num >= min_valid_track_length

            points3D, extrinsics, intrinsics = global_BA(
                best_triangulated_points,
                valid_tracks,
                pred_tracks,
                best_inlier_mask,
                extrinsics,
                intrinsics,
                image_size,
                device,
            )

            try:
                ba_options = pycolmap.BundleAdjustmentOptions()
                ba_options.print_summary = False

                for BA_iter in range(BA_iters):
                    if BA_iter == (BA_iters - 1):
                        ba_options.print_summary = True

                    points3D, extrinsics, intrinsics, valid_tracks, reconstruction = iterative_global_BA(
                        pred_tracks,
                        intrinsics,
                        extrinsics,
                        pred_vis,
                        pred_score,
                        valid_tracks,
                        points3D,
                        image_size,
                        min_valid_track_length=min_valid_track_length,
                        max_reproj_error=max_reproj_error,
                        ba_options=ba_options,
                    )
                    max_reproj_error = max_reproj_error // 2
            except:
                print("oops BA fails")

            # From OpenCV/COLMAP to PyTorch3D
            rot_BA = extrinsics[:, :3, :3]
            trans_BA = extrinsics[:, :3, 3]

            if return_in_pt3d:
                rot_BA = rot_BA.clone().permute(0, 2, 1)
                trans_BA = trans_BA.clone()
                trans_BA[:, :2] *= -1
                rot_BA[:, :, :2] *= -1

            BA_cameras = PerspectiveCameras(R=rot_BA, T=trans_BA, device=device)

            return BA_cameras, extrinsics, intrinsics, points3D, reconstruction


def pt3d_camera_to_opencv_EFP(pred_cameras, image_size, B, S):
    """
    Converting PyTorch3D cameras to extrinsics, intrinsics matrix

    Return extrinsics, intrinsics, focal_length, principal_point
    """
    scale = image_size.min()

    focal_length = pred_cameras.focal_length

    principal_point = torch.zeros_like(focal_length)

    focal_length = focal_length * scale / 2
    principal_point = (image_size[None] - principal_point * scale) / 2

    Rots = pred_cameras.R.clone()
    Trans = pred_cameras.T.clone()

    Rots[:, :, :2] *= -1
    Trans[:, :2] *= -1
    Rots = Rots.permute(0, 2, 1)

    extrinsics = torch.cat([Rots, Trans[..., None]], dim=-1)

    # reshape
    extrinsics = extrinsics.reshape(B, S, 3, 4)
    focal_length = focal_length.reshape(B, S, 2)
    principal_point = principal_point.reshape(B, S, 2)

    # only one dof focal length
    focal_length = focal_length.mean(dim=-1, keepdim=True).expand(-1, -1, 2)
    focal_length = focal_length.clamp(0.2 * scale, 5 * scale)

    intrinsics = create_intri_matrix(focal_length, principal_point)
    return extrinsics, intrinsics, focal_length, principal_point
