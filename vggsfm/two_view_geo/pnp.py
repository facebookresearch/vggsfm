# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Literal, Optional, Tuple

import torch

from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers import solve_cubic
from kornia.utils._compat import torch_version_ge

import math
from torch.cuda.amp import autocast


import numpy as np
import kornia
from .perspective_n_points import efficient_pnp
from kornia.geometry.calibration.pnp import solve_pnp_dlt


from .utils import (
    generate_samples,
    sampson_epipolar_distance_batched,
    calculate_residual_indicator,
    normalize_points_masked,
    local_refinement,
    _torch_svd_cast,
)


def conduct_pnp(
    points3D,
    points2D,
    intrinsics,
    max_ransac_iters=1024,
    max_error=8,
    lo_num=50,
    f_trials=51,
):
    """
    Solve PnP problem by 6p algorithm + ePnP + LORANSAC
    points2D and intrinsics is defined in pixel
    """

    max_thres = max_error**2

    oriB = points3D.shape[0]
    if f_trials > 0:
        # Search for potential focal lengths
        # Refer to
        # https://github.com/colmap/colmap/blob/0ea2d5ceee1360bba427b2ef61f1351e59a46f91/src/colmap/estimators/pose.cc#L87
        # for more details

        B, P, _ = points3D.shape
        f_factors = generate_focal_factors(f_trials - 1)
        f_factors = torch.FloatTensor(f_factors).to(points2D.device)

        points3D = points3D[:, None].expand(-1, f_trials, -1, -1)
        points2D = points2D[:, None].expand(-1, f_trials, -1, -1)
        intrinsics = intrinsics[:, None].expand(-1, f_trials, -1, -1).clone()
        intrinsics[:, :, 0, 0] = intrinsics[:, :, 0, 0] * f_factors[None, :]
        intrinsics[:, :, 1, 1] = intrinsics[:, :, 1, 1] * f_factors[None, :]

        points3D = points3D.reshape(B * f_trials, P, 3)
        points2D = points2D.reshape(B * f_trials, P, 2)
        intrinsics = intrinsics.reshape(B * f_trials, 3, 3)
    else:
        f_trials = 1

    # update B
    B, P, _ = points3D.shape

    point_per_sample = 6  # 7p algorithm
    ransac_idx = generate_samples(P, max_ransac_iters, point_per_sample)

    points3D_ransac = points3D[:, ransac_idx].view(
        B * max_ransac_iters, point_per_sample, 3
    )
    points2D_ransac = points2D[:, ransac_idx].view(
        B * max_ransac_iters, point_per_sample, 2
    )
    intrinsics_ransac = (
        intrinsics[:, None]
        .expand(-1, max_ransac_iters, -1, -1)
        .reshape(B * max_ransac_iters, 3, 3)
    )
    pred_world_to_cam = solve_pnp_dlt(
        points3D_ransac, points2D_ransac, intrinsics_ransac
    )

    pred_world_to_cam_4x4 = kornia.eye_like(4, pred_world_to_cam)
    pred_world_to_cam_4x4[:, :3, :] = pred_world_to_cam

    points3D_expand = (
        points3D[:, None]
        .expand(-1, max_ransac_iters, -1, -1)
        .reshape(B * max_ransac_iters, P, 3)
    )
    points2D_expand = (
        points2D[:, None]
        .expand(-1, max_ransac_iters, -1, -1)
        .reshape(B * max_ransac_iters, P, 2)
    )
    cam_points = kornia.geometry.transform_points(
        pred_world_to_cam_4x4, points3D_expand
    )

    img_points = kornia.geometry.project_points(
        cam_points, intrinsics_ransac[:, None]
    )

    che_invalid = cam_points[..., -1] <= 0
    residuals = (img_points - points2D_expand).norm(dim=-1) ** 2
    residuals[che_invalid] = 1e6  # fails for che Chirality

    inlier_mask = residuals <= max_thres

    inlier_mask = inlier_mask.reshape(B, max_ransac_iters, P)
    inlier_num = inlier_mask.sum(dim=-1)

    sorted_values, sorted_indices = torch.sort(
        inlier_num, dim=1, descending=True
    )

    focal_length = intrinsics[:, [0, 1], [0, 1]]
    principal_point = intrinsics[:, [0, 1], [2, 2]]
    points2D_normalized = (points2D - principal_point[:, None]) / focal_length[
        :, None
    ]

    # LORANSAC refinement
    transform_lo = local_refinement(
        efficient_pnp,
        points3D,
        points2D_normalized,
        inlier_mask,
        sorted_indices,
        lo_num=lo_num,
        skip_resize=True,
    )

    pred_world_to_cam_4x4_lo = kornia.eye_like(4, transform_lo.R)
    pred_world_to_cam_4x4_lo[:, :3, :3] = transform_lo.R.permute(0, 2, 1)
    pred_world_to_cam_4x4_lo[:, :3, 3] = transform_lo.T

    all_pmat = pred_world_to_cam_4x4_lo.reshape(B, lo_num, 4, 4)

    all_pmat_num = all_pmat.shape[1]
    # all
    points3D_expand = (
        points3D[:, None]
        .expand(-1, all_pmat_num, -1, -1)
        .reshape(B * all_pmat_num, P, 3)
    )
    points2D_expand = (
        points2D[:, None]
        .expand(-1, all_pmat_num, -1, -1)
        .reshape(B * all_pmat_num, P, 2)
    )
    intrinsics_all = (
        intrinsics[:, None]
        .expand(-1, all_pmat_num, -1, -1)
        .reshape(B * all_pmat_num, 3, 3)
    )

    cam_points = kornia.geometry.transform_points(
        all_pmat.reshape(B * all_pmat_num, 4, 4), points3D_expand
    )
    img_points = kornia.geometry.project_points(
        cam_points, intrinsics_all[:, None]
    )

    residuals_all = (img_points - points2D_expand).norm(dim=-1) ** 2

    che_invalid_all = cam_points[..., -1] <= 0
    residuals_all[che_invalid_all] = 1e6  # fails for che Chirality

    residuals_all = residuals_all.reshape(B, all_pmat_num, P)
    residuals_all = residuals_all.reshape(
        oriB, f_trials, all_pmat_num, P
    ).reshape(oriB, f_trials * all_pmat_num, P)

    (residual_indicator, inlier_num_all, inlier_mask_all) = (
        calculate_residual_indicator(residuals_all, max_thres, debug=True)
    )

    # update B back to original B
    B = residual_indicator.shape[0]
    batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)

    best_p_indices = torch.argmax(residual_indicator, dim=1)

    all_pmat = all_pmat.reshape(B, f_trials, all_pmat_num, 4, 4).reshape(
        B, f_trials * all_pmat_num, 4, 4
    )
    all_intri = intrinsics_all.reshape(B, f_trials, all_pmat_num, 3, 3).reshape(
        B, f_trials * all_pmat_num, 3, 3
    )

    best_pmat = all_pmat[batch_index[:, 0], best_p_indices]
    best_intri = all_intri[batch_index[:, 0], best_p_indices]

    best_inlier_num = inlier_num_all[batch_index[:, 0], best_p_indices]
    best_inlier_mask = inlier_mask_all[batch_index[:, 0], best_p_indices]

    return best_pmat, best_intri, best_inlier_num, best_inlier_mask


def generate_focal_factors(
    num_focal_length_samples=10,
    max_focal_length_ratio=5,
    min_focal_length_ratio=0.2,
):
    focal_length_factors = []
    fstep = 1.0 / num_focal_length_samples
    fscale = max_focal_length_ratio - min_focal_length_ratio
    focal = 0.0
    for i in range(num_focal_length_samples):
        focal_length_factors.append(
            min_focal_length_ratio + fscale * focal * focal
        )
        focal += fstep
    focal_length_factors.append(1.0)
    return focal_length_factors
