# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Adapted from https://github.com/kornia

from typing import Literal, Optional, Tuple
import numpy as np
import torch
import cv2
import math

# Importing Kornia core functionalities
from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE, KORNIA_CHECK_IS_TENSOR

# Importing Kornia geometry functionalities
from kornia.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers import solve_cubic
from kornia.geometry.epipolar.fundamental import (
    normalize_points,
    normalize_transformation,
)

# Importing PyTorch functionalities
from torch.cuda.amp import autocast

# Importing Kornia utils
from kornia.utils._compat import torch_version_ge


def generate_samples(N, target_num, sample_num, expand_ratio=2):
    """
    This function generates random samples of indices without duplicates.

    Parameters:
    N (int): The upper limit for generating random integers.
    max_num_trials (int): The maximum number of trials for generating samples.
    sample_num (int): The number of samples to generate.

    Returns:
    np.array: An array of indices without duplicates.
    """
    sample_idx = np.random.randint(
        0, N, size=(target_num * expand_ratio, sample_num)
    )
    sorted_array = np.sort(sample_idx, axis=1)
    diffs = np.diff(sorted_array, axis=1)
    has_duplicates = (diffs == 0).any(axis=1)
    indices_wo_duplicates = np.where(~has_duplicates)[0]
    sample_idx_safe = sample_idx[indices_wo_duplicates][:target_num]

    return sample_idx_safe


def calculate_residual_indicator(
    residuals, max_residual, debug=False, check=False, nanvalue=1e6
):
    with autocast(dtype=torch.double):
        B, S, N = residuals.shape
        inlier_mask = residuals <= max_residual

        inlier_num = inlier_mask.sum(dim=-1)

        # only consider the residuals of inliers, BxSxN
        residual_indicator = inlier_mask.float() * residuals
        # the average residual for inliers
        residual_indicator = residual_indicator.sum(dim=-1) / inlier_num
        # remove zero dividing
        residual_indicator = torch.nan_to_num(
            residual_indicator, nan=nanvalue, posinf=nanvalue, neginf=nanvalue
        )
        # we want the min average one, but don't want it to change the choice of inlier num
        thres = residual_indicator.max() + 1e-6

        residual_indicator = (thres - residual_indicator) / thres
        # choose the one with the higher inlier number and smallest (valid) residual
        residual_indicator = residual_indicator.double() + inlier_num.double()

        return residual_indicator, inlier_num, inlier_mask


def sampson_epipolar_distance_batched(
    pts1: Tensor,
    pts2: Tensor,
    Fm: Tensor,
    squared: bool = True,
    eps: float = 1e-8,
    debug=False,
    evaluation=False,
) -> Tensor:
    """Return Sampson distance for correspondences given the fundamental matrices.

    Args:
        pts1: correspondences from the left images with shape :math:`(B, N, (2|3))`.
        pts2: correspondences from the right images with shape :math:`(B, N, (2|3))`.
        Fm: Batch of fundamental matrices with shape :math:`(B, K, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(B, K, N)`.
    """
    # TODO: check why this would take a high GPU memory

    if not isinstance(Fm, Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if Fm.shape[-2:] != (3, 3):
        raise ValueError(f"Fm must be a (B, K, 3, 3) tensor. Got {Fm.shape}")

    dtype = pts1.dtype
    efficient_dtype = torch.float32

    with autocast(dtype=efficient_dtype):
        if pts1.shape[-1] == 2:
            pts1 = convert_points_to_homogeneous(pts1)

        if pts2.shape[-1] == 2:
            pts2 = convert_points_to_homogeneous(pts2)

        # Expand pts1 and pts2 to match Fm's batch and K dimensions for broadcasting
        B, K, _, _ = Fm.shape
        N = pts1.shape[1]

        pts1_expanded = pts1[:, None, :, :].expand(
            -1, K, -1, -1
        )  # Shape: (B, K, N, 3)
        pts2_expanded = pts2[:, None, :, :].expand(
            -1, K, -1, -1
        )  # Shape: (B, K, N, 3)

        Fm = Fm.to(efficient_dtype)
        F_t = Fm.transpose(-2, -1)  # Shape: (B, K, 3, 3)

        # pts1_expanded @ F_t
        line1_in_2 = torch.einsum(
            "bkij,bkjn->bkin", pts1_expanded, F_t
        )  # Shape: (B, K, N, 3)
        if evaluation:
            torch.cuda.empty_cache()
        line2_in_1 = torch.einsum(
            "bkij,bkjn->bkin", pts2_expanded, Fm
        )  # Shape: (B, K, N, 3)
        if evaluation:
            torch.cuda.empty_cache()

        numerator = (
            (pts2_expanded * line1_in_2).sum(dim=-1).pow(2)
        )  # Shape: (B, K, N)
        denominator = line1_in_2[..., :2].norm(2, dim=-1).pow(2) + line2_in_1[
            ..., :2
        ].norm(2, dim=-1).pow(
            2
        )  # Shape: (B, K, N)

        out = numerator / denominator

    out = out.to(dtype)
    if debug:
        return numerator, denominator, out, line1_in_2, line2_in_1

    if squared:
        return out
    return (out + eps).sqrt()


def normalize_points_masked(
    points: Tensor, masks: Tensor, eps: float = 1e-8, colmap_style=False
) -> Tuple[Tensor, Tensor]:
    """
    Normalizes points using a boolean mask to exclude certain points.

    Args:
        points: Tensor containing the points to be normalized with shape (B, N, 2).
        masks: Bool tensor indicating which points to include with shape (B, N).
        eps: epsilon value to avoid numerical instabilities.

    Returns:
        Tuple containing the normalized points in the shape (B, N, 2) and the transformation matrix in the shape (B, 3, 3).
    """
    if len(points.shape) != 3 or points.shape[-1] != 2:
        raise ValueError(
            f"Expected points with shape (B, N, 2), got {points.shape}"
        )

    if masks is None:
        masks = ones_like(points[..., 0])

    if masks.shape != points.shape[:-1]:
        raise ValueError(
            f"Expected masks with shape {points.shape[:-1]}, got {masks.shape}"
        )

    # Convert masks to float and apply it
    mask_f = masks.float().unsqueeze(-1)  # BxNx1
    masked_points = points * mask_f

    # Compute mean only over masked (non-zero) points
    num_valid_points = mask_f.sum(dim=1, keepdim=True)  # Bx1x1
    x_mean = masked_points.sum(dim=1, keepdim=True) / (
        num_valid_points + eps
    )  # Bx1x2

    diffs = (
        masked_points - x_mean
    )  # BxNx2, Apply mask before subtraction to zero-out invalid points

    if colmap_style:
        sum_squared_diffs = (diffs**2).sum(dim=-1).sum(dim=-1)  # Shape: (B, N)
        mean_squared_diffs = sum_squared_diffs / (
            num_valid_points[:, 0, 0] + eps
        )  # Shape: (B,)
        rms_mean_dist = torch.sqrt(mean_squared_diffs)  # Shape: (B,)
        rms_mean_dist = torch.clamp(rms_mean_dist, min=eps)
        scale = torch.sqrt(torch.tensor(2.0)) / rms_mean_dist  # Shape: (B,)
    else:
        # Compute scale only over masked points
        scale = (diffs.norm(dim=-1, p=2) * masks).sum(dim=-1) / (
            num_valid_points[:, 0, 0] + eps
        )  # B
        scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)  # B

    # Prepare transformation matrix components
    ones = torch.ones_like(scale)
    zeros = torch.zeros_like(scale)

    transform = stack(
        [
            scale,
            zeros,
            -scale * x_mean[..., 0, 0],
            zeros,
            scale,
            -scale * x_mean[..., 0, 1],
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    )  # Bx3x3

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = transform_points(transform, points)  # BxNx2

    return points_norm, transform


def local_refinement(
    local_estimator,
    points1,
    points2,
    inlier_mask,
    sorted_indices,
    lo_num=50,
    essential=False,
    skip_resize=False,
):
    # Running local refinement by local_estimator based on inlier_mask
    # as in LORANSAC

    B, N, _ = points1.shape
    batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)

    points1_expand = points1.unsqueeze(1).expand(-1, lo_num, -1, -1)
    points2_expand = points2.unsqueeze(1).expand(-1, lo_num, -1, -1)

    # The sets selected for local refinement
    lo_indices = sorted_indices[:, :lo_num]

    # Find the points that would be used for local_estimator
    lo_mask = inlier_mask[batch_index, lo_indices]
    lo_points1 = torch.zeros_like(points1_expand)
    lo_points1[lo_mask] = points1_expand[lo_mask]
    lo_points2 = torch.zeros_like(points2_expand)
    lo_points2[lo_mask] = points2_expand[lo_mask]

    lo_points1 = lo_points1.reshape(B * lo_num, N, -1)
    lo_points2 = lo_points2.reshape(B * lo_num, N, -1)
    lo_mask = lo_mask.reshape(B * lo_num, N)

    pred_mat = local_estimator(lo_points1, lo_points2, masks=lo_mask)

    if skip_resize:
        return pred_mat

    if essential:
        return pred_mat.reshape(B, lo_num, 10, 3, 3)

    return pred_mat.reshape(B, lo_num, 3, 3)


def inlier_by_fundamental(fmat, tracks, max_error=0.5):
    """
    Given tracks and fundamental matrix, compute the inlier mask for each 2D match
    """

    B, S, N, _ = tracks.shape
    left = tracks[:, 0:1].expand(-1, S - 1, -1, -1).reshape(B * (S - 1), N, 2)
    right = tracks[:, 1:].reshape(B * (S - 1), N, 2)

    fmat = fmat.reshape(B * (S - 1), 3, 3)

    max_thres = max_error**2

    residuals = sampson_epipolar_distance_batched(
        left, right, fmat[:, None], squared=True
    )

    residuals = residuals[:, 0]

    inlier_mask = residuals <= max_thres

    inlier_mask = inlier_mask.reshape(B, S - 1, -1)
    return inlier_mask


def remove_cheirality(
    R, t, points1, points2, focal_length=None, principal_point=None
):
    # TODO: merge this function with triangulation utils
    with autocast(dtype=torch.double):
        if focal_length is not None:
            principal_point = principal_point.unsqueeze(1)
            focal_length = focal_length.unsqueeze(1)

            points1 = (points1 - principal_point[..., :2]) / focal_length[
                ..., :2
            ]
            points2 = (points2 - principal_point[..., 2:]) / focal_length[
                ..., 2:
            ]

        B, cheirality_dim, _, _ = R.shape
        Bche = B * cheirality_dim
        _, N, _ = points1.shape
        points1_expand = points1[:, None].expand(-1, cheirality_dim, -1, -1)
        points2_expand = points2[:, None].expand(-1, cheirality_dim, -1, -1)
        points1_expand = points1_expand.reshape(Bche, N, 2)
        points2_expand = points2_expand.reshape(Bche, N, 2)

        cheirality_num, points3D = check_cheirality_batch(
            R.reshape(Bche, 3, 3),
            t.reshape(Bche, 3),
            points1_expand,
            points2_expand,
        )
        cheirality_num = cheirality_num.reshape(B, cheirality_dim)

        cheirality_idx = torch.argmax(cheirality_num, dim=1)

        batch_idx = torch.arange(B)
        R_cheirality = R[batch_idx, cheirality_idx]
        t_cheirality = t[batch_idx, cheirality_idx]

        return R_cheirality, t_cheirality


def triangulate_point_batch(cam1_from_world, cam2_from_world, points1, points2):
    # TODO: merge this function with triangulation utils

    B, N, _ = points1.shape
    A = torch.zeros(B, N, 4, 4, dtype=points1.dtype, device=points1.device)

    A[:, :, 0, :] = (
        points1[:, :, 0, None] * cam1_from_world[:, None, 2, :]
        - cam1_from_world[:, None, 0, :]
    )
    A[:, :, 1, :] = (
        points1[:, :, 1, None] * cam1_from_world[:, None, 2, :]
        - cam1_from_world[:, None, 1, :]
    )
    A[:, :, 2, :] = (
        points2[:, :, 0, None] * cam2_from_world[:, None, 2, :]
        - cam2_from_world[:, None, 0, :]
    )
    A[:, :, 3, :] = (
        points2[:, :, 1, None] * cam2_from_world[:, None, 2, :]
        - cam2_from_world[:, None, 1, :]
    )

    # Perform SVD on A
    _, _, Vh = torch.linalg.svd(A.view(-1, 4, 4), full_matrices=True)
    V = Vh.transpose(-2, -1)  # Transpose Vh to get V

    # Extract the last column of V for each batch and point, then reshape to the original batch and points shape
    X = V[..., -1].view(B, N, 4)
    return X[..., :3] / X[..., 3, None]


def calculate_depth_batch(proj_matrices, points3D):
    # TODO: merge this function with triangulation utils

    # proj_matrices: Bx3x4
    # points3D: BxNx3
    B, N, _ = points3D.shape
    points3D_homo = torch.cat(
        (
            points3D,
            torch.ones(B, N, 1, dtype=points3D.dtype, device=points3D.device),
        ),
        dim=-1,
    )
    points2D_homo = torch.einsum("bij,bkj->bki", proj_matrices, points3D_homo)
    return points2D_homo[..., 2]


def check_cheirality_batch(R, t, points1, points2):
    # TODO: merge this function with triangulation utils

    B, N, _ = points1.shape
    assert points1.shape == points2.shape

    proj_matrix1 = torch.eye(3, 4, dtype=R.dtype, device=R.device).expand(
        B, -1, -1
    )
    proj_matrix2 = torch.zeros(B, 3, 4, dtype=R.dtype, device=R.device)
    proj_matrix2[:, :, :3] = R
    proj_matrix2[:, :, 3] = t

    kMinDepth = torch.finfo(R.dtype).eps
    max_depth = 1000.0 * torch.linalg.norm(
        R.transpose(-2, -1) @ t[:, :, None], dim=1
    )

    points3D = triangulate_point_batch(
        proj_matrix1, proj_matrix2, points1, points2
    )

    depths1 = calculate_depth_batch(proj_matrix1, points3D)
    depths2 = calculate_depth_batch(proj_matrix2, points3D)

    valid_depths = (
        (depths1 > kMinDepth)
        & (depths1 < max_depth)
        & (depths2 > kMinDepth)
        & (depths2 < max_depth)
    )

    valid_nums = valid_depths.sum(dim=-1)
    return valid_nums, points3D


######################################################################################################


def sampson_epipolar_distance_forloop_wrapper(
    pts1: Tensor,
    pts2: Tensor,
    Fm: Tensor,
    squared: bool = True,
    eps: float = 1e-8,
    debug=False,
) -> Tensor:
    """Wrapper function for sampson_epipolar_distance_batched to loop over B dimension.

    Args:
        pts1: correspondences from the left images with shape :math:`(B, N, (2|3))`.
        pts2: correspondences from the right images with shape :math:`(B, N, (2|3))`.
        Fm: Batch of fundamental matrices with shape :math:`(B, K, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(B, K, N)`.
    """
    B = Fm.shape[0]
    output_list = []

    for b in range(B):
        output = sampson_epipolar_distance_batched(
            pts1[b].unsqueeze(0),
            pts2[b].unsqueeze(0),
            Fm[b].unsqueeze(0),
            squared=squared,
            eps=eps,
            debug=debug,
            evaluation=True,
        )
        output_list.append(output)

    return torch.cat(output_list, dim=0)


def get_default_intri(width, height, device, dtype, ratio=1.0):
    # assume same focal length for hw
    max_size = max(width, height)
    focal_length = max_size * ratio

    principal_point = [width / 2, height / 2]

    K = torch.tensor(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        device=device,
        dtype=dtype,
    )

    return (
        torch.tensor(focal_length, device=device, dtype=dtype),
        torch.tensor(principal_point, device=device, dtype=dtype),
        K,
    )


def _torch_svd_cast(input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Helper function to make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    out1, out2, out3H = torch.linalg.svd(input)
    if torch_version_ge(1, 11):
        out3 = out3H.mH
    else:
        out3 = out3H.transpose(-1, -2)
    return (out1, out2, out3)


def oneway_transfer_error_batched(
    pts1: Tensor,
    pts2: Tensor,
    H: Tensor,
    squared: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    r"""Return transfer error in image 2 for correspondences given the homography matrix.

    Args:
        pts1: correspondences from the left images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (B, N, 2 or 3). If they are homogeneous, converted automatically.
        H: Homographies with shape :math:`(B, K, 3, 3)`.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(B, K, N)`.
    """

    # From Hartley and Zisserman, Error in one image (4.6)
    # dist = \sum_{i} ( d(x', Hx)**2)

    if pts1.shape[-1] == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    B, K, _, _ = H.shape
    N = pts1.shape[1]

    pts1_expanded = pts1[:, None, :, :].expand(
        -1, K, -1, -1
    )  # Shape: (B, K, N, 3)

    H_transpose = H.permute(0, 1, 3, 2)

    pts1_in_2_h = torch.einsum("bkij,bkjn->bkin", pts1_expanded, H_transpose)

    pts1_in_2 = convert_points_from_homogeneous(pts1_in_2_h)
    pts2_expanded = pts2[:, None, :, :].expand(
        -1, K, -1, -1
    )  # Shape: (B, K, N, 2)

    error_squared = (pts1_in_2 - pts2_expanded).pow(2).sum(dim=-1)

    if squared:
        return error_squared
    return (error_squared + eps).sqrt()
