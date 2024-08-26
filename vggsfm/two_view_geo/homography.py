# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Adapted from https://github.com/kornia
from typing import Literal, Optional, Tuple

import torch

from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
from kornia.geometry.linalg import transform_points
from kornia.geometry.solvers import solve_cubic
from kornia.utils._compat import torch_version_ge

import math
from torch.cuda.amp import autocast

from .utils import (
    generate_samples,
    calculate_residual_indicator,
    normalize_points_masked,
    local_refinement,
    _torch_svd_cast,
    oneway_transfer_error_batched,
)

from kornia.core.check import KORNIA_CHECK_IS_TENSOR

import warnings

from kornia.utils import (
    _extract_device_dtype,
    safe_inverse_with_mask,
    safe_solve_with_mask,
)

from kornia.geometry.homography import oneway_transfer_error


# The code structure learned from https://github.com/kornia/kornia
# Some funtions adapted from https://github.com/kornia/kornia
# The minimal solvers learned from https://github.com/colmap/colmap


def estimate_homography(
    points1, points2, max_ransac_iters=1024, max_error=4, lo_num=50
):
    max_thres = max_error**2
    # points1, points2: BxNx2
    B, N, _ = points1.shape
    point_per_sample = 4  # 4p algorithm

    ransac_idx = generate_samples(N, max_ransac_iters, point_per_sample)
    left = points1[:, ransac_idx].view(
        B * max_ransac_iters, point_per_sample, 2
    )
    right = points2[:, ransac_idx].view(
        B * max_ransac_iters, point_per_sample, 2
    )

    hmat_ransac = run_homography_dlt(left.float(), right.float())
    hmat_ransac = hmat_ransac.reshape(B, max_ransac_iters, 3, 3)

    residuals = oneway_transfer_error_batched(
        points1, points2, hmat_ransac, squared=True
    )

    inlier_mask = residuals <= max_thres

    inlier_num = inlier_mask.sum(dim=-1)

    sorted_values, sorted_indices = torch.sort(
        inlier_num, dim=1, descending=True
    )

    hmat_lo = local_refinement(
        run_homography_dlt,
        points1,
        points2,
        inlier_mask,
        sorted_indices,
        lo_num=lo_num,
    )

    # choose the one with the higher inlier number and smallest (valid) residual
    all_hmat = torch.cat([hmat_ransac, hmat_lo], dim=1)
    residuals_all = oneway_transfer_error_batched(
        points1, points2, all_hmat, squared=True
    )
    (residual_indicator, inlier_num_all, inlier_mask_all) = (
        calculate_residual_indicator(residuals_all, max_thres)
    )

    batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)
    best_indices = torch.argmax(residual_indicator, dim=1)

    best_hmat = all_hmat[batch_index[:, 0], best_indices]
    best_inlier_num = inlier_num_all[batch_index[:, 0], best_indices]
    best_inlier_mask = inlier_mask_all[batch_index[:, 0], best_indices]

    return best_hmat, best_inlier_num, best_inlier_mask


def run_homography_dlt(
    points1: torch.Tensor,
    points2: torch.Tensor,
    masks=None,
    weights: Optional[torch.Tensor] = None,
    solver: str = "svd",
    colmap_style=False,
) -> torch.Tensor:
    r"""Compute the homography matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        solver: variants: svd, lu.


    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    # with autocast(dtype=torch.double):
    with autocast(dtype=torch.float32):
        if points1.shape != points2.shape:
            raise AssertionError(points1.shape)
        if points1.shape[1] < 4:
            raise AssertionError(points1.shape)
        KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
        KORNIA_CHECK_SHAPE(points2, ["B", "N", "2"])

        device, dtype = _extract_device_dtype([points1, points2])

        eps: float = 1e-8

        if masks is None:
            masks = ones_like(points1[..., 0])

        points1_norm, transform1 = normalize_points_masked(
            points1, masks=masks, colmap_style=colmap_style
        )
        points2_norm, transform2 = normalize_points_masked(
            points2, masks=masks, colmap_style=colmap_style
        )

        x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
        x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
        ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

        # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501

        if colmap_style:
            # should be the same
            ax = torch.cat(
                [-x1, -y1, -ones, zeros, zeros, zeros, x1 * x2, y1 * x2, x2],
                dim=-1,
            )
            ay = torch.cat(
                [zeros, zeros, zeros, -x1, -y1, -ones, x1 * y2, y1 * y2, y2],
                dim=-1,
            )
        else:
            ax = torch.cat(
                [zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2],
                dim=-1,
            )
            ay = torch.cat(
                [x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2],
                dim=-1,
            )

        # if masks is not valid, force the cooresponding rows (points) to all zeros
        if masks is not None:
            masks = masks.unsqueeze(-1)
            ax = ax * masks
            ay = ay * masks

        A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

        if weights is None:
            # All points are equally important
            A = A.transpose(-2, -1) @ A
        else:
            # We should use provided weights
            if not (
                len(weights.shape) == 2 and weights.shape == points1.shape[:2]
            ):
                raise AssertionError(weights.shape)
            w_diag = torch.diag_embed(
                weights.unsqueeze(dim=-1)
                .repeat(1, 1, 2)
                .reshape(weights.shape[0], -1)
            )
            A = A.transpose(-2, -1) @ w_diag @ A

        if solver == "svd":
            try:
                _, _, V = _torch_svd_cast(A)
            except RuntimeError:
                warnings.warn("SVD did not converge", RuntimeWarning)
                return torch.empty(
                    (points1_norm.size(0), 3, 3), device=device, dtype=dtype
                )
            H = V[..., -1].view(-1, 3, 3)
        else:
            raise NotImplementedError

        H = transform2.inverse() @ (H @ transform1)
        H_norm = H / (H[..., -1:, -1:] + eps)
        return H_norm


def normalize_to_unit(M: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.
    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val = M.norm(dim=-1, keepdim=True)
    return where(norm_val.abs() > eps, M / (norm_val + eps), M)


############ decompose


def decompose_homography_matrix(H, left, right, K1, K2):
    # WE FORCE FLOAT64 here to avoid the problem in SVD
    B, _, _ = H.shape  # Assuming H is Bx3x3
    H = H.double()
    K1 = K1.double()
    K2 = K2.double()

    with autocast(dtype=torch.double):
        # Adjust calibration removal for batched input
        K2_inv = torch.linalg.inv(K2)  # Assuming K2 is Bx3x3
        H_normalized = torch.matmul(torch.matmul(K2_inv, H), K1)

        # Adjust scale removal for batched input
        _, s, _ = torch.linalg.svd(H_normalized)
        s_mid = s[:, 1].unsqueeze(1).unsqueeze(2)
        H_normalized /= s_mid

        # Ensure that we always return rotations, and never reflections
        det_H = torch.linalg.det(H_normalized)
        H_normalized[det_H < 0] *= -1.0

        I_3 = torch.eye(3, device=H.device).unsqueeze(0).expand(B, 3, 3)
        S = torch.matmul(H_normalized.transpose(-2, -1), H_normalized) - I_3

        kMinInfinityNorm = 1e-3
        rotation_only_mask = (
            torch.linalg.norm(S, ord=float("inf"), dim=(-2, -1))
            < kMinInfinityNorm
        )

        M00 = compute_opposite_of_minor(S, 0, 0)
        M11 = compute_opposite_of_minor(S, 1, 1)
        M22 = compute_opposite_of_minor(S, 2, 2)

        rtM00 = torch.sqrt(M00)
        rtM11 = torch.sqrt(M11)
        rtM22 = torch.sqrt(M22)

        M01 = compute_opposite_of_minor(S, 0, 1)
        M12 = compute_opposite_of_minor(S, 1, 2)
        M02 = compute_opposite_of_minor(S, 0, 2)

        e12 = torch.sign(M12)
        e02 = torch.sign(M02)
        e01 = torch.sign(M01)

        nS = torch.stack(
            [S[:, 0, 0].abs(), S[:, 1, 1].abs(), S[:, 2, 2].abs()], dim=1
        )
        idx = torch.argmax(nS, dim=1)

        np1, np2 = compute_np1_np2(idx, S, rtM22, rtM11, rtM00, e12, e02, e01)

        traceS = torch.einsum("bii->b", S)  # Batched trace
        v = 2.0 * torch.sqrt(1.0 + traceS - M00 - M11 - M22)

        ESii = torch.sign(torch.stack([S[i, idx[i], idx[i]] for i in range(B)]))

        r_2 = 2 + traceS + v
        nt_2 = 2 + traceS - v

        r = torch.sqrt(r_2)
        n_t = torch.sqrt(nt_2)

        # normalize there
        np1_valid_mask = torch.linalg.norm(np1, dim=-1) != 0
        np1_valid_scale = torch.linalg.norm(np1[np1_valid_mask], dim=-1)
        np1[np1_valid_mask] = np1[np1_valid_mask] / np1_valid_scale.unsqueeze(1)

        np2_valid_mask = torch.linalg.norm(np2, dim=-1) != 0
        np2_valid_scale = torch.linalg.norm(np2[np2_valid_mask], dim=-1)
        np2[np2_valid_mask] = np2[np2_valid_mask] / np2_valid_scale.unsqueeze(1)

        half_nt = 0.5 * n_t
        esii_t_r = ESii * r
        t1_star = half_nt.unsqueeze(-1) * (
            esii_t_r.unsqueeze(-1) * np2 - n_t.unsqueeze(-1) * np1
        )
        t2_star = half_nt.unsqueeze(-1) * (
            esii_t_r.unsqueeze(-1) * np1 - n_t.unsqueeze(-1) * np2
        )

        R1 = compute_homography_rotation(H_normalized, t1_star, np1, v)
        t1 = torch.bmm(R1, t1_star.unsqueeze(-1)).squeeze(-1)

        R2 = compute_homography_rotation(H_normalized, t2_star, np2, v)
        t2 = torch.bmm(R2, t2_star.unsqueeze(-1)).squeeze(-1)

        # normalize to norm-1 vector
        t1 = normalize_to_unit(t1)
        t2 = normalize_to_unit(t2)

        R_return = torch.cat(
            [R1[:, None], R1[:, None], R2[:, None], R2[:, None]], dim=1
        )
        t_return = torch.cat(
            [t1[:, None], -t1[:, None], t2[:, None], -t2[:, None]], dim=1
        )

        np_return = torch.cat(
            [-np1[:, None], np1[:, None], -np2[:, None], np2[:, None]], dim=1
        )

        return R_return, t_return, np_return


def compute_homography_rotation(H_normalized, tstar, n, v):
    B, _, _ = H_normalized.shape
    identity_matrix = (
        torch.eye(3, device=H_normalized.device).unsqueeze(0).repeat(B, 1, 1)
    )
    outer_product = tstar.unsqueeze(2) * n.unsqueeze(1)
    R = H_normalized @ (
        identity_matrix - (2.0 / v.unsqueeze(-1).unsqueeze(-1)) * outer_product
    )
    return R


def compute_np1_np2(idx, S, rtM22, rtM11, rtM00, e12, e02, e01):
    B = S.shape[0]
    np1 = torch.zeros(B, 3, dtype=S.dtype, device=S.device)
    np2 = torch.zeros(B, 3, dtype=S.dtype, device=S.device)

    # Masks for selecting indices
    idx0 = idx == 0
    idx1 = idx == 1
    idx2 = idx == 2

    # Compute np1 and np2 for idx == 0
    np1[idx0, 0], np2[idx0, 0] = S[idx0, 0, 0], S[idx0, 0, 0]
    np1[idx0, 1], np2[idx0, 1] = (
        S[idx0, 0, 1] + rtM22[idx0],
        S[idx0, 0, 1] - rtM22[idx0],
    )
    np1[idx0, 2], np2[idx0, 2] = (
        S[idx0, 0, 2] + e12[idx0] * rtM11[idx0],
        S[idx0, 0, 2] - e12[idx0] * rtM11[idx0],
    )

    # Compute np1 and np2 for idx == 1
    np1[idx1, 0], np2[idx1, 0] = (
        S[idx1, 0, 1] + rtM22[idx1],
        S[idx1, 0, 1] - rtM22[idx1],
    )
    np1[idx1, 1], np2[idx1, 1] = S[idx1, 1, 1], S[idx1, 1, 1]
    np1[idx1, 2], np2[idx1, 2] = (
        S[idx1, 1, 2] - e02[idx1] * rtM00[idx1],
        S[idx1, 1, 2] + e02[idx1] * rtM00[idx1],
    )

    # Compute np1 and np2 for idx == 2
    np1[idx2, 0], np2[idx2, 0] = (
        S[idx2, 0, 2] + e01[idx2] * rtM11[idx2],
        S[idx2, 0, 2] - e01[idx2] * rtM11[idx2],
    )
    np1[idx2, 1], np2[idx2, 1] = (
        S[idx2, 1, 2] + rtM00[idx2],
        S[idx2, 1, 2] - rtM00[idx2],
    )
    np1[idx2, 2], np2[idx2, 2] = S[idx2, 2, 2], S[idx2, 2, 2]

    return np1, np2


def compute_opposite_of_minor(matrix, row, col):
    col1 = 1 if col == 0 else 0
    col2 = 1 if col == 2 else 2
    row1 = 1 if row == 0 else 0
    row2 = 1 if row == 2 else 2
    return (
        matrix[:, row1, col2] * matrix[:, row2, col1]
        - matrix[:, row1, col1] * matrix[:, row2, col2]
    )
