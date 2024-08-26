# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# The code structure learned from https://github.com/kornia/kornia
# Some funtions adapted from https://github.com/kornia/kornia
# The minimal solvers learned from https://github.com/colmap/colmap


import torch
from torch.cuda.amp import autocast
from kornia.geometry import solvers
from kornia.core import eye, ones_like, stack, where, zeros
from kornia.core.check import (
    KORNIA_CHECK,
    KORNIA_CHECK_SAME_SHAPE,
    KORNIA_CHECK_SHAPE,
)


from typing import Optional, Tuple


from .utils import (
    generate_samples,
    calculate_residual_indicator,
    local_refinement,
    _torch_svd_cast,
    sampson_epipolar_distance_batched,
)


def decompose_essential_matrix(E_mat: torch.Tensor):
    r"""Decompose an essential matrix to possible rotations and translation.

    This function decomposes the essential matrix E using svd decomposition
    and give the possible solutions: :math:`R1, R2, t`.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.
    """

    with autocast(dtype=torch.double):
        if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:]):
            raise AssertionError(E_mat.shape)

        # decompose matrix by its singular values
        U, _, V = _torch_svd_cast(E_mat)
        Vt = V.transpose(-2, -1)

        mask = ones_like(E_mat)
        mask[..., -1:] *= -1.0  # fill last column with negative values

        maskt = mask.transpose(-2, -1)

        # avoid singularities
        U = where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
        Vt = where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

        W = cross_product_matrix(torch.tensor([[0.0, 0.0, 1.0]]).type_as(E_mat))
        W[..., 2, 2] += 1.0

        # reconstruct rotations and retrieve translation vector
        U_W_Vt = U @ W @ Vt
        U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

        # return values
        R1 = U_W_Vt
        R2 = U_Wt_Vt
        T = U[..., -1:]

        # compbine and returns the four possible solutions
        Rs = stack([R1, R1, R2, R2], dim=1)
        Ts = stack([T, -T, T, -T], dim=1)

        return Rs, Ts[..., 0]


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = stack(
        [zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1
    )
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)


def estimate_essential(
    points1,
    points2,
    focal_length,
    principal_point,
    max_ransac_iters=1024,
    max_error=4,
    lo_num=50,
):
    """
    Estimate essential matrix by 5 point algorithm with LORANSAC

    points1, points2: Pytorch Tensor, BxNx2

    best_emat: Bx3x3
    """
    with autocast(dtype=torch.double):
        # normalize points for 5p
        principal_point = principal_point.unsqueeze(1)
        focal_length = focal_length.unsqueeze(1)

        points1 = (points1 - principal_point[..., :2]) / focal_length[..., :2]
        points2 = (points2 - principal_point[..., 2:]) / focal_length[..., 2:]

        max_error = max_error / focal_length.mean(dim=-1, keepdim=True)
        max_thres = max_error**2

        B, N, _ = points1.shape
        point_per_sample = 5  # 5p algorithm

        # randomly sample 5 point set by max_ransac_iters times
        # ransac_idx: torch matirx Nx5
        ransac_idx = generate_samples(N, max_ransac_iters, point_per_sample)
        left = points1[:, ransac_idx].view(
            B * max_ransac_iters, point_per_sample, 2
        )
        right = points2[:, ransac_idx].view(
            B * max_ransac_iters, point_per_sample, 2
        )

        # 5p algorithm will provide 10 potential answers
        # so the shape of emat_ransac is
        # B x (max_ransac_iters*10) x 3 x 3
        ####################################################################################
        emat_ransac = run_5point(left, right)
        emat_ransac = emat_ransac.reshape(
            B, max_ransac_iters, 10, 3, 3
        ).reshape(B, max_ransac_iters * 10, 3, 3)

        residuals = sampson_epipolar_distance_batched(
            points1, points2, emat_ransac, squared=True
        )

        inlier_mask = residuals <= max_thres
        inlier_num = inlier_mask.sum(dim=-1)

        _, sorted_indices = torch.sort(inlier_num, dim=1, descending=True)

        # Local Refinement by
        # 5p algorithm with inliers
        emat_lo = local_refinement(
            run_5point,
            points1,
            points2,
            inlier_mask,
            sorted_indices,
            lo_num=lo_num,
            essential=True,
        )

        emat_lo = emat_lo.reshape(B, 10 * lo_num, 3, 3)

        # choose the one with the higher inlier number and smallest (valid) residual
        all_emat = torch.cat([emat_ransac, emat_lo], dim=1)
        residuals_all = sampson_epipolar_distance_batched(
            points1, points2, all_emat, squared=True
        )

        (residual_indicator, inlier_num_all, inlier_mask_all) = (
            calculate_residual_indicator(residuals_all, max_thres)
        )

        batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)
        best_e_indices = torch.argmax(residual_indicator, dim=1)

        best_emat = all_emat[batch_index[:, 0], best_e_indices]
        best_inlier_num = inlier_num_all[batch_index[:, 0], best_e_indices]
        best_inlier_mask = inlier_mask_all[batch_index[:, 0], best_e_indices]

        return best_emat, best_inlier_num, best_inlier_mask


def run_5point(
    points1: torch.Tensor,
    points2: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized].

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`.
    """
    with autocast(dtype=torch.double):
        KORNIA_CHECK_SHAPE(points1, ["B", "N", "2"])
        KORNIA_CHECK_SAME_SHAPE(points1, points2)
        KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")

        if masks is None:
            masks = ones_like(points1[..., 0])

        if weights is not None:
            KORNIA_CHECK_SAME_SHAPE(points1[:, :, 0], weights)

        batch_size, _, _ = points1.shape
        x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
        x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
        ones = ones_like(x1)

        # build equations system and find null space.
        # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
        # BxNx9
        X = torch.cat(
            [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1
        )

        # if masks is not valid, force the cooresponding rows (points) to all zeros
        if masks is not None:
            X = X * masks.unsqueeze(-1)

        # apply the weights to the linear system
        if weights is None:
            X = X.transpose(-2, -1) @ X
        else:
            w_diag = torch.diag_embed(weights)
            X = X.transpose(-2, -1) @ w_diag @ X

        # compute eigenvectors and retrieve the one with the smallest eigenvalue, using SVD
        # turn off the grad check due to the unstable gradients from SVD.
        # several close to zero values of eigenvalues.
        _, _, V = _torch_svd_cast(X)  # torch.svd

        # use Nister's method to solve essential matrix

        E_Nister = null_to_Nister_solution(V, batch_size)
        return E_Nister


def fun_select(null_mat, i: int, j: int, ratio=3) -> torch.Tensor:
    return null_mat[:, ratio * j + i]


def null_to_Nister_solution(V, batch_size):
    null_ = V[:, :, -4:]  # the last four rows
    nullSpace = V.transpose(-1, -2)[:, -4:, :]

    coeffs = zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)

    coeffs[:, 9] = (
        solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 1), fun_select(null_, 1, 2)
            )
            - solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 2), fun_select(null_, 1, 1)
            ),
            fun_select(null_, 2, 0),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 2), fun_select(null_, 1, 0)
            )
            - solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 0), fun_select(null_, 1, 2)
            ),
            fun_select(null_, 2, 1),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 0), fun_select(null_, 1, 1)
            )
            - solvers.multiply_deg_one_poly(
                fun_select(null_, 0, 1), fun_select(null_, 1, 0)
            ),
            fun_select(null_, 2, 2),
        )
    )

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (Eqn. 20 in the paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j] : indices[i, j] + 10] = (
                solvers.multiply_deg_one_poly(
                    fun_select(null_, i, 0), fun_select(null_, j, 0)
                )
                + solvers.multiply_deg_one_poly(
                    fun_select(null_, i, 1), fun_select(null_, j, 1)
                )
                + solvers.multiply_deg_one_poly(
                    fun_select(null_, i, 2), fun_select(null_, j, 2)
                )
            )

    for i in range(10):
        t = 0.5 * (
            d[:, indices[0, 0] + i]
            + d[:, indices[1, 1] + i]
            + d[:, indices[2, 2] + i]
        )
        d[:, indices[0, 0] + i] -= t
        d[:, indices[1, 1] + i] -= t
        d[:, indices[2, 2] + i] -= t

    cnt = 0
    for i in range(3):
        for j in range(3):
            row = (
                solvers.multiply_deg_two_one_poly(
                    d[:, indices[i, 0] : indices[i, 0] + 10],
                    fun_select(null_, 0, j),
                )
                + solvers.multiply_deg_two_one_poly(
                    d[:, indices[i, 1] : indices[i, 1] + 10],
                    fun_select(null_, 1, j),
                )
                + solvers.multiply_deg_two_one_poly(
                    d[:, indices[i, 2] : indices[i, 2] + 10],
                    fun_select(null_, 2, j),
                )
            )
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]

    # NOTE Some operations are filtered here
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs),
        ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10])) * 10,
    )

    if len(singular_filter) == 0:
        return (
            torch.eye(3, dtype=coeffs.dtype, device=coeffs.device)[None]
            .expand(batch_size, 10, -1, -1)
            .clone()
        )

    eliminated_mat = torch.linalg.solve(
        coeffs[singular_filter, :, :10], b[singular_filter]
    )

    coeffs_ = torch.cat(
        (coeffs[singular_filter, :, :10], eliminated_mat), dim=-1
    )

    batch_size_filtered = coeffs_.shape[0]
    A = zeros(
        batch_size_filtered, 3, 13, device=coeffs_.device, dtype=coeffs_.dtype
    )

    for i in range(3):
        A[:, i, 0] = 0.0
        A[:, i : i + 1, 1:4] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 10:13]
        A[:, i : i + 1, 0:3] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 10:13]
        A[:, i, 4] = 0.0
        A[:, i : i + 1, 5:8] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 13:16]
        A[:, i : i + 1, 4:7] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 13:16]
        A[:, i, 8] = 0.0
        A[:, i : i + 1, 9:13] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 16:20]
        A[:, i : i + 1, 8:12] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 16:20]

    # Bx11
    cs = solvers.determinant_to_polynomial(A)
    E_models = []

    # A: Bx3x13
    # nullSpace: Bx4x9

    C = zeros((batch_size_filtered, 10, 10), device=cs.device, dtype=cs.dtype)
    eye_mat = eye(C[0, 0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
    C[:, 0:-1, 1:] = eye_mat

    cs_de = cs[:, -1].unsqueeze(-1)
    cs_de = torch.where(cs_de == 0, 1e-8, cs_de)
    C[:, -1, :] = -cs[:, :-1] / cs_de

    roots = torch.real(torch.linalg.eigvals(C))

    roots_unsqu = roots.unsqueeze(1)
    Bs = stack(
        (
            A[:, :3, :1] * (roots_unsqu**3)
            + A[:, :3, 1:2] * roots_unsqu.square()
            + A[:, 0:3, 2:3] * roots_unsqu
            + A[:, 0:3, 3:4],
            A[:, 0:3, 4:5] * (roots_unsqu**3)
            + A[:, 0:3, 5:6] * roots_unsqu.square()
            + A[:, 0:3, 6:7] * roots_unsqu
            + A[:, 0:3, 7:8],
        ),
        dim=1,
    )
    Bs = Bs.transpose(1, -1)

    bs = (
        (
            A[:, 0:3, 8:9] * (roots_unsqu**4)
            + A[:, 0:3, 9:10] * (roots_unsqu**3)
            + A[:, 0:3, 10:11] * roots_unsqu.square()
            + A[:, 0:3, 11:12] * roots_unsqu
            + A[:, 0:3, 12:13]
        )
        .transpose(1, 2)
        .unsqueeze(-1)
    )

    xzs = torch.matmul(torch.inverse(Bs[:, :, 0:2, 0:2]), bs[:, :, 0:2])

    mask = (
        abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3
    ).flatten()

    # mask: bx10x1x1
    mask = (
        abs(
            torch.matmul(Bs[:, :, 2, :].unsqueeze(2), xzs)
            - bs[:, :, 2, :].unsqueeze(2)
        )
        > 1e-3
    )  # .flatten(start_dim=1)
    # bx10
    mask = mask.squeeze(3).squeeze(2)

    if torch.any(mask):
        q_batch, r_batch = torch.linalg.qr(Bs[mask])
        xyz_to_feed = torch.linalg.solve(
            r_batch, torch.matmul(q_batch.transpose(-1, -2), bs[mask])
        )
        xzs[mask] = xyz_to_feed

    nullSpace_filtered = nullSpace[singular_filter]

    Es = (
        nullSpace_filtered[:, 0:1] * (-xzs[:, :, 0])
        + nullSpace_filtered[:, 1:2] * (-xzs[:, :, 1])
        + nullSpace_filtered[:, 2:3] * roots.unsqueeze(-1)
        + nullSpace_filtered[:, 3:4]
    )

    inv = 1.0 / torch.sqrt(
        (-xzs[:, :, 0]) ** 2
        + (-xzs[:, :, 1]) ** 2
        + roots.unsqueeze(-1) ** 2
        + 1.0
    )
    Es *= inv

    Es = Es.view(batch_size_filtered, -1, 3, 3).transpose(-1, -2)
    E_return = (
        torch.eye(3, dtype=Es.dtype, device=Es.device)[None]
        .expand(batch_size, 10, -1, -1)
        .clone()
    )
    E_return[singular_filter] = Es

    return E_return
