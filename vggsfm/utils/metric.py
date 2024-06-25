# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import torch

from pytorch3d.transforms import so3_relative_angle
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_size, gt_se3.shape[0] // batch_size)
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
        rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])

    return rel_rangle_deg, rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def calculate_auc(r_error, t_error, max_threshold=30, return_list=False):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    if return_list:
        return torch.cumsum(normalized_histogram, dim=0).mean(), normalized_histogram
    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def closed_form_inverse_OpenCV(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    
    
    | R t |
    | 0 1 |
    -->
    | R^T  -R^T t|
    | 0       1  |
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, :3, 3:]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # -R^T t
    top_right = - R_transposed.bmm(T)
    
    inverted_matrix = torch.eye(4,4)[None].repeat(len(se3), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)
    
    inverted_matrix[:, :3,:3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def closed_form_inverse(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.
    This function assumes PyTorch3D coordinate.
    

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, 3:, :3]

    # NOTE THIS ASSUMES PYTORCH3D CAMERA COORDINATE
    
    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t
