# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pycolmap

from torch.cuda.amp import autocast
from itertools import combinations


def triangulate_multi_view_point_batched(
    cams_from_world, points, mask=None, compute_tri_angle=False, check_cheirality=False
):
    # cams_from_world: BxNx3x4
    # points: BxNx2

    B, N, _ = points.shape
    assert (
        cams_from_world.shape[0] == B and cams_from_world.shape[1] == N
    ), "The number of cameras and points must be equal for each batch."

    # Convert points to homogeneous coordinates and normalize
    points_homo = torch.cat(
        (points, torch.ones(B, N, 1, dtype=cams_from_world.dtype, device=cams_from_world.device)), dim=-1
    )
    points_norm = points_homo / torch.norm(points_homo, dim=-1, keepdim=True)

    # Compute the outer product of each point with itself
    outer_products = torch.einsum("bni,bnj->bnij", points_norm, points_norm)

    # Compute the term for each camera-point pair
    terms = cams_from_world - torch.einsum("bnij,bnik->bnjk", outer_products, cams_from_world)

    if mask is not None:
        terms = terms * mask[:, :, None, None]

    A = torch.einsum("bnij,bnik->bjk", terms, terms)

    # Compute eigenvalues and eigenvectors
    try:
        _, eigenvectors = torch.linalg.eigh(A)
    except:
        print("Meet CUSOLVER_STATUS_INVALID_VALUE ERROR during torch.linalg.eigh()")
        print("SWITCH TO torch.linalg.eig()")
        _, eigenvectors = torch.linalg.eig(A)
        eigenvectors = torch.real(eigenvectors)
        
        
    # Select the first eigenvector
    first_eigenvector = eigenvectors[:, :, 0]

    # Perform homogeneous normalization: divide by the last component
    first_eigenvector_hnormalized = first_eigenvector / first_eigenvector[..., -1:]

    # Return the first eigenvector normalized to make its first component 1
    triangulated_points = first_eigenvector_hnormalized[..., :-1]

    if check_cheirality:
        points3D_homogeneous = torch.cat(
            [triangulated_points, torch.ones_like(triangulated_points[..., 0:1])], dim=1
        )  # Nx4

        points3D_homogeneous = points3D_homogeneous.unsqueeze(1).unsqueeze(-1)
        points_cam = torch.matmul(cams_from_world, points3D_homogeneous).squeeze(-1)

        invalid_cheirality_mask = points_cam[..., -1] <= 0
        invalid_cheirality_mask = invalid_cheirality_mask.any(dim=1)

    if compute_tri_angle:
        triangles = calculate_triangulation_angle_batched(cams_from_world, triangulated_points)

    if check_cheirality and compute_tri_angle:
        return triangulated_points, triangles, invalid_cheirality_mask

    if compute_tri_angle:
        return triangulated_points, triangles

    if check_cheirality:
        return triangulated_points, invalid_cheirality_mask

    return triangulated_points


def filter_all_points3D(
    points3D,
    points2D,
    extrinsics,
    intrinsics,
    max_reproj_error=4,
    min_tri_angle=1.5,
    check_triangle=True,
    return_detail=False,
    hard_max = 100,
):
    """
    Filter 3D points based on reprojection error and triangulation angle error.

    Adapted from https://github.com/colmap/colmap/blob/0ea2d5ceee1360bba427b2ef61f1351e59a46f91/src/colmap/sfm/incremental_mapper.cc#L828

    """
    # points3D Px3
    # points2D BxPx2
    # extrinsics Bx3x4
    # intrinsics Bx3x3

    # compute reprojection error
    projected_points2D, projected_points_cam = project_3D_points(
        points3D, extrinsics, intrinsics, return_points_cam=True
    )

    reproj_error = (projected_points2D - points2D).norm(dim=-1) ** 2  # sqaure
    # ensure all the points stay in front of the cameras
    reproj_error[projected_points_cam[:, -1] <= 0] = 1e6

    inlier = reproj_error <= (max_reproj_error**2)
    valid_track_length = inlier.sum(dim=0)

    valid_track_mask = valid_track_length >= 2  # at least two frames to form a track
    
    if hard_max>0:
        valid_value_mask = (points3D.abs()<=hard_max).all(-1)
        valid_track_mask = torch.logical_and(valid_track_mask, valid_value_mask)

    if check_triangle:
        # update points3D
        points3D = points3D[valid_track_mask]
        inlier = inlier[:, valid_track_mask]
        # https://github.com/colmap/colmap/blob/0ea2d5ceee1360bba427b2ef61f1351e59a46f91/src/colmap/geometry/triangulation.cc#L130

        B = len(extrinsics)

        triangles = calculate_triangulation_angle_exhaustive(extrinsics, points3D)

        # only when both the pair are within reporjection thres,
        # the triangles can be counted
        inlier_row = inlier[:, None].expand(-1, B, -1).reshape(B * B, -1)
        inlier_col = inlier[None].expand(B, -1, -1).reshape(B * B, -1)
        inlier_grid = torch.logical_and(inlier_row, inlier_col)

        triangles_valid_mask = torch.logical_and((triangles >= min_tri_angle), inlier_grid)

        # if any pair meets the standard, it is okay
        triangles_valid_any = triangles_valid_mask.sum(dim=0) > 0

        triangles_valid_any_full_size = torch.zeros_like(valid_track_mask)
        triangles_valid_any_full_size[valid_track_mask] = triangles_valid_any

        return_mask = torch.logical_and(triangles_valid_any_full_size, valid_track_mask)
    else:
        return_mask = valid_track_mask

    if check_triangle and return_detail:
        inlier_detail = reproj_error <= (max_reproj_error**2)
        inlier_detail = triangles_valid_any_full_size[None] * inlier_detail
        return return_mask, inlier_detail

    return return_mask


def project_3D_points(points3D, extrinsics, intrinsics=None, return_points_cam=False, default=0, only_points_cam=False):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        points3D (torch.Tensor): 3D points of shape Px3.
        extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    with autocast(dtype=torch.double):
        N = points3D.shape[0]  # Number of points
        B = extrinsics.shape[0]  # Batch size, i.e., number of cameras
        points3D_homogeneous = torch.cat([points3D, torch.ones_like(points3D[..., 0:1])], dim=1)  # Nx4
        # Reshape for batch processing
        points3D_homogeneous = points3D_homogeneous.unsqueeze(0).expand(B, -1, -1)  # BxNx4
        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        points_cam = torch.bmm(extrinsics, points3D_homogeneous.transpose(-1, -2))

        if only_points_cam:
            return points_cam

        # Step 2: Apply intrinsic parameters
        # Intrinsic multiplication requires a transpose to match dimensions (Bx3x3 * Bx3xN -> Bx3xN)
        points2D_homogeneous = torch.bmm(intrinsics, points_cam)  # Still Bx3xN
        points2D_homogeneous = points2D_homogeneous.transpose(1, 2)  # BxNx3
        points2D = points2D_homogeneous[..., :2] / points2D_homogeneous[..., 2:3]  # BxNx2
        # Performs safe division, replacing NaNs with a default value
        points2D[torch.isnan(points2D)] = default
        if return_points_cam:
            return points2D, points_cam
        return points2D


def calculate_normalized_angular_error_batched(point2D, point3D, cam_from_world, to_degree=False):
    """
    Please note the normalized angular error is different from triangulation angle
    """
    # point2D: BxNx2
    # point3D: PxNx3
    # cam_from_world: Bx3x4

    B, N, _ = point2D.shape
    P, _, _ = point3D.shape
    assert len(cam_from_world) == B

    # homogeneous
    point2D_homo = torch.cat([point2D, torch.ones_like(point2D[..., 0:1])], dim=-1)
    point3D_homo = torch.cat([point3D, torch.ones_like(point3D[..., 0:1])], dim=-1)

    point3D_homo_tran = point3D_homo.transpose(-1, -2)

    ray1 = point2D_homo
    ray2 = cam_from_world[None].expand(P, -1, -1, -1) @ point3D_homo_tran[:, None].expand(-1, B, -1, -1)

    ray1 = F.normalize(ray1, dim=-1)
    # PxBxNx3
    ray2 = F.normalize(ray2.transpose(-1, -2), dim=-1)

    ray1 = ray1[None].expand(P, -1, -1, -1)
    cos_angle = (ray1 * ray2).sum(dim=-1)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    triangles = torch.acos(cos_angle)

    if to_degree:
        triangles = triangles * (180.0 / torch.pi)

    return triangles, cos_angle


def calculate_triangulation_angle_batched(extrinsics, points3D, eps=1e-12):
    # points3D: Bx3
    # extrinsics: BxSx3x4

    B, S, _, _ = extrinsics.shape
    assert len(points3D) == B

    R = extrinsics[:, :, :, :3]  # B x S x 3 x 3
    t = extrinsics[:, :, :, 3]  # B x S x 3

    proj_centers = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    # tmp = -R.transpose(-1, -2)[0].bmm(t.unsqueeze(-1)[0])

    proj_center1 = proj_centers[:, :, None].expand(-1, -1, S, -1)
    proj_center2 = proj_centers[:, None].expand(-1, S, -1, -1)

    # Bx(S*S)x3
    # TODO not using S*S any more, instead of C( )
    proj_center1 = proj_center1.reshape(B, S * S, 3)
    proj_center2 = proj_center2.reshape(B, S * S, 3)

    # Bx(S*S)
    baseline_length_squared = (proj_center1 - proj_center2).norm(dim=-1) ** 2

    # Bx(S*S)
    ray_length_squared1 = (points3D[:, None] - proj_center1).norm(dim=-1) ** 2
    ray_length_squared2 = (points3D[:, None] - proj_center2).norm(dim=-1) ** 2

    denominator = 2.0 * torch.sqrt(ray_length_squared1 * ray_length_squared2)
    nominator = ray_length_squared1 + ray_length_squared2 - baseline_length_squared
    # if denominator is zero, angle is zero
    # so we set nominator and denominator as one
    # acos(1) = 0
    nonvalid = denominator <= eps
    nominator = torch.where(nonvalid, torch.ones_like(nominator), nominator)
    denominator = torch.where(nonvalid, torch.ones_like(denominator), denominator)
    cos_angle = nominator / denominator
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    triangles = torch.abs(torch.acos(cos_angle))
    triangles = torch.min(triangles, torch.pi - triangles)
    triangles = triangles * (180.0 / torch.pi)

    return triangles


def calculate_triangulation_angle_exhaustive(extrinsics, points3D):
    # points3D: Px3
    # extrinsics: Bx3x4

    R = extrinsics[:, :, :3]  # B x 3 x 3
    t = extrinsics[:, :, 3]  # B x 3
    # Compute projection centers
    proj_centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    B = len(proj_centers)

    # baseline_length_squared = (proj_centers[:,None] - proj_centers[None]).norm(dim=-1) ** 2
    proj_center1 = proj_centers[:, None].expand(-1, B, -1)
    proj_center2 = proj_centers[None].expand(B, -1, -1)
    proj_center1 = proj_center1.reshape(B * B, 3)
    proj_center2 = proj_center2.reshape(B * B, 3)

    triangles = calculate_triangulation_angle(proj_center1, proj_center2, points3D)

    return triangles


def calculate_triangulation_angle(proj_center1, proj_center2, point3D, eps=1e-12):
    # proj_center1: Bx3
    # proj_center2: Bx3
    # point3D: Px3
    # returned: (B*B)xP, in degree

    # B
    baseline_length_squared = (proj_center1 - proj_center2).norm(dim=-1) ** 2  # B*(S-1)x1

    # BxP
    ray_length_squared1 = (point3D[None] - proj_center1[:, None]).norm(dim=-1) ** 2
    ray_length_squared2 = (point3D[None] - proj_center2[:, None]).norm(dim=-1) ** 2

    denominator = 2.0 * torch.sqrt(ray_length_squared1 * ray_length_squared2)
    nominator = ray_length_squared1 + ray_length_squared2 - baseline_length_squared.unsqueeze(-1)
    # if denominator is zero, angle is zero
    # so we set nominator and denominator as one
    # acos(1) = 0
    nonvalid = denominator <= eps
    nominator = torch.where(nonvalid, torch.ones_like(nominator), nominator)
    denominator = torch.where(nonvalid, torch.ones_like(denominator), denominator)
    cos_angle = nominator / denominator
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    triangles = torch.abs(torch.acos(cos_angle))
    triangles = torch.min(triangles, torch.pi - triangles)
    triangles = triangles * (180.0 / torch.pi)
    return triangles


def create_intri_matrix(focal_length, principal_point):
    """
    Creates a intri matrix from focal length and principal point.

    Args:
        focal_length (torch.Tensor): A Bx2 or BxSx2 tensor containing the focal lengths (fx, fy) for each image.
        principal_point (torch.Tensor): A Bx2 or BxSx2 tensor containing the principal point coordinates (cx, cy) for each image.

    Returns:
        torch.Tensor: A Bx3x3 or BxSx3x3 tensor containing the camera matrix for each image.
    """

    if len(focal_length.shape) == 2:
        B = focal_length.shape[0]
        intri_matrix = torch.zeros(B, 3, 3, dtype=focal_length.dtype, device=focal_length.device)
        intri_matrix[:, 0, 0] = focal_length[:, 0]
        intri_matrix[:, 1, 1] = focal_length[:, 1]
        intri_matrix[:, 2, 2] = 1.0
        intri_matrix[:, 0, 2] = principal_point[:, 0]
        intri_matrix[:, 1, 2] = principal_point[:, 1]
    else:
        B, S = focal_length.shape[0], focal_length.shape[1]
        intri_matrix = torch.zeros(B, S, 3, 3, dtype=focal_length.dtype, device=focal_length.device)
        intri_matrix[:, :, 0, 0] = focal_length[:, :, 0]
        intri_matrix[:, :, 1, 1] = focal_length[:, :, 1]
        intri_matrix[:, :, 2, 2] = 1.0
        intri_matrix[:, :, 0, 2] = principal_point[:, :, 0]
        intri_matrix[:, :, 1, 2] = principal_point[:, :, 1]

    return intri_matrix


def prepare_ba_options():
    ba_options_tmp = pycolmap.BundleAdjustmentOptions()
    ba_options_tmp.solver_options.function_tolerance *= 10
    ba_options_tmp.solver_options.gradient_tolerance *= 10
    ba_options_tmp.solver_options.parameter_tolerance *= 10

    ba_options_tmp.solver_options.max_num_iterations = 50
    ba_options_tmp.solver_options.max_linear_solver_iterations = 200
    ba_options_tmp.print_summary = False
    return ba_options_tmp


def generate_combinations(N):
    # Create an array of numbers from 0 to N-1
    indices = np.arange(N)
    # Generate all C(N, 2) combinations
    comb = list(combinations(indices, 2))
    # Convert list of tuples into a NumPy array
    comb_array = np.array(comb)
    return comb_array


def local_refinement_tri(points1, extrinsics, inlier_mask, sorted_indices, lo_num=50):
    """
    Local Refinement for triangulation
    """
    B, N, _ = points1.shape
    batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)

    points1_expand = points1.unsqueeze(1).expand(-1, lo_num, -1, -1)
    extrinsics_expand = extrinsics.unsqueeze(1).expand(-1, lo_num, -1, -1, -1)

    # The sets selected for local refinement
    lo_indices = sorted_indices[:, :lo_num]

    # Find the points that would be used for local_estimator
    lo_mask = inlier_mask[batch_index, lo_indices]
    lo_points1 = torch.zeros_like(points1_expand)
    lo_points1[lo_mask] = points1_expand[lo_mask]

    lo_points1 = lo_points1.reshape(B * lo_num, N, -1)
    lo_mask = lo_mask.reshape(B * lo_num, N)
    lo_extrinsics = extrinsics_expand.reshape(B * lo_num, N, 3, 4)

    # triangulate the inliers
    triangulated_points, tri_angles, invalid_che_mask = triangulate_multi_view_point_batched(
        lo_extrinsics, lo_points1, mask=lo_mask, compute_tri_angle=True, check_cheirality=True
    )

    triangulated_points = triangulated_points.reshape(B, lo_num, 3)
    tri_angles = tri_angles.reshape(B, lo_num, -1)

    invalid_che_mask = invalid_che_mask.reshape(B, lo_num)

    return triangulated_points, tri_angles, invalid_che_mask
