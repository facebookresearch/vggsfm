# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Adapted from https://github.com/amyxlase/relpose-plus-plus

import torch
import numpy as np
import math


from minipytorch3d.cameras import PerspectiveCameras
from minipytorch3d.transform3d import Rotate, Translate
from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

# from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
# from pytorch3d.transforms import Rotate, Translate
# from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


def bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh


def adjust_camera_to_bbox_crop_(
    fl, pp, image_size_wh: torch.Tensor, clamp_bbox_xywh: torch.Tensor
):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        fl, pp, image_size_wh
    )

    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px, principal_point_px_cropped, clamp_bbox_xywh[2:]
    )

    return focal_length, principal_point_cropped


def adjust_camera_to_image_scale_(
    fl, pp, original_size_wh: torch.Tensor, new_size_wh: torch.LongTensor
):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        fl, pp, original_size_wh
    )

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (
        (image_size_wh_output / original_size_wh)
        .min(dim=-1, keepdim=True)
        .values
    )
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled, principal_point_px_scaled, image_size_wh_output
    )
    return focal_length_scaled, principal_point_scaled


def _convert_ndc_to_pixels(
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    image_size_wh: torch.Tensor,
):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor,
    principal_point_px: torch.Tensor,
    image_size_wh: torch.Tensor,
):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point


def normalize_cameras(
    cameras,
    compute_optical=True,
    first_camera=True,
    normalize_trans=True,
    scale=1.0,
    points=None,
    max_norm=False,
):
    """
    Normalizes cameras such that
    (1) the optical axes point to the origin and the average distance to the origin is 1
    (2) the first camera is the origin
    (3) the translation vector is normalized

    TODO: some transforms overlap with others. no need to do so many transforms
    Args:
        cameras (List[camera]).
    """
    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()

    if compute_optical:
        new_cameras, points = compute_optical_transform(
            new_cameras, points=points
        )

    if first_camera:
        new_cameras, points = first_camera_transform(new_cameras, points=points)

    if normalize_trans:
        new_cameras, points = normalize_translation(
            new_cameras, points=points, max_norm=max_norm
        )

    return new_cameras, points


def compute_optical_transform(new_cameras, points=None):
    """
    adapted from https://github.com/amyxlase/relpose-plus-plus
    """

    new_transform = new_cameras.get_world_to_view_transform()
    (p_intersect, dist, p_line_intersect, pp, r) = (
        compute_optical_axis_intersection(new_cameras)
    )
    t = Translate(p_intersect)
    scale = dist.squeeze()[0]

    if points is not None:
        points = t.inverse().transform_points(points)
        points = points / scale

    # Degenerate case
    if scale == 0:
        scale = torch.norm(new_cameras.T, dim=(0, 1))
        scale = torch.sqrt(scale)
        new_cameras.T = new_cameras.T / scale
    else:
        new_matrix = t.compose(new_transform).get_matrix()
        new_cameras.R = new_matrix[:, :3, :3]
        new_cameras.T = new_matrix[:, 3, :3] / scale

    return new_cameras, points


def compute_optical_axis_intersection(cameras):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1))
    optical_axis = torch.cat((principal_points, one_vec), -1)

    pp = cameras.unproject_points(
        optical_axis, from_ndc=True, world_coordinates=True
    )

    pp2 = pp[torch.arange(pp.shape[0]), torch.arange(pp.shape[0])]

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(
        p=centers, r=directions, mask=None
    )

    p_intersect = p_intersect.squeeze().unsqueeze(0)
    dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    _, p_line_intersect = _point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = (
        (p_line_intersect - p_intersect[..., None, :]) ** 2
    ).sum(dim=-1)
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[
        ..., 0
    ]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        raise ValueError(f"p_intersect is NaN")

    return p_intersect, r


def _point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def first_camera_transform(cameras, rotation_only=False, points=None):
    """
    Transform so that the first camera is the origin
    """

    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()

    tR = Rotate(new_cameras.R[0].unsqueeze(0))
    if rotation_only:
        t = tR.inverse()
    else:
        tT = Translate(new_cameras.T[0].unsqueeze(0))
        t = tR.compose(tT).inverse()

    if points is not None:
        points = t.inverse().transform_points(points)

    new_matrix = t.compose(new_transform).get_matrix()

    new_cameras.R = new_matrix[:, :3, :3]
    new_cameras.T = new_matrix[:, 3, :3]

    return new_cameras, points


def normalize_translation(new_cameras, points=None, max_norm=False):
    t_gt = new_cameras.T.clone()
    t_gt = t_gt[1:, :]

    if max_norm:
        t_gt_scale = torch.norm(t_gt, dim=(-1))
        t_gt_scale = t_gt_scale.max()
        t_gt_scale = t_gt_scale.clamp(min=0.01, max=100)
    else:
        t_gt_scale = torch.norm(t_gt, dim=(0, 1))
        t_gt_scale = t_gt_scale / math.sqrt(len(t_gt))
        t_gt_scale = t_gt_scale / 2
        t_gt_scale = t_gt_scale.clamp(min=0.01, max=100)

    new_cameras.T = new_cameras.T / t_gt_scale

    if points is not None:
        points = points / t_gt_scale

    return new_cameras, points
