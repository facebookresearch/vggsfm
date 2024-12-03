# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import pycolmap

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.cuda.amp import autocast

from .tensor_to_pycolmap import (
    batch_matrix_to_pycolmap,
    pycolmap_to_batch_matrix,
)

from .triangulation_helpers import (
    cam_from_img,
    triangulate_multi_view_point_batched,
    filter_all_points3D,
    project_3D_points,
    calculate_normalized_angular_error_batched,
    calculate_triangulation_angle_batched,
    calculate_triangulation_angle_exhaustive,
    calculate_triangulation_angle,
    create_intri_matrix,
    prepare_ba_options,
    generate_combinations,
    local_refinement_tri,
)

from ..two_view_geo.utils import (
    calculate_depth_batch,
    calculate_residual_indicator,
)


def triangulate_by_pair(extrinsics, tracks_normalized, eps=1e-12):
    """
    Given B x S x 3 x 4 extrinsics and B x S x N x 2 tracks_normalized,
    triangulate point clouds for B*(S-1) query-reference pairs

    Return:

    points_3d_pair: B*(S-1) x N x 3
    cheirality_mask: B*(S-1) x N
    triangles: B*(S-1) x N
    """
    B, S, N, _ = tracks_normalized.shape

    # build pair wise extrinsics and matches
    extrinsics_left = extrinsics[:, 0:1].expand(-1, S - 1, -1, -1)
    extrinsics_right = extrinsics[:, 1:]
    extrinsics_pair = torch.cat(
        [extrinsics_left.unsqueeze(2), extrinsics_right.unsqueeze(2)], dim=2
    )

    tracks_normalized_left = tracks_normalized[:, 0:1].expand(-1, S - 1, -1, -1)
    tracks_normalized_right = tracks_normalized[:, 1:]
    tracks_normalized_pair = torch.cat(
        [
            tracks_normalized_left.unsqueeze(2),
            tracks_normalized_right.unsqueeze(2),
        ],
        dim=2,
    )

    extrinsics_pair = extrinsics_pair.reshape(B * (S - 1), 2, 3, 4)
    tracks_normalized_pair = tracks_normalized_pair.reshape(
        B * (S - 1), 2, N, 2
    )

    # triangulate
    points_3d_pair, cheirality_mask = triangulate_multi_view_point_from_tracks(
        extrinsics_pair, tracks_normalized_pair
    )

    # check triangulation angles
    # B*(S-1)x3x1
    # Learned from
    # https://github.com/colmap/colmap/blob/c0d8926841cf6325eb031c873eaedb95204a1845/src/colmap/geometry/triangulation.cc#L155
    rot_left = extrinsics_pair[:, 0, :3, :3]
    t_left = extrinsics_pair[:, 0, :3, 3:4]
    project_center1 = torch.bmm(-rot_left.transpose(-2, -1), t_left)

    rot_right = extrinsics_pair[:, 1, :3, :3]
    t_right = extrinsics_pair[:, 1, :3, 3:4]
    project_center2 = torch.bmm(-rot_right.transpose(-2, -1), t_right)

    baseline_length_squared = (project_center2 - project_center1).norm(
        dim=1
    ) ** 2  # B*(S-1)x1
    ray_length_squared1 = (
        points_3d_pair - project_center1.transpose(-2, -1)
    ).norm(
        dim=-1
    ) ** 2  # BxN
    ray_length_squared2 = (
        points_3d_pair - project_center2.transpose(-2, -1)
    ).norm(
        dim=-1
    ) ** 2  # BxN

    denominator = 2.0 * torch.sqrt(ray_length_squared1 * ray_length_squared2)
    nominator = (
        ray_length_squared1 + ray_length_squared2 - baseline_length_squared
    )

    # if denominator is zero, angle is zero
    # so we set nominator and denominator as one acos(1) = 0
    nonvalid = denominator <= eps
    nominator = torch.where(nonvalid, torch.ones_like(nominator), nominator)
    denominator = torch.where(
        nonvalid, torch.ones_like(denominator), denominator
    )

    cos_angle = nominator / denominator
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    # rad to deg
    triangles = torch.abs(torch.acos(cos_angle))

    # take the min of (angle, pi-angle)
    # to avoid the effect of acute angles (far away points) and obtuse angles (close points)
    triangles = torch.min(triangles, torch.pi - triangles)
    triangles = triangles * (180.0 / torch.pi)

    return points_3d_pair, cheirality_mask, triangles


def init_BA(
    extrinsics,
    intrinsics,
    extra_params,
    tracks,
    points_3d_pair,
    inlier,
    image_size,
    shared_camera=False,
    init_max_reproj_error=0.5,
    camera_type="SIMPLE_PINHOLE",
):
    """
    This function first optimizes the init point cloud
    and the cameras of its corresponding frames by BA,

    Input:
        extrinsics:         Sx3x4
        intrinsics:         Sx3x3
        tracks:             SxPx2
        points_3d_pair:     (S-1)xPx3
        inlier:             (S-1)xP
    """

    # Find the frame that has the highest inlier (inlier for triangulation angle and cheirality check)
    # Please note the init_idx was defined in 0 to S-1
    init_idx = torch.argmax(inlier.sum(dim=-1)).item()

    # init_indices include the query frame and the frame with highest inlier number (i.e., the init pair)
    # init_idx+1 means shifting to the range of 0 to S
    # TODO: consider initializing by not only a pair, but more frames
    init_indices = [0, init_idx + 1]

    # Pick the camera parameters for the init pair
    toBA_extrinsics = extrinsics[init_indices]
    toBA_intrinsics = intrinsics[init_indices]

    toBA_extra_params = (
        extra_params[init_indices] if extra_params is not None else None
    )
    toBA_tracks = tracks[init_indices]

    # points_3d_pair and inlier has a shape of (S-1), *, *, ...
    toBA_points3D = points_3d_pair[init_idx]
    toBA_masks = inlier[init_idx].unsqueeze(0)
    # all the points are assumed valid at query
    # TODO: remove this assumption in the next version
    toBA_masks = torch.cat([torch.ones_like(toBA_masks), toBA_masks], dim=0)

    # Only if a track has more than 2 inliers,
    # it is viewed as valid
    toBA_valid_track_mask = toBA_masks.sum(dim=0) >= 2
    toBA_masks = toBA_masks[:, toBA_valid_track_mask]
    toBA_points3D = toBA_points3D[toBA_valid_track_mask]
    toBA_tracks = toBA_tracks[:, toBA_valid_track_mask]

    # Convert PyTorch tensors to the format of Pycolmap
    # Prepare for the Bundle Adjustment Optimization
    # NOTE although we use pycolmap for BA here, but any BA library should be able to achieve the same result
    reconstruction = batch_matrix_to_pycolmap(
        toBA_points3D,
        toBA_extrinsics,
        toBA_intrinsics,
        toBA_tracks,
        toBA_masks,
        image_size,
        extra_params=toBA_extra_params,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    # Prepare BA options
    ba_options = prepare_ba_options()

    # Conduct BA
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    # reconstruction = filter_reconstruction(reconstruction)

    # Get the optimized 3D points, extrinsics, and intrinsics
    (points3D_opt, extrinsics_opt, intrinsics_opt, extra_params_opt) = (
        pycolmap_to_batch_matrix(
            reconstruction,
            device=toBA_extrinsics.device,
            camera_type=camera_type,
        )
    )

    # Filter those invalid 3D points
    valid_poins3D_mask, _ = filter_all_points3D(
        points3D_opt,
        toBA_tracks,
        extrinsics_opt,
        intrinsics_opt,
        extra_params_opt,
        check_triangle=False,
        max_reproj_error=init_max_reproj_error,
    )

    points3D_opt = points3D_opt[valid_poins3D_mask]

    # If a 3D point is invalid, all of its 2D matches are invalid
    filtered_valid_track_mask = toBA_valid_track_mask.clone()
    filtered_valid_track_mask[toBA_valid_track_mask] = valid_poins3D_mask

    # Replace the original cameras by the optimized ones
    extrinsics[init_indices] = extrinsics_opt.to(extrinsics.dtype)
    intrinsics[init_indices] = intrinsics_opt.to(intrinsics.dtype)
    if extra_params is not None:
        extra_params[init_indices] = extra_params_opt.to(extra_params.dtype)

    # NOTE: filtered_valid_track_mask or toBA_valid_track_mask?
    return (
        points3D_opt,
        extrinsics,
        intrinsics,
        extra_params,
        filtered_valid_track_mask,
        reconstruction,
        init_idx,
    )


def refine_pose(
    extrinsics,
    intrinsics,
    extra_params,
    inlier,
    points3D,
    tracks,
    valid_track_mask,
    image_size,
    shared_camera=False,
    max_reproj_error=12,
    camera_type="SIMPLE_PINHOLE",
    force_estimate=False,
):
    # extrinsics: Sx3x4
    # intrinsics: Sx3x3
    # inlier: SxP
    # points3D: P' x 3
    # tracks: SxPx2
    # valid_track_mask: P

    S, _, _ = extrinsics.shape
    _, P, _ = tracks.shape

    assert len(intrinsics) == S
    assert inlier.shape[0] == S
    assert inlier.shape[1] == P
    assert len(valid_track_mask) == P

    empty_mask = points3D.abs().sum(-1) <= 0
    if empty_mask.sum() > 0:
        non_empty_mask = ~empty_mask
        tmp_mask = valid_track_mask.clone()
        tmp_mask[valid_track_mask] = non_empty_mask
        valid_track_mask = tmp_mask.clone()
        points3D = points3D[non_empty_mask]

    tracks2D = tracks[:, valid_track_mask]

    # compute reprojection error
    projected_points2D, projected_points_cam = project_3D_points(
        points3D,
        extrinsics,
        intrinsics,
        extra_params=extra_params,
        return_points_cam=True,
    )

    reproj_error = (projected_points2D - tracks2D).norm(dim=-1) ** 2  # sqaure
    # ensure all the points stay in front of the cameras
    reproj_error[projected_points_cam[:, -1] <= 0] = 1e9

    reproj_inlier = reproj_error <= (max_reproj_error**2)

    inlier_nongeo = inlier[:, valid_track_mask]
    inlier_absrefine = torch.logical_and(inlier_nongeo, reproj_inlier)

    inlier_nongeo = inlier_nongeo.cpu().numpy()
    inlier_absrefine = inlier_absrefine.cpu().numpy()
    # P' x 3
    points3D = points3D.cpu().numpy()
    # S x P' x 2
    tracks2D = tracks2D.cpu().numpy()

    estoptions = pycolmap.AbsolutePoseEstimationOptions()
    estoptions.estimate_focal_length = True
    estoptions.ransac.max_error = max_reproj_error

    refoptions = pycolmap.AbsolutePoseRefinementOptions()
    refoptions.refine_focal_length = True
    refoptions.refine_extra_params = True
    refoptions.print_summary = False

    refined_extrinsics = []
    refined_intrinsics = []
    refined_extra_params = [] if extra_params is not None else None

    scale = image_size.max()

    pycamera = None

    for ridx in range(S):
        if pycamera is None or (not shared_camera):
            if camera_type == "SIMPLE_RADIAL":
                pycolmap_intri = np.array(
                    [
                        intrinsics[ridx][0, 0].cpu(),
                        intrinsics[ridx][0, 2].cpu(),
                        intrinsics[ridx][1, 2].cpu(),
                        extra_params[ridx][0].cpu(),
                    ]
                )
            elif camera_type == "SIMPLE_PINHOLE":
                pycolmap_intri = np.array(
                    [
                        intrinsics[ridx][0, 0].cpu(),
                        intrinsics[ridx][0, 2].cpu(),
                        intrinsics[ridx][1, 2].cpu(),
                    ]
                )
            else:
                raise ValueError(
                    f"Camera type {camera_type} is not supported yet"
                )

            pycamera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=ridx,
            )

        if ridx > 0 and shared_camera:
            refoptions.refine_focal_length = False
            refoptions.refine_extra_params = False

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[ridx][:3, :3].cpu()),
            extrinsics[ridx][:3, 3].cpu(),
        )  # Rot and Trans
        points2D = tracks2D[ridx]
        inlier_mask = inlier_absrefine[ridx]

        estimate_abs_pose = False

        if inlier_mask.sum() > 100:
            answer = pycolmap.pose_refinement(
                cam_from_world,
                points2D,
                points3D,
                inlier_mask,
                pycamera,
                refoptions,
            )
            cam_from_world = answer["cam_from_world"]

            intri_mat = pycamera.calibration_matrix()
            focal = intri_mat[0, 0]
            if (focal < 0.1 * scale) or (focal > 30 * scale):
                # invalid focal length
                estimate_abs_pose = True
        else:
            estimate_abs_pose = True
            print(f"Frame {ridx} only has {inlier_mask.sum()} geo_vis inliers")

        if estimate_abs_pose and force_estimate:
            inlier_mask = inlier_nongeo[ridx]

            if inlier_mask.sum() > 50:
                print(
                    f"Estimating absolute poses by visible matches for frame {ridx}"
                )
                estanswer = pycolmap.absolute_pose_estimation(
                    points2D[inlier_mask],
                    points3D[inlier_mask],
                    pycamera,
                    estoptions,
                    refoptions,
                )
                if estanswer is None:
                    estanswer = pycolmap.absolute_pose_estimation(
                        points2D, points3D, pycamera, estoptions, refoptions
                    )
            else:
                print(
                    f"Warning! Estimating absolute poses by non visible matches for frame {ridx}"
                )
                estanswer = pycolmap.absolute_pose_estimation(
                    points2D, points3D, pycamera, estoptions, refoptions
                )

            if estanswer is not None:
                cam_from_world = estanswer["cam_from_world"]

        extri_mat = cam_from_world.matrix()
        intri_mat = pycamera.calibration_matrix()

        refined_extrinsics.append(extri_mat)
        refined_intrinsics.append(intri_mat)
        if extra_params is not None:
            refined_extra_params.append(pycamera.params[3])

    # get the optimized cameras
    refined_extrinsics = torch.from_numpy(np.stack(refined_extrinsics)).to(
        tracks.device
    )
    refined_intrinsics = torch.from_numpy(np.stack(refined_intrinsics)).to(
        tracks.device
    )
    
    if extra_params is not None:
        refined_extra_params = torch.from_numpy(
            np.stack(refined_extra_params)
        ).to(tracks.device)
        if len(refined_extra_params.shape) == 1:
            refined_extra_params = refined_extra_params[:, None]


    valid_frame_mask = get_valid_frame_mask(refined_intrinsics, refined_extrinsics, refined_extra_params, scale)

    if (~valid_frame_mask).sum() > 0:
        print("some frames are invalid after BA refinement")
        refined_extrinsics[~valid_frame_mask] = extrinsics[
            ~valid_frame_mask
        ].to(refined_extrinsics.dtype)
        refined_intrinsics[~valid_frame_mask] = intrinsics[
            ~valid_frame_mask
        ].to(refined_extrinsics.dtype)
        if extra_params is not None:
            refined_extra_params[~valid_frame_mask] = extra_params[
                ~valid_frame_mask
            ].to(refined_extrinsics.dtype)

    return (
        refined_extrinsics,
        refined_intrinsics,
        refined_extra_params,
        valid_frame_mask,
    )


def init_refine_pose(
    extrinsics,
    intrinsics,
    extra_params,
    inlier,
    points3D,
    tracks,
    valid_track_mask_init,
    image_size,
    init_idx,
    max_reproj_error=12,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Refine the extrinsics and intrinsics by points3D and tracks,
    which conducts bundle adjustment but does not modify points3D
    """
    # extrinsics: Sx3x4
    # intrinsics: Sx3x3
    # inlier: (S-1)xP
    # points3D: P' x 3
    # tracks: SxPx2
    # valid_track_mask_init: P

    S, _, _ = extrinsics.shape
    _, P, _ = tracks.shape

    assert len(intrinsics) == S
    assert inlier.shape[0] == (S - 1)
    assert inlier.shape[1] == P
    assert len(valid_track_mask_init) == P

    # TODO check this
    # remove all zeros points3D
    # non_empty_mask = points3D.abs().sum(-1) >0
    # valid_track_mask_tmp = valid_track_mask_init.clone()
    # valid_track_mask_tmp[valid_track_mask_init] = non_empty_mask
    # valid_track_mask_init = valid_track_mask_tmp.clone()

    # Prepare the inlier mask
    inlier_absrefine = torch.cat([torch.ones_like(inlier[0:1]), inlier], dim=0)
    inlier_absrefine = inlier_absrefine[:, valid_track_mask_init]
    inlier_absrefine = inlier_absrefine.cpu().numpy()

    # P' x 3
    points3D = points3D.cpu().numpy()
    # S x P' x 2
    tracks2D = tracks[:, valid_track_mask_init].cpu().numpy()

    refoptions = pycolmap.AbsolutePoseRefinementOptions()
    refoptions.refine_focal_length = True
    refoptions.refine_extra_params = True
    refoptions.print_summary = False

    refined_extrinsics = []
    refined_intrinsics = []
    refined_extra_params = [] if extra_params is not None else None

    pycamera = None
    for ridx in range(S):
        if pycamera is None or (not shared_camera):
            if camera_type == "SIMPLE_RADIAL":
                pycolmap_intri = np.array(
                    [
                        intrinsics[ridx][0, 0].cpu(),
                        intrinsics[ridx][0, 2].cpu(),
                        intrinsics[ridx][1, 2].cpu(),
                        extra_params[ridx][0].cpu(),
                    ]
                )
            elif camera_type == "SIMPLE_PINHOLE":
                pycolmap_intri = np.array(
                    [
                        intrinsics[ridx][0, 0].cpu(),
                        intrinsics[ridx][0, 2].cpu(),
                        intrinsics[ridx][1, 2].cpu(),
                    ]
                )
            else:
                raise ValueError(
                    f"Camera type {camera_type} is not supported yet"
                )

            pycamera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=ridx,
            )

        if ridx > 0 and shared_camera:
            refoptions.refine_focal_length = False
            refoptions.refine_extra_params = False

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[ridx][:3, :3].cpu()),
            extrinsics[ridx][:3, 3].cpu(),
        )  # Rot and Trans
        points2D = tracks2D[ridx]
        inlier_mask = inlier_absrefine[ridx]

        if ridx != 0 and ridx != (init_idx + 1):
            # ridx==0 or ==(init_idx+1) means they are init pair, no need to optimize again
            if inlier_mask.sum() > 50:
                # If too few inliers, ignore it
                # Bundle adjustment without optimizing 3D point
                answer = pycolmap.pose_refinement(
                    cam_from_world,
                    points2D,
                    points3D,
                    inlier_mask,
                    pycamera,
                    refoptions,
                )
                cam_from_world = answer["cam_from_world"]
            else:
                print("This frame only has inliers:", inlier_mask.sum())

        extri_mat = cam_from_world.matrix()
        intri_mat = pycamera.calibration_matrix()
        refined_extrinsics.append(extri_mat)
        refined_intrinsics.append(intri_mat)

        if extra_params is not None:
            refined_extra_params.append(pycamera.params[3])

    # get the optimized cameras
    refined_extrinsics = torch.from_numpy(np.stack(refined_extrinsics)).to(
        tracks.device
    )
    refined_intrinsics = torch.from_numpy(np.stack(refined_intrinsics)).to(
        tracks.device
    )
    if extra_params is not None:
        refined_extra_params = torch.from_numpy(
            np.stack(refined_extra_params)
        ).to(tracks.device)
        if len(refined_extra_params.shape) == 1:
            refined_extra_params = refined_extra_params[:, None]


    scale = image_size.max()

    valid_frame_mask = get_valid_frame_mask(refined_intrinsics, refined_extrinsics, refined_extra_params, scale)

    if (~valid_frame_mask).sum() > 0:
        print("some frames are invalid after BA refinement")
        refined_extrinsics[~valid_frame_mask] = extrinsics[
            ~valid_frame_mask
        ].to(refined_extrinsics.dtype)
        refined_intrinsics[~valid_frame_mask] = intrinsics[
            ~valid_frame_mask
        ].to(refined_extrinsics.dtype)
        if extra_params is not None:
            refined_extra_params[~valid_frame_mask] = extra_params[
                ~valid_frame_mask
            ].to(refined_extrinsics.dtype)

    return (
        refined_extrinsics,
        refined_intrinsics,
        refined_extra_params,
        valid_frame_mask,
    )


def triangulate_multi_view_point_from_tracks(
    cams_from_world, tracks, mask=None
):
    with autocast(dtype=torch.float32):
        B, S, _, _ = cams_from_world.shape
        _, _, N, _ = tracks.shape  # B S N 2
        tracks = tracks.permute(0, 2, 1, 3)

        tracks = tracks.reshape(B * N, S, 2)
        if mask is not None:
            mask = mask.permute(0, 2, 1).reshape(B * N, S)

        cams_from_world = cams_from_world[:, None].expand(-1, N, -1, -1, -1)
        cams_from_world = cams_from_world.reshape(B * N, S, 3, 4)

        (points3d, invalid_cheirality_mask) = (
            triangulate_multi_view_point_batched(
                cams_from_world, tracks, mask, check_cheirality=True
            )
        )

        points3d = points3d.reshape(B, N, 3)
        invalid_cheirality_mask = invalid_cheirality_mask.reshape(B, N)
        cheirality_mask = ~invalid_cheirality_mask
        return points3d, cheirality_mask


def triangulate_tracks(
    extrinsics,
    tracks_normalized,
    max_ransac_iters=256,
    lo_num=50,
    max_angular_error=2,
    min_tri_angle=1.5,
    track_vis=None,
    track_score=None,
    max_tri_points_num=200000,
):
    """
    Triangulate Tracks. If necessary, process tracks in smaller chunks to avoid memory issues during triangulation.

    Args:
        extrinsics (torch.Tensor): S x 3 x 4 tensor, where S is the number of frames.
                                   Extrinsics follow the convention of OpenCV R|t.
        tracks_normalized (torch.Tensor): S x N x 2 tensor, where N is the number of tracks.
                                          Tracks are normalized by intrinsics (i.e., cam_from_img).
                                          Generally, if not considering distortion, it is in the form:
                                          tracks_normalized = (pred_tracks - principal_point) / focal_length.
                                          Refer to the following link for details:
                                          https://github.com/facebookresearch/vggsfm/blob/5899a99ff263519c254110fde2036e5ce11f6874/vggsfm/utils/triangulation_helpers.py#L310
        lo_num (int): The number of trials for local refinement for LORANSAC.
                      If your GPU memory can afford it, higher is better, but usually doesn't need to go beyond 300.
        max_angular_error (float): The maximum angular error for a track to be considered as an inlier.
        min_tri_angle (float): The minimum triangulation angle to avoid some triangulation points leading to infinity.
        track_vis (torch.Tensor, optional): Tensor to filter out low-quality correspondences.
        track_score (torch.Tensor, optional): Tensor to filter out low-quality correspondences.

    Note:
        To filter out low-quality correspondences, use:
        invalid_vis_conf_mask = torch.logical_or(track_vis <= 0.05, track_score <= 0.5)
    """

    all_tri_points_num = extrinsics.shape[0] * tracks_normalized.shape[1]

    if all_tri_points_num > max_tri_points_num:
        print('Triangulate tracks in chunks to fit in memory')
        
        num_splits = (
            all_tri_points_num + max_tri_points_num - 1
        ) // max_tri_points_num

        split_tracks_normalized = torch.chunk(
            tracks_normalized, num_splits, dim=1
        )
        split_track_vis = (
            torch.chunk(track_vis, num_splits, dim=1)
            if track_vis is not None
            else [None] * num_splits
        )
        split_track_score = (
            torch.chunk(track_score, num_splits, dim=1)
            if track_score is not None
            else [None] * num_splits
        )

        triangulated_points_list = []
        inlier_num_list = []
        inlier_mask_list = []

        for i in range(num_splits):
            triangulated_points, inlier_num, inlier_mask = (
                triangulate_tracks_single_chunk(
                    extrinsics,
                    split_tracks_normalized[i],
                    max_ransac_iters=max_ransac_iters,
                    lo_num=lo_num,
                    max_angular_error=max_angular_error,
                    min_tri_angle=min_tri_angle,
                    track_vis=split_track_vis[i],
                    track_score=split_track_score[i],
                )
            )
            triangulated_points_list.append(triangulated_points)
            inlier_num_list.append(inlier_num)
            inlier_mask_list.append(inlier_mask)

        triangulated_points = torch.cat(triangulated_points_list, dim=0)
        inlier_num = torch.cat(inlier_num_list, dim=0)
        inlier_mask = torch.cat(inlier_mask_list, dim=0)
    else:
        triangulated_points, inlier_num, inlier_mask = (
            triangulate_tracks_single_chunk(
                extrinsics,
                tracks_normalized,
                max_ransac_iters,
                lo_num,
                max_angular_error,
                min_tri_angle,
                track_vis,
                track_score,
            )
        )

    return triangulated_points, inlier_num, inlier_mask


def triangulate_tracks_single_chunk(
    extrinsics,
    tracks_normalized,
    max_ransac_iters=256,
    lo_num=50,
    max_angular_error=2,
    min_tri_angle=1.5,
    track_vis=None,
    track_score=None,
):
    """
    This function conduct triangulation over all the input frames

    It adopts LORANSAC, which means
    (1) first triangulate 3d points by random 2-view pairs
    (2) compute the inliers of these triangulated points
    (3) do re-triangulation using the inliers
    (4) check the ones with most inliers
    """
    max_rad_error = max_angular_error * (torch.pi / 180)

    with autocast(dtype=torch.float32):
        tracks_normalized = tracks_normalized.transpose(0, 1)
        B, S, _ = tracks_normalized.shape
        extrinsics_expand = extrinsics[None].expand(B, -1, -1, -1)

        point_per_sample = 2  # first triangulate points by 2 points

        ransac_idx = generate_combinations(S)
        ransac_idx = torch.from_numpy(ransac_idx).to(extrinsics.device)

        # Prevent max_ransac_iters from being unnecessarily high
        if max_ransac_iters > len(ransac_idx):
            max_ransac_iters = len(ransac_idx)
        else:
            ransac_idx = ransac_idx[
                torch.randperm(len(ransac_idx))[:max_ransac_iters]
            ]
        lo_num = lo_num if max_ransac_iters >= lo_num else max_ransac_iters

        # Prepare the input
        points_ransac = tracks_normalized[:, ransac_idx].view(
            B * max_ransac_iters, point_per_sample, 2
        )
        extrinsics_ransac = extrinsics_expand[:, ransac_idx].view(
            B * max_ransac_iters, point_per_sample, 3, 4
        )

        # triangulated_points: (B * max_ransac_iters) x 3
        # tri_angles: (B * max_ransac_iters) x (point_per_sample * point_per_sample)
        # invalid_che_mask: (B * max_ransac_iters)
        (triangulated_points, tri_angles, invalid_che_mask) = (
            triangulate_multi_view_point_batched(
                extrinsics_ransac,
                points_ransac,
                compute_tri_angle=True,
                check_cheirality=True,
            )
        )

        triangulated_points = triangulated_points.reshape(
            B, max_ransac_iters, 3
        )
        invalid_che_mask = invalid_che_mask.reshape(B, max_ransac_iters)

        # if any of the pair fits the minimum triangulation angle, we view it as valid
        tri_masks = (tri_angles >= min_tri_angle).any(dim=-1)
        invalid_tri_mask = (~tri_masks).reshape(B, max_ransac_iters)

        # a point is invalid if it does not meet the minimum triangulation angle or fails the cheirality test
        # B x max_ransac_iters

        invalid_mask = torch.logical_or(invalid_tri_mask, invalid_che_mask)

        # Please note angular error is not triangulation angle
        # For a quick understanding,
        # angular error: lower the better
        # triangulation angle: higher the better (within a reasonable range)
        angular_error, _ = calculate_normalized_angular_error_batched(
            tracks_normalized.transpose(0, 1),
            triangulated_points.permute(1, 0, 2),
            extrinsics,
        )
        # max_ransac_iters x S x B -> B x max_ransac_iters x S
        angular_error = angular_error.permute(2, 0, 1)

        # If some tracks are invalid, give them a very high error
        angular_error[invalid_mask] = angular_error[invalid_mask] + torch.pi

        # Also, we hope the tracks also meet the visibility and score requirement
        # logical_or: invalid if does not meet any requirement
        if track_score is not None:
            invalid_vis_conf_mask = torch.logical_or(
                track_vis <= 0.05, track_score <= 0.5
            )
        else:
            invalid_vis_conf_mask = track_vis <= 0.05

        invalid_vis_conf_mask = invalid_vis_conf_mask.permute(1, 0)
        angular_error[
            invalid_vis_conf_mask[:, None].expand(-1, max_ransac_iters, -1)
        ] += torch.pi

        # wow, finally, inlier
        inlier_mask = (angular_error) <= (max_rad_error)

        #############################################################################
        # LOCAL REFINEMENT

        # Triangulate based on the inliers
        # and compute the errors
        (lo_triangulated_points, _, lo_angular_error) = (
            local_refine_and_compute_error(
                tracks_normalized,
                extrinsics,
                extrinsics_expand,
                inlier_mask,
                lo_num,
                min_tri_angle,
                invalid_vis_conf_mask,
                max_rad_error=max_rad_error,
            )
        )

        # Refine it again
        # if you want to, you can repeat the local refine more and more
        lo_num_sec = 10
        if lo_num <= lo_num_sec:
            lo_num_sec = lo_num

        lo_inlier_mask = (lo_angular_error) <= (max_rad_error)
        (lo_triangulated_points_2, _, lo_angular_error_2) = (
            local_refine_and_compute_error(
                tracks_normalized,
                extrinsics,
                extrinsics_expand,
                lo_inlier_mask,
                lo_num_sec,
                min_tri_angle,
                invalid_vis_conf_mask,
                max_rad_error=max_rad_error,
            )
        )

        # combine the first and second local refinement results
        lo_num += lo_num_sec
        lo_triangulated_points = torch.cat(
            [lo_triangulated_points, lo_triangulated_points_2], dim=1
        )
        lo_angular_error = torch.cat(
            [lo_angular_error, lo_angular_error_2], dim=1
        )
        # lo_tri_angles = torch.cat([lo_tri_angles, lo_tri_angles_2], dim=1)
        #############################################################################

        all_triangulated_points = torch.cat(
            [triangulated_points, lo_triangulated_points], dim=1
        )
        all_angular_error = torch.cat([angular_error, lo_angular_error], dim=1)

        (residual_indicator, inlier_num_all, inlier_mask_all) = (
            calculate_residual_indicator(
                all_angular_error,
                max_rad_error,
                check=True,
                nanvalue=2 * torch.pi,
            )
        )

        batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)

        best_indices = torch.argmax(residual_indicator, dim=1)

        # Pick the triangulated points with most inliers
        best_triangulated_points = all_triangulated_points[
            batch_index[:, 0], best_indices
        ]
        best_inlier_num = inlier_num_all[batch_index[:, 0], best_indices]
        best_inlier_mask = inlier_mask_all[batch_index[:, 0], best_indices]

    return best_triangulated_points, best_inlier_num, best_inlier_mask


def local_refine_and_compute_error(
    tracks_normalized,
    extrinsics,
    extrinsics_expand,
    inlier_mask,
    lo_num,
    min_tri_angle,
    invalid_vis_conf_mask,
    max_rad_error,
):
    B, S, _ = tracks_normalized.shape

    inlier_num = inlier_mask.sum(dim=-1)
    sorted_values, sorted_indices = torch.sort(
        inlier_num, dim=1, descending=True
    )

    # local refinement
    (lo_triangulated_points, lo_tri_masks, lo_invalid_che_mask) = (
        local_refinement_tri(
            tracks_normalized,
            extrinsics_expand,
            min_tri_angle,
            inlier_mask,
            sorted_indices,
            lo_num=lo_num,
        )
    )

    # lo_tri_masks = (lo_tri_angles >= min_tri_angle).any(dim=-1)
    lo_invalid_tri_mask = (~lo_tri_masks).reshape(B, lo_num)

    lo_invalid_mask = torch.logical_or(lo_invalid_tri_mask, lo_invalid_che_mask)

    lo_angular_error, _ = calculate_normalized_angular_error_batched(
        tracks_normalized.transpose(0, 1),
        lo_triangulated_points.permute(1, 0, 2),
        extrinsics,
    )
    lo_angular_error = lo_angular_error.permute(2, 0, 1)

    # avoid nan and inf
    lo_angular_error = torch.nan_to_num(
        lo_angular_error,
        nan=100 * torch.pi,
        posinf=100 * torch.pi,
        neginf=100 * torch.pi,
    )

    # penalty to invalid points
    lo_angular_error[lo_invalid_mask] = (
        lo_angular_error[lo_invalid_mask] + torch.pi
    )

    lo_angular_error[
        invalid_vis_conf_mask[:, None].expand(-1, lo_num, -1)
    ] += torch.pi

    return lo_triangulated_points, lo_tri_masks, lo_angular_error


def global_BA(
    triangulated_points,
    valid_tracks,
    pred_tracks,
    inlier_mask,
    extrinsics,
    intrinsics,
    extra_params,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    ba_options = prepare_ba_options()

    # triangulated_points
    BA_points = triangulated_points[valid_tracks]
    BA_tracks = pred_tracks[:, valid_tracks]
    BA_inlier_masks = inlier_mask[valid_tracks].transpose(0, 1)
    reconstruction = batch_matrix_to_pycolmap(
        BA_points,
        extrinsics,
        intrinsics,
        BA_tracks,
        BA_inlier_masks,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
        extra_params=extra_params,
    )

    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction = filter_reconstruction(reconstruction)

    extrinsics_tmp = extrinsics.clone()
    intrinsics_tmp = intrinsics.clone()
    extra_params_tmp = (
        extra_params.clone() if extra_params is not None else None
    )

    (points3D_opt, extrinsics, intrinsics, extra_params) = (
        pycolmap_to_batch_matrix(
            reconstruction, device=BA_points.device, camera_type=camera_type
        )
    )

    if (intrinsics[:, 0, 0] < 0).any():
        invalid_mask = intrinsics[:, 0, 0] < 0
        extrinsics[invalid_mask] = extrinsics_tmp[invalid_mask]
        intrinsics[invalid_mask] = intrinsics_tmp[invalid_mask]
        if extra_params is not None:
            extra_params[invalid_mask] = extra_params_tmp[invalid_mask]

    return points3D_opt, extrinsics, intrinsics, extra_params, reconstruction


def iterative_global_BA(
    pred_tracks,
    intrinsics,
    extrinsics,
    pred_vis,
    pred_score,
    valid_tracks,
    points3D_opt,
    image_size,
    shared_camera=False,
    min_valid_track_length=2,
    max_reproj_error=1,
    ba_options=None,
    lastBA=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,  # Add extra_params to the function signature
):
    # normalize points from pixel

    tracks_normalized_refined = cam_from_img(
        pred_tracks, intrinsics, extra_params
    )

    # triangulate tracks by LORANSAC
    (best_triangulated_points, best_inlier_num, best_inlier_mask) = (
        triangulate_tracks(
            extrinsics,
            tracks_normalized_refined,
            track_vis=pred_vis,
            track_score=pred_score,
            max_ransac_iters=128,
        )
    )

    best_triangulated_points[valid_tracks] = points3D_opt

    # well do we need this? best_inlier_mask may be enough already
    valid_poins3D_mask, filtered_inlier_mask = filter_all_points3D(
        best_triangulated_points,
        pred_tracks,
        extrinsics,
        intrinsics,
        extra_params=extra_params,  # Pass extra_params to filter_all_points3D
        max_reproj_error=max_reproj_error,
        return_detail=True,
    )

    valid_tracks = filtered_inlier_mask.sum(dim=0) >= min_valid_track_length
    BA_points = best_triangulated_points[valid_tracks]
    BA_tracks = pred_tracks[:, valid_tracks]
    BA_inlier_masks = filtered_inlier_mask[:, valid_tracks]

    if ba_options is None:
        ba_options = pycolmap.BundleAdjustmentOptions()

    reconstruction = batch_matrix_to_pycolmap(
        BA_points,
        extrinsics,
        intrinsics,
        BA_tracks,
        BA_inlier_masks,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
        extra_params=extra_params,  # Pass extra_params to batch_matrix_to_pycolmap
    )
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    reconstruction = filter_reconstruction(reconstruction)

    extrinsics_tmp = extrinsics.clone()
    intrinsics_tmp = intrinsics.clone()
    extra_params_tmp = (
        extra_params.clone() if extra_params is not None else None
    )  # Clone extra_params if not None

    (points3D_opt, extrinsics, intrinsics, extra_params) = (
        pycolmap_to_batch_matrix(
            reconstruction, device=pred_tracks.device, camera_type=camera_type
        )
    )

    if (intrinsics[:, 0, 0] < 0).any():
        invalid_mask = intrinsics[:, 0, 0] < 0
        extrinsics[invalid_mask] = extrinsics_tmp[invalid_mask]
        intrinsics[invalid_mask] = intrinsics_tmp[invalid_mask]
        if extra_params is not None:
            extra_params[invalid_mask] = extra_params_tmp[
                invalid_mask
            ]  # Restore extra_params if invalid

    valid_poins3D_mask, filtered_inlier_mask = filter_all_points3D(
        points3D_opt,
        pred_tracks[:, valid_tracks],
        extrinsics,
        intrinsics,
        extra_params=extra_params,  # Pass extra_params to filter_all_points3D
        max_reproj_error=max_reproj_error,
        return_detail=True,
    )

    valid_tracks_afterBA = (
        filtered_inlier_mask.sum(dim=0) >= min_valid_track_length
    )
    valid_tracks_tmp = valid_tracks.clone()
    valid_tracks_tmp[valid_tracks] = valid_tracks_afterBA
    valid_tracks = valid_tracks_tmp.clone()
    points3D_opt = points3D_opt[valid_tracks_afterBA]
    BA_inlier_masks = filtered_inlier_mask[:, valid_tracks_afterBA]

    if lastBA:
        reconstruction = batch_matrix_to_pycolmap(
            points3D_opt,
            extrinsics,
            intrinsics,
            pred_tracks[:, valid_tracks],
            BA_inlier_masks,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
            extra_params=extra_params, 
        )

        reconstruction = filter_reconstruction(reconstruction)
    return (
        points3D_opt,
        extrinsics,
        intrinsics,
        extra_params,  # Return extra_params
        valid_tracks,
        BA_inlier_masks,
        # filtered_inlier_mask,
        reconstruction,
    )


def filter_reconstruction(reconstruction, filter_points=False):
    if filter_points:
        observation_manager = pycolmap.ObservationManager(reconstruction)
        observation_manager.filter_all_points3D(4.0, 1.5)
        observation_manager.filter_observations_with_negative_depth()
    reconstruction.normalize(5.0, 0.1, 0.9, True)
    return reconstruction



def get_valid_frame_mask(intrinsics, extrinsics, extra_params, scale):
    valid_param_mask = torch.logical_and(
        intrinsics[:, 0, 0] >= 0.1 * scale,
        intrinsics[:, 0, 0] <= 30 * scale,
    )

    if extra_params is not None:
        # Do not allow the extra params to be too large

        if len(extra_params.shape) == 1:
            extra_params = extra_params[:, None]

        valid_param_mask = torch.logical_and(
            valid_param_mask, (extra_params.abs() <= 1.0).all(dim=-1)
        )

    valid_trans_mask = (extrinsics[:, :, 3].abs() <= 30).all(-1)

    valid_frame_mask = torch.logical_and(valid_param_mask, valid_trans_mask)
    
    return valid_frame_mask
