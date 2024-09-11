# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

import pycolmap
import numpy as np
from torch.cuda.amp import autocast

# #####################
from .utils import get_EFP

from ..utils.triangulation import (
    triangulate_by_pair,
    init_BA,
    init_refine_pose,
    refine_pose,
    triangulate_tracks,
    global_BA,
    iterative_global_BA,
)
from ..utils.triangulation_helpers import filter_all_points3D, cam_from_img


class Triangulator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        """
        The module for triangulation and BA adjustment 
        
        NOTE After VGGSfM v1.1, we remove the learnable parameters of Triangulator.
        Now this uses RANSAC DLT to triangulate and bundle adjust.
        We plan to bring back deep Triangulator in v2.1
        Check this for more details: 
        https://github.com/facebookresearch/vggsfm/issues/47
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
        init_max_reproj_error=0.5,
        BA_iters=2,
        shared_camera=False,
        max_reproj_error=4,
        init_tri_angle_thres=16,
        min_valid_track_length=3,
        robust_refine=2,
        extract_color=True,
        camera_type="SIMPLE_PINHOLE",
    ):
        """
        Conduct triangulation and bundle adjustment given
        the input pred_cameras, pred_tracks, pred_vis, and pred_score

        We use the pred_cameras from camera_predictor but it can be any init.
        Please note pred_tracks are defined in pixels.
        """
        # for safety
        torch.cuda.empty_cache()

        device = pred_tracks.device

        with autocast(dtype=torch.float32):
            B, S, _, H, W = images.shape
            _, _, N, _ = pred_tracks.shape

            assert (
                B == 1
            )  # The released implementation now only supports batch=1 during inference

            image_size = torch.tensor(
                [W, H], dtype=pred_tracks.dtype, device=device
            )
            # extrinsics: B x S x 3 x 4
            # intrinsics: B x S x 3 x 3
            # focal_length, principal_point : B x S x 2

            extrinsics, intrinsics = get_EFP(pred_cameras, image_size, B, S)

            extrinsics = extrinsics.double()
            inlier_fmat = preliminary_dict["fmat_inlier_mask"]

            # Remove B dim
            # To simplify the code, now we only support B==1 during inference
            extrinsics = extrinsics[0]
            intrinsics = intrinsics[0]
            pred_tracks = pred_tracks[0]
            pred_vis = pred_vis[0]
            pred_score = pred_score[0]
            inlier_fmat = inlier_fmat[0]

            if shared_camera:
                fx = intrinsics[:, 0, 0].mean()
                fy = intrinsics[:, 1, 1].mean()

                intrinsics[:, 0, 0] = fx
                intrinsics[:, 1, 1] = fy

            extra_params = None
            if camera_type == "SIMPLE_RADIAL":
                extra_params = torch.zeros_like(extrinsics[:, 0, 0:1])

            tracks_normalized = cam_from_img(pred_tracks, intrinsics)
            # Visibility inlier
            inlier_vis = pred_vis > 0.05  # TODO: avoid hardcoded
            inlier_vis = inlier_vis[1:]

            # Intersection of inlier_fmat and inlier_vis
            inlier_geo_vis = torch.logical_and(inlier_fmat, inlier_vis)

            # For initialization
            # we first triangulate a point cloud for each pair of query-reference images,
            # i.e., we have S-1 3D point clouds
            # points_3d_pair: S-1 x N x 3
            (points_3d_pair, cheirality_mask_pair, triangle_value_pair) = (
                triangulate_by_pair(extrinsics[None], tracks_normalized[None])
            )

            # Check which point cloud can provide sufficient inliers
            # that pass the triangulation angle and cheirality check
            # Pick the highest inlier_geo_vis one as the initial point cloud
            inlier_total, valid_tri_angle_thres = find_best_initial_pair(
                inlier_geo_vis,
                cheirality_mask_pair,
                triangle_value_pair,
                init_tri_angle_thres,
            )

            # Conduct BA on the init point cloud and init pair
            (
                points3D_init,
                extrinsics,
                intrinsics,
                extra_params,
                track_init_mask,
                reconstruction,
                init_idx,
            ) = init_BA(
                extrinsics,
                intrinsics,
                extra_params,
                pred_tracks,
                points_3d_pair,
                inlier_total,
                image_size,
                shared_camera=shared_camera,
                init_max_reproj_error=init_max_reproj_error,
                camera_type=camera_type,
            )
            print("Finished init BA")

            # Given we have a well-conditioned point cloud,
            # we can optimize all the cameras by absolute pose refinement as in
            # https://github.com/colmap/colmap/blob/4ced4a5bc72fca93a2ffaea2c7e193bc62537416/src/colmap/estimators/pose.cc#L207
            # Basically it is a bundle adjustment without optmizing 3D points
            # It is fine even this step fails

            (extrinsics, intrinsics, extra_params, valid_param_mask) = (
                init_refine_pose(
                    extrinsics,
                    intrinsics,
                    extra_params,
                    inlier_geo_vis,
                    points3D_init,
                    pred_tracks,
                    track_init_mask,
                    image_size,
                    init_idx,
                    shared_camera=shared_camera,
                    camera_type=camera_type,
                )
            )
            print("Finished init refine pose")  
            
            (
                points3D,
                extrinsics,
                intrinsics,
                extra_params,
                valid_tracks,
                reconstruction,
            ) = self.triangulate_tracks_and_BA(
                pred_tracks,
                intrinsics,
                extrinsics,
                extra_params,
                pred_vis,
                pred_score,
                image_size,
                min_valid_track_length,
                max_reproj_error,
                shared_camera=shared_camera,
                camera_type=camera_type,
            )
            print("Finished track triangulation and BA")


            if robust_refine > 0:
                for refine_idx in range(robust_refine):
                    # Helpful for some turnable videos
                    inlier_vis_all = pred_vis > 0.05

                    force_estimate = refine_idx == (robust_refine - 1)

                    (extrinsics, intrinsics, extra_params, valid_param_mask) = (
                        refine_pose(
                            extrinsics,
                            intrinsics,
                            extra_params,
                            inlier_vis_all,
                            points3D,
                            pred_tracks,
                            valid_tracks,
                            image_size,
                            force_estimate=force_estimate,
                            shared_camera=shared_camera,
                            camera_type=camera_type,
                        )
                    )

                    (
                        points3D,
                        extrinsics,
                        intrinsics,
                        extra_params,
                        valid_tracks,
                        reconstruction,
                    ) = self.triangulate_tracks_and_BA(
                        pred_tracks,
                        intrinsics,
                        extrinsics,
                        extra_params,
                        pred_vis,
                        pred_score,
                        image_size,
                        min_valid_track_length,
                        max_reproj_error,
                        shared_camera=shared_camera,
                        camera_type=camera_type,
                    )
                    print(f"Finished robust refine {refine_idx}")

            ba_options = pycolmap.BundleAdjustmentOptions()
            ba_options.print_summary = False

            print(f"Running iterative BA by {BA_iters} times")
            for BA_iter in range(BA_iters):
                if BA_iter == (BA_iters - 1):
                    ba_options.print_summary = True
                    lastBA = True
                else:
                    lastBA = False

                (
                    points3D,
                    extrinsics,
                    intrinsics,
                    extra_params,
                    valid_tracks,
                    BA_inlier_masks,
                    reconstruction,
                ) = iterative_global_BA(
                    pred_tracks,
                    intrinsics,
                    extrinsics,
                    pred_vis,
                    pred_score,
                    valid_tracks,
                    points3D,
                    image_size,
                    lastBA=lastBA,
                    extra_params=extra_params,
                    shared_camera=shared_camera,
                    min_valid_track_length=min_valid_track_length,
                    max_reproj_error=max_reproj_error,
                    ba_options=ba_options,
                    camera_type=camera_type,
                )
                
                print(f"Finished iterative BA {BA_iter}")
                
                max_reproj_error = max_reproj_error // 2
                if max_reproj_error <= 1:
                    max_reproj_error = 1

            rot_BA = extrinsics[:, :3, :3]
            trans_BA = extrinsics[:, :3, 3]

            # find the invalid predictions
            scale = image_size.max()
            valid_param_mask = torch.logical_and(
                intrinsics[:, 0, 0] >= 0.1 * scale,
                intrinsics[:, 0, 0] <= 30 * scale,
            )
            if extra_params is not None:
                valid_extra_params_mask = (extra_params.abs() <= 1.0).all(-1)
                valid_param_mask = torch.logical_and(
                    valid_param_mask, valid_extra_params_mask
                )

            valid_trans_mask = (trans_BA.abs() <= 30).all(-1)
            valid_frame_mask = torch.logical_and(
                valid_param_mask, valid_trans_mask
            )

            valid_2D_mask = torch.ones_like(pred_tracks[..., 0]).bool()
            valid_2D_mask[:, ~valid_tracks] = False
            valid_2D_mask[:, valid_tracks] = BA_inlier_masks

            if extract_color:
                from vggsfm.models.utils import sample_features4d

                pred_track_rgb = sample_features4d(
                    images.squeeze(0), pred_tracks
                )

                valid_track_rgb = pred_track_rgb[:, valid_tracks]

                sum_rgb = (
                    BA_inlier_masks.float()[..., None] * valid_track_rgb
                ).sum(dim=0)
                points3D_rgb = sum_rgb / BA_inlier_masks.sum(dim=0)[:, None]

                if points3D_rgb.shape[0] == max(reconstruction.point3D_ids()):
                    for point3D_id in reconstruction.points3D:
                        color_255 = (
                            points3D_rgb[point3D_id - 1].cpu().numpy() * 255
                        )
                        reconstruction.points3D[point3D_id].color = np.round(
                            color_255
                        ).astype(np.uint8)
                else:
                    print(
                        "Cannot save point rgb colors to colmap reconstruction object. Please file an issue in github. "
                    )
                    import pdb

                    pdb.set_trace()
            else:
                points3D_rgb = None

            return (
                extrinsics,
                intrinsics,
                extra_params,
                points3D,
                points3D_rgb,
                reconstruction,
                valid_frame_mask,
                valid_2D_mask,
                valid_tracks,
            )

    def triangulate_tracks_and_BA(
        self,
        pred_tracks,
        intrinsics,
        extrinsics,
        extra_params,
        pred_vis,
        pred_score,
        image_size,
        min_valid_track_length,
        max_reproj_error=4,
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
    ):
        """ """
        # Normalize the tracks

        tracks_normalized_refined = cam_from_img(
            pred_tracks, intrinsics, extra_params
        )

        # Conduct triangulation to all the frames
        # We adopt LORANSAC here again

        (best_triangulated_points, best_inlier_num, best_inlier_mask) = (
            triangulate_tracks(
                extrinsics,
                tracks_normalized_refined,  # TxNx2
                track_vis=pred_vis,  # TxN
                track_score=pred_score,  # TxN
            )
        )

        # Determine valid tracks based on inlier numbers
        valid_tracks = best_inlier_num >= min_valid_track_length

        # Perform global bundle adjustment
        (points3D, extrinsics, intrinsics, extra_params, reconstruction) = (
            global_BA(
                best_triangulated_points,
                valid_tracks,
                pred_tracks,
                best_inlier_mask,
                extrinsics,
                intrinsics,
                extra_params,
                image_size,
                shared_camera=shared_camera,
                camera_type=camera_type,
            )
        )

        valid_poins3D_mask, _ = filter_all_points3D(
            points3D,
            pred_tracks[:, valid_tracks],
            extrinsics,
            intrinsics,
            extra_params,
            check_triangle=False,
            max_reproj_error=max_reproj_error,
        )
        points3D = points3D[valid_poins3D_mask]

        valid_tracks_tmp = valid_tracks.clone()
        valid_tracks_tmp[valid_tracks] = valid_poins3D_mask
        valid_tracks = valid_tracks_tmp.clone()

        return (
            points3D,
            extrinsics,
            intrinsics,
            extra_params,
            valid_tracks,
            reconstruction,
        )


def find_best_initial_pair(
    inlier_geo_vis,
    cheirality_mask_pair,
    triangle_value_pair,
    init_tri_angle_thres,
):
    """
    Find the initial point cloud by checking which point cloud can provide sufficient inliers
    that pass the triangulation angle and cheirality check.
    """
    trial_count = 0
    N = inlier_geo_vis.shape[-1]
    while trial_count < 5:
        # If no success, relax the constraint
        # try at most 5 times
        triangle_mask = triangle_value_pair >= init_tri_angle_thres
        inlier_total = torch.logical_and(inlier_geo_vis, cheirality_mask_pair)
        inlier_total = torch.logical_and(inlier_total, triangle_mask)
        inlier_num_per_frame = inlier_total.sum(dim=-1)

        max_num_inlier = inlier_num_per_frame.max()
        max_num_inlier_ratio = max_num_inlier / N

        # We accept a pair only when the number of inliers and the ratio
        # is higher than a threshold
        if (max_num_inlier >= 100) and (max_num_inlier_ratio >= 0.25):
            break

        if init_tri_angle_thres < 2:
            break

        init_tri_angle_thres = init_tri_angle_thres // 2
        trial_count += 1

    return inlier_total, init_tri_angle_thres
