
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
import copy
import kornia
import pycolmap
from torch.cuda.amp import autocast

from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


# #####################
# from ..two_view_geo.utils import inlier_by_fundamental
from ..utils.triangulation import (
    create_intri_matrix,
    triangulate_by_pair,
    init_BA,
    init_refine_pose,
    refine_pose,
    triangulate_tracks,
    global_BA,
    iterative_global_BA,
)
from ..utils.triangulation_helpers import filter_all_points3D

from .utils import get_EFP



class Triangulator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        """
        The module for triangulation and BA adjustment 
        
        NOTE After VGGSfM v1.1, we remove the learnable parameters of Triangulator
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
        init_max_reproj_error=0.5,
        BA_iters=2,
        max_reproj_error=4,
        init_tri_angle_thres=16,
        min_valid_track_length=3,
        image_paths=None,
        crop_params=None,
        cfg= None,
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

            assert B == 1  # The released implementation now only supports batch=1 during inference

            image_size = torch.tensor([W, H], dtype=pred_tracks.dtype, device=device)
            # extrinsics: B x S x 3 x 4
            # intrinsics: B x S x 3 x 3
            # focal_length, principal_point : B x S x 2
            
            extrinsics, intrinsics, _, _ = get_EFP(
                pred_cameras, image_size, B, S
            )
                
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
            
            
            
            

            tracks_normalized = normalize_tracks(pred_tracks, intrinsics)
            # Visibility inlier
            inlier_vis = pred_vis > 0.05  # TODO: avoid hardcoded
            inlier_vis = inlier_vis[1:]

            # Intersection of inlier_fmat and inlier_vis
            inlier_geo_vis = torch.logical_and(inlier_fmat, inlier_vis)

            # For initialization
            # we first triangulate a point cloud for each pair of query-reference images,
            # i.e., we have S-1 3D point clouds
            # points_3d_pair: S-1 x N x 3
            points_3d_pair, cheirality_mask_pair, triangle_value_pair = triangulate_by_pair(
                extrinsics[None], tracks_normalized[None]
            )


            # Check which point cloud can provide sufficient inliers
            # that pass the triangulation angle and cheirality check
            # Pick the highest inlier_geo_vis one as the initial point cloud
            trial_count = 0
            while trial_count < 5:
                # If no success, relax the constraint
                # try at most 5 times
                triangle_mask = triangle_value_pair >= init_tri_angle_thres
                inlier_total = torch.logical_and(inlier_geo_vis, cheirality_mask_pair)
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


            # Conduct BA on the init point cloud and init pair
            points3D_init, extrinsics, intrinsics, track_init_mask, reconstruction, init_idx = init_BA(
                extrinsics, intrinsics, pred_tracks, points_3d_pair, inlier_total, image_size, 
                init_max_reproj_error=init_max_reproj_error
            )
            
            

            # Given we have a well-conditioned point cloud,
            # we can optimize all the cameras by absolute pose refinement as in
            # https://github.com/colmap/colmap/blob/4ced4a5bc72fca93a2ffaea2c7e193bc62537416/src/colmap/estimators/pose.cc#L207
            # Basically it is a bundle adjustment without optmizing 3D points
            # It is fine even this step fails
            
            extrinsics, intrinsics, valid_intri_mask = init_refine_pose(
                extrinsics, intrinsics, 
                inlier_geo_vis, 
                points3D_init, pred_tracks, track_init_mask, image_size, init_idx
            )
            

            points3D, extrinsics, intrinsics, valid_tracks, reconstruction = self.triangulate_tracks_and_BA(pred_tracks, 
                                                            intrinsics, extrinsics, pred_vis, 
                                                            pred_score, image_size, device, min_valid_track_length)


            if cfg.robust_refine >0:
                for refine_idx in range(cfg.robust_refine):
                    # Helpful for some turnable videos
                    inlier_vis_all = pred_vis > 0.05
                    
                    force_estimate = refine_idx==(cfg.robust_refine-1)

                    extrinsics, intrinsics, valid_intri_mask = refine_pose(
                        extrinsics, intrinsics, 
                        inlier_vis_all, 
                        points3D, pred_tracks, valid_tracks, image_size, force_estimate = force_estimate
                    )
                    
                    points3D, extrinsics, intrinsics, valid_tracks, reconstruction = self.triangulate_tracks_and_BA(pred_tracks, 
                                                                    intrinsics, extrinsics, pred_vis, 
                                                                    pred_score, image_size, device, min_valid_track_length)


            # try:
            ba_options = pycolmap.BundleAdjustmentOptions()
            ba_options.print_summary = False

            print(f"Running iterative BA by {BA_iters} times")
            for BA_iter in range(BA_iters):
                if BA_iter == (BA_iters - 1):
                    ba_options.print_summary = True
                    lastBA= True
                else:
                    lastBA = False
                    
                try:
                    points3D, extrinsics, intrinsics, valid_tracks, BA_inlier_masks, reconstruction = iterative_global_BA(
                        pred_tracks,
                        intrinsics,
                        extrinsics,
                        pred_vis,
                        pred_score,
                        valid_tracks,
                        points3D,
                        image_size,
                        lastBA=lastBA,
                        min_valid_track_length=min_valid_track_length,
                        max_reproj_error=max_reproj_error,
                        ba_options=ba_options,
                    )           
                    max_reproj_error = max_reproj_error//2
                    if max_reproj_error<=1: max_reproj_error = 1  
                except:
                    print(f"Oh BA fails at iter {BA_iter}! Careful")

            rot_BA = extrinsics[:, :3, :3]
            trans_BA = extrinsics[:, :3, 3]

            # find the invalid predictions
            scale = image_size.max()
            valid_intri_mask = torch.logical_and(
                intrinsics[:, 0, 0] >= 0.1 * scale, intrinsics[:, 0, 0] <= 30 * scale
            )
            valid_trans_mask = (trans_BA.abs() <= 30).all(-1)
            valid_frame_mask = torch.logical_and(valid_intri_mask, valid_trans_mask)
            
            
            for pyimageid in reconstruction.images: 
                # scale from resized image size to the real size
                # rename the images to the original names
                pyimage = reconstruction.images[pyimageid]
                pycamera = reconstruction.cameras[pyimage.camera_id]
                
                pyimage.name = image_paths[pyimageid]
                
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = crop_params[0, pyimageid][:2]
                real_focal = real_image_size.max()/cfg.img_size * pred_params[0]
                
                real_pp = real_image_size.cpu().numpy()//2
                
                pred_params[0] = real_focal
                pred_params[1:3] = real_pp
                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]
                
                
            if cfg.extract_color:
                from vggsfm.models.utils import sample_features4d
                pred_track_rgb = sample_features4d(images.squeeze(0), pred_tracks)
                valid_track_rgb = pred_track_rgb[:, valid_tracks]
                
                sum_rgb = (BA_inlier_masks.float()[..., None] * valid_track_rgb).sum(dim=0)
                points3D_rgb = sum_rgb/BA_inlier_masks.sum(dim=0)[:,None]                    
            else:
                points3D_rgb = None
            
            
            # From OpenCV/COLMAP to PyTorch3D            
            rot_BA = rot_BA.clone().permute(0, 2, 1)
            trans_BA = trans_BA.clone()
            trans_BA[:, :2] *= -1
            rot_BA[:, :, :2] *= -1
            BA_cameras_PT3D = PerspectiveCameras(R=rot_BA, T=trans_BA, device=device)

            return BA_cameras_PT3D, extrinsics, intrinsics, points3D, points3D_rgb, reconstruction, valid_frame_mask


    def triangulate_tracks_and_BA(self, pred_tracks, intrinsics, extrinsics, pred_vis, pred_score, image_size, device, min_valid_track_length):
        """
        """
        # Normalize the tracks
        tracks_normalized_refined = normalize_tracks(pred_tracks, intrinsics)
        
        # Conduct triangulation to all the frames
        # We adopt LORANSAC here again
        
        best_triangulated_points, best_inlier_num, best_inlier_mask = triangulate_tracks(
            extrinsics, tracks_normalized_refined, track_vis=pred_vis, track_score=pred_score, max_ransac_iters=128
        )
        # Determine valid tracks based on inlier numbers
        valid_tracks = best_inlier_num >= min_valid_track_length
        # Perform global bundle adjustment
        points3D, extrinsics, intrinsics, reconstruction = global_BA(
            best_triangulated_points,
            valid_tracks,
            pred_tracks,
            best_inlier_mask,
            extrinsics,
            intrinsics,
            image_size,
            device,
        )
        
        
        valid_poins3D_mask = filter_all_points3D(
            points3D, pred_tracks[:, valid_tracks], extrinsics, intrinsics, check_triangle=False, max_reproj_error=4
        )
        points3D = points3D[valid_poins3D_mask]

        valid_tracks_tmp = valid_tracks.clone()
        valid_tracks_tmp[valid_tracks] = valid_poins3D_mask
        valid_tracks = valid_tracks_tmp.clone()

        return points3D, extrinsics, intrinsics, valid_tracks, reconstruction



def normalize_tracks(pred_tracks, intrinsics):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """
    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length
    return tracks_normalized
