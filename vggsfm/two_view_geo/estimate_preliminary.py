# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle


from torch.cuda.amp import autocast

from .fundamental import estimate_fundamental, essential_from_fundamental
from .homography import estimate_homography, decompose_homography_matrix
from .essential import estimate_essential, decompose_essential_matrix
from .utils import get_default_intri, remove_cheirality

# TODO remove the .. and . that may look confusing
from ..utils.metric import closed_form_inverse

try:
    import poselib
    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")
    

def estimate_preliminary_cameras_poselib(tracks,
    tracks_vis,
    width,
    height,
    tracks_score=None,
    max_error=0.5,
    max_ransac_iters=20000,
    predict_essential=False,
    lo_num=None,
    predict_homo=False,
    loopresidual=False,
):
    B, S, N, _ = tracks.shape

    query_points = tracks[:, 0:1].reshape(B, N , 2) 
    reference_points = tracks[:, 1:].reshape(B * (S - 1), N, 2)

    valid_mask = (tracks_vis >= 0.05)[:, 1:].reshape(B * (S - 1), N)

    fmat = []
    inlier_mask = []
    for idx in range(len(reference_points)):
        kps_left = query_points[0].cpu().numpy()
        kps_right = reference_points[idx].cpu().numpy()
        
        cur_inlier_mask = valid_mask[idx].cpu().numpy()
        
        kps_left = kps_left[cur_inlier_mask]
        kps_right = kps_right[cur_inlier_mask]
        
        cur_fmat, info = poselib.estimate_fundamental(kps_left, kps_right, 
                                                  {'max_epipolar_error': max_error, 
                                                   "max_iterations": max_ransac_iters, "min_iterations": 1000, 
                                                   "real_focal_check": True, "progressive_sampling": False})
        
        cur_inlier_mask[cur_inlier_mask] = np.array(info['inliers'])
        
        fmat.append(cur_fmat)
        inlier_mask.append(cur_inlier_mask)
    
    fmat = torch.from_numpy(np.array(fmat)).to(query_points.device)
    inlier_mask = torch.from_numpy(np.array(inlier_mask)).to(query_points.device)

    preliminary_dict = {
        "fmat": fmat[None],
        "fmat_inlier_mask": inlier_mask[None],
    }

    return None, preliminary_dict



def estimate_preliminary_cameras(
    tracks,
    tracks_vis,
    width,
    height,
    tracks_score=None,
    max_error=0.5,
    lo_num=300,
    max_ransac_iters=4096,
    predict_essential=False,
    predict_homo=False,
    loopresidual=False,
):
    # TODO: also clean the code for predict_essential and predict_homo

    with autocast(dtype=torch.double):
        # batch_num, frame_num, point_num
        B, S, N, _ = tracks.shape

        # We have S-1 reference frame per batch
        query_points = tracks[:, 0:1].expand(-1, S - 1, -1, -1).reshape(B * (S - 1), N, 2)
        reference_points = tracks[:, 1:].reshape(B * (S - 1), N, 2)

        # Filter out some matches based on track vis and score

        valid_mask = (tracks_vis >= 0.05)[:, 1:].reshape(B * (S - 1), N)

        if tracks_score is not None:
            valid_tracks_score_mask = (tracks_score >= 0.5)[:, 1:].reshape(B * (S - 1), N)
            valid_mask = torch.logical_and(valid_mask, valid_tracks_score_mask)

        # Estimate Fundamental Matrix by Batch
        # fmat: (B*(S-1))x3x3
        fmat, fmat_inlier_num, fmat_inlier_mask, fmat_residuals = estimate_fundamental(
            query_points,
            reference_points,
            max_error=max_error,
            lo_num=lo_num,
            max_ransac_iters=max_ransac_iters,
            valid_mask=valid_mask,
            loopresidual=loopresidual,
            return_residuals=True,
        )

        # kmat1, kmat2: (B*(S-1))x3x3
        kmat1, kmat2, fl, pp = build_default_kmat(
            width, height, B, S, N, device=query_points.device, dtype=query_points.dtype
        )

        emat_fromf, _, _ = essential_from_fundamental(fmat, kmat1, kmat2)

        R_emat_fromf, t_emat_fromf = decompose_essential_matrix(emat_fromf)
        R_emat_fromf, t_emat_fromf = remove_cheirality(
            R_emat_fromf, t_emat_fromf, query_points, reference_points, fl, pp
        )

        # TODO: clean the code for R_hmat, t_hmat, R_emat, t_emat and add them here
        R_preliminary = R_emat_fromf
        t_preliminary = t_emat_fromf

        R_preliminary = R_preliminary.reshape(B, S - 1, 3, 3)
        t_preliminary = t_preliminary.reshape(B, S - 1, 3)

        # pad for the first camera
        R_pad = torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B, 1, 1).unsqueeze(1)
        t_pad = torch.zeros(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B, 1).unsqueeze(1)

        R_preliminary = torch.cat([R_pad, R_preliminary], dim=1).reshape(B * S, 3, 3)
        t_preliminary = torch.cat([t_pad, t_preliminary], dim=1).reshape(B * S, 3)

        R_opencv = R_preliminary.clone()
        t_opencv = t_preliminary.clone()

        # From OpenCV/COLMAP camera convention to PyTorch3D
        # TODO: Remove the usage of PyTorch3D convention in all the codebase
        # So that we don't need to do such conventions any more
        R_preliminary = R_preliminary.clone().permute(0, 2, 1)
        t_preliminary = t_preliminary.clone()
        t_preliminary[:, :2] *= -1
        R_preliminary[:, :, :2] *= -1

        pred_cameras = PerspectiveCameras(R=R_preliminary, T=t_preliminary, device=R_preliminary.device)

        with autocast(dtype=torch.double):
            # Optional in the future
            # make all the cameras relative to the first one
            pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

            rel_transform = closed_form_inverse(pred_se3[0:1, :, :])
            rel_transform = rel_transform.expand(B * S, -1, -1)

            pred_se3_rel = torch.bmm(rel_transform, pred_se3)
            pred_se3_rel[..., :3, 3] = 0.0
            pred_se3_rel[..., 3, 3] = 1.0

            pred_cameras.R = pred_se3_rel[:, :3, :3].clone()
            pred_cameras.T = pred_se3_rel[:, 3, :3].clone()

        # Record them in case we may need later
        fmat = fmat.reshape(B, S - 1, 3, 3)
        fmat_inlier_mask = fmat_inlier_mask.reshape(B, S - 1, -1)
        kmat1 = kmat1.reshape(B, S - 1, 3, 3)
        R_opencv = R_opencv.reshape(B, S, 3, 3)
        t_opencv = t_opencv.reshape(B, S, 3)

        fmat_residuals = fmat_residuals.reshape(B, S - 1, -1)

        preliminary_dict = {
            "fmat": fmat,
            "fmat_inlier_mask": fmat_inlier_mask,
            "R_opencv": R_opencv,
            "t_opencv": t_opencv,
            "default_intri": kmat1,
            "emat_fromf": emat_fromf,
            "fmat_residuals": fmat_residuals,
        }

        return pred_cameras, preliminary_dict


def build_default_kmat(width, height, B, S, N, device=None, dtype=None):
    # focal length is set as max(width, height)
    # principal point is set as (width//2, height//2,)

    fl, pp, _ = get_default_intri(width, height, device, dtype)

    # :2 for left frame, 2: for right frame
    fl = torch.ones((B, S - 1, 4), device=device, dtype=dtype) * fl
    pp = torch.cat([pp, pp])[None][None].expand(B, S - 1, -1)

    fl = fl.reshape(B * (S - 1), 4)
    pp = pp.reshape(B * (S - 1), 4)

    # build kmat
    kmat1 = torch.eye(3, device=device, dtype=dtype)[None].repeat(B * (S - 1), 1, 1)
    kmat2 = torch.eye(3, device=device, dtype=dtype)[None].repeat(B * (S - 1), 1, 1)

    # assign them to the corresponding locations of kmats
    kmat1[:, [0, 1], [0, 1]] = fl[:, :2]
    kmat1[:, [0, 1], 2] = pp[:, :2]

    kmat2[:, [0, 1], [0, 1]] = fl[:, 2:]
    kmat2[:, [0, 1], 2] = pp[:, 2:]

    return kmat1, kmat2, fl, pp
