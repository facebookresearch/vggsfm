# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
from util.run_hloc_utils import run_sfm
import tempfile
import os

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding

import models
from hydra.utils import instantiate
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from util.train_util import get_laf_mask
from pytorch3d.renderer import HarmonicEmbedding

from lightglue import LightGlue, SuperPoint, DISK
from util.relative_pose import compute_pose_by_LG, RT_by_track, RT_by_track_5p
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.checkpoint as checkpoint

# from util.vggBA import BAwrapper
from util.two_view_geo.fundamental import estimate_fundamental, essential_from_fundamental
from util.two_view_geo.homography import estimate_homography, decompose_homography_matrix
from util.two_view_geo.essential import estimate_essential, decompose_essential_matrix
from util.metric import closed_form_inverse

# from util.relative_pose import RT_by_track_5p
from util.two_view_geo.utils import get_default_intri, remove_cheirality
from torch.cuda.amp import autocast


from models.triangulator import camera_to_EFP

# from kornia.geometry.epipolar.fundamental import  normalize_transformation

# , estimate_fundamental_local
# from kornia.geometry.epipolar.fundamental import run_7point as run_7point_kornia

logger = logging.getLogger(__name__)


class VGGSfMV5(nn.Module):
    # def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, ENCODER: Dict, POINT: Dict, TRACKER: Dict, DIFFUSER: Dict, DENOISER: Dict, cfg=None):
    def __init__(
        self, pose_encoding_type: str, ENCODER: Dict, TRACKER: Dict, CAMERAPRED: Dict, TRIANGULATOR: Dict, cfg=None
    ):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.cfg = cfg
        self.enable_track = cfg.enable_track
        self.enable_pose = cfg.enable_pose
        self.enable_point = cfg.enable_point


        if (not self.enable_track) and (not self.enable_pose) and (not self.enable_point):
            raise ValueError("You have to have one to predict")


        if self.enable_track:
            self.fnet = instantiate(ENCODER, _recursive_=False)

            TRACKER.stride = ENCODER.stride
            # tracker has its own init method
            self.tracker = instantiate(TRACKER, _recursive_=False, cfg=cfg)

            # No, don't use this anymore
            # if ENCODER._target_ == "models.BasicEncoder":
            #     fnet_pips = torch.load("/data/home/jianyuan/src/ReconstructionJ/pose/pose_diffusion/tmp/pips_pretrained.pth")["model_state_dict"]
            #     state_dict = {k.replace("fnet.", ""): v for k, v in fnet_pips.items()}
            #     self.fnet.load_state_dict(state_dict, strict=False)

            cfg.track_latent_dim = self.tracker.latent_dim
        else:
            cfg.track_latent_dim = None

        if cfg.freeze_ctrack:
            for param in self.fnet.parameters():
                param.requires_grad = False
            for param in self.tracker.parameters():
                param.requires_grad = False
            for param in self.tracker.shallowfnet.parameters():
                param.requires_grad = True
            for param in self.tracker.finetracker.parameters():
                param.requires_grad = True

        if cfg.freeze_track:
            for param in self.fnet.parameters():
                param.requires_grad = False
            for param in self.tracker.parameters():
                param.requires_grad = False

        if self.enable_pose:
            self.pose_encoding_type = pose_encoding_type
            self.camera_predictor = instantiate(CAMERAPRED, _recursive_=False, cfg=cfg)
            self.target_dim = self.camera_predictor.target_dim

        if self.enable_point:
            self.triangulator = instantiate(TRIANGULATOR, _recursive_=False, cfg=cfg)

    def eval_tracks(self, tracks, image, fmaps, extra_fmaps, tracks_visibility, images_hd=None):
        with torch.no_grad():
            pred_tracks, _, vis_e, track_ffeat, track_score, track_losses = self.tracker(
                tracks[:, 0],
                rgbs=image,
                fmaps=fmaps,
                extra_fmaps=extra_fmaps,
                coords_init=None,
                iters=self.cfg.track_teseit,
                trajs_g=tracks,
                vis_g=tracks_visibility,
                valids=None,
                is_train=False,
                return_feat=True,
            )

            seq_loss, vis_loss, tconf_loss, refine_loss = track_losses
            # sig_sq = None

        return pred_tracks, vis_e, seq_loss, vis_loss, tconf_loss, refine_loss, track_ffeat, track_score

    def forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        training=True,
        points=None,
        points_rgb=None,
        tracks=None,
        tracks_visibility=None,
        tracks_ndc=None,
        epoch=None,
        imgpaths=None,
        crop_params=None,
        images_hd=None,
        batch=None,
    ):
        """
        Forward pass of the VGGSfMV4.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        # print(image.shape)
        batch_num, frame_num, image_dim, height, width = image.shape

        reshaped_image = image.reshape(batch_num * frame_num, image_dim, height, width)
        predictions = {}
        extra_dict = {}

        # if crop_params is not None:
        #     extra_dict['ndc_fl'] = crop_params[..., 2:3]

        if training:
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            #               Training
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            loss = 0

            track_score = None

            if self.enable_track:
                if self.cfg.track_downr > 1:
                    if self.cfg.ckpt_updatef:
                        fmaps = torch.utils.checkpoint.checkpoint(
                            self.fnet,
                            F.interpolate(
                                reshaped_image,
                                scale_factor=1 / self.cfg.track_downr,
                                mode="bilinear",
                                align_corners=True,
                            ),
                            use_reentrant=False,
                        )
                    else:
                        fmaps = self.fnet(
                            F.interpolate(
                                reshaped_image,
                                scale_factor=1 / self.cfg.track_downr,
                                mode="bilinear",
                                align_corners=True,
                            )
                        )
                else:
                    fmaps = self.fnet(reshaped_image)

                fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

                pred_tracks, _, vis_e, track_ffeat, track_score, track_losses = self.tracker(
                    tracks[:, 0],
                    rgbs=image,
                    fmaps=fmaps,
                    coords_init=None,
                    iters=self.cfg.track_trainit,
                    trajs_g=tracks,
                    vis_g=tracks_visibility,
                    valids=None,
                    is_train=True,
                    return_feat=True,
                )

                seq_loss, vis_loss, tconf_loss, refine_loss = track_losses

                predictions["loss_track"] = seq_loss
                predictions["loss_vis"] = vis_loss * 10
                predictions["loss_tconf"] = tconf_loss * 10
                predictions["loss_re"] = refine_loss

                loss_for_tracking = (
                    predictions["loss_track"]
                    + predictions["loss_vis"]
                    + predictions["loss_tconf"]
                    + predictions["loss_re"]
                )

                loss_for_tracking = loss_for_tracking * self.cfg.train.track_weight
                loss = loss + loss_for_tracking
                predictions["pred_tracks"] = pred_tracks
                predictions["pred_vis"] = vis_e

                predictions["pred_score"] = track_score

            if self.enable_pose:
                ### first compute 8p relative poses
                # check how colmap does here
                # start from GT poses
                # also check the AUC of such 8p results

                if self.cfg.prelimi_cam:
                    with torch.no_grad():
                        if "pred_tracks" in predictions:
                            tracks_for_pre = pred_tracks[self.cfg.trackl]
                            tracks_vis_for_pre = vis_e
                        else:
                            tracks_for_pre = tracks
                            tracks_vis_for_pre = tracks_visibility

                        preliminary_cameras, preliminary_dict = self.estimate_preliminary_cameras(
                            tracks_for_pre,
                            tracks_vis_for_pre,
                            width,
                            height,
                            score=track_score,
                            max_ransac_iters=self.cfg.pre_ransac // 2,
                            lo_num=100,
                            max_error=self.cfg.pre_max_error,
                            estimate_fl=False,
                            predict_essential=False,
                            predict_homo=False,
                            gt_cameras=gt_cameras,
                        )
                else:
                    preliminary_cameras = None
                    preliminary_dict = None

                pose_predictions = self.camera_predictor(
                    reshaped_image,
                    batch_size=batch_num,
                    gt_cameras=gt_cameras,
                    iters=self.cfg.camera_iter,
                    pose_pre=preliminary_cameras,
                    crop_params=crop_params,
                )

                # noisy_track

                predictions["preliminary_cameras"] = preliminary_cameras
                predictions["preliminary_dict"] = preliminary_dict

                predictions["pred_cameras"] = pose_predictions["pred_cameras"]
                predictions["loss_pose"] = pose_predictions["loss_pose"]
                loss_for_pose = predictions["loss_pose"]
                loss = loss + loss_for_pose

            if self.enable_point:
                for _ in range(100):
                    print("DROP THIS VERSION ")
                    print("USING THESEUS CUSTOMIZED IMPLEMTATION!!!!!!!!")
                raise NotImplementedError
                pred_cameras = pose_predictions["pred_cameras"]
                pred_tracks = None
                vis_e = None
                BA_cameras = self.triangulator(
                    pred_cameras,
                    pred_tracks,
                    vis_e,
                    gt_cameras,
                    tracks,
                    tracks_visibility,
                    image,
                    batch,
                    preliminary_dict,
                )
                predictions["pred_cameras"] = BA_cameras
            predictions["loss"] = loss
            return predictions
        else:
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            #               EVALUATION
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            track_score = None
            if self.enable_track:
                if self.cfg.track_downr > 1:
                    fmaps = self.fnet(
                        F.interpolate(
                            reshaped_image, scale_factor=1 / self.cfg.track_downr, mode="bilinear", align_corners=True
                        )
                    )
                else:
                    fmaps = self.fnet(reshaped_image)

                fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])
                extra_fmaps = None
                (
                    pred_tracks,
                    vis_e,
                    seq_loss,
                    vis_loss,
                    tconf_loss,
                    refine_loss,
                    track_ffeat,
                    track_score,
                ) = self.eval_tracks(tracks, image, fmaps, extra_fmaps, tracks_visibility)

                ################################################################################
                # # # # #
                # TODO: check it carefully here
                if crop_params is not None:
                    boundaries = crop_params[:, :, -4:-2].abs()
                    boundaries = torch.cat([boundaries, reshaped_image.shape[-1] - boundaries], dim=-1)
                    final_pred = pred_tracks[-1]
                    hvis = torch.logical_and(
                        final_pred[..., 1] >= boundaries[:, :, 1:2], final_pred[..., 1] <= boundaries[:, :, 3:4]
                    )
                    wvis = torch.logical_and(
                        final_pred[..., 0] >= boundaries[:, :, 0:1], final_pred[..., 0] <= boundaries[:, :, 2:3]
                    )
                    force_vis = torch.logical_and(hvis, wvis)
                    vis_e = vis_e * force_vis.float()
                # # # # #

                predictions["loss_track"] = seq_loss.mean()
                predictions["loss_vis"] = vis_loss.mean() * 10
                predictions["loss_tconf"] = tconf_loss.mean() * 10
                predictions["loss_re"] = refine_loss

                predictions["pred_tracks"] = pred_tracks
                predictions["pred_vis"] = vis_e
                predictions["pred_score"] = track_score

            if self.enable_pose:
                ### first compute 8p relative poses
                # check how colmap does here
                # start from GT poses
                # also check the AUC of such 8p results

                ### then feed them into camera predictor

                if self.cfg.prelimi_cam:
                    if "pred_tracks" in predictions:
                        tracks_for_pre = pred_tracks[self.cfg.trackl]
                        tracks_vis_for_pre = vis_e
                    else:
                        tracks_for_pre = tracks
                        tracks_vis_for_pre = tracks_visibility

                    preliminary_cameras, preliminary_dict = self.estimate_preliminary_cameras(
                        tracks_for_pre,
                        tracks_vis_for_pre,
                        width,
                        height,
                        max_ransac_iters=self.cfg.pre_ransac,
                        lo_num=300,
                        score=track_score,
                        max_error=self.cfg.pre_max_error,  # 0.5
                        estimate_fl=False,
                        predict_essential=False,
                        # estimate_fl=True,
                        # predict_essential = True,
                        # max_error=0.25,
                        ########################
                        # fl=gt_focal[None], pp=gt_pp[None],
                        ########################
                        predict_homo=False,
                        gt_cameras=gt_cameras,
                    )
                else:
                    preliminary_cameras = None
                    preliminary_dict = None

                pose_predictions = self.camera_predictor(
                    reshaped_image,
                    batch_size=batch_num,
                    gt_cameras=gt_cameras,
                    iters=self.cfg.camera_iter,
                    pose_pre=preliminary_cameras,
                    crop_params=crop_params,
                )

                predictions["preliminary_cameras"] = preliminary_cameras
                predictions["preliminary_dict"] = preliminary_dict

                predictions["loss_pose"] = pose_predictions["loss_pose"]
                predictions["pred_cameras"] = pose_predictions["pred_cameras"]

            if self.enable_point:
                # return predictions
                pred_cameras = predictions["pred_cameras"]
                BA_cameras = self.triangulator(
                    pred_cameras,
                    pred_tracks[self.cfg.trackl],
                    vis_e,
                    gt_cameras,
                    tracks,
                    tracks_visibility,
                    points,
                    image,
                    batch,
                    preliminary_dict,
                    track_vis=tracks_vis_for_pre,
                    track_score=track_score,
                )
                predictions["pred_cameras"] = BA_cameras

            return predictions

    def estimate_preliminary_cameras(
        self,
        tracks,
        tracks_vis,
        width,
        height,
        score=None,
        fl=None,
        pp=None,
        return_rawRt=False,
        max_error=0.5,
        lo_num=50,
        max_ransac_iters=1024,
        predict_essential=False,
        predict_homo=False,
        estimate_fl=False,
        loopresidual=False,
        second_refine=False,
        intri_bag=None,
        gt_cameras=None,
    ):
        
        if self.cfg.noisy_track > 0:
            track_noise = torch.normal(mean=0, std=self.cfg.noisy_track, size=tracks.size()).to(tracks.device)
            tracks[:, 1:] = tracks[:, 1:] + track_noise[:, 1:]

        if self.cfg.pre_good_fl:
            if np.random.rand() <= 0.33:
                batch_num, frame_num, N, _ = tracks.shape
                image_size = torch.tensor([width, height], dtype=tracks.dtype, device=tracks.device)

                augratio = 0.85 + (1.15 - 0.85) * torch.rand(batch_num, frame_num).to(tracks.device)

                _, intrinsics, focal_length, principal_point = camera_to_EFP(
                    gt_cameras, image_size, batch_num, frame_num
                )
                focal_length = focal_length * augratio[..., None]
                intrinsics[:, :, [0, 1], [0, 1]] = focal_length
                intri_bag = {"intrinsics": intrinsics, "focal_length": focal_length, "principal_point": principal_point}

        with autocast(dtype=torch.double):
            # # # # #
            # tracks = tracks + 0.5
            # # # # #
            B, S, N, _ = tracks.shape
            left = tracks[:, 0:1].expand(-1, S - 1, -1, -1).reshape(B * (S - 1), N, 2)
            right = tracks[:, 1:].reshape(B * (S - 1), N, 2)

            # TODO: check this, are they really helpful?
            # valid_mask = None

            valid_mask = (tracks_vis >= 0.05)[:, 1:].reshape(B * (S - 1), N)

            if score is not None:
                valid_score_mask = (score >= 0.5)[:, 1:].reshape(B * (S - 1), N)
                valid_mask = torch.logical_and(valid_mask, valid_score_mask)

            fmat, fmat_inlier_num, fmat_inlier_mask, pred_fl, fmat_residuals = estimate_fundamental(
                left,
                right,
                max_error=max_error,
                lo_num=lo_num,
                max_ransac_iters=max_ransac_iters,
                estimate_fl=estimate_fl,
                maxsize=max(width, height),
                valid_mask=valid_mask,
                loopresidual=loopresidual,
                second_refine=second_refine,
                return_residuals=True,
            )

            # pre_good_fl
            if intri_bag is not None:
                # kmat1, kmat2, fl, pp = build_kmat(tracks, fl, pp, width, height, pred_fl)
                # intri_bag
                # intri_bag = {"intrinsics": intrinsics, "focal_length": focal_length, "principal_point": principal_point}
                intrinsics = intri_bag["intrinsics"]
                focal_length = intri_bag["focal_length"]
                principal_point = intri_bag["principal_point"]

                kmat1 = intrinsics[:, 0:1].expand(-1, S - 1, -1, -1).reshape(B * (S - 1), 3, 3)
                kmat2 = intrinsics[:, 1:].reshape(B * (S - 1), 3, 3)

                fl = torch.cat([focal_length[:, 0:1].expand(-1, S - 1, -1), focal_length[:, 1:]], dim=2).reshape(
                    B * (S - 1), 4
                )
                pp = torch.cat([principal_point[:, 0:1].expand(-1, S - 1, -1), principal_point[:, 1:]], dim=2).reshape(
                    B * (S - 1), 4
                )
            else:
                kmat1, kmat2, fl, pp = build_kmat(tracks, fl, pp, width, height, pred_fl)

            # print(fl)
            emat_fromf, emat_fromf_inlier_num, emat_fromf_inlier_mask = essential_from_fundamental(
                left, right, fmat, fl, pp, max_error=max_error, kmat1=kmat1, kmat2=kmat2
            )

            if predict_essential:
                emat, emat_inlier_num, emat_inlier_mask = estimate_essential(
                    left, right, fl, pp, max_error=max_error, lo_num=lo_num, max_ransac_iters=max_ransac_iters
                )
                R_emat, t_emat = decompose_essential_matrix(emat)
                R_emat, t_emat = remove_cheirality(R_emat, t_emat, left, right, fl, pp)
            else:
                emat_inlier_num = 0

            # NOTE the number is S-1 here, because the first image is the query one
            if predict_homo:
                hmat, hmat_inlier_num, hmat_inlier_mask = estimate_homography(
                    left, right, max_error=max_error, lo_num=lo_num, max_ransac_iters=max_ransac_iters
                )
                max_H_ratio = 0.9
                H_F_inlier_ratio = hmat_inlier_num.float() / (fmat_inlier_num.float() + 1e-6)
                H_E_inlier_ratio = hmat_inlier_num.float() / (emat_inlier_num.float() + 1e-6)
                R_hmat, t_hmat, _ = decompose_homography_matrix(hmat, left, right, kmat1, kmat2)
                R_hmat, t_hmat = remove_cheirality(R_hmat, t_hmat, left, right)
                use_hmat = (H_F_inlier_ratio > max_H_ratio) & (H_E_inlier_ratio > max_H_ratio)

                use_emat = torch.logical_and(~use_hmat, emat_inlier_num >= emat_fromf_inlier_num)
                use_emat_fromf = torch.logical_and(~use_hmat, emat_inlier_num < emat_fromf_inlier_num)
            else:
                use_emat = emat_inlier_num > emat_fromf_inlier_num
                use_emat_fromf = emat_inlier_num <= emat_fromf_inlier_num
                use_hmat = torch.zeros_like(use_emat)

            R_emat_fromf, t_emat_fromf = decompose_essential_matrix(emat_fromf)
            R_emat_fromf, t_emat_fromf = remove_cheirality(R_emat_fromf, t_emat_fromf, left, right, fl, pp)

            # print(fmat_inlier_num.min())

            # remember to go from B*(S-1) to B, S
            # remember, have to use torch.repeat here, because expand will not really creat B*(S-1) tensors
            R_preliminary = torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B * (S - 1), 1, 1)
            t_preliminary = torch.zeros(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B * (S - 1), 1)

            if use_hmat.sum() > 0:
                assert predict_homo
                R_preliminary[use_hmat] = R_hmat[use_hmat].float()
                t_preliminary[use_hmat] = t_hmat[use_hmat].float()

            if use_emat_fromf.sum() > 0:
                R_preliminary[use_emat_fromf] = R_emat_fromf[use_emat_fromf].float()
                t_preliminary[use_emat_fromf] = t_emat_fromf[use_emat_fromf].float()

            if use_emat.sum() > 0:
                assert predict_essential
                R_preliminary[use_emat] = R_emat[use_emat].float()
                t_preliminary[use_emat] = t_emat[use_emat].float()

            R_preliminary = R_preliminary.reshape(B, S - 1, 3, 3)
            t_preliminary = t_preliminary.reshape(B, S - 1, 3)

            if return_rawRt:
                return R_preliminary, t_preliminary

            if self.cfg.noisy_pre:
                R_preliminary = R_preliminary + torch.normal(mean=0, std=0.06, size=R_preliminary.size()).to(
                    R_preliminary.device
                )
                t_preliminary = t_preliminary + torch.normal(mean=0, std=0.1, size=t_preliminary.size()).to(
                    t_preliminary.device
                )

            # pad for the first camera
            R_pad = torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B, 1, 1).unsqueeze(1)
            t_pad = torch.zeros(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B, 1).unsqueeze(1)

            R_preliminary = torch.cat([R_pad, R_preliminary], dim=1).reshape(B * S, 3, 3)
            t_preliminary = torch.cat([t_pad, t_preliminary], dim=1).reshape(B * S, 3)

            R_opencv = R_preliminary.clone()
            t_opencv = t_preliminary.clone()

            # to pt3d
            R_preliminary = R_preliminary.clone().permute(0, 2, 1)
            t_preliminary = t_preliminary.clone()
            t_preliminary[:, :2] *= -1
            R_preliminary[:, :, :2] *= -1

            pred_cameras = PerspectiveCameras(R=R_preliminary, T=t_preliminary, device=R_preliminary.device)

            with autocast(dtype=torch.double):
                pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

                rel_transform = closed_form_inverse(pred_se3[0:1, :, :])
                rel_transform = rel_transform.expand(B * S, -1, -1)

                pred_se3_rel = torch.bmm(rel_transform, pred_se3)
                pred_se3_rel[..., :3, 3] = 0.0
                pred_se3_rel[..., 3, 3] = 1.0

                pred_cameras.R = pred_se3_rel[:, :3, :3].clone()
                pred_cameras.T = pred_se3_rel[:, 3, :3].clone()

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


def build_kmat(tracks, fl=None, pp=None, width=None, height=None, pred_fl=None):
    B, S, N, _ = tracks.shape

    if pred_fl is not None:
        f1, f2 = pred_fl

        _, pp, _ = get_default_intri(width, height, tracks.device, tracks.dtype)

        f2 = f2.reshape(B, S - 1).unsqueeze(-1).repeat(1, 1, 2)
        f1 = torch.ones_like(f2[:, 0:1]) * f1
        f1 = f1.expand(-1, S - 1, -1)
        fl = torch.cat([f1, f2], dim=-1)

        pp = torch.cat([pp, pp])[None][None].expand(B, S - 1, -1)
    elif fl is None:
        fl, pp, _ = get_default_intri(width, height, tracks.device, tracks.dtype)
        fl = torch.ones((B, S - 1, 4), device=tracks.device, dtype=tracks.dtype) * fl
        pp = torch.cat([pp, pp])[None][None].expand(B, S - 1, -1)
    else:
        if pp is None:
            pp = torch.zeros_like(fl)
            pp[:, 0] *= width
            pp[:, 1] *= height

        # BxSx2 -> BxS-1x4
        fl = torch.cat([fl[:, 0:1].expand(-1, S - 1, -1), fl[:, 1:]], dim=-1)
        pp = torch.cat([pp[:, 0:1].expand(-1, S - 1, -1), pp[:, 1:]], dim=-1)

    # :2 for left frame, 2: for right frame
    fl = fl.reshape(B * (S - 1), 4)
    pp = pp.reshape(B * (S - 1), 4)

    # build kmat
    kmat1 = torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B * (S - 1), 1, 1)
    kmat2 = torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None].repeat(B * (S - 1), 1, 1)

    kmat1[:, [0, 1], [0, 1]] = fl[:, :2]
    kmat1[:, [0, 1], 2] = pp[:, :2]

    kmat2[:, [0, 1], [0, 1]] = fl[:, 2:]
    kmat2[:, [0, 1], 2] = pp[:, 2:]

    return kmat1, kmat2, fl, pp