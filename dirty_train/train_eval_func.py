import argparse
import cProfile
import datetime
import glob
import io
import json
import os
import pickle
import pstats
import re
import time
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union
from visdom import Visdom

# Related third-party imports
from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from hydra.utils import instantiate, get_original_cwd
from torch.cuda.amp import GradScaler, autocast

import cv2
import hydra
import models
import numpy as np
import psutil
import torch
import tqdm
import visdom
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from griddle.utils import is_aws_cluster
from test_category import test_co3d, test_imc
from util.load_img_folder import load_and_preprocess_images
from util.metric import camera_to_rel_deg, calculate_auc, calculate_auc_np, camera_to_rel_deg_pair
from util.triangulation import intersect_skew_line_groups
from util.train_util import *
from inference import run_inference


def train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, viz=None, epoch=-1
):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it = len(dataloader)

    if cfg.track_by_spsg:
        from gluefactory.models.extractors.superpoint_open import SuperPoint
        from gluefactory.models.extractors.sift import SIFT
        from gluefactory.models.extractors.grid_extractor import GridExtractor
        from gluefactory.models.extractors.disk_kornia import DISK

        sp = SuperPoint({"nms_radius": 4, "force_num_keypoints": True}).cuda().eval()
        disk = DISK({"dense_outputs": False, "force_num_keypoints": True}).cuda().eval()
        # sift = SIFT({"backend":"pycolmap_cuda"}).cuda().eval()
        sift = SIFT({}).cuda().eval()
        gridextractor = GridExtractor({"cell_size": 16}).cuda().eval()

    AUC_scene_dict = {}
    
    for step, batch in enumerate(dataloader):
        # print(batch["seq_name"])
        if step == 100:
            record_and_print_cpu_memory_and_usage()

        images_hd = None

        gt_cameras = None
        points_rgb = None

        (
            images,
            crop_params,
            translation,
            rotation,
            fl,
            pp,
            points,
            points_rgb,
            tracks,
            tracks_visibility,
            tracks_ndc,
        ) = process_co3d_data(batch, accelerator.device, cfg)

        
        if cfg.track_by_spsg and (not cfg.inference):
            # use keypoints as the starting points of tracks
            images_for_kp = images
            bbb, nnn, ppp, _ = tracks.shape
            pred0_sp = sp({"image": images_for_kp[:, 0]})
            kp0_sp = pred0_sp["keypoints"]

            # pred0_disk = disk({'image': images_for_kp[:, 0]})
            # kp0_disk = pred0_disk['keypoints']
            # sift.conf.force_num_keypoints

            pred0_sift = sift({"image": images_for_kp[:, 0]})
            kp0_sift = pred0_sift["keypoints"]

            # pred0_sp = sp({'image': images_for_kp[:, 0]})

            # import pdb;pdb.set_trace()
            # pred0_grid = gridextractor({'image': images_for_kp[0, 0:1]})
            # kp0_grid = pred0_grid['keypoints']
            # kp0 = torch.cat([kp0_sp, kp0_disk, kp0_sift, kp0_grid], dim=1)
            # kp0 = torch.cat([kp0_sp, kp0_disk, kp0_sift], dim=1)

            kp0 = torch.cat([kp0_sp, kp0_sift], dim=1)

            # kp0 = kp0_sp

            new_track_num = kp0.shape[-2]
            if new_track_num > cfg.train.track_num:
                indices = torch.randperm(new_track_num)[: cfg.train.track_num]
                kp0 = kp0[:, indices, :]

            # print(kp0)
            kp0 = kp0[None].repeat(1, nnn, 1, 1)
            tracks = kp0.clone()
            tracks_visibility = torch.ones(bbb, nnn, kp0.shape[-2]).bool().cuda().clone()

        frame_size = images.shape[1]

        
        if rotation is not None:
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                principal_point=pp.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=accelerator.device,
            )
            batch_size = len(images)


            if training and cfg.train.batch_repeat > 0:
                # repeat samples by several times
                # to accelerate training
                br = cfg.train.batch_repeat
                gt_cameras = PerspectiveCameras(
                    focal_length=fl.reshape(-1, 2).repeat(br, 1),
                    R=rotation.reshape(-1, 3, 3).repeat(br, 1, 1),
                    T=translation.reshape(-1, 3).repeat(br, 1),
                    device=accelerator.device,
                )
                batch_size = len(images) * br



        if training:
            predictions = model(
                images,
                gt_cameras=gt_cameras,
                training=True,
                # batch_repeat=cfg.train.batch_repeat,
                points=points,
                points_rgb=points_rgb,
                tracks=tracks,
                tracks_visibility=tracks_visibility,
                tracks_ndc=tracks_ndc,
                epoch=epoch,
                crop_params=crop_params,
            )

            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            # for _ in range(10):
            #     print("GOOOOOOOOOOOOO")
            imgpaths = None
            if cfg.inference:
                predictions = run_inference(
                    model,
                    images,
                    gt_cameras=gt_cameras,
                    training=False,
                    points=points,
                    points_rgb=points_rgb,
                    tracks=tracks,
                    tracks_visibility=tracks_visibility,
                    tracks_ndc=tracks_ndc,
                    epoch=epoch,
                    imgpaths=imgpaths,
                    crop_params=crop_params,
                    images_hd=images_hd,
                    batch=batch,
                    cfg=cfg,
                )
            else:
                with torch.no_grad():
                    predictions = model(
                        images,
                        gt_cameras=gt_cameras,
                        training=False,
                        points=points,
                        points_rgb=points_rgb,
                        tracks=tracks,
                        tracks_visibility=tracks_visibility,
                        tracks_ndc=tracks_ndc,
                        epoch=epoch,
                        imgpaths=imgpaths,
                        crop_params=crop_params,
                        images_hd=images_hd,
                        batch=batch,
                    )

        # Computing Metrics
        with torch.no_grad():
            if "pred_cameras" in predictions:
                with autocast(dtype=torch.double):
                    pred_cameras = predictions["pred_cameras"]

                    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(
                        pred_cameras, gt_cameras, accelerator.device, batch_size
                    )

                    # metrics to report
                    thresholds = [5, 15, 30]
                    for threshold in thresholds:
                        predictions[f"Racc_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
                        predictions[f"Tacc_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

                    Auc_30, normalized_histogram = calculate_auc(
                        rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
                    )
                    auc_thresholds = [30, 10, 5, 3]
                    for auc_threshold in auc_thresholds:
                        predictions[f"Auc_{auc_threshold}"] = torch.cumsum(
                            normalized_histogram[:auc_threshold], dim=0
                        ).mean()

                    AUC_scene_dict[batch["seq_name"][0]] = torch.cumsum(normalized_histogram[:10], dim=0).mean()

                    # not pair-wise
                    pair_rangle_deg, pair_tangle_deg = camera_to_rel_deg_pair(
                        pred_cameras, gt_cameras, accelerator.device, batch_size
                    )
                    pair_Auc_30, pair_normalized_histogram = calculate_auc(
                        pair_rangle_deg, pair_tangle_deg, max_threshold=30, return_list=True
                    )
                    predictions[f"PairAuc_10"] = torch.cumsum(pair_normalized_histogram[:10], dim=0).mean()
                    predictions[f"PairAuc_3"] = torch.cumsum(pair_normalized_histogram[:3], dim=0).mean()

            if "pred_tracks" in predictions:
                pred_tracks = predictions["pred_tracks"]

                pred_tracks = pred_tracks[cfg.trackl]

                pred_vis = predictions["pred_vis"]

                # if cfg.debug:
                #     tracks = tracks * 2
                #     pred_tracks = pred_tracks * 2

                # if torch.isnan(pred_tracks).any() or torch.isinf(pred_tracks).any():
                #     for _ in range(100):
                #         print("Predicting NaN!!!!")
                # print(batch["seq_name"])

                track_dis = (tracks - pred_tracks) ** 2
                track_dis = torch.sqrt(torch.sum(track_dis, dim=-1))

                # if torch.isnan(track_dis).any() or torch.isinf(track_dis).any():
                #     track_dis_tmp = track_dis.sum(dim=-1).sum(dim=-1)
                #     print_idx = torch.logical_or(torch.isnan(track_dis_tmp), torch.isinf(track_dis_tmp))
                #     print_idx = torch.nonzero(print_idx)
                #     for _ in range(100):
                #         print("Meet NaN here!!!!")
                #         print(batch["seq_name"][print_idx])

                predictions["Track_m"] = track_dis.mean()
                predictions["Track_mvis"] = track_dis[tracks_visibility].mean()
                # TODO: need to check here again

                # import pdb;pdb.set_trace()

                predictions["Track_a1"] = (track_dis[:, 1:][tracks_visibility[:, 1:]] < 1).float().mean()

                if check_ni(predictions["Track_a1"]):
                    # TODO: something is wrong with Megadepth dataset loader
                    print("track a1 is NaN")
                    print(track_dis.shape)
                    print("dis:")
                    print(track_dis[:, 1:])
                    print("visi:")
                    print(tracks_visibility[:, 1:])
                    # print(batch['seq_id'])
                    # np.save("/data/home/jianyuan/src/ReconstructionJ/pose/pose_diffusion/tmp/debugNaN.npy", images.detach().cpu().numpy())

                pred_vis_binary = pred_vis > 0.5
                correct_predictions = (pred_vis_binary == tracks_visibility).float()
                predictions["Vis_acc"] = correct_predictions.sum() / correct_predictions.numel()

            if "pred_points" in predictions:
                pred_points = predictions["pred_points"]
                chamfer = chamfer_distance(points, pred_points)[0]
                predictions["chamfer"] = chamfer

        ################################################################################################################
        if cfg.visualize:
            # TODO: clean here
            viz = vis_utils.get_visdom_connection(
                server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
            )
            # viz = Visdom(server=f"10.200.160.58", port=10089)

            # import pdb;pdb.set_trace()
            # Visualize GT Cameras
            if gt_cameras is not None:
                # pred_cameras
                # gt_cameras

                first_seq_idx = torch.arange(frame_size)
                # pred_cameras_aligned = corresponding_cameras_alignment(cameras_src=pred_cameras[first_seq_idx], cameras_tgt=gt_cameras[first_seq_idx], estimate_scale=True, mode="extrinsics", eps=1e-9)
                cams_show = {"gt_cameras": gt_cameras[first_seq_idx]}

                first_seq_name = batch["seq_name"][0]

                if "pred_cameras" in predictions:
                    from pytorch3d.ops import corresponding_cameras_alignment
                    pred_cameras = predictions["pred_cameras"]
                    pred_cameras_aligned = corresponding_cameras_alignment(cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="centers", eps=1e-9)

                    cams_show["pred_cameras"] = pred_cameras[first_seq_idx]
                    cams_show["pred_cameras_aligned"] = pred_cameras_aligned[first_seq_idx]
                    
                    
                fig = plot_scene({f"{first_seq_name}": cams_show})
                env_name = f"visual_{cfg.exp_name}_{step}"
                viz.plotlyplot(fig, env=env_name, win="cams")
                viz.images((images[0]).clamp(0, 1), env=env_name, win="img")
                # viz.images((images[0]).clamp(0, 1), env=env_name, win="img")

                # log_to_filename
                # viz_new =  Visdom(server=f"10.200.160.58", port=10089)

                # 10.200.160.58
                # viz_new.plotlyplot(fig, env="heyhey", win="cams")
                # viz_new.images((images[0]).clamp(0, 1), env="heyhey", win="img")
                print(env_name)
                import pdb;pdb.set_trace()

            # Visualize GT Tracks
            if "pred_tracks" in predictions:
                # if True:
                save_dir = f"{cfg.exp_dir}/visual_track"

                n_points = np.random.randint(20, 30)

                res_combined, _ = visualize_track(
                    predictions, images, tracks, tracks_visibility, cfg, step, viz, n_points=n_points, save_dir=save_dir,
                    visual_gt=True
                )

                save_track_visual = False
                if save_track_visual:
                    save_visual_track(res_combined, save_dir + ".png")

                import pdb;pdb.set_trace()

                # res_combined, _ = visualize_track(predictions, images, tracks, tracks_visibility, cfg, step, viz, n_points=100, save_dir= save_dir)
                # res_combined, _ = visualize_track(predictions, images, tracks, tracks_visibility, cfg, step, viz, n_points=n_points, save_dir= save_dir)
                # save_visual_track(res_combined, save_dir +"tmp" + ".png")
                # save_visual_track(res_combined, save_dir + ".png")
                # batch['seq_id']
                # ['25bag_003_lincoln_memorial_statue']                                                                                                                                                  │··········
                # res_combined, _ = visualize_track(predictions, images, tracks, tracks_visibility, cfg, step, viz, n_points=100, save_dir= save_dir)
                # import pdb;pdb.set_trace()
                # m=1

                # _, num_frames, channels, height, width = res_combined.shape
                # res_row = res_combined.squeeze(0).permute(1, 2, 0, 3).reshape(channels, height, num_frames * width)
                # res_row_np = res_row.numpy()
                # res_row_np = ((res_row_np - res_row_np.min()) / (res_row_np.max() - res_row_np.min()) * 255).astype(np.uint8)
                # res_row_np = np.transpose(res_row_np, (1, 2, 0))
                # cv2.imwrite('combined_frames.png', res_row_np)

                # import pdb;pdb.set_trace()

            if False:
                # camera_dict = {"pred_cameras": {}, "gt_cameras": {}}

                # for visidx in range(frame_size):
                # frame_size = 2
                # for visidx in range(frame_size):
                #     camera_dict["pred_cameras"][visidx] = pred_cameras[visidx]
                #     camera_dict["gt_cameras"][visidx] = gt_cameras[visidx]

                # # pcl = Pointclouds(points=points[0:1])
                # # camera_dict["points"] = {"pc": pcl}

                # fig = plotly_scene_visualization(camera_dict, frame_size)

                # viz.plotlyplot(fig, env="comeon", win="cams")
                # import pdb;pdb.set_trace()

                ########################################################
                # if "pred_points" in predictions:
                pcl = Pointclouds(points=points[0:1])
                # pred_points = predictions["pred_points"]
                # pred_pcl = Pointclouds(points=pred_points[0:1])
                # camera_dict["points"] = {"pc": pcl}

                pred_cameras_aligned = corresponding_cameras_alignment(
                    cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
                )
                combined_dict = {
                    "scenes": {
                        "pred_cameras": pred_cameras[torch.range(0, frame_size - 1).long()],
                        "pred_cameras_aligned": pred_cameras_aligned[torch.range(0, frame_size - 1).long()],
                        "gt_cameras": gt_cameras[torch.range(0, frame_size - 1).long()],
                        "points": pcl,
                    }
                }

                fig = plot_scene(combined_dict)
                ########################################################
                viz.plotlyplot(fig, env=f"Perfect1", win="cams")

                show_img = view_color_coded_images_for_visdom(images[0])
                # viz.images(show_img.clamp(0,1), env=cfg.exp_name, win="imgs")

                viz.images(show_img.clamp(0, 1), env=f"Perfect1", win="imgs")
                # rel_rangle_deg.mean()
                # rel_tangle_deg
                import pdb

                pdb.set_trace()
                m = 1

            # viz.images(images[0], env=cfg.exp_name, win="imgs")
            # batch['seq_id']
            # cfg.train.normalize_cameras
        ################################################################################################################

        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="train", max_it=max_it))
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="eval", max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)

            # if cfg.debug:
            #     for name, param in model.named_parameters():
            #         if "backbone" not in name:
            #             if param.grad is None:
            #                 print(f"Parameter '{name}' is unused.")
            #             else:
            #                 # Optionally check for zero gradients
            #                 # Note: This might not be necessary for just checking unused parameters
            #                 if torch.all(param.grad == 0):
            #                     print(f"Parameter '{name}' has zero gradient.")

            if cfg.train.clip_grad > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                # total_norm_before_clipping = accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                # print(total_norm_before_clipping)

            optimizer.step()
            lr_scheduler.step()

    if cfg.debug:
        import pdb

        pdb.set_trace()
        # AUC_scene_dict
        m = 1

    return True