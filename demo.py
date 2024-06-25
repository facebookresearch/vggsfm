# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from lightglue import LightGlue, SuperPoint, SIFT, ALIKED

import pycolmap

from visdom import Visdom

from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras


from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras

try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras_poselib
    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")
    
from vggsfm.utils.utils import (
    set_seed_and_print,
    farthest_point_sampling,
    calculate_index_mappings,
    switch_tensor_order,
)


@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed
    seed_all_random_engines(cfg.seed)

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Prepare test dataset
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR, img_size=cfg.img_size, normalize_cameras=False, load_gt=cfg.load_gt, cfg=cfg
    )

    if cfg.resume_ckpt:
        # Reload model
        checkpoint = torch.load(cfg.resume_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Successfully resumed from {cfg.resume_ckpt}")

    if cfg.visualize:
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import plot_scene
        
        # viz = Visdom()


        from pytorch3d.implicitron.tools import model_io, vis_utils
        viz = vis_utils.get_visdom_connection(
            server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
        )

    sequence_list = test_dataset.sequence_list

    for seq_name in sequence_list:
        print("*" * 50 + f" Testing on Scene {seq_name} " + "*" * 50)

        # Load the data
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)

        # Send to GPU
        images = batch["image"].to(device)
        crop_params = batch["crop_params"].to(device)

        if cfg.load_gt:
            translation = batch["T"].to(device)
            rotation = batch["R"].to(device)
            fl = batch["fl"].to(device)
            pp = batch["pp"].to(device)

            # Prepare gt cameras
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                principal_point=pp.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=device,
            )

        # Unsqueeze to have batch size = 1
        images = images.unsqueeze(0)
        crop_params = crop_params.unsqueeze(0)

        batch_size = len(images)
        
        with torch.no_grad():
            # Run the model
            assert cfg.mixed_precision in ("None", "bf16", "fp16")
            if cfg.mixed_precision == "None":
                dtype = torch.float32
            elif cfg.mixed_precision == "bf16":
                dtype = torch.bfloat16
            elif cfg.mixed_precision == "fp16":
                dtype = torch.float16
            else:
                raise NotImplementedError(f"dtype {cfg.mixed_precision} is not supported now")
            
            predictions = run_one_scene(
                model,
                images,
                crop_params=crop_params,
                query_frame_num=cfg.query_frame_num,
                image_paths=image_paths,
                dtype = dtype,
                cfg=cfg,
            )

        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        output_path = os.path.join("output", seq_name)
        print("-"*50)
        print(f"The output has been saved in COLMAP style at: {output_path} ")
        os.makedirs(output_path, exist_ok=True)
        reconstruction_pycolmap.write(output_path)

        pred_cameras_PT3D = predictions["pred_cameras_PT3D"]

        if cfg.visualize:
            if "points3D_rgb" in predictions:
                pcl = Pointclouds(points=predictions["points3D"][None], features=predictions["points3D_rgb"][None])
            else:
                pcl = Pointclouds(points=predictions["points3D"][None])
                
            visual_dict = {"scenes": {"points": pcl, "cameras": pred_cameras_PT3D}}

            fig = plot_scene(visual_dict, camera_scale=0.05)
            
            env_name = f"demo_visual_{seq_name}"
            print(f"Visualizing the scene by visdom at env: {env_name}")
            viz.plotlyplot(fig, env=env_name, win="3D")


    return True









def run_one_scene(model, images, crop_params=None, query_frame_num=3, image_paths=None, dtype=None, cfg=None):
    """
    images have been normalized to the range [0, 1] instead of [0, 255]
    """
    batch_num, frame_num, image_dim, height, width = images.shape
    device = images.device
    reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)

    predictions = {}
    extra_dict = {}

    camera_predictor = model.camera_predictor
    track_predictor = model.track_predictor
    triangulator = model.triangulator

    # Find the query frames
    # First use DINO to find the most common frame among all the input frames
    # i.e., the one has highest (average) cosine similarity to all others
    # Then use farthest_point_sampling to find the next ones
    # The number of query frames is determined by query_frame_num
    
    with autocast(dtype=dtype):
        query_frame_indexes = find_query_frame_indexes(reshaped_image, camera_predictor, frame_num)

    image_paths = [os.path.basename(imgpath) for imgpath in image_paths]
    
    if cfg.center_order:
        # The code below switchs the first frame (frame 0) to the most common frame
        center_frame_index = query_frame_indexes[0]
        center_order = calculate_index_mappings(center_frame_index, frame_num, device=device)

        images, crop_params = switch_tensor_order([images, crop_params], center_order, dim=1)
        reshaped_image = switch_tensor_order([reshaped_image], center_order, dim=0)[0]

        image_paths = [image_paths[i] for i in center_order.cpu().numpy().tolist()]
        
        # Also update query_frame_indexes:
        query_frame_indexes = [center_frame_index if x == 0 else x for x in query_frame_indexes]
        query_frame_indexes[0] = 0


    # only pick query_frame_num 
    query_frame_indexes = query_frame_indexes[:query_frame_num]

    # Prepare image feature maps for tracker
    fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)

    # Predict tracks
    with autocast(dtype=dtype):
        pred_track, pred_vis, pred_score = predict_tracks(cfg.query_method, cfg.max_query_pts,
                                                          track_predictor, images, fmaps_for_tracker, 
                                                      query_frame_indexes, frame_num, device, cfg)



        if cfg.comple_nonvis:
            pred_track, pred_vis, pred_score = comple_nonvis_frames(track_predictor, images, fmaps_for_tracker, 
                                                                    frame_num, device, pred_track, pred_vis, 
                                                                    pred_score, 500,cfg=cfg)


                
    torch.cuda.empty_cache()

    # If necessary, force all the predictions at the padding areas as non-visible
    if crop_params is not None:
        boundaries = crop_params[:, :, -4:-2].abs().to(device)
        boundaries = torch.cat([boundaries, reshaped_image.shape[-1] - boundaries], dim=-1)
        hvis = torch.logical_and(
            pred_track[..., 1] >= boundaries[:, :, 1:2], pred_track[..., 1] <= boundaries[:, :, 3:4]
        )
        wvis = torch.logical_and(
            pred_track[..., 0] >= boundaries[:, :, 0:1], pred_track[..., 0] <= boundaries[:, :, 2:3]
        )
        force_vis = torch.logical_and(hvis, wvis)
        pred_vis = pred_vis * force_vis.float()

    # TODO: plot 2D matches
    if cfg.use_poselib:
        estimate_preliminary_cameras_fn = estimate_preliminary_cameras_poselib
    else:
        estimate_preliminary_cameras_fn = estimate_preliminary_cameras
    
    
    # Estimate preliminary_cameras by recovering fundamental/essential/homography matrix from 2D matches
    # By default, we use fundamental matrix estimation with 7p/8p+LORANSAC
    # All the operations are batched and differentiable (if necessary)
    # except when you enable use_poselib to save GPU memory
    _, preliminary_dict = estimate_preliminary_cameras_fn(
        pred_track,
        pred_vis,
        width,
        height,
        tracks_score=pred_score,
        max_error = cfg.fmat_thres,
        loopresidual=True,
        # max_ransac_iters=cfg.max_ransac_iters,
    )
    
    pose_predictions = camera_predictor(reshaped_image, batch_size=batch_num)

    pred_cameras = pose_predictions["pred_cameras"]

    # Conduct Triangulation and Bundle Adjustment
    BA_cameras_PT3D, extrinsics_opencv, intrinsics_opencv, points3D, points3D_rgb, reconstruction, valid_frame_mask= triangulator(
        pred_cameras,
        pred_track,
        pred_vis,
        images,
        preliminary_dict,
        image_paths=image_paths,
        crop_params=crop_params,
        pred_score=pred_score,
        fmat_thres=cfg.fmat_thres,
        BA_iters=cfg.BA_iters,
        init_max_reproj_error=cfg.init_max_reproj_error,
        cfg= cfg,
    )
    
    
    if cfg.center_order:
        # NOTE we changed the image order previously, now we need to switch it back    
        BA_cameras_PT3D = BA_cameras_PT3D[center_order]
        extrinsics_opencv = extrinsics_opencv[center_order]
        intrinsics_opencv = intrinsics_opencv[center_order]


    predictions["pred_cameras_PT3D"] = BA_cameras_PT3D
    predictions["extrinsics_opencv"] = extrinsics_opencv
    predictions["intrinsics_opencv"] = intrinsics_opencv
    predictions["points3D"] = points3D
    predictions["points3D_rgb"] = points3D_rgb
    predictions["reconstruction"] = reconstruction
    return predictions



def predict_tracks(query_method, max_query_pts, track_predictor, images, fmaps_for_tracker, query_frame_indexes, frame_num, device, cfg=None):
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for query_index in query_frame_indexes:
        print(f"Predicting tracks with query_index = {query_index}")
        
        # Find query_points at the query frame
        query_points = get_query_points(images[:, query_index], query_method, max_query_pts)

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(query_index, frame_num, device=device)
        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], new_order)


        # Feed into track predictor
        fine_pred_track, _, pred_vis, pred_score = track_predictor(images_feed, query_points, fmaps=fmaps_feed)

        # Switch back the predictions
        fine_pred_track, pred_vis, pred_score = switch_tensor_order([fine_pred_track, pred_vis, pred_score], new_order)

        # Append predictions for different queries
        pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)


    pred_track = torch.cat(pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    pred_score = torch.cat(pred_score_list, dim=2)

    return pred_track, pred_vis, pred_score


def comple_nonvis_frames(track_predictor, images, fmaps_for_tracker, frame_num, device, pred_track, pred_vis, pred_score, min_vis=500,cfg=None):
    # if a frame has too few visible inlier, use it as a query  
    non_vis_frames = torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis).squeeze(-1).tolist()
    last_query = -1
    while len(non_vis_frames) > 0:
        print("Processing non visible frames")
        print(non_vis_frames)
        if non_vis_frames[0] == last_query:
            print("The non vis frame still does not has enough 2D matches")
            pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(
                "sp+sift+aliked", cfg.max_query_pts // 2,
                track_predictor, images,
                fmaps_for_tracker, non_vis_frames,
                frame_num, device, cfg)
            # concat predictions
            pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
            pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
            pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
            break

        non_vis_query_list = [non_vis_frames[0]]
        last_query = non_vis_frames[0]
        pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(
            cfg.query_method, cfg.max_query_pts,
            track_predictor, images,
            fmaps_for_tracker, non_vis_query_list,
            frame_num, device, cfg)

        # concat predictions
        pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
        pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
        pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
        non_vis_frames = torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis).squeeze(-1).tolist()
    return pred_track, pred_vis, pred_score



def find_query_frame_indexes(reshaped_image, camera_predictor, query_frame_num, image_size=336):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = F.interpolate(reshaped_image, (image_size, image_size), mode="bilinear", align_corners=True)
    rgbs = camera_predictor._resnet_normalize_image(rgbs)

    # Get the image features (patch level)
    frame_feat = camera_predictor.backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
    similarity_matrix = similarity_matrix.mean(dim=0)
    distance_matrix = 1 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(0)

    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling
    # Starting from the most_common_frame_index,
    # try to find the farthest frame,
    # then the farthest to the last found frame
    # (frames are not allowed to be found twice)
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    return fps_idx



def get_query_points(query_image, query_method, max_query_num=4096, det_thres=0.005):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own

    methods = query_method.split('+')
    pred_points = []
    
    for method in methods:
        if "sp" in method:
            extractor = SuperPoint(max_num_keypoints=max_query_num, detection_threshold=det_thres).cuda().eval()
        elif "sift" in method:
            extractor = SIFT(max_num_keypoints=max_query_num).cuda().eval()            
        elif "aliked" in method:
            extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres).cuda().eval()
        else:
            raise NotImplementedError(f"query method {method} is not supprted now")
        
        query_points = extractor.extract(query_image)["keypoints"]
        pred_points.append(query_points)

    query_points = torch.cat(pred_points, dim=1)

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
