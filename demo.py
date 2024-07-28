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

import glob
import copy
import cv2
import scipy
from visdom import Visdom
from collections import defaultdict
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras

from vggsfm.utils.utils import (
    read_array,
    write_array,
    generate_rank_by_interval,
    farthest_point_sampling,
    calculate_index_mappings,
    align_dense_depth_maps,
    switch_tensor_order,
    visual_query_points,
    create_depth_map_visual,
    create_video_with_reprojections,
)

try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras_poselib

    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")

try:
    from dependency.depth_any_v2.depth_anything_v2.dpt import DepthAnythingV2

    print("DepthAnythingV2 is available")
except:
    print("DepthAnythingV2 is not installed. Please disable dense_depth")


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
        from pytorch3d.renderer.cameras import PerspectiveCameras as PerspectiveCamerasVisual

        viz = Visdom()

    sequence_list = test_dataset.sequence_list

    for seq_name in sequence_list:
        print("*" * 50 + f" Testing on Scene {seq_name} " + "*" * 50)

        # Load the data
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)

        # Send to GPU
        images = batch["image"].to(device)
        crop_params = batch["crop_params"].to(device)

        # Unsqueeze to have batch size = 1
        images = images.unsqueeze(0)
        crop_params = crop_params.unsqueeze(0)


        if batch["masks"] is not None:
            masks = batch["masks"].to(device)
            masks = masks.unsqueeze(0)   
        else:
            masks = None 


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
                masks=masks,
                crop_params=crop_params,
                query_frame_num=cfg.query_frame_num,
                image_paths=image_paths,
                dtype=dtype,
                cfg=cfg,
            )

        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        output_path = os.path.join("output", seq_name)
        print("-" * 50)
        print(f"The output has been saved in COLMAP style at: {output_path} ")
        os.makedirs(output_path, exist_ok=True)
        reconstruction_pycolmap.write(output_path)

        pred_cameras_PT3D = predictions["pred_cameras_PT3D"]

        if cfg.visualize:
            if "points3D_rgb" in predictions:
                pcl = Pointclouds(points=predictions["points3D"][None], features=predictions["points3D_rgb"][None])
            else:
                pcl = Pointclouds(points=predictions["points3D"][None])

            visual_cameras = PerspectiveCamerasVisual(
                R=pred_cameras_PT3D.R, T=pred_cameras_PT3D.T, device=pred_cameras_PT3D.device
            )

            visual_dict = {"scenes": {"points": pcl, "cameras": visual_cameras}}

            unproj_dense_points3D = predictions["unproj_dense_points3D"]
            if unproj_dense_points3D is not None:
                unproj_pts = unproj_dense_points3D[0]
                unproj_pts_rgb = unproj_dense_points3D[1]
                # mask out too far away 3D points due to depth unprojection 
                mask_for_visual = unproj_pts.abs().max(-1)[0] <= 512
                dense_pcl = Pointclouds(points=unproj_pts[mask_for_visual][None], 
                                        features=unproj_pts_rgb[mask_for_visual][None])
                visual_dict["scenes"]["dense_points"] = dense_pcl

            fig = plot_scene(visual_dict, camera_scale=0.05)

            env_name = f"demo_visual_{seq_name}"
            print(f"Visualizing the scene by visdom at env: {env_name}")
            
            viz.plotlyplot(fig, env=env_name, win="3D")

    return True


def run_one_scene(model, images, masks=None, crop_params=None, query_frame_num=3, image_paths=None, dtype=None, cfg=None):
    """
    images have been normalized to the range [0, 1] instead of [0, 255]
    """

    batch_num, frame_num, image_dim, height, width = images.shape
    device = images.device
    reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)

    if cfg.dense_depth:
        print("Extracting dense depth maps")
        depth_dir = os.path.join(cfg.SCENE_DIR, "depths")
        os.makedirs(depth_dir, exist_ok=True)
        # Build DepthAnythingV2 model
        model_config = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
        depth_model = DepthAnythingV2(**model_config)
        _DEPTH_ANYTHING_V2_URL = (
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        )
        checkpoint = torch.hub.load_state_dict_from_url(_DEPTH_ANYTHING_V2_URL)
        depth_model.load_state_dict(checkpoint)
        depth_model = depth_model.to(device).eval()

        for idx in range(len(image_paths)):
            img_fname = image_paths[idx]
            raw_img = cv2.imread(img_fname)
            # raw resolution
            disp_map = depth_model.infer_image(
                raw_img, min(1024, max(raw_img.shape[:2]))
            )  # HxW raw depth_map map in numpy

            if cfg.visual_depths:
                create_depth_map_visual(disp_map, raw_img, img_fname, depth_dir)

            write_array(disp_map, img_fname.replace("images", "depths") + ".bin")

        print("Monocular depth maps complete. Depth alignment to be conducted.")

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
        if cfg.query_by_interval:
            query_frame_indexes = generate_rank_by_interval(frame_num, frame_num//query_frame_num)
        else:
            query_frame_indexes = generate_rank_by_dino(reshaped_image, camera_predictor, frame_num)


    image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

    if cfg.center_order:
        # The code below switchs the first frame (frame 0) to the most common frame
        center_frame_index = query_frame_indexes[0]
        if center_frame_index !=0:
            center_order = calculate_index_mappings(center_frame_index, frame_num, device=device)
            
            images, crop_params, masks = switch_tensor_order([images, crop_params, masks], center_order, dim=1)
            reshaped_image = switch_tensor_order([reshaped_image], center_order, dim=0)[0]

            image_paths = [image_paths[i] for i in center_order.cpu().numpy().tolist()]

            # Also update query_frame_indexes:
            query_frame_indexes = [center_frame_index if x == 0 else x for x in query_frame_indexes]
            query_frame_indexes[0] = 0

    # only pick query_frame_num
    query_frame_indexes = query_frame_indexes[:query_frame_num]

    # Prepare image feature maps for tracker
    fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)


    if crop_params is not None:
        bound_bboxes = crop_params[:, :, -4:-2].abs().to(device)

        # also remove those near the boundary
        remove_borders = 4
        bound_bboxes[bound_bboxes != 0] += remove_borders

        bound_bboxes = torch.cat([bound_bboxes, reshaped_image.shape[-1] - bound_bboxes], dim=-1)


    # Predict tracks
    with autocast(dtype=dtype):
        pred_track, pred_vis, pred_score = predict_tracks(
            cfg.query_method,
            cfg.max_query_pts,
            track_predictor,
            images,
            masks,
            fmaps_for_tracker,
            query_frame_indexes,
            frame_num,
            device,
            bound_bboxes,
            cfg,
        )

        if cfg.comple_nonvis:
            pred_track, pred_vis, pred_score = comple_nonvis_frames(
                track_predictor,
                images,
                masks,
                fmaps_for_tracker,
                frame_num,
                device,
                pred_track,
                pred_vis,
                pred_score,
                500,
                bound_bboxes,
                cfg=cfg,
            )

    torch.cuda.empty_cache()

    # If necessary, force all the predictions at the padding areas as non-visible
    if crop_params is not None:
        hvis = torch.logical_and(
            pred_track[..., 1] >= bound_bboxes[:, :, 1:2], pred_track[..., 1] <= bound_bboxes[:, :, 3:4]
        )
        wvis = torch.logical_and(
            pred_track[..., 0] >= bound_bboxes[:, :, 0:1], pred_track[..., 0] <= bound_bboxes[:, :, 2:3]
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
        pred_track, pred_vis, width, height, tracks_score=pred_score, max_error=cfg.fmat_thres, loopresidual=True
    )

    pose_predictions = camera_predictor(reshaped_image, batch_size=batch_num)

    pred_cameras = pose_predictions["pred_cameras"]

    # Conduct Triangulation and Bundle Adjustment
    (
        BA_cameras_PT3D,
        extrinsics_opencv,
        intrinsics_opencv,
        points3D,
        points3D_rgb,
        reconstruction,
        valid_frame_mask,
    ) = triangulator(
        pred_cameras,
        pred_track,
        pred_vis,
        images,
        preliminary_dict,
        pred_score=pred_score,
        BA_iters=cfg.BA_iters,
        shared_camera=cfg.shared_camera,
        max_reproj_error=cfg.max_reproj_error,
        init_max_reproj_error=cfg.init_max_reproj_error,
        cfg=cfg,
    )
    
    rescale_camera = True
    
    for pyimageid in reconstruction.images:
        # scale from resized image size to the real size
        # rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = crop_params[0, pyimageid][:2]
            real_focal = real_image_size.max() / cfg.img_size * pred_params[0]
            real_pp = real_image_size.cpu().numpy() // 2

            pred_params[0] = real_focal
            pred_params[1:3] = real_pp
            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if cfg.shared_camera:
            # if shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False
            

    if cfg.dense_depth or cfg.make_reproj_video:
        sparse_depth = defaultdict(list)
        sparse_point = defaultdict(list)
        # Extract sparse depths from SfM points
        for point3D_idx in reconstruction.points3D:
            pt3D = reconstruction.points3D[point3D_idx]
            for track_element in pt3D.track.elements:
                pyimg = reconstruction.images[track_element.image_id]
                pycam = reconstruction.cameras[pyimg.camera_id]
                img_name = pyimg.name
                projection = pyimg.cam_from_world * pt3D.xyz
                depth = projection[-1]
                # NOTE: uv here cooresponds to the (x, y)
                # at the original image coordinate
                # instead of the cropped one
                uv = pycam.img_from_cam(projection)
                sparse_depth[img_name].append(np.append(uv, depth))
                sparse_point[img_name].append(np.append(pt3D.xyz, point3D_idx))


    if cfg.make_reproj_video:
        max_hw = crop_params[0,:,:2].max(dim=0)[0].long()
        video_size = (max_hw[0].item(), max_hw[1].item())
        output_path = os.path.join(cfg.SCENE_DIR, "reproj.mp4")
        fname_prefix = os.path.join(cfg.SCENE_DIR, "images")
        create_video_with_reprojections(output_path, fname_prefix, video_size, 
                                        reconstruction, image_paths, 
                                        sparse_depth, sparse_point)


    if cfg.dense_depth:
        print("Aligning dense depth maps by sparse SfM points")
        unproj_dense_points3D = align_dense_depth_maps(reconstruction, sparse_depth, depth_dir, cfg)

    if cfg.visual_dense_point_cloud:
        assert cfg.dense_depth, "dense_depth must be True when visual_dense_point_cloud is True."
        unproj_dense_points3D = np.concatenate(unproj_dense_points3D, axis=1)
        unproj_dense_points3D = torch.from_numpy(unproj_dense_points3D)
    else:
        unproj_dense_points3D = None
        
    predictions["pred_cameras_PT3D"] = BA_cameras_PT3D
    predictions["extrinsics_opencv"] = extrinsics_opencv
    predictions["intrinsics_opencv"] = intrinsics_opencv
    predictions["points3D"] = points3D
    predictions["points3D_rgb"] = points3D_rgb
    predictions["reconstruction"] = reconstruction
    predictions["unproj_dense_points3D"] = unproj_dense_points3D

    return predictions


################################################ Helper Functions


def predict_tracks(
    query_method,
    max_query_pts,
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    query_frame_indexes,
    frame_num,
    device,
    bound_bboxes=None,
    cfg=None,
):
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []
    
    if cfg.visual_query_points:
        query_dir = os.path.join(cfg.SCENE_DIR, "query")
        os.makedirs(query_dir, exist_ok=True)

    for query_index in query_frame_indexes:
        print(f"Predicting tracks with query_index = {query_index}")

        if bound_bboxes is not None:
            bound_bbox = bound_bboxes[:, query_index]
        else:
            bound_bbox = None
            
        mask = masks[:, query_index] if masks is not None else None

        # Find query_points at the query frame
        query_points = get_query_points(images[:, query_index], mask, query_method, max_query_pts, bound_bbox=bound_bbox)        
        
        if cfg.visual_query_points:
            save_name = os.path.join(query_dir, f"query_{query_index}.png")
            visual_query_points(images, query_index, query_points, save_name=save_name)
            
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


def comple_nonvis_frames(
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    frame_num,
    device,
    pred_track,
    pred_vis,
    pred_score,
    min_vis=500,
    bound_bboxes=None,
    cfg=None,
):
    # if a frame has too few visible inlier, use it as a query
    non_vis_frames = torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis).squeeze(-1).tolist()
    last_query = -1
    while len(non_vis_frames) > 0:
        print("Processing non visible frames")
        print(non_vis_frames)
        if non_vis_frames[0] == last_query:
            print("The non vis frame still does not has enough 2D matches")
            pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(
                "sp+sift+aliked",
                cfg.max_query_pts // 2,
                track_predictor,
                images,
                masks,
                fmaps_for_tracker,
                non_vis_frames,
                frame_num,
                device,
                bound_bboxes,
                cfg,
            )
            # concat predictions
            pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
            pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
            pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
            break

        non_vis_query_list = [non_vis_frames[0]]
        last_query = non_vis_frames[0]
        pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(
            cfg.query_method,
            cfg.max_query_pts,
            track_predictor,
            images,
            masks,
            fmaps_for_tracker,
            non_vis_query_list,
            frame_num,
            device,
            bound_bboxes,
            cfg,
        )

        # concat predictions
        pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
        pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
        pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
        non_vis_frames = torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis).squeeze(-1).tolist()
    return pred_track, pred_vis, pred_score


def generate_rank_by_dino(reshaped_image, camera_predictor, query_frame_num, image_size=336):
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
    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)

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


def get_query_points(query_image, seg_invalid_mask, query_method, max_query_num=4096, det_thres=0.005, bound_bbox=None):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own

    methods = query_method.split("+")
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

        invalid_mask = None

        if bound_bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bound_bbox[0])
            bbox_valid_mask = torch.zeros_like(query_image[:, 0], dtype=torch.bool)
            bbox_valid_mask[:, y_min:y_max, x_min:x_max] = 1
            invalid_mask = ~bbox_valid_mask

        if seg_invalid_mask is not None:
            seg_invalid_mask = seg_invalid_mask.squeeze(1).bool()
            invalid_mask = seg_invalid_mask if invalid_mask is None else torch.logical_or(invalid_mask, seg_invalid_mask)
    
        query_points = extractor.extract(query_image, invalid_mask=invalid_mask)["keypoints"]
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
