# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
import hydra

import cv2
import sys
import glob
import copy
import scipy

import random
import pycolmap
import datetime


import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tqdm import tqdm
from visdom import Visdom

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from lightglue import LightGlue, SuperPoint, SIFT, ALIKED


from collections import defaultdict
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.visualizer import Visualizer
from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras


from vggsfm.utils.utils import (
    read_array,
    write_array,
    generate_rank_by_dino,
    generate_rank_by_interval,
    farthest_point_sampling,
    calculate_index_mappings,
    extract_dense_depth_maps_and_save,
    align_dense_depth_maps_and_save,
    switch_tensor_order,
    visual_query_points,
    create_depth_map_visual,
    average_camera_prediction,
    create_video_with_reprojections,
)

try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import (
        estimate_preliminary_cameras_poselib,
    )

    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")

try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.vis.plotly_vis import plot_scene
    from pytorch3d.renderer.cameras import (
        PerspectiveCameras as PerspectiveCamerasVisual,
    )
except:
    print("PyTorch3d is not available. Please disable visdom. ")


class VGGSfMRunner:
    def __init__(self, cfg):
        """
        Wrapper for running VGGSfM
        """
        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.build_vggsfm_model()

        self.camera_predictor = self.vggsfm_model.camera_predictor
        self.track_predictor = self.vggsfm_model.track_predictor
        self.triangulator = self.vggsfm_model.triangulator

        if cfg.dense_depth:
            self.build_monocular_depth_model()

        if cfg.visualize:
            self.build_visdom()

        assert cfg.mixed_precision in ("None", "bf16", "fp16")
        if cfg.mixed_precision == "None":
            self.dtype = torch.float32
        elif cfg.mixed_precision == "bf16":
            # it seems bf16 may lead to some strange behaviors
            # best to use fp16 is possible
            self.dtype = torch.bfloat16
        elif cfg.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError(
                f"dtype {cfg.mixed_precision} is not supported now"
            )

        self.remove_borders = 4

    def build_vggsfm_model(self):
        print("Building VGGSfM")
        model = instantiate(self.cfg.MODEL, _recursive_=False, cfg=self.cfg)

        print("Loading VGGSfM Checkpoint")
        if self.cfg.auto_download_ckpt:
            _VGGSFM_URL = (
                "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_0_0.bin"
            )
            checkpoint = torch.hub.load_state_dict_from_url(_VGGSFM_URL)
        else:
            checkpoint = torch.load(self.cfg.resume_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Successfully resumed VGGSfM ckpt")
        self.vggsfm_model = model.to(self.device).eval()

    def build_monocular_depth_model(self):
        parent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        sys.path.append(parent_path)
        from dependency.depth_any_v2.depth_anything_v2.dpt import DepthAnythingV2

        print("DepthAnythingV2 is available")

        model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        depth_model = DepthAnythingV2(**model_config)
        _DEPTH_ANYTHING_V2_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        checkpoint = torch.hub.load_state_dict_from_url(_DEPTH_ANYTHING_V2_URL)
        depth_model.load_state_dict(checkpoint)
        depth_model = depth_model.to(self.device).eval()
        self.depth_model = depth_model
        print(f"Successfully built DepthAnythingV2")

    def build_visdom(self):
        viz = Visdom()
        self.viz = viz

    def run(
        self,
        images,
        masks=None,
        crop_params=None,
        image_paths=None,
        query_frame_num=None,
        seq_name=None,
        output_dir=None,
    ):
        
        if output_dir is None:
            now = datetime.datetime.now()            
            # Format the date and time as year_month_day_hour_minute
            timestamp = now.strftime("%Y%m%d_%H%M")            
            output_dir = f"output_{timestamp}"
            
        with torch.no_grad():
            images = images.to(self.device)
            
            if masks is not None:
                masks = masks.to(self.device)
            if crop_params is not None:
                crop_params = crop_params.to(self.device)

            # Check if images are in the shape of Tx3xHxW (single batch)
            if len(images.shape) == 4:
                # Add a batch dimension to images
                images = images.unsqueeze(0)
                if masks is not None: 
                    masks = masks.unsqueeze(0)
                if crop_params is not None: 
                    crop_params = crop_params.unsqueeze(0)


            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num

            predictions = self.run_one_scene(
                images,
                masks=masks,
                crop_params=crop_params,
                image_paths=image_paths,
                query_frame_num=query_frame_num,
                dtype=self.dtype,
                seq_name=seq_name,
                output_dir=output_dir,
            )
            
            self.save_reconstruction(predictions, seq_name, output_dir)
            return predictions

    def run_one_scene(
        self,
        images,
        masks=None,
        crop_params=None,
        query_frame_num=3,
        image_paths=None,
        seq_name=None,
        output_dir=None,
        dtype=None,
    ):
        """
        images have been normalized to the range [0, 1] instead of [0, 255]
        """
        
        print(f"Run Reconstruction for Scene {seq_name}")
        batch_num, frame_num, image_dim, height, width = images.shape
        device = images.device
        reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
        unproj_dense_points3D = None
        visual_dir = os.path.join(output_dir, "visuals")

        if self.cfg.dense_depth:
            print("Predicting dense depth maps via monocular depth estimation.")
            depth_dir = os.path.join(output_dir, "depths")
            extract_dense_depth_maps_and_save(self.depth_model, depth_dir, image_paths)

        predictions = {}
        extra_dict = {}

        # Find the query frames
        # First use DINO to find the most common frame among all the input frames
        # i.e., the one has highest (average) cosine similarity to all others
        # Then use farthest_point_sampling to find the next ones
        # The number of query frames is determined by query_frame_num

        with autocast(dtype=dtype):
            if self.cfg.query_by_interval:
                query_frame_indexes = generate_rank_by_interval(
                    frame_num, frame_num // query_frame_num
                )
            else:
                query_frame_indexes = generate_rank_by_dino(
                    reshaped_image, self.camera_predictor, frame_num
                )

        image_dir_prefix = os.path.dirname(image_paths[0])
        image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

        if self.cfg.center_order:
            # The code below switchs the first frame (frame 0) to the most common frame
            center_frame_index = query_frame_indexes[0]
            if center_frame_index != 0:
                center_order = calculate_index_mappings(
                    center_frame_index, frame_num, device=device
                )

                images, crop_params, masks = switch_tensor_order(
                    [images, crop_params, masks], center_order, dim=1
                )
                reshaped_image = switch_tensor_order(
                    [reshaped_image], center_order, dim=0
                )[0]

                image_paths = [
                    image_paths[i] for i in center_order.cpu().numpy().tolist()
                ]

                # Also update query_frame_indexes:
                query_frame_indexes = [
                    center_frame_index if x == 0 else x for x in query_frame_indexes
                ]
                query_frame_indexes[0] = 0

        # only pick query_frame_num
        query_frame_indexes = query_frame_indexes[:query_frame_num]

        # Prepare image feature maps for tracker
        fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(images)

        if crop_params is not None:
            bound_bboxes = crop_params[:, :, -4:-2].abs().to(device)
            # also remove those near the boundary
            bound_bboxes[bound_bboxes != 0] += self.remove_borders
            bound_bboxes = torch.cat(
                [bound_bboxes, reshaped_image.shape[-1] - bound_bboxes], dim=-1
            )

        # Predict tracks
        with autocast(dtype=dtype):
            pred_track, pred_vis, pred_score = predict_tracks(
                self.cfg.query_method,
                self.cfg.max_query_pts,
                self.track_predictor,
                images,
                masks,
                fmaps_for_tracker,
                query_frame_indexes,
                bound_bboxes,
            )

            if self.cfg.visual_tracks:
                vis = Visualizer(save_dir=visual_dir, linewidth=1)
                vis.visualize(images * 255, pred_track, pred_vis[..., None])

            if self.cfg.comple_nonvis:
                pred_track, pred_vis, pred_score = comple_nonvis_frames(
                    self.cfg.query_method,
                    self.cfg.max_query_pts,
                    self.track_predictor,
                    images,
                    masks,
                    fmaps_for_tracker,
                    [pred_track, pred_vis, pred_score],
                    bound_bboxes,
                )

        torch.cuda.empty_cache()

        # If necessary, force all the predictions at the padding areas as non-visible
        if crop_params is not None:
            hvis = torch.logical_and(
                pred_track[..., 1] >= bound_bboxes[:, :, 1:2],
                pred_track[..., 1] <= bound_bboxes[:, :, 3:4],
            )
            wvis = torch.logical_and(
                pred_track[..., 0] >= bound_bboxes[:, :, 0:1],
                pred_track[..., 0] <= bound_bboxes[:, :, 2:3],
            )
            force_vis = torch.logical_and(hvis, wvis)
            pred_vis = pred_vis * force_vis.float()

        if self.cfg.use_poselib:
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
            max_error=self.cfg.fmat_thres,
            loopresidual=True,
        )

        if self.cfg.avg_camera:
            pred_cameras = average_camera_prediction(
                self.camera_predictor,
                reshaped_image,
                batch_num,
                query_indices=query_frame_indexes,
            )
        else:
            pred_cameras = self.camera_predictor(reshaped_image, batch_size=batch_num)[
                "pred_cameras"
            ]

        # Conduct Triangulation and Bundle Adjustment
        # Force torch.float32 for triangulator
        with autocast(dtype=torch.float32):
            (
                extrinsics_opencv,
                intrinsics_opencv,
                points3D,
                points3D_rgb,
                reconstruction,
                valid_frame_mask,
            ) = self.triangulator(
                pred_cameras,
                pred_track,
                pred_vis,
                images,
                preliminary_dict,
                pred_score=pred_score,
                BA_iters=self.cfg.BA_iters,
                shared_camera=self.cfg.shared_camera,
                max_reproj_error=self.cfg.max_reproj_error,
                init_max_reproj_error=self.cfg.init_max_reproj_error,
                cfg=self.cfg,
            )

        rescale_camera = True
        img_size = images.shape[-1] # H/W
        
        print("*"*100)
        print("Renstruction Complete!")
        print("Prepare the output format now.")
        print("*"*100)
        
        for pyimageid in reconstruction.images:
            # scale from resized image size to the real size
            # rename the images to the original names
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid]

            if rescale_camera:
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = crop_params[0, pyimageid][:2]
                resize_ratio = real_image_size.max() / img_size
                real_focal = resize_ratio * pred_params[0]
                real_pp = real_image_size.cpu().numpy() // 2

                pred_params[0] = real_focal
                pred_params[1:3] = real_pp
                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

            if self.cfg.shift_point2d_to_original_res:
                top_left = crop_params[0, pyimageid][-4:-2].abs().cpu().numpy()
                for point2D in pyimage.points2D: 
                    point2D.xy = (point2D.xy - top_left) * resize_ratio.item()

            if self.cfg.shared_camera:
                # if shared_camera, all images share the same camera
                # no need to rescale any more
                rescale_camera = False


        if self.cfg.dense_depth or self.cfg.make_reproj_video:
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
                            
        if self.cfg.make_reproj_video:
            max_hw = crop_params[0, :, :2].max(dim=0)[0].long()
            video_size = (max_hw[0].item(), max_hw[1].item())
            create_video_with_reprojections(
                os.path.join(visual_dir, "reproj.mp4"),
                image_dir_prefix,
                video_size,
                reconstruction,
                image_paths,
                sparse_depth,
                sparse_point,
            )

        if self.cfg.dense_depth:
            print("Aligning dense depth maps by sparse SfM points")
            unproj_dense_points3D = align_dense_depth_maps_and_save(
                reconstruction, sparse_depth, depth_dir, image_dir_prefix,
                visual_dense_point_cloud=self.cfg.visual_dense_point_cloud,
            )

        if center_order is not None:
            # NOTE we changed the image order previously, now we need to scwitch it back
            extrinsics_opencv = extrinsics_opencv[center_order]
            intrinsics_opencv = intrinsics_opencv[center_order]

        predictions["extrinsics_opencv"] = extrinsics_opencv
        predictions["intrinsics_opencv"] = intrinsics_opencv
        predictions["points3D"] = points3D
        predictions["points3D_rgb"] = points3D_rgb
        predictions["reconstruction"] = reconstruction
        predictions["unproj_dense_points3D"] = unproj_dense_points3D

        return predictions




    def save_reconstruction(self, predictions, seq_name=None, output_dir=None):
        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        if output_dir is None:
            output_dir = os.path.join("output", seq_name)
            
        print("-" * 50)
        print(f"The output has been saved in COLMAP style at: {output_dir} ")
        os.makedirs(output_dir, exist_ok=True)
        reconstruction_pycolmap.write(output_dir)

        if self.cfg.visualize:
            if "points3D_rgb" in predictions:
                pcl = Pointclouds(
                    points=predictions["points3D"][None],
                    features=predictions["points3D_rgb"][None],
                )
            else:
                pcl = Pointclouds(points=predictions["points3D"][None])

            extrinsics_opencv = predictions["extrinsics_opencv"]
            # From OpenCV/COLMAP to PyTorch3D
            rot_PT3D = extrinsics_opencv[:, :3, :3].clone().permute(0, 2, 1)
            trans_PT3D = extrinsics_opencv[:, :3, 3].clone()
            trans_PT3D[:, :2] *= -1
            rot_PT3D[:, :, :2] *= -1
            visual_cameras = PerspectiveCamerasVisual(
                R=rot_PT3D, T=trans_PT3D, device=trans_PT3D.device
            )

            visual_dict = {"scenes": {"points": pcl, "cameras": visual_cameras}}

            unproj_dense_points3D = predictions["unproj_dense_points3D"]
            if unproj_dense_points3D is not None:
                unprojected_rgb_points_list = []
                for unproj_img_name in sorted(unproj_dense_points3D.keys()):
                    unprojected_rgb_points = torch.from_numpy(
                        unproj_dense_points3D[unproj_img_name]
                    )
                    unprojected_rgb_points_list.append(unprojected_rgb_points)

                    # Separate 3D point locations and RGB colors
                    point_locations = unprojected_rgb_points[0]  # 3D point location
                    rgb_colors = unprojected_rgb_points[1]  # RGB color

                    # Create a mask for points within the specified range
                    valid_mask = point_locations.abs().max(-1)[0] <= 512

                    # Create a Pointclouds object with valid points and their RGB colors
                    point_cloud = Pointclouds(
                        points=point_locations[valid_mask][None],
                        features=rgb_colors[valid_mask][None],
                    )

                    # Add the point cloud to the visual dictionary
                    visual_dict["scenes"][f"unproj_{unproj_img_name}"] = point_cloud

            fig = plot_scene(visual_dict, camera_scale=0.05)

            env_name = f"demo_visual_{seq_name}"
            print(f"Visualizing the scene by visdom at env: {env_name}")

            self.viz.plotlyplot(fig, env=env_name, win="3D")


################################################ Helper Functions


def predict_tracks(
    query_method,
    max_query_pts,
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    query_frame_indexes,
    bound_bboxes=None,
):
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    frame_num = images.shape[1]
    device = images.device

    for query_index in query_frame_indexes:
        print(f"Predicting tracks with query_index = {query_index}")

        if bound_bboxes is not None:
            bound_bbox = bound_bboxes[:, query_index]
        else:
            bound_bbox = None

        mask = masks[:, query_index] if masks is not None else None

        # Find query_points at the query frame
        query_points = get_query_points(
            images[:, query_index],
            mask,
            query_method,
            max_query_pts,
            bound_bbox=bound_bbox,
        )

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(query_index, frame_num, device=device)
        images_feed, fmaps_feed = switch_tensor_order(
            [images, fmaps_for_tracker], new_order
        )

        # Feed into track predictor
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed, query_points, fmaps=fmaps_feed
        )

        # Switch back the predictions
        fine_pred_track, pred_vis, pred_score = switch_tensor_order(
            [fine_pred_track, pred_vis, pred_score], new_order
        )

        # Append predictions for different queries
        pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    pred_track = torch.cat(pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    pred_score = torch.cat(pred_score_list, dim=2)

    return pred_track, pred_vis, pred_score


def comple_nonvis_frames(
    query_method,
    max_query_pts,
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    preds,
    bound_bboxes=None,
    min_vis=500,
):
    pred_track, pred_vis, pred_score = preds
    # if a frame has too few visible inlier, use it as a query
    non_vis_frames = (
        torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis)
        .squeeze(-1)
        .tolist()
    )
    last_query = -1
    final_trial = False

    while len(non_vis_frames) > 0:
        print("Processing non visible frames: ", non_vis_frames)

        if non_vis_frames[0] == last_query:
            print("The non visible frame still does not has enough 2D matches")
            final_trial = True
            query_method = "sp+sift+aliked"
            max_query_pts = max_query_pts // 2
            non_vis_query_list = non_vis_frames
        else:
            non_vis_query_list = [non_vis_frames[0]]

        last_query = non_vis_frames[0]
        pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(
            query_method,
            max_query_pts,
            track_predictor,
            images,
            masks,
            fmaps_for_tracker,
            non_vis_query_list,
            bound_bboxes,
        )

        # concat predictions
        pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
        pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
        pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
        non_vis_frames = (
            torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis)
            .squeeze(-1)
            .tolist()
        )

        if final_trial:
            break
    return pred_track, pred_vis, pred_score


def get_query_points(
    query_image,
    seg_invalid_mask,
    query_method,
    max_query_num=4096,
    det_thres=0.005,
    bound_bbox=None,
):
    # Run superpoint, sift, or aliked on the target frame
    # Feel free to modify for your own

    methods = query_method.split("+")
    pred_points = []

    for method in methods:
        if "sp" in method:
            extractor = (
                SuperPoint(
                    max_num_keypoints=max_query_num, detection_threshold=det_thres
                )
                .cuda()
                .eval()
            )
        elif "sift" in method:
            extractor = SIFT(max_num_keypoints=max_query_num).cuda().eval()
        elif "aliked" in method:
            extractor = (
                ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
                .cuda()
                .eval()
            )
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
            invalid_mask = (
                seg_invalid_mask
                if invalid_mask is None
                else torch.logical_or(invalid_mask, seg_invalid_mask)
            )

        query_points = extractor.extract(query_image, invalid_mask=invalid_mask)[
            "keypoints"
        ]
        pred_points.append(query_points)

    query_points = torch.cat(pred_points, dim=1)

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points


if __name__ == "__main__":
    from hydra import initialize, compose

    yaml_path = "../../cfgs/demo.yaml"
    config_dir = os.path.dirname(yaml_path)
    config_name = os.path.basename(yaml_path)

    with initialize(config_path=config_dir):
        config = compose(config_name=os.path.splitext(config_name)[0])

    config.make_reproj_video = True
    config.visual_tracks = True
    config.dense_depth = True
    config.query_frame_num = 3
    config.shift_point2d_to_original_res = True
    config.visualize = True
    config.visual_dense_point_cloud = True

    vggsfm_runner = VGGSfMRunner(config)

    debug_SCENE_DIR = "../../examples/kitchen"
    # debug_SCENE_DIR = "examples/kitchen"
    # Prepare test dataset
    test_dataset = DemoLoader(
        SCENE_DIR=debug_SCENE_DIR,
        img_size=config.img_size,
        normalize_cameras=False,
        load_gt=config.load_gt,
        cfg=config,
    )
    sequence_list = test_dataset.sequence_list

    # ensure deterministic
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    for seq_name in sequence_list:
        # Load the data
        batch, image_paths = test_dataset.get_data(
            sequence_name=seq_name, return_path=True
        )

        # Send to GPU
        images = batch["image"]
        crop_params = batch["crop_params"]
        # Unsqueeze to have batch size = 1
        images = images.unsqueeze(0)
        crop_params = crop_params.unsqueeze(0)

        # import pdb;pdb.set_trace()
        # FOR DEPTH ANYTHING
        # images_bgr_unpadded = batch['images_bgr_unpadded']

        if batch["masks"] is not None:
            masks = batch["masks"].unsqueeze(0)
        else:
            masks = None
            
        output_dir = "output_tmp/"
        
        vggsfm_runner.run(
            images,
            masks,
            crop_params,
            image_paths,
            seq_name=seq_name,
            output_dir=output_dir,
        )
        
    import pdb
    pdb.set_trace()
    m = 1
