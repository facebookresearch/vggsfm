# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import copy
import torch
import pycolmap
import datetime

import time
import numpy as np
from visdom import Visdom
from torch.cuda.amp import autocast
from hydra.utils import instantiate
from lightglue import SuperPoint, SIFT, ALIKED

from collections import defaultdict
from vggsfm.utils.visualizer import Visualizer
from vggsfm.two_view_geo.estimate_preliminary import (
    estimate_preliminary_cameras,
)

from vggsfm.utils.utils import (
    write_array,
    generate_grid_samples,
    generate_rank_by_midpoint,
    generate_rank_by_dino,
    generate_rank_by_interval,
    calculate_index_mappings,
    extract_dense_depth_maps,
    align_dense_depth_maps,
    switch_tensor_order,
    sample_subrange,
    average_camera_prediction,
    create_video_with_reprojections,
    save_video_with_reprojections,
)


from vggsfm.utils.triangulation import triangulate_tracks
from vggsfm.utils.triangulation_helpers import cam_from_img, filter_all_points3D


# Optional imports
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
    print("PyTorch3d is not available. Please disable visdom.")


class VGGSfMRunner:
    def __init__(self, cfg):
        """
        A runner class for the VGGSfM (Structure from Motion) pipeline.

        This class encapsulates the entire SfM process, including model initialization,
        sparse and dense reconstruction, and visualization.

        Args:
            cfg: Configuration object containing pipeline settings.
        """

        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.build_vggsfm_model()
        self.camera_predictor = self.vggsfm_model.camera_predictor
        self.track_predictor = self.vggsfm_model.track_predictor
        self.triangulator = self.vggsfm_model.triangulator

        if cfg.dense_depth:
            self.build_monocular_depth_model()

        if cfg.viz_visualize:
            self.build_visdom()

        # Set up mixed precision
        assert cfg.mixed_precision in ("None", "bf16", "fp16")
        self.dtype = {
            "None": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(cfg.mixed_precision, None)

        if self.dtype is None:
            raise NotImplementedError(
                f"dtype {cfg.mixed_precision} is not supported now"
            )

        # Remove the pixels too close to the border
        self.remove_borders = 4

    def build_vggsfm_model(self):
        """
        Builds the VGGSfM model and loads the checkpoint.

        Initializes the VGGSfM model and loads the weights from a checkpoint.
        The model is then moved to the appropriate device and set to evaluation mode.
        """

        print("Building VGGSfM")

        vggsfm = instantiate(self.cfg.MODEL, _recursive_=False, cfg=self.cfg)

        if self.cfg.auto_download_ckpt:
            vggsfm.from_pretrained(self.cfg.model_name)
        else:
            checkpoint = torch.load(self.cfg.resume_ckpt)
            vggsfm.load_state_dict(checkpoint, strict=True)
        self.vggsfm_model = vggsfm.to(self.device).eval()
        print("VGGSfM built successfully")

    def build_monocular_depth_model(self):
        """
        Builds the monocular depth model and loads the checkpoint.

        This function initializes the DepthAnythingV2 model,
        downloads the pre-trained weights from a URL, and loads these weights into the model.
        The model is then moved to the appropriate device and set to evaluation mode.
        """
        # Import DepthAnythingV2 inside the function to avoid unnecessary imports

        parent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        sys.path.append(parent_path)
        from dependency.depth_any_v2.depth_anything_v2.dpt import (
            DepthAnythingV2,
        )

        print("Building DepthAnythingV2")
        model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        depth_model = DepthAnythingV2(**model_config)
        _DEPTH_ANYTHING_V2_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        checkpoint = torch.hub.load_state_dict_from_url(_DEPTH_ANYTHING_V2_URL)
        depth_model.load_state_dict(checkpoint)
        self.depth_model = depth_model.to(self.device).eval()
        print(f"DepthAnythingV2 built successfully")

    def build_visdom(self):
        """
        Set up a Visdom server for visualization.
        """
        self.viz = Visdom()

    def run(
        self,
        images,
        masks=None,
        original_images=None,
        image_paths=None,
        crop_params=None,
        query_frame_num=None,
        seq_name=None,
        output_dir=None,
    ):
        """
        Executes the full VGGSfM pipeline on a set of input images.

        This method orchestrates the entire reconstruction process, including sparse
        reconstruction, dense reconstruction (if enabled), and visualization.

        Args:
            images (torch.Tensor): Input images with shape Tx3xHxW or BxTx3xHxW, where T is
                the number of frames, B is the batch size, H is the height, and W is the
                width. The values should be in the range (0,1).
            masks (torch.Tensor, optional): Input masks with shape Tx1xHxW or BxTx1xHxW.
                Binary masks where 1 indicates the pixel is filtered out.
            original_images (dict, optional): Dictionary with image basename as keys and original
                numpy images (rgb) as values.
            image_paths (list of str, optional): List of paths to input images. If not
                provided, you can use placeholder names such as image0000.png, image0001.png.
            crop_params (torch.Tensor, optional): A tensor with shape Tx8 or BxTx8. Crop parameters
                indicating the mapping from the original image to the processed one (We pad
                and resize the original images to a fixed size.).
            query_frame_num (int, optional): Number of query frames to be used. If not
                specified, will use self.cfg.query_frame_num.
            seq_name (str, optional): Name of the sequence.
            output_dir (str, optional): Directory to save the output. If not specified,
                the directory will be named as f"{seq_name}_{timestamp}".
        Returns:
            dict: A dictionary containing the predictions from the reconstruction process.
        """
        if output_dir is None:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            output_dir = f"{seq_name}_{timestamp}"

        with torch.no_grad():
            images = move_to_device(images, self.device)
            masks = move_to_device(masks, self.device)
            crop_params = move_to_device(crop_params, self.device)

            # Add batch dimension if necessary
            if len(images.shape) == 4:
                images = add_batch_dimension(images)
                masks = add_batch_dimension(masks)
                crop_params = add_batch_dimension(crop_params)

            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num

            # Perform sparse reconstruction
            predictions = self.sparse_reconstruct(
                images,
                masks=masks,
                crop_params=crop_params,
                image_paths=image_paths,
                query_frame_num=query_frame_num,
                seq_name=seq_name,
                output_dir=output_dir,
            )

            # Save the sparse reconstruction results
            if self.cfg.save_to_disk:
                self.save_sparse_reconstruction(
                    predictions, seq_name, output_dir
                )
                
                if predictions["additional_points_dict"] is not None:
                    additional_dir = os.path.join(output_dir, "additional")
                    os.makedirs(additional_dir, exist_ok=True)
                    torch.save(predictions["additional_points_dict"], os.path.join(additional_dir, "additional_points_dict.pt"))


            # Extract sparse depth and point information if needed for further processing
            if self.cfg.dense_depth or self.cfg.make_reproj_video:
                predictions = (
                    self.extract_sparse_depth_and_point_from_reconstruction(
                        predictions
                    )
                )

            # Perform dense reconstruction if enabled
            if self.cfg.dense_depth:
                predictions = self.dense_reconstruct(
                    predictions, image_paths, original_images
                )

                # Save the dense depth maps
                if self.cfg.save_to_disk:
                    self.save_dense_depth_maps(
                        predictions["depth_dict"], output_dir
                    )

            # Create reprojection video if enabled
            if self.cfg.make_reproj_video:
                max_hw = crop_params[0, :, :2].max(dim=0)[0].long()
                video_size = (max_hw[0].item(), max_hw[1].item())
                img_with_circles_list = self.make_reprojection_video(
                    predictions, video_size, image_paths, original_images
                )
                predictions["reproj_video"] = img_with_circles_list
                if self.cfg.save_to_disk:
                    self.save_reprojection_video(
                        img_with_circles_list, video_size, output_dir
                    )

            # Visualize the 3D reconstruction if enabled
            if self.cfg.viz_visualize:
                self.visualize_3D_in_visdom(predictions, seq_name, output_dir)

            if self.cfg.gr_visualize:
                self.visualize_3D_in_gradio(predictions, seq_name, output_dir)

            return predictions

    def sparse_reconstruct(
        self,
        images,
        masks=None,
        crop_params=None,
        query_frame_num=3,
        image_paths=None,
        seq_name=None,
        output_dir=None,
        dtype=None,
        back_to_original_resolution=True,
    ):
        """
        Perform sparse reconstruction on the given images.

        This function implements the core SfM pipeline, including:
        1. Selecting query frames
        2. Estimating camera poses
        3. Predicting feature tracks across frames
        4. Triangulating 3D points
        5. Performing bundle adjustment

        Args:
            images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
            masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images.
            crop_params (torch.Tensor): A tensor of shape (B, T, 4), indicating the mapping from the original image
                                    to the processed one (We pad and resize the original images to a fixed size.).
            query_frame_num (int): The number of query frames to use for reconstruction. Default is 3.
            image_paths (list): A list of image file paths corresponding to the input images.
            seq_name (str): The name of the sequence being processed.
            output_dir (str): The directory to save the output files.
            dtype (torch.dtype): The data type to use for computations.

            NOTE During inference we force B=1 now.
        Returns:
            dict: A dictionary containing the reconstruction results, including camera parameters and 3D points.
        """

        print(f"Run Sparse Reconstruction for Scene {seq_name}")
        batch_num, frame_num, image_dim, height, width = images.shape
        device = images.device
        reshaped_image = images.reshape(
            batch_num * frame_num, image_dim, height, width
        )
        visual_dir = os.path.join(output_dir, "visuals")

        if dtype is None:
            dtype = self.dtype

        predictions = {}

        # Find the query frames using DINO or frame names
        with autocast(dtype=dtype):
            if self.cfg.query_by_midpoint:
                query_frame_indexes = generate_rank_by_midpoint(frame_num)
            elif self.cfg.query_by_interval:
                query_frame_indexes = generate_rank_by_interval(
                    frame_num, (frame_num // query_frame_num + 1)
                )
            else:
                query_frame_indexes = generate_rank_by_dino(
                    reshaped_image, self.camera_predictor, frame_num
                )

        # Extract base names from image paths
        image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

        center_order = None
        # Reorder frames if center_order is enabled
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
                    center_frame_index if x == 0 else x
                    for x in query_frame_indexes
                ]
                query_frame_indexes[0] = 0

        # Select only the specified number of query frames
        query_frame_indexes = query_frame_indexes[:query_frame_num]

        # Predict Camera Parameters by camera_predictor
        if self.cfg.avg_pose:
            # Conduct several times with different frames as the query frame
            # self.camera_predictor is super fast and this is almost a free-lunch
            pred_cameras = average_camera_prediction(
                self.camera_predictor,
                reshaped_image,
                batch_num,
                query_indices=query_frame_indexes,
            )
        else:
            pred_cameras = self.camera_predictor(
                reshaped_image, batch_size=batch_num
            )["pred_cameras"]

        # Prepare image feature maps for tracker
        fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(images)

        # Calculate bounding boxes if crop parameters are provided
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
                self.cfg.fine_tracking,
                bound_bboxes,
            )

            # Complement non-visible frames if enabled
            if self.cfg.comple_nonvis:
                pred_track, pred_vis, pred_score = comple_nonvis_frames(
                    self.cfg.query_method,
                    self.cfg.max_query_pts,
                    self.track_predictor,
                    images,
                    masks,
                    fmaps_for_tracker,
                    [pred_track, pred_vis, pred_score],
                    self.cfg.fine_tracking,
                    bound_bboxes,
                )

        # Visualize tracks as a video if enabled
        if self.cfg.visual_tracks:
            vis = Visualizer(save_dir=visual_dir, linewidth=1)
            vis.visualize(
                images * 255, pred_track, pred_vis[..., None], filename="track"
            )

        torch.cuda.empty_cache()

        # Force predictions in padding areas as non-visible
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
            estimate_preliminary_cameras_fn = (
                estimate_preliminary_cameras_poselib
            )
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

        # Perform triangulation and bundle adjustment
        with autocast(dtype=torch.float32):
            (
                extrinsics_opencv,
                intrinsics_opencv,
                extra_params,
                points3D,
                points3D_rgb,
                reconstruction,
                valid_frame_mask,
                valid_2D_mask,
                valid_tracks,
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
                extract_color=self.cfg.extract_color,
                robust_refine=self.cfg.robust_refine,
                camera_type=self.cfg.camera_type,
            )

        
        additional_points_dict = None
        
        if self.cfg.extra_pt_pixel_interval > 0:
            additional_points_dict = self.triangulate_extra_points(
                images,
                masks,
                fmaps_for_tracker,
                bound_bboxes,
                intrinsics_opencv,
                extra_params,
                extrinsics_opencv,
                image_paths,
                frame_num,
            )
            additional_points3D = torch.cat(
                [
                    additional_points_dict[img_name]["points3D"]
                    for img_name in image_paths
                ],
                dim=0,
            )
            additional_points3D_rgb = torch.cat(
                [
                    additional_points_dict[img_name]["points3D_rgb"]
                    for img_name in image_paths
                ],
                dim=0,
            )

            additional_points_dict["sfm_points_num"] = len(points3D)
            additional_points_dict["additional_points_num"] = len(additional_points3D)

            if self.cfg.concat_extra_points:
                additional_points3D_numpy = additional_points3D.cpu().numpy()
                additional_points3D_rgb_numpy = (
                    (additional_points3D_rgb * 255).long().cpu().numpy()
                )
                for extra_point_idx in range(len(additional_points3D)):
                    reconstruction.add_point3D(
                        additional_points3D_numpy[extra_point_idx],
                        pycolmap.Track(),
                        additional_points3D_rgb_numpy[extra_point_idx],
                    )
                    
                points3D = torch.cat([points3D, additional_points3D], dim=0)
                points3D_rgb = torch.cat(
                    [points3D_rgb, additional_points3D_rgb], dim=0
                )

        if self.cfg.filter_invalid_frame:
            extrinsics_opencv = extrinsics_opencv[valid_frame_mask]
            intrinsics_opencv = intrinsics_opencv[valid_frame_mask]
            if extra_params is not None:
                extra_params = extra_params[valid_frame_mask]
            invalid_ids = torch.nonzero(~valid_frame_mask).squeeze(1)
            invalid_ids = invalid_ids.cpu().numpy().tolist()
            if len(invalid_ids) > 0:
                for invalid_id in invalid_ids:
                    reconstruction.deregister_image(invalid_id)

        img_size = images.shape[-1]  # H or W, the same for square

        if center_order is not None:
            # NOTE we changed the image order previously, now we need to scwitch it back
            extrinsics_opencv = extrinsics_opencv[center_order]
            intrinsics_opencv = intrinsics_opencv[center_order]
            if extra_params is not None:
                extra_params = extra_params[center_order]
            pred_track = pred_track[:, center_order]
            pred_vis = pred_vis[:, center_order]
            if pred_score is not None:
                pred_score = pred_score[:, center_order]


        if back_to_original_resolution:
            reconstruction = self.rename_colmap_recons_and_rescale_camera(
                reconstruction,
                image_paths,
                crop_params,
                img_size,
                shared_camera=self.cfg.shared_camera,
                shift_point2d_to_original_res=self.cfg.shift_point2d_to_original_res,
            )

            # Also rescale the intrinsics_opencv tensor
            fname_to_id = {
                reconstruction.images[imgid].name: imgid
                for imgid in reconstruction.images
            }
            intrinsics_original_res = []
            # We assume the returned extri and intri cooresponds to the order of sorted image_paths
            for fname in sorted(image_paths):
                pyimg = reconstruction.images[fname_to_id[fname]]
                pycam = reconstruction.cameras[pyimg.camera_id]
                intrinsics_original_res.append(pycam.calibration_matrix())
            intrinsics_opencv = torch.from_numpy(
                np.stack(intrinsics_original_res)
            ).to(device)

        predictions["extrinsics_opencv"] = extrinsics_opencv
        # NOTE! If not back_to_original_resolution, then intrinsics_opencv
        # cooresponds to the resized one (e.g., 1024x1024)
        predictions["intrinsics_opencv"] = intrinsics_opencv
        predictions["points3D"] = points3D
        predictions["points3D_rgb"] = points3D_rgb
        predictions["reconstruction"] = reconstruction
        predictions["extra_params"] = extra_params
        predictions["unproj_dense_points3D"] = None  # placeholder here
        predictions["valid_2D_mask"] = valid_2D_mask
        predictions["pred_track"] = pred_track
        predictions["pred_vis"] = pred_vis
        predictions["pred_score"] = pred_score
        predictions["valid_tracks"] = valid_tracks
        
        predictions["additional_points_dict"] = additional_points_dict
        
        return predictions

    def triangulate_extra_points(
        self,
        images,
        masks,
        fmaps_for_tracker,
        bound_bboxes,
        intrinsics_opencv,
        extra_params,
        extrinsics_opencv,
        image_paths,
        frame_num,
    ):
        """
        Triangulate extra points for each frame and return a dictionary containing 3D points and their RGB values.

        Returns:
            dict: A dictionary containing 3D points and their RGB values for each frame.
        """
        from vggsfm.models.utils import sample_features4d

        additional_points_dict = {}
        for frame_idx in range(frame_num):
            rect_for_sample = bound_bboxes[:, frame_idx].clone()
            rect_for_sample = rect_for_sample.floor()
            rect_for_sample[:, :2] += self.cfg.extra_pt_pixel_interval // 2
            rect_for_sample[:, 2:] -= self.cfg.extra_pt_pixel_interval // 2
            grid_points = generate_grid_samples(
                rect_for_sample, pixel_interval=self.cfg.extra_pt_pixel_interval
            )
            grid_points = grid_points.floor()

            grid_rgb = sample_features4d(
                images[:, frame_idx], grid_points[None]
            ).squeeze(0)

            if self.cfg.extra_by_neighbor > 0:
                neighbor_start, neighbor_end = sample_subrange(
                    frame_num, frame_idx, self.cfg.extra_by_neighbor
                )
            else:
                neighbor_start = 0
                neighbor_end = frame_num

            rel_frame_idx = frame_idx - neighbor_start

            extra_track, extra_vis, extra_score = predict_tracks(
                self.cfg.query_method,
                self.cfg.max_query_pts,
                self.track_predictor,
                images[:, neighbor_start:neighbor_end],
                (
                    masks[:, neighbor_start:neighbor_end]
                    if masks is not None
                    else masks
                ),
                fmaps_for_tracker[:, neighbor_start:neighbor_end],
                [rel_frame_idx],
                fine_tracking=False,
                bound_bboxes=bound_bboxes[:, neighbor_start:neighbor_end],
                query_points_dict={rel_frame_idx: grid_points[None]},
            )

            extra_params_neighbor = (
                extra_params[neighbor_start:neighbor_end]
                if extra_params is not None
                else None
            )
            extrinsics_neighbor = extrinsics_opencv[neighbor_start:neighbor_end]
            intrinsics_neighbor = intrinsics_opencv[neighbor_start:neighbor_end]

            extra_track_normalized = cam_from_img(
                extra_track, intrinsics_neighbor, extra_params_neighbor
            )

            (extra_triangulated_points, extra_inlier_num, extra_inlier_mask) = (
                triangulate_tracks(
                    extrinsics_neighbor,
                    extra_track_normalized.squeeze(0),
                    track_vis=extra_vis.squeeze(0),
                    track_score=extra_score.squeeze(0),
                )
            )

            valid_triangulation_mask = extra_inlier_num > 3

            valid_poins3D_mask, _ = filter_all_points3D(
                extra_triangulated_points,
                extra_track.squeeze(0),
                extrinsics_neighbor,
                intrinsics_neighbor,
                extra_params=extra_params_neighbor,  # Pass extra_params to filter_all_points3D
                max_reproj_error=self.cfg.max_reproj_error,
            )

            valid_triangulation_mask = torch.logical_and(
                valid_triangulation_mask, valid_poins3D_mask
            )

            extra_points3D = extra_triangulated_points[valid_triangulation_mask]
            extra_points3D_rgb = grid_rgb[valid_triangulation_mask]

            additional_points_dict[image_paths[frame_idx]] = {
                "points3D": extra_points3D,
                "points3D_rgb": extra_points3D_rgb,
                "uv": grid_points[valid_triangulation_mask],
            }

        return additional_points_dict

    def extract_sparse_depth_and_point_from_reconstruction(self, predictions):
        """
        Extracts sparse depth and 3D points from the reconstruction.

        Args:
            predictions (dict): Contains reconstruction data with a 'reconstruction' key.

        Returns:
            dict: Updated predictions with 'sparse_depth' and 'sparse_point' keys.
        """
        reconstruction = predictions["reconstruction"]
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
                # instead of the padded&resized one
                uv = pycam.img_from_cam(projection)
                sparse_depth[img_name].append(np.append(uv, depth))
                sparse_point[img_name].append(np.append(pt3D.xyz, point3D_idx))

        predictions["sparse_depth"] = sparse_depth
        predictions["sparse_point"] = sparse_point
        return predictions

    def dense_reconstruct(self, predictions, image_paths, original_images, min_res=512):
        """
        Args:
            predictions (dict): A dictionary containing the sparse reconstruction results.
            image_paths (list): A list of paths to the input images.
            original_images (dict): Dictionary with image basename as keys and original
                numpy images (rgb) as values.

        The function performs the following steps:
        1. Predicts dense depth maps using a monocular depth estimation model, e.g., DepthAnything V2.
        2. Extracts sparse depths from the SfM reconstruction.
        3. Aligns the dense depth maps with the sparse reconstruction.
        4. Updates the predictions dictionary with the dense point cloud data.
        """

        print("Predicting dense depth maps via monocular depth estimation.")

        disp_dict = extract_dense_depth_maps(
            self.depth_model, image_paths, original_images, min_res=min_res,
        )

        sparse_depth = predictions["sparse_depth"]
        reconstruction = predictions["reconstruction"]

        # Align dense depth maps
        print("Aligning dense depth maps by sparse SfM points")
        depth_dict, unproj_dense_points3D = align_dense_depth_maps(
            sparse_depth,
            disp_dict,
            original_images,
            reconstruction=reconstruction,
            visual_dense_point_cloud=self.cfg.visual_dense_point_cloud,
        )

        # Update predictions with dense reconstruction results
        predictions["depth_dict"] = depth_dict
        predictions["unproj_dense_points3D"] = unproj_dense_points3D

        return predictions

    def save_dense_depth_maps(self, depth_dict, output_dir):
        """
        Save the dense depth maps to disk.

        Args:
            depth_dict (dict): Dictionary containing depth maps.
            output_dir (str): Directory to save the depth maps.
        """
        depth_dir = os.path.join(output_dir, "depths")
        os.makedirs(depth_dir, exist_ok=True)
        for img_basename in depth_dict:
            depth_map = depth_dict[img_basename]
            depth_map_path = os.path.join(depth_dir, img_basename)

            name_wo_extension = os.path.splitext(depth_map_path)[0]
            out_fname_with_bin = name_wo_extension + ".bin"
            write_array(depth_map, out_fname_with_bin)

    def make_reprojection_video(
        self, predictions, video_size, image_paths, original_images
    ):
        """
        Create a video with reprojections of the 3D points onto the original images.

        Args:
            predictions (dict): A dictionary containing the reconstruction results,
                                including 3D points and camera parameters.
            video_size (tuple): A tuple specifying the size of the output video (width, height).
            image_paths (list): A list of paths to the input images.
            output_dir (str): The directory to save the output video.
            original_images (dict): Dictionary with image basename as keys and original
                numpy images (rgb) as values.
        """
        reconstruction = predictions["reconstruction"]
        sparse_depth = predictions["sparse_depth"]
        sparse_point = predictions["sparse_point"]

        image_dir_prefix = os.path.dirname(image_paths[0])
        image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

        img_with_circles_list = create_video_with_reprojections(
            image_dir_prefix,
            video_size,
            image_paths,
            sparse_depth,
            sparse_point,
            original_images,
            reconstruction=reconstruction,
        )

        return img_with_circles_list

    def save_reprojection_video(
        self, img_with_circles_list, video_size, output_dir
    ):
        """
        Save the reprojection video to disk.

        Args:
            img_with_circles_list (list): List of images with circles to be included in the video.
            video_size (tuple): A tuple specifying the size of the output video (width, height).
            output_dir (str): The directory to save the output video.
        """
        visual_dir = os.path.join(output_dir, "visuals")
        os.makedirs(visual_dir, exist_ok=True)
        save_video_with_reprojections(
            os.path.join(visual_dir, "reproj.mp4"),
            img_with_circles_list,
            video_size,
        )

    def save_sparse_reconstruction(
        self, predictions, seq_name=None, output_dir=None
    ):
        """
        Save the reconstruction results in COLMAP format.

        Args:
            predictions (dict): Reconstruction results including camera parameters and 3D points.
            seq_name (str, optional): Sequence name for default output directory.
            output_dir (str, optional): Directory to save the reconstruction.

        Saves camera parameters, 3D points, and other data in COLMAP-compatible format.
        """
        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        if output_dir is None:
            output_dir = os.path.join("output", seq_name)

        sfm_output_dir = os.path.join(output_dir, "sparse")
        print("-" * 50)
        print(
            f"The output has been saved in COLMAP style at: {sfm_output_dir} "
        )
        os.makedirs(sfm_output_dir, exist_ok=True)
        reconstruction_pycolmap.write(sfm_output_dir)

    def visualize_3D_in_visdom(
        self, predictions, seq_name=None, output_dir=None
    ):
        """
        This function takes the predictions from the reconstruction process and visualizes
        the 3D point cloud and camera positions in Visdom. It handles both sparse and dense
        reconstructions if available. Requires a running Visdom server and PyTorch3D library.

        Args:
            predictions (dict): Reconstruction results including 3D points and camera parameters.
            seq_name (str, optional): Sequence name for visualization.
            output_dir (str, optional): Directory for saving output files.
        """

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

    def visualize_3D_in_gradio(
        self, predictions, seq_name=None, output_dir=None
    ):
        from vggsfm.utils.gradio import (
            vggsfm_predictions_to_glb,
            visualize_by_gradio,
        )

        # Convert predictions to GLB scene
        glbscene = vggsfm_predictions_to_glb(predictions)

        visual_dir = os.path.join(output_dir, "visuals")

        os.makedirs(visual_dir, exist_ok=True)

        sparse_glb_file = os.path.join(visual_dir, "sparse.glb")

        # Export the GLB scene to the specified file
        glbscene.export(file_obj=sparse_glb_file)

        # Visualize the GLB file using Gradio
        visualize_by_gradio(sparse_glb_file)

        unproj_dense_points3D = predictions["unproj_dense_points3D"]
        if unproj_dense_points3D is not None:
            print(
                "Dense point cloud visualization in Gradio is not supported due to time constraints."
            )

    def rename_colmap_recons_and_rescale_camera(
        self,
        reconstruction,
        image_paths,
        crop_params,
        img_size,
        shift_point2d_to_original_res=False,
        shared_camera=False,
    ):
        rescale_camera = True

        for pyimageid in reconstruction.images:
            # Reshaped the padded&resized image to the original size
            # Rename the images to the original names
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid]

            if rescale_camera:
                # Rescale the camera parameters
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

                resize_ratio = resize_ratio.item()

            if shift_point2d_to_original_res:
                # Also shift the point2D to original resolution
                top_left = crop_params[0, pyimageid][-4:-2].abs().cpu().numpy()
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                # If shared_camera, all images share the same camera
                # no need to rescale any more
                rescale_camera = False

        return reconstruction


################################################ Helper Functions


def move_to_device(tensor, device):
    return tensor.to(device) if tensor is not None else None


def add_batch_dimension(tensor):
    return tensor.unsqueeze(0) if tensor is not None else None


def predict_tracks(
    query_method,
    max_query_pts,
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    query_frame_indexes,
    fine_tracking,
    bound_bboxes=None,
    query_points_dict=None,
    max_points_num=163840,
):
    """
    Predict tracks for the given images and masks.

    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        query_method (str): The methods to use for querying points (e.g., "sp", "sift", "aliked", or "sp+sift).
        max_query_pts (int): The maximum number of query points.
        track_predictor (object): The track predictor object used for predicting tracks.
        images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images. 1 indicates ignored.
        fmaps_for_tracker (torch.Tensor): A tensor of feature maps for the tracker.
        query_frame_indexes (list): A list of indices representing the query frames.
        fine_tracking (bool): Whether to perform fine tracking.
        bound_bboxes (torch.Tensor, optional): A tensor of shape (B, T, 4) representing bounding boxes for the images.
        max_points_num (int): The maximum number of points to process in one chunk.
                              If the total number of points (T * N) exceeds max_points_num,
                              the query points are split into smaller chunks.
                              Default is 163840, suitable for 40GB GPUs.

    Returns:
        tuple: A tuple containing the predicted tracks, visibility, and scores.
            - pred_track (torch.Tensor): A tensor of shape (B, T, N, 2) representing the predicted tracks.
            - pred_vis (torch.Tensor): A tensor of shape (B, T, N) representing the visibility of the predicted tracks.
            - pred_score (torch.Tensor): A tensor of shape (B, T, N) representing the scores of the predicted tracks.
    """
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    frame_num = images.shape[1]
    device = images.device

    if fmaps_for_tracker is None:
        fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)

    for query_index in query_frame_indexes:
        print(f"Predicting tracks with query_index = {query_index}")

        if bound_bboxes is not None:
            bound_bbox = bound_bboxes[:, query_index]
        else:
            bound_bbox = None

        mask = masks[:, query_index] if masks is not None else None

        # Find query_points at the query frame
        if query_points_dict is None:
            query_points = get_query_points(
                images[:, query_index],
                mask,
                query_method,
                max_query_pts,
                bound_bbox=bound_bbox,
            )
        else:
            query_points = query_points_dict[query_index]

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(
            query_index, frame_num, device=device
        )
        images_feed, fmaps_feed = switch_tensor_order(
            [images, fmaps_for_tracker], new_order
        )

        all_points_num = images_feed.shape[1] * max_query_pts

        if all_points_num > max_points_num:
            print('Predict tracks in chunks to fit in memory')

            # Split query_points into smaller chunks to avoid memory issues
            all_points_num = images_feed.shape[1] * query_points.shape[1]
            
            shuffle_indices = torch.randperm(query_points.size(1))
            query_points = query_points[:, shuffle_indices]
            
            num_splits = (all_points_num + max_points_num - 1) // max_points_num
            fine_pred_track, pred_vis, pred_score = predict_tracks_in_chunks(
                track_predictor,
                images_feed,
                query_points,
                fmaps_feed,
                fine_tracking,
                num_splits,
            )
        else:
            # Feed into track predictor
            fine_pred_track, _, pred_vis, pred_score = track_predictor(
                images_feed,
                query_points,
                fmaps=fmaps_feed,
                fine_tracking=fine_tracking,
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
    fine_tracking,
    bound_bboxes=None,
    min_vis=500,
):
    """
    Completes non-visible frames by predicting additional 2D matches.

    This function identifies frames with insufficient visible inliers and uses them as query frames
    to predict additional 2D matches. It iteratively processes these non-visible frames until they
    have enough 2D matches or a final trial is reached.

    Args:
        query_method (str): The methods to use for querying points
                            (e.g., "sp", "sift", "aliked", or "sp+sift).
        max_query_pts (int): The maximum number of query points to use.
        track_predictor (object): The track predictor model.
        images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images.
        fmaps_for_tracker (torch.Tensor): Feature maps for the tracker.
        preds (tuple): A tuple containing predicted tracks, visibility, and scores.
        fine_tracking (bool): Whether to perform fine tracking.
        bound_bboxes (torch.Tensor, optional): Bounding boxes for the images.
        min_vis (int, optional): The minimum number of visible inliers required. Default is 500.
    Returns:
        tuple: A tuple containing updated predicted tracks, visibility, and scores.
    """
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
            fine_tracking,
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


def predict_tracks_in_chunks(
    track_predictor,
    images_feed,
    query_points,
    fmaps_feed,
    fine_tracking,
    num_splits,
):
    """
    Process query points in smaller chunks to avoid memory issues.

    Args:
        track_predictor (object): The track predictor object used for predicting tracks.
        images_feed (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        query_points (torch.Tensor): A tensor of shape (B, N, 2) representing the query points.
        fmaps_feed (torch.Tensor): A tensor of feature maps for the tracker.
        fine_tracking (bool): Whether to perform fine tracking.
        num_splits (int): The number of chunks to split the query points into.

    Returns:
        tuple: A tuple containing the concatenated predicted tracks, visibility, and scores.
    """
    split_query_points = torch.chunk(query_points, num_splits, dim=1)

    fine_pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for split_points in split_query_points:
        # Feed into track predictor for each split
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed,
            split_points,
            fmaps=fmaps_feed,
            fine_tracking=fine_tracking,
        )
        fine_pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    # Concatenate the results from all splits
    fine_pred_track = torch.cat(fine_pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    if pred_score is not None:
        pred_score = torch.cat(pred_score_list, dim=2)
    else:
        pred_score = None

    return fine_pred_track, pred_vis, pred_score


def get_query_points(
    query_image,
    seg_invalid_mask,
    query_method,
    max_query_num=4096,
    det_thres=0.005,
    bound_bbox=None,
):
    """
    Extract query points from the given query image using the specified method.

    This function extracts query points from the given query image using the specified method(s).
    It supports multiple methods such as "sp" (SuperPoint), "sift" (SIFT), and "aliked" (ALIKED).
    The function also handles invalid masks and bounding boxes to filter out unwanted regions.

    Args:
        query_image (torch.Tensor): A tensor of shape (B, C, H, W) representing the query image.
        seg_invalid_mask (torch.Tensor, optional):
                        A tensor of shape (B, 1, H, W) representing the segmentation invalid mask.
        query_method (str): The method(s) to use for extracting query points
                            (e.g., "sp", "sift", "aliked", or combinations like "sp+sift").
        max_query_num (int, optional): The maximum number of query points to extract. Default is 4096.
        det_thres (float, optional): The detection threshold for keypoint extraction. Default is 0.005.
        bound_bbox (torch.Tensor, optional): A tensor of shape (B, 4) representing bounding boxes for the images.

    Returns:
        torch.Tensor: A tensor of shape (B, N, 2) representing the extracted query points,
                        where N is the number of query points.
    """

    methods = query_method.split("+")
    pred_points = []

    for method in methods:
        if "sp" in method:
            extractor = SuperPoint(
                max_num_keypoints=max_query_num, detection_threshold=det_thres
            )
        elif "sift" in method:
            extractor = SIFT(max_num_keypoints=max_query_num)
        elif "aliked" in method:
            extractor = ALIKED(
                max_num_keypoints=max_query_num, detection_threshold=det_thres
            )
        else:
            raise NotImplementedError(
                f"query method {method} is not supprted now"
            )
        extractor = extractor.cuda().eval()
        invalid_mask = None

        if bound_bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bound_bbox[0])
            bbox_valid_mask = torch.zeros_like(
                query_image[:, 0], dtype=torch.bool
            )
            bbox_valid_mask[:, y_min:y_max, x_min:x_max] = 1
            invalid_mask = ~bbox_valid_mask

        if seg_invalid_mask is not None:
            seg_invalid_mask = seg_invalid_mask.squeeze(1).bool()
            invalid_mask = (
                seg_invalid_mask
                if invalid_mask is None
                else torch.logical_or(invalid_mask, seg_invalid_mask)
            )

        query_points = extractor.extract(
            query_image, invalid_mask=invalid_mask
        )["keypoints"]
        pred_points.append(query_points)

    query_points = torch.cat(pred_points, dim=1)

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[
            :max_query_num
        ]
        query_points = query_points[:, random_point_indices, :]

    return query_points
