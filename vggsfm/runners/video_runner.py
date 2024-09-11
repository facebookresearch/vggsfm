# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2
import os
import copy
import time
import math
import random
import pyceres
import pycolmap
import numpy as np
import datetime
import logging

from collections import defaultdict


from .runner import (
    VGGSfMRunner,
    move_to_device,
    add_batch_dimension,
    predict_tracks,
    get_query_points,
)
from vggsfm.utils.utils import average_camera_prediction, generate_grid_samples

from vggsfm.utils.tensor_to_pycolmap import (
    batch_matrix_to_pycolmap,
    pycolmap_to_batch_matrix,
)
from vggsfm.utils.align import align_camera_extrinsics, apply_transformation

from vggsfm.utils.triangulation import triangulate_tracks
from vggsfm.utils.triangulation_helpers import filter_all_points3D, cam_from_img


class VideoRunner(VGGSfMRunner):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.point_dict = {}
        self.frame_dict = defaultdict(dict)
        self.crop_params = None
        self.intrinsics = None

        assert (
            self.cfg.shared_camera == True
        ), "Currently only shared camera is supported for video runner"

        # TODO: add a loop detection
        # TODO: support the handle of invalid frames
        # TODO: support camera parameter change in the future

        if self.cfg.extra_pt_pixel_interval > 0:
            raise ValueError(
                "Extra points have not been supported for video runner; Stay tuned Please!"
            )

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
        init_window_size=32,
        window_size=16,
        joint_BA_interval=6,
    ):
        # NOTE
        # We assume crop_params, intrinsics, and extra_params are the same for a video

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

            # Calculate bounding boxes if crop parameters are provided
            # NOTE: We assume crop_params are the same for a video
            self.bound_bboxes = self.calculate_bounding_boxes(
                crop_params, images
            )
            self.crop_params = crop_params[:, 0:1].clone()

            B, T, C, H, W = images.shape
            self.B = B
            self.img_dim = C
            self.H = H
            self.W = W

            self.image_size = torch.tensor(
                [W, H], dtype=images.dtype, device=self.device
            )

            self.images = images
            self.masks = masks
            self.image_paths = image_paths

            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num

            start_idx = 0
            end_idx = start_idx + init_window_size

            # Temporally double the max_query_pts to ensure
            # the initial window has enough points to track
            self.cfg.max_query_pts = self.cfg.max_query_pts * 2
            # Run reconstruction for the initial window
            # which calls VGGSfMRunner.sparse_reconstruct
            init_pred = self.process_initial_window(
                start_idx,
                end_idx,
                images,
                masks,
                crop_params,
                image_paths,
                query_frame_num,
                seq_name,
                output_dir,
            )
            self.cfg.max_query_pts = self.cfg.max_query_pts // 2

            _, init_intri, init_extra = (
                init_pred["extrinsics_opencv"],
                init_pred["intrinsics_opencv"],
                init_pred["extra_params"],
            )

            if init_extra is None:
                init_extra = torch.zeros(len(init_intri), 1).to(self.device)

            self.intrinsics = init_intri[0:1].clone()  # 1x3x3
            self.extra_params = init_extra[0:1].clone()  # 1xnum_extra_params

            self.convert_pred_to_point_frame_dict(init_pred, start_idx, end_idx)

            window_counter = 0  # Initialize a counter for windows

            while end_idx < T:
                if (T - end_idx) <= int(1.25 * window_size):
                    start_idx, end_idx, move_success, _ = self.move_window(
                        start_idx, end_idx, T - end_idx
                    )
                else:
                    start_idx, end_idx, move_success, _ = self.move_window(
                        start_idx, end_idx, window_size
                    )

                if not move_success:
                    print(
                        "Moving window failed, trying again. (This should not happen in most cases)"
                    )
                    self.cfg.max_query_pts = self.cfg.max_query_pts * 2
                    start_idx, end_idx, move_success, _ = self.move_window(
                        start_idx, end_idx, window_size
                    )
                    self.cfg.max_query_pts = self.cfg.max_query_pts // 2

                if window_counter % joint_BA_interval == 0:
                    print("Running joint BA:")
                    start_time = time.time()

                    self.joint_BA(0, end_idx, normalize=True)

                    end_time = time.time()
                    print(f"Joint BA took {end_time - start_time:.2f} seconds")

                window_counter += 1  # Increment the window counter

            # Conduct a BA for the entire sequence
            print("Running joint BA for the entire sequence:")
            self.joint_BA(0, T, normalize=True)

            print("Updating points color")
            self._update_points_color()  # TODO: _update_points_color is too slow, fix it
            print("Points color updated")

            predictions = self.dicts_to_output(
                0, T, back_to_original_resolution=True
            )

            # Save the sparse reconstruction results
            if self.cfg.save_to_disk:
                self.save_sparse_reconstruction(
                    predictions, seq_name, output_dir
                )

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

    def dicts_to_output(
        self, start_idx, end_idx, back_to_original_resolution=False
    ):
        print("Converting Predictions to the Output Format")
        predictions = {}

        T = self.images.shape[1]
        reconstruction = self.dicts_to_reconstruction(
            start_idx, end_idx, extract_color=True
        )

        basenames = [os.path.basename(path) for path in self.image_paths]

        if self.cfg.extra_pt_pixel_interval > 0:
            raise ValueError(
                "Extra points have not been supported for video runner; Stay tuned Please!"
            )

        if back_to_original_resolution:
            reconstruction = self.rename_colmap_recons_and_rescale_camera(
                reconstruction,
                basenames,
                self.crop_params.expand(-1, T, -1),
                self.image_size[-1],
                shift_point2d_to_original_res=self.cfg.shift_point2d_to_original_res,
                shared_camera=self.cfg.shared_camera,
            )

        predictions["reconstruction"] = reconstruction

        # NOTE:
        # We can do this only when we know
        # the images inside reconstruction is sorted by order
        # i.e., reconstruction.images[1] cooresponds to extrinsics[1]
        _, extrinsics, intrinsics, extra_params = pycolmap_to_batch_matrix(
            reconstruction, self.device, camera_type=self.cfg.camera_type
        )

        predictions["extrinsics_opencv"] = extrinsics
        predictions["intrinsics_opencv"] = intrinsics
        predictions["extra_params"] = extra_params

        points_xyz = []
        points_rgb = []

        for point_id in self.point_dict.keys():
            point_data = self.point_dict[point_id]
            points_xyz.append(point_data["xyz"])
            points_rgb.append(point_data["rgb"])

        predictions["points3D"] = torch.stack(points_xyz).to(self.device)
        predictions["points3D_rgb"] = torch.stack(points_rgb).to(self.device)

        predictions["unproj_dense_points3D"] = None
        predictions["valid_2D_mask"] = None
        predictions["pred_track"] = None
        predictions["pred_vis"] = None
        predictions["pred_score"] = None
        predictions["valid_tracks"] = None
        return predictions

    def calculate_bounding_boxes(self, crop_params, images):
        """
        Calculate bounding boxes if crop parameters are provided.
        """
        if crop_params is not None:
            # We know bound_bboxes is the same for a video
            bound_bboxes = crop_params[:, 0:1, -4:-2].abs().to(self.device)
            # also remove those near the boundary
            bound_bboxes[bound_bboxes != 0] += self.remove_borders

            bound_bboxes = torch.cat(
                [bound_bboxes, images.shape[-1] - bound_bboxes], dim=-1
            )
            return bound_bboxes
        return None

    def process_initial_window(
        self,
        start_idx,
        end_idx,
        images,
        masks,
        crop_params,
        image_paths,
        query_frame_num,
        seq_name,
        output_dir,
    ):
        init_images, init_masks, init_crop_params = extract_window(
            start_idx, end_idx, images, masks, crop_params
        )

        init_pred = self.sparse_reconstruct(
            init_images,
            masks=init_masks,
            crop_params=init_crop_params,
            image_paths=image_paths[start_idx:end_idx],
            query_frame_num=query_frame_num,
            seq_name=seq_name,
            output_dir=output_dir,
            back_to_original_resolution=False,
        )
        return init_pred

    def convert_pred_to_point_frame_dict(self, pred, start_idx, end_idx):
        (
            pred_track,
            pred_vis,
            valid_2D_mask,
            valid_tracks,
            points3D,
            points3D_rgb,
        ) = (
            pred["pred_track"],
            pred["pred_vis"],
            pred["valid_2D_mask"],
            pred["valid_tracks"],
            pred["points3D"],
            pred["points3D_rgb"],
        )

        point_to_track_mapping = valid_tracks.nonzero().squeeze(1).cpu().numpy()

        if "points3D_idx" not in pred:
            points3D_idx = np.arange(len(points3D))
        else:
            points3D_idx = pred["points3D_idx"]

        extrinsics = pred["extrinsics_opencv"]

        # save them in cpu
        extrinsics = extrinsics.cpu()

        for frame_idx in range(start_idx, end_idx):
            relative_frame_idx = frame_idx - start_idx
            self.frame_dict[frame_idx]["extri"] = extrinsics[relative_frame_idx]
            if "visible_points" not in self.frame_dict[frame_idx]:
                self.frame_dict[frame_idx]["visible_points"] = []

        if len(self.point_dict.keys()) == 0:
            exist_max_point = 0
        else:
            exist_max_point = max(self.point_dict.keys()) + 1

        self._update_points_to_dict(
            start_idx,
            end_idx,
            valid_2D_mask,
            pred_track,
            pred_vis,
            points3D_idx,
            point_to_track_mapping,
            points3D,
            points3D_rgb,
            existing_max_point_idx=exist_max_point,
        )

    def _update_points_to_dict(
        self,
        start_idx,
        end_idx,
        valid_2D_mask,
        pred_track,
        pred_vis,
        points3D_idx,
        point_to_track_mapping=None,
        points3D=None,
        points3D_rgb=None,
        existing_max_point_idx=0,
    ):

        pred_track = pred_track.squeeze(0).cpu()
        pred_vis = pred_vis.squeeze(0).cpu()
        if points3D is not None:
            points3D = points3D.cpu()
        if points3D_rgb is not None:
            points3D_rgb = points3D_rgb.cpu()

        # TODO: write unit test for this function
        if point_to_track_mapping is None:
            # this can also be a dict, is it okay?
            point_to_track_mapping = np.arange(len(points3D_idx))

        for point_idx in points3D_idx:
            abs_point_idx = point_idx + existing_max_point_idx

            track_idx = point_to_track_mapping[point_idx]

            point_valid_2D_mask = valid_2D_mask[:, track_idx]

            if abs_point_idx in self.point_dict:
                # if the point is already in the point dict, we need to update the track dict
                point_track_dict = self.point_dict[abs_point_idx]["track"]
            else:
                point_track_dict = {}

            for frame_idx in range(start_idx, end_idx):
                relative_frame_idx = frame_idx - start_idx
                if point_valid_2D_mask[relative_frame_idx]:
                    point_track_dict[frame_idx] = {
                        "uv": pred_track[relative_frame_idx, track_idx],
                        "vis": pred_vis[relative_frame_idx, track_idx],
                    }

                    self.frame_dict[frame_idx]["visible_points"].append(
                        abs_point_idx
                    )

            if abs_point_idx not in self.point_dict:
                # if the point is already in the point dict,
                # we don't need to update id, xyz, and rgb

                point_dict = {
                    "id": abs_point_idx,
                    "xyz": points3D[point_idx],
                    "rgb": (
                        points3D_rgb[point_idx]
                        if points3D_rgb is not None
                        else None
                    ),
                    "track": point_track_dict,
                }

                self.point_dict[abs_point_idx] = point_dict

    def _update_points_color(self, reverse=False):
        for point_id, point_data in self.point_dict.items():
            colors = []
            for i, frame_id in enumerate(point_data["track"].keys()):
                uu, vv = torch.floor(point_data["track"][frame_id]["uv"]).long()
                # vv, uu or uu, vv?
                if vv < self.images.shape[-2] and uu < self.images.shape[-1]:
                    if reverse:
                        colors.append(self.images[0, frame_id, :, (uu), (vv)])
                    else:
                        colors.append(self.images[0, frame_id, :, (vv), (uu)])

            if len(colors) > 0:
                colors = torch.stack(colors)
                avg_color = colors.float().mean(dim=0)

                # Update point_dict with the computed RGB value
                self.point_dict[point_id]["rgb"] = avg_color.cpu()

    def joint_BA(
        self,
        start_idx,
        end_idx,
        reproj_error=2.0,
        tri_angle=1.5,
        normalize=True,
    ):
        reconstruction = self.dicts_to_reconstruction(start_idx, end_idx)

        if normalize:
            reconstruction.normalize(5.0, 0.1, 0.9, True)

        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        observation_manager = pycolmap.ObservationManager(reconstruction)
        observation_manager.filter_all_points3D(reproj_error, tri_angle)
        observation_manager.filter_observations_with_negative_depth()

        if normalize:
            reconstruction.normalize(5.0, 0.1, 0.9, True)

        intrinsics_opt = (
            torch.from_numpy(reconstruction.cameras[0].calibration_matrix())
            .to(self.device)
            .float()
        )
        self.intrinsics = intrinsics_opt[None].clone()  # 1x3x3

        if self.cfg.camera_type == "SIMPLE_RADIAL":
            extra_params_opt = (
                torch.from_numpy(reconstruction.cameras[0].params)
                .to(self.device)
                .float()
            )
            self.extra_params = (
                extra_params_opt[-1].reshape(1, 1).clone()
            )  # 1 x num_extra_params

        if normalize:
            self.point_dict = {}
            self.frame_dict = defaultdict(dict)
            self.reconstruction_to_dicts(reconstruction)
        else:
            self.update_dicts_by_reconstruction(
                reconstruction, start_idx, end_idx
            )

    def dicts_to_reconstruction(self, start_idx, end_idx, extract_color=False):
        reconstruction = pycolmap.Reconstruction()
        points3d_idx_list = sorted(list(self.point_dict.keys()))
        for pidx in points3d_idx_list:
            if extract_color:
                point_color = np.round(
                    self.point_dict[pidx]["rgb"].numpy() * 255
                ).astype(np.uint8)
            else:
                point_color = np.zeros(3)

            reconstruction.add_point3D(
                self.point_dict[pidx]["xyz"].numpy(),
                pycolmap.Track(),
                point_color,
            )

        pycam = self.build_camera_for_video()
        reconstruction.add_camera(pycam)

        for image_idx in range(start_idx, end_idx):
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(
                    self.frame_dict[image_idx]["extri"][:3, :3]
                ),
                self.frame_dict[image_idx]["extri"][:3, 3],
            )

            pyimg = pycolmap.Image(
                id=image_idx,
                name=f"image_{image_idx}",
                camera_id=pycam.camera_id,
                cam_from_world=cam_from_world,
            )

            points2D_list = []

            point2D_idx = 0
            for point3D_id in self.frame_dict[image_idx]["visible_points"]:
                pycolmap_point3D_id = point3D_id + 1
                point2D_xy = self.point_dict[point3D_id]["track"][image_idx][
                    "uv"
                ].numpy()
                points2D_list.append(
                    pycolmap.Point2D(point2D_xy, pycolmap_point3D_id)
                )

                track = reconstruction.points3D[pycolmap_point3D_id].track
                track.add_element(image_idx, point2D_idx)
                point2D_idx += 1
            assert point2D_idx == len(points2D_list)

            try:
                pyimg.points2D = pycolmap.ListPoint2D(points2D_list)
                pyimg.registered = True
            except:
                print(f"frame {image_idx} is out of BA")
                pyimg.registered = False

            reconstruction.add_image(pyimg)

        return reconstruction

    def reconstruction_to_dicts(self, reconstruction):
        # Convert reconstruction to frame_dict
        for image_id, image in reconstruction.images.items():
            self.frame_dict[image_id]["extri"] = torch.tensor(
                image.cam_from_world.matrix()
            )
            self.frame_dict[image_id]["visible_points"] = []

        point3D_id = 0
        # Convert reconstruction to point_dict
        for pycolmap_point3D_id in sorted(reconstruction.point3D_ids()):
            point3D = reconstruction.points3D[pycolmap_point3D_id]
            point_dict = {
                "id": point3D_id,
                "xyz": torch.from_numpy(point3D.xyz).float(),
                "rgb": torch.from_numpy(point3D.color),
                "track": {},
            }

            for track_element in point3D.track.elements:
                image_id = track_element.image_id
                point2D_idx = track_element.point2D_idx
                point2D = reconstruction.images[image_id].points2D[point2D_idx]

                point_dict["track"][image_id] = {
                    "uv": torch.from_numpy(point2D.xy).float(),
                    "vis": torch.from_numpy(np.array([1.0])).float(),
                }

                self.frame_dict[image_id]["visible_points"].append(point3D_id)

            self.point_dict[point3D_id] = point_dict
            point3D_id += 1

    def move_window(
        self, start_idx, end_idx, window_size, min_valid_track_length=3
    ):
        # Move forward to the right first
        last_window_size = end_idx - start_idx
        assert last_window_size > 0, "last_window_size should be positive"

        last_start_idx = start_idx
        start_idx = end_idx
        end_idx = start_idx + window_size

        print(f"Processing window from {start_idx} to {end_idx}")

        # Include the last window in the next window for average_camera_prediction
        # because we need to align the camera prediction of the next window with the last window

        two_window_images = extract_window(
            last_start_idx, end_idx, self.images
        )[0]
        two_window_size = two_window_images.shape[1]

        # Predict extri for the combination of (last_window, current_window)
        pred_cameras = average_camera_prediction(
            self.camera_predictor,
            two_window_images.reshape(-1, self.img_dim, self.H, self.W),
            self.B,
            query_indices=[0, two_window_size // 2, two_window_size - 1],
        )

        last_extri = torch.stack(
            [
                self.frame_dict[frame_idx]["extri"]
                for frame_idx in range(last_start_idx, start_idx)
            ]
        ).to(self.device)

        pred_extri = torch.cat(
            (pred_cameras.R, pred_cameras.T.unsqueeze(-1)), dim=-1
        )

        # Align to the last window
        rel_r, rel_t, rel_s = align_camera_extrinsics(
            pred_extri[:last_window_size], last_extri
        )
        aligned_pred_extri_next_window = apply_transformation(
            pred_extri[last_window_size:], rel_r, rel_t, rel_s
        )

        (
            last_end_visible_points_3D,
            _,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            window_tracks_for_exist_points,
            window_vis_for_exist_points,
            window_vis_inlier_for_exist_points,
            extri_window_plus_one,
            last_end_visible_points_idx,
        ) = self.prepare_window_data(
            start_idx, end_idx, aligned_pred_extri_next_window
        )

        ###############################################################
        # from vggsfm.utils.utils import visual_query_points
        # visual_query_points(window_images, 0, window_pred_track[0:1])
        # from vggsfm.utils.visualizer import Visualizer
        # vis = Visualizer(save_dir="visual_debug", linewidth=1)
        # vis.visualize(window_images * 255, window_tracks_for_exist_points[None], window_vis_inlier_for_exist_points[None][..., None],
        #               filename=f"start_{start_idx}_end_{end_idx}")
        ###############################################################

        per_frame_enough_flag = (
            window_vis_inlier_for_exist_points.sum(dim=1) < 50
        )

        if per_frame_enough_flag.any():
            first_invalid_index = (
                (per_frame_enough_flag == True)
                .nonzero(as_tuple=True)[0][0]
                .item()
            )

            if first_invalid_index > 2:
                window_size = first_invalid_index - 1  # TODO: -1 or not?
                print(
                    f"Shrink the window from {start_idx}-{end_idx} to {start_idx}-{start_idx + window_size}"
                )
                end_idx = start_idx + window_size

                window_images = window_images[:, : (window_size + 1)]
                if window_masks is not None:
                    window_masks = window_masks[:, : (window_size + 1)]
                window_fmaps_for_tracker = window_fmaps_for_tracker[
                    :, : (window_size + 1)
                ]
                window_tracks_for_exist_points = window_tracks_for_exist_points[
                    : (window_size + 1)
                ]
                window_vis_for_exist_points = window_vis_for_exist_points[
                    : (window_size + 1)
                ]
                window_vis_inlier_for_exist_points = (
                    window_vis_inlier_for_exist_points[: (window_size + 1)]
                )
                extri_window_plus_one = extri_window_plus_one[
                    : (window_size + 1)
                ]
            else:
                # TODO drop some frames and get them back later
                print("No valid frame, step back")
                return last_start_idx - 1, start_idx - 1, False, None

        align_extri_window_plus_one = self.align_next_window(
            extri_window_plus_one,
            window_tracks_for_exist_points,
            window_vis_inlier_for_exist_points,
            last_end_visible_points_3D,
        )

        (
            exist_valid_points,
            exist_valid_tracks,
            exist_valid_inlier_masks,
            exist_valid_tracks_mask,
        ) = self.filter_points_and_compute_masks(
            last_end_visible_points_3D.to(self.device),
            window_tracks_for_exist_points,
            align_extri_window_plus_one,
        )

        exist_valid_points_idx = np.array(last_end_visible_points_idx)[
            exist_valid_tracks_mask.cpu().numpy()
        ]

        (
            window_new_triangulated_points,
            window_new_tracks,
            window_new_inlier_masks,
            window_new_vis,
        ) = self.triangulate_window_points(
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            window_size,
            align_extri_window_plus_one,
        )

        exist_points_3D_num = len(exist_valid_points)

        window_points_all = torch.cat(
            [exist_valid_points, window_new_triangulated_points], dim=0
        )
        window_tracks_all = torch.cat(
            [exist_valid_tracks, window_new_tracks], dim=1
        )
        window_inlier_masks_all = torch.cat(
            [exist_valid_inlier_masks, window_new_inlier_masks], dim=1
        )

        rec = batch_matrix_to_pycolmap(
            window_points_all,
            align_extri_window_plus_one,
            self.intrinsics.expand(window_size + 1, -1, -1),
            window_tracks_all,
            window_inlier_masks_all,
            self.image_size,
            shared_camera=self.cfg.shared_camera,
            camera_type=self.cfg.camera_type,
            extra_params=self.extra_params.expand(window_size + 1, -1),
        )

        # NOTE It is window_size + 1 instead of window_size
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = False
        ba_options.refine_extra_params = False

        ba_config = pycolmap.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids():
            ba_config.add_image(image_id)

        # Fix frame 0, i.e, the end frame of the last window
        ba_config.set_constant_cam_pose(rec.reg_image_ids()[0])

        for fixp_idx in rec.point3D_ids():
            if fixp_idx < (exist_points_3D_num + 1):
                # Set existing points to be fixed
                ba_config.add_constant_point(fixp_idx)
            else:
                ba_config.add_variable_point(fixp_idx)

        summary = solve_bundle_adjustment(rec, ba_options, ba_config)

        ba_success = log_ba_summary(summary)

        if not ba_success:
            raise RuntimeError("Bundle adjustment failed")

        window_points3D_opt, extrinsics, _, _ = pycolmap_to_batch_matrix(
            rec, device=self.device, camera_type=window_points_all.dtype
        )

        assert window_points3D_opt.shape[0] == window_tracks_all.shape[1]

        (
            new_valid_points,
            new_valid_tracks,
            new_valid_inlier_masks,
            new_valid_tracks_mask,
        ) = self.filter_points_and_compute_masks(
            window_points3D_opt[exist_points_3D_num:],
            window_tracks_all[:, exist_points_3D_num:],
            extrinsics,
        )

        new_pred = {}
        new_pred["extrinsics_opencv"] = extrinsics[1:]
        new_pred["pred_track"] = new_valid_tracks[1:]
        new_pred["pred_vis"] = window_new_vis[:, new_valid_tracks_mask][1:]
        new_pred["valid_2D_mask"] = new_valid_inlier_masks[1:]
        new_pred["valid_tracks"] = new_valid_tracks_mask[new_valid_tracks_mask]
        new_pred["points3D"] = new_valid_points
        new_pred["points3D_rgb"] = None

        self.convert_pred_to_point_frame_dict(new_pred, start_idx, end_idx)

        (
            refiltered_exist_points,
            refiltered_exist_tracks,
            refiltered_exist_inlier_masks,
            refiltered_exist_tracks_mask,
        ) = self.filter_points_and_compute_masks(
            window_points3D_opt[:exist_points_3D_num],
            window_tracks_all[:, :exist_points_3D_num],
            extrinsics,
        )

        refiltered_exist_vis = window_vis_for_exist_points[
            :, exist_valid_tracks_mask
        ][:, refiltered_exist_tracks_mask]

        refiltered_exist_valid_points_idx = exist_valid_points_idx[
            refiltered_exist_tracks_mask.cpu().numpy()
        ]

        refilter_point_to_track_mapping = {}

        for num in range(len(refiltered_exist_valid_points_idx)):
            refilter_point_to_track_mapping[
                refiltered_exist_valid_points_idx[num]
            ] = num

        # update existing points3D
        # does not change point xyz, rgb or idx
        # but changes the track linked to these existing points
        self._update_points_to_dict(
            start_idx,
            end_idx,
            refiltered_exist_inlier_masks[1:],
            refiltered_exist_tracks[1:],
            refiltered_exist_vis[1:],
            refiltered_exist_valid_points_idx,
            point_to_track_mapping=refilter_point_to_track_mapping,
        )

        return start_idx, end_idx, True, None

    def filter_points_and_compute_masks(
        self,
        points,
        tracks,
        extrinsics,
        min_valid_track_length=3,
        max_reproj_error=4,
    ):
        # points: P x 3
        # tracks: S x P x 2
        # extrinsics: S x 3 x 4
        S = extrinsics.shape[0]
        valid_poins3D_mask, inlier_mask = filter_all_points3D(
            points,
            tracks,
            extrinsics,
            self.intrinsics.expand(S, -1, -1),
            extra_params=self.extra_params.expand(S, -1),
            max_reproj_error=max_reproj_error,
            return_detail=True,
            hard_max=-1,
        )

        valid_tracks_mask = inlier_mask.sum(dim=0) >= min_valid_track_length
        filtered_points = points[valid_tracks_mask]
        filtered_tracks = tracks[:, valid_tracks_mask]
        filtered_inlier_masks = inlier_mask[:, valid_tracks_mask]
        return (
            filtered_points,
            filtered_tracks,
            filtered_inlier_masks,
            valid_tracks_mask,
        )

    def align_next_window(
        self,
        extrinsics,
        tracks,
        inlier,
        points3D,
        use_pnp=False,
        min_vis_num=50,
    ):
        # extrinsics: Sx3x4
        # tracks: SxPx2
        # inlier: SxP
        # points3D: P x 3

        S, _, _ = extrinsics.shape
        _, P, _ = tracks.shape

        points3D = points3D.cpu().numpy()
        tracks2D = tracks.cpu().numpy()
        inlier = inlier.cpu().numpy()
        refoptions = pycolmap.AbsolutePoseRefinementOptions()
        refoptions.refine_focal_length = False
        refoptions.refine_extra_params = False
        refoptions.print_summary = False

        refined_extrinsics = []

        pycam = self.build_camera_for_video()

        for ridx in range(S):
            if ridx == 0:
                refined_extrinsics.append(extrinsics[ridx].cpu().numpy())
                continue
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(extrinsics[ridx][:3, :3].cpu()),
                extrinsics[ridx][:3, 3].cpu(),
            )  # Rot and Trans
            points2D = tracks2D[ridx]
            inlier_mask = inlier[ridx]

            if inlier_mask.sum() <= min_vis_num:
                # If too few inliers, ignore it
                # use all the points
                print("Too small inliers")
                inlier_mask[:] = 1

            if use_pnp:
                estoptions = pycolmap.AbsolutePoseEstimationOptions()
                estoptions.ransac.max_error = 12

                estanswer = pycolmap.absolute_pose_estimation(
                    points2D[inlier_mask],
                    points3D[inlier_mask],
                    pycam,
                    estoptions,
                    refoptions,
                )
                cam_from_world = estanswer["cam_from_world"]
                print(estanswer["inliers"].mean())

            answer = pycolmap.pose_refinement(
                cam_from_world,
                points2D,
                points3D,
                inlier_mask,
                pycam,
                refoptions,
            )

            cam_from_world = answer["cam_from_world"]
            refined_extrinsics.append(cam_from_world.matrix())

        # get the optimized cameras
        refined_extrinsics = torch.from_numpy(np.stack(refined_extrinsics)).to(
            tracks.device
        )
        return refined_extrinsics

    def build_camera_for_video(self):
        if self.cfg.camera_type == "SIMPLE_RADIAL":
            pycolmap_intri = np.array(
                [
                    self.intrinsics[0][0, 0].cpu().numpy(),
                    self.intrinsics[0][0, 2].cpu().numpy(),
                    self.intrinsics[0][1, 2].cpu().numpy(),
                    self.extra_params[0][0].cpu().numpy(),
                ]
            )
        elif self.cfg.camera_type == "SIMPLE_PINHOLE":
            pycolmap_intri = np.array(
                [
                    self.intrinsics[0][0, 0].cpu().numpy(),
                    self.intrinsics[0][0, 2].cpu().numpy(),
                    self.intrinsics[0][1, 2].cpu().numpy(),
                ]
            )
        else:
            raise NotImplementedError(
                f"Camera type {self.cfg.camera_type} not implemented"
            )
        # We assume the same camera for all frames in a video
        pycam = pycolmap.Camera(
            model=self.cfg.camera_type,
            width=self.image_size[0],
            height=self.image_size[1],
            params=pycolmap_intri,
            camera_id=0,
        )
        return pycam

    def prepare_window_data(
        self,
        start_idx,
        end_idx,
        aligned_pred_extri_next_window,
        back_step=1,
        max_ratio=1,
        use_support_points=True,
        random_support_points=True,
        track_vis_thres=0.05,
    ):
        last_end_visible_points_idx = self.frame_dict[start_idx - back_step][
            "visible_points"
        ]

        max_exist_points_num = self.cfg.max_query_pts * max_ratio

        if len(last_end_visible_points_idx) > max_exist_points_num:
            last_end_visible_points_idx = sorted(
                random.sample(last_end_visible_points_idx, max_exist_points_num)
            )

        last_end_visible_points_3D = [
            self.point_dict[point3D_idx]["xyz"]
            for point3D_idx in last_end_visible_points_idx
        ]
        last_end_visible_points_3D = torch.stack(last_end_visible_points_3D)

        last_end_visible_points_2D = [
            self.point_dict[point3D_idx]["track"][start_idx - back_step]["uv"]
            for point3D_idx in last_end_visible_points_idx
        ]
        last_end_visible_points_2D = torch.stack(last_end_visible_points_2D)

        if back_step > 1:
            window_images, window_masks = extract_window(
                start_idx, end_idx, self.images, self.masks
            )
            sub_window_images, sub_window_masks = extract_window(
                start_idx - back_step,
                start_idx - back_step + 1,
                self.images,
                self.masks,
            )
            window_images = torch.cat([sub_window_images, window_images], dim=1)
            if window_masks is not None:
                window_masks = torch.cat(
                    [sub_window_masks, window_masks], dim=1
                )
        else:
            window_images, window_masks = extract_window(
                start_idx - back_step, end_idx, self.images, self.masks
            )

        window_fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(
            window_images
        )

        if use_support_points:
            if random_support_points:
                support_points = generate_grid_samples(
                    self.bound_bboxes[0], self.cfg.max_query_pts
                ).to(self.device)[None]
            else:
                support_points = get_query_points(
                    window_images[:, 0],
                    window_masks[:, 0] if window_masks is not None else None,
                    self.cfg.query_method,
                    self.cfg.max_query_pts,
                    bound_bbox=self.bound_bboxes[0],
                )
            exist_points_num = len(last_end_visible_points_2D)
            query_points = torch.cat(
                [
                    last_end_visible_points_2D.to(self.device)[None],
                    support_points,
                ],
                dim=1,
            )
        else:
            query_points = last_end_visible_points_2D.to(self.device)[None]

        (
            window_tracks_for_exist_points,
            window_vis_for_exist_points,
            window_score_for_exist_points,
        ) = predict_tracks(
            self.cfg.query_method,
            self.cfg.max_query_pts,
            self.track_predictor,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            [0],
            self.cfg.fine_tracking,
            self.bound_bboxes.expand(-1, window_images.shape[1], -1),
            query_points_dict={0: query_points},
        )

        if use_support_points:
            # only pick the real query ones instead of the support points
            window_tracks_for_exist_points = window_tracks_for_exist_points[
                :, :, :exist_points_num
            ]
            window_vis_for_exist_points = window_vis_for_exist_points[
                :, :, :exist_points_num
            ]

        window_tracks_for_exist_points = window_tracks_for_exist_points.squeeze(
            0
        )
        window_vis_for_exist_points = window_vis_for_exist_points.squeeze(0)

        window_vis_inlier_for_exist_points = (
            window_vis_for_exist_points > track_vis_thres
        )

        last_end_extri = self.frame_dict[start_idx - back_step]["extri"][
            None
        ].to(self.device)

        extri_window_plus_one = torch.cat(
            [last_end_extri, aligned_pred_extri_next_window], dim=0
        )

        return (
            last_end_visible_points_3D,
            last_end_visible_points_2D,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            window_tracks_for_exist_points,
            window_vis_for_exist_points,
            window_vis_inlier_for_exist_points,
            extri_window_plus_one,
            last_end_visible_points_idx,
        )

    def triangulate_window_points(
        self,
        window_images,
        window_masks,
        window_fmaps_for_tracker,
        window_size,
        extrinsics,
        max_reproj_error=4,
        min_valid_track_length=3,
    ):
        """
        Add more points by predicting tracks and performing triangulation.

        Args:
            window_images (torch.Tensor): The images in the current window.
            window_masks (torch.Tensor): The masks for the images in the current window.
            window_fmaps_for_tracker (torch.Tensor): Feature maps for the tracker.
            window_size (int): The size of the window.
            extrinsics (align_extri_window_plus_one) (torch.Tensor): Aligned extrinsics for the window.
        """
        window_pred_track, window_pred_vis, window_pred_score = predict_tracks(
            self.cfg.query_method,
            self.cfg.max_query_pts,
            self.track_predictor,
            window_images,
            window_masks,
            window_fmaps_for_tracker,
            [window_size // 2, window_size],
            self.cfg.fine_tracking,
            self.bound_bboxes.expand(-1, window_images.shape[1], -1),
        )

        window_pred_track = window_pred_track.squeeze(0)
        window_pred_vis = window_pred_vis.squeeze(0)
        if window_pred_score is not None:
            window_pred_score = window_pred_score.squeeze(0)

        tracks_normalized_refined = cam_from_img(
            window_pred_track,
            self.intrinsics.expand(window_size + 1, -1, -1),
            self.extra_params.expand(window_size + 1, -1),
        )

        # Conduct triangulation to all the frames using LORANSAC
        best_triangulated_points, best_inlier_num, best_inlier_mask = (
            triangulate_tracks(
                extrinsics,
                tracks_normalized_refined,
                track_vis=window_pred_vis,
                track_score=window_pred_score,
            )
        )

        (
            filtered_points,
            filtered_tracks,
            filtered_inlier_masks,
            valid_tracks_mask,
        ) = self.filter_points_and_compute_masks(
            best_triangulated_points,
            window_pred_track,
            extrinsics,
            max_reproj_error=max_reproj_error,
            min_valid_track_length=min_valid_track_length,
        )

        filtered_vis = window_pred_vis[:, valid_tracks_mask]

        return (
            filtered_points,
            filtered_tracks,
            filtered_inlier_masks,
            filtered_vis,
        )

    def update_dicts_by_reconstruction(
        self, reconstruction, start_idx, end_idx
    ):
        # Update dicts by reconstruction
        for image_id, image in reconstruction.images.items():
            self.frame_dict[image_id]["extri"] = torch.tensor(
                image.cam_from_world.matrix()
            )

        for point3D_id in sorted(self.point_dict.keys()):
            pycolmap_point3D_id = point3D_id + 1
            if pycolmap_point3D_id in reconstruction.point3D_ids():
                point3D = reconstruction.points3D[pycolmap_point3D_id]
                self.point_dict[point3D_id]["xyz"] = torch.from_numpy(
                    point3D.xyz
                ).float()
                visible_frames = []
                for track_element in point3D.track.elements:
                    visible_frames.append(track_element.image_id)

                # Remove the filtered 2D observations
                for frame_idx in range(start_idx, end_idx):
                    if frame_idx in self.point_dict[point3D_id]["track"]:
                        if frame_idx not in visible_frames:
                            del self.point_dict[point3D_id]["track"][frame_idx]
                            self.frame_dict[frame_idx]["visible_points"].remove(
                                point3D_id
                            )
            else:
                for frame_idx in range(start_idx, end_idx):
                    if frame_idx in self.point_dict[point3D_id]["track"]:
                        del self.point_dict[point3D_id]["track"][frame_idx]
                        self.frame_dict[frame_idx]["visible_points"].remove(
                            point3D_id
                        )


def log_ba_summary(summary):
    logging.info(f"Residuals : {summary.num_residuals_reduced}")
    if summary.num_residuals_reduced > 0:
        logging.info(f"Parameters : {summary.num_effective_parameters_reduced}")
        logging.info(
            f"Iterations : {summary.num_successful_steps + summary.num_unsuccessful_steps}"
        )
        logging.info(f"Time : {summary.total_time_in_seconds} [s]")
        logging.info(
            f"Initial cost : {np.sqrt(summary.initial_cost / summary.num_residuals_reduced)} [px]"
        )
        logging.info(
            f"Final cost : {np.sqrt(summary.final_cost / summary.num_residuals_reduced)} [px]"
        )
        return True
    else:
        print("No residuals reduced")
        return False


def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
    bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
    bundle_adjuster.set_up_problem(
        reconstruction, ba_options.create_loss_function()
    )
    solver_options = bundle_adjuster.set_up_solver_options(
        bundle_adjuster.problem, ba_options.solver_options
    )
    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options, bundle_adjuster.problem, summary)
    return summary


def extract_window(start_idx, end_idx, *vars):
    """
    Extracts a window from start_idx to end_idx along dimension 1 for each variable in vars.
    """
    return [
        var[:, start_idx:end_idx, ...] if var is not None else None
        for var in vars
    ]


def remove_query(*vars):
    """
    Removes the first element along dimension 1 for each variable in vars.
    """
    return [var[:, 1:, ...] if var is not None else None for var in vars]
