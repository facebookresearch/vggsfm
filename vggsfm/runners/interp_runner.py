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
from vggsfm.utils.utils import average_camera_prediction, generate_grid_samples, generate_rank_by_interval, sample_subrange, closed_form_inverse_OpenCV, create_video_with_reprojections
from vggsfm.utils.pose_interp import interpolate_transformations

from vggsfm.utils.tensor_to_pycolmap import (
    batch_matrix_to_pycolmap,
    pycolmap_to_batch_matrix,
)
from vggsfm.utils.align import align_camera_extrinsics, apply_transformation

from vggsfm.utils.triangulation import triangulate_tracks
from vggsfm.utils.triangulation_helpers import filter_all_points3D, cam_from_img, project_3D_points


class InterpRunner(VGGSfMRunner):
    def __init__(self, cfg):
        super().__init__(cfg)

        assert self.cfg.shared_camera==True
        assert self.cfg.query_by_interval==True
        
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
        key_frame_indexes=None,
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
        print("NOTE THAT WE ASSUME THE INPUT TO INTERP RUNNER IS A SEQUENCE OF IMAGES, i.e., ORDERED!")
        
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

            image_paths = [os.path.basename(imgpath) for imgpath in image_paths]
            
            total_frames = images.shape[1]
            if key_frame_indexes is None:
                key_frame_indexes = list(range(0, total_frames, self.cfg.subsample_fps))
                if (total_frames -1) not in key_frame_indexes:
                    key_frame_indexes.append(total_frames - 1)


            key_frame_images = images[:, key_frame_indexes]
            key_frame_masks = masks[:, key_frame_indexes] if masks is not None else None
            key_frame_crop_params = crop_params[:, key_frame_indexes] if crop_params is not None else None
            key_frame_image_paths = [image_paths[i] for i in key_frame_indexes]
            
            
            # Perform sparse reconstruction
            print("Performing sparse reconstruction on key frames...")
            key_frame_predictions = self.sparse_reconstruct(
                key_frame_images,
                masks=key_frame_masks,
                crop_params=key_frame_crop_params,
                image_paths=key_frame_image_paths,
                query_frame_num=query_frame_num,
                seq_name=seq_name,
                output_dir=output_dir,
                back_to_original_resolution=False,
            )

            extrinsics = key_frame_predictions["extrinsics_opencv"]
            
            interp_timesteps = np.arange(0, total_frames)
            extrinsics_numpy = extrinsics.cpu().numpy()
            
            all_interped_rot, all_interped_trans = interpolate_transformations(extrinsics_numpy[:,:,:3], extrinsics_numpy[:,:,-1], key_frame_indexes, interp_timesteps)
            all_interped_rot = torch.from_numpy(all_interped_rot).to(self.device)
            all_interped_trans = torch.from_numpy(all_interped_trans).to(self.device)
            
            all_interped_extrinsics = torch.cat([all_interped_rot, all_interped_trans[..., None]], dim=-1)
            all_interped_intrinsics = key_frame_predictions["intrinsics_opencv"][0:1].expand(total_frames, -1, -1)
            
            
            if key_frame_predictions["extra_params"] is not None:
                all_interped_extra_params = key_frame_predictions["extra_params"][0:1].expand(total_frames, -1)   
            else:
                all_interped_extra_params = None
                
            if self.cfg.interp_extra_pt_pixel_interval>0:
                
                fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(images)
                
                # Calculate bounding boxes if crop parameters are provided
                if crop_params is not None:
                    bound_bboxes = crop_params[:, :, -4:-2].abs().to(self.device)
                    # also remove those near the boundary
                    bound_bboxes[bound_bboxes != 0] += self.remove_borders
                    bound_bboxes = torch.cat(
                        [bound_bboxes, images.shape[-1] - bound_bboxes], dim=-1
                    )


                # key_frame_fmaps_for_tracker = fmaps_for_tracker[:, key_frame_indexes]
                # key_frame_bound_bboxes = bound_bboxes[:, key_frame_indexes]
                additional_points_dict = self.triangulate_extra_points_interp(
                    images,
                    masks,
                    fmaps_for_tracker,
                    bound_bboxes,
                    all_interped_intrinsics,
                    all_interped_extra_params,
                    all_interped_extrinsics,
                    image_paths,
                    total_frames,
                    key_frame_indexes,
                )


            intrinsics_original_resolution = all_interped_intrinsics.clone()

            real_image_size = crop_params[0, 0][:2]
            resize_ratio = real_image_size.max() / images.shape[-1] 
            
            real_focal = resize_ratio * intrinsics_original_resolution[0, 0,0]
            real_pp = real_image_size // 2
            
            intrinsics_original_resolution[:, 0,0] = real_focal
            intrinsics_original_resolution[:, 1,1] = real_focal

            intrinsics_original_resolution[:, 0,2] = real_pp[0]
            intrinsics_original_resolution[:, 1,2] = real_pp[1]
            
            # dirty...
            all_interped_intrinsics = intrinsics_original_resolution
            
            predictions = {}
            

            predictions["extrinsics_opencv"] = all_interped_extrinsics
            predictions["intrinsics_opencv"] = intrinsics_original_resolution

            predictions["points3D"] = key_frame_predictions["points3D"]
            predictions["points3D_rgb"] = key_frame_predictions["points3D_rgb"]
            predictions["extra_params"] = all_interped_extra_params
                
                
            if self.cfg.dense_depth:
            
                interp_sparse_depth = defaultdict(list)
                
                for frame_idx in range(total_frames):
                    cur_image_name = image_paths[frame_idx]
                    additional_points3D = additional_points_dict[cur_image_name]["points3D"]
                    if len(additional_points3D)<20:
                        raise ValueError(f"Too few points3D for frame {frame_idx} to align depth maps, in total {len(additional_points3D)}")
                    points2D, points_cam = project_3D_points(additional_points3D, all_interped_extrinsics[frame_idx][None], 
                                                            intrinsics_original_resolution[frame_idx][None], all_interped_extra_params[frame_idx][None], return_points_cam=True)
                    depth = points_cam[:, -1][...,None]
                    uvd = torch.cat([points2D, depth], dim=-1)[0].cpu().numpy()
                    interp_sparse_depth[cur_image_name].append(uvd)
                predictions["sparse_depth"] = interp_sparse_depth
                predictions["reconstruction"] = None


                predictions = self.dense_reconstruct(
                    predictions, image_paths, original_images, min_res=self.cfg.dense_depth_min_res
                )
                
                
                # all_additional_points3D = torch.cat( [ additional_points_dict[img_name]["points3D"] for img_name in image_paths ], dim=0, )
                # all_additional_points3D_rgb = torch.cat( [ additional_points_dict[img_name]["points3D_rgb"] for img_name in image_paths ], dim=0, )

                # unproj_dense_points3D = {}
                
            
                
                if self.cfg.visual_dense_point_cloud:
                    depth_dict = predictions["depth_dict"]
                    # Call the function within your existing code
                    unproj_dense_points3D = unproject_dense_points(
                        depth_dict, image_paths, total_frames, all_interped_extrinsics, 
                        intrinsics_original_resolution, all_interped_extra_params, 
                        original_images, self.device
                    )
                    predictions["unproj_dense_points3D"] = unproj_dense_points3D
                    # haha_point = np.concatenate([unproj_dense_points3D[image_paths[0]][0], unproj_dense_points3D[image_paths[1]][0], unproj_dense_points3D[image_paths[10]][0]], axis=0)
                    # haha_rgb = np.concatenate([unproj_dense_points3D[image_paths[0]][1], unproj_dense_points3D[image_paths[1]][1], unproj_dense_points3D[image_paths[10]][1]], axis=0)
                    
                    # haha_point = np.concatenate([unproj_dense_points3D[image_paths[0]][0], unproj_dense_points3D[image_paths[1]][0], ], axis=0)
                    # haha_rgb = np.concatenate([unproj_dense_points3D[image_paths[0]][1], unproj_dense_points3D[image_paths[1]][1], ], axis=0)
                    # predictions["points3D"] = torch.from_numpy(haha_point).to(self.device)
                    # predictions["points3D_rgb"] = torch.from_numpy(haha_rgb).to(self.device)
                    
                    # self.visualize_3D_in_gradio(predictions, seq_name, output_dir)
            
                
            # Create reprojection video if enabled
            if self.cfg.make_reproj_video:
                
                sparse_depth, sparse_point = compute_sparse_depth_and_points_for_interp(
                    key_frame_predictions["points3D"], all_interped_extrinsics, 
                    intrinsics_original_resolution, all_interped_extra_params, 
                    image_paths, total_frames
                )
                
                
                image_dir_prefix = os.path.dirname(image_paths[0])
                image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

                
                max_hw = crop_params[0, :, :2].max(dim=0)[0].long()
                video_size = (max_hw[0].item(), max_hw[1].item())
                
                
                img_with_circles_list = create_video_with_reprojections(
                    image_dir_prefix,
                    video_size,
                    image_paths,
                    sparse_depth,
                    sparse_point,
                    original_images,
                    points3D=key_frame_predictions["points3D"].cpu().numpy(),
                    reconstruction=None,
                )

                
                predictions["reproj_video"] = img_with_circles_list
                if self.cfg.save_to_disk:
                    self.save_reprojection_video(
                        img_with_circles_list, video_size, output_dir
                    )

            import pdb;pdb.set_trace()
            
            return predictions



    def triangulate_extra_points_interp(
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
        key_frame_indexes,
    ):
        """
        Triangulate extra points for each frame and return a dictionary containing 3D points and their RGB values.

        Returns:
            dict: A dictionary containing 3D points and their RGB values for each frame.
        """
        from vggsfm.models.utils import sample_features4d


        key_frame_images = images[:, key_frame_indexes]
        key_frame_masks = masks[:, key_frame_indexes] if masks is not None else None
        key_frame_fmaps_for_tracker = fmaps_for_tracker[:, key_frame_indexes]
        key_frame_bound_bboxes = bound_bboxes[:, key_frame_indexes]
        
        key_frame_extrinsics = extrinsics_opencv[key_frame_indexes]
        key_frame_intrinsics = intrinsics_opencv[key_frame_indexes]
        key_frame_extra_params = extra_params[key_frame_indexes] if extra_params is not None else None
        
        additional_points_dict = {}
        for frame_idx in range(frame_num):
            print("Triangulating more points for frame: ", frame_idx)
            rect_for_sample = bound_bboxes[:, frame_idx].clone()
            rect_for_sample = rect_for_sample.floor()
            rect_for_sample[:, :2] += self.cfg.interp_extra_pt_pixel_interval // 2
            rect_for_sample[:, 2:] -= self.cfg.interp_extra_pt_pixel_interval // 2
            grid_points = generate_grid_samples(
                rect_for_sample, pixel_interval=self.cfg.interp_extra_pt_pixel_interval
            )
            grid_points = grid_points.floor()

            grid_rgb = sample_features4d(
                images[:, frame_idx], grid_points[None]
            ).squeeze(0)

            feed_images = torch.cat([images[:, frame_idx:frame_idx+1], key_frame_images], dim=1)
            feed_masks = torch.cat([masks[:, frame_idx:frame_idx+1], key_frame_masks], dim=1) if masks is not None else None
            feed_fmaps_for_tracker = torch.cat([fmaps_for_tracker[:, frame_idx:frame_idx+1], key_frame_fmaps_for_tracker], dim=1)
            feed_bound_bboxes = torch.cat([bound_bboxes[:, frame_idx:frame_idx+1], key_frame_bound_bboxes], dim=1)

            extra_track, extra_vis, extra_score = predict_tracks(
                self.cfg.query_method,
                self.cfg.max_query_pts,
                self.track_predictor,
                feed_images,
                feed_masks,
                feed_fmaps_for_tracker,
                [0],
                fine_tracking=False,
                bound_bboxes=feed_bound_bboxes,
                query_points_dict={0: grid_points[None]},
            )

            feed_extra_params = torch.cat([extra_params[frame_idx:frame_idx+1], key_frame_extra_params], dim=0) if extra_params is not None else None
            feed_extrinsics = torch.cat([extrinsics_opencv[frame_idx:frame_idx+1], key_frame_extrinsics], dim=0)
            feed_intrinsics = torch.cat([intrinsics_opencv[frame_idx:frame_idx+1], key_frame_intrinsics], dim=0)

            extra_track_normalized = cam_from_img(
                extra_track, feed_intrinsics, feed_extra_params
            )

            (extra_triangulated_points, extra_inlier_num, extra_inlier_mask) = (
                triangulate_tracks(
                    feed_extrinsics,
                    extra_track_normalized.squeeze(0),
                    track_vis=extra_vis.squeeze(0),
                    track_score=extra_score.squeeze(0),
                )
            )

            valid_triangulation_mask = extra_inlier_num > 3

            valid_poins3D_mask, _ = filter_all_points3D(
                extra_triangulated_points,
                extra_track.squeeze(0),
                feed_extrinsics,
                feed_intrinsics,
                extra_params=feed_extra_params,  # Pass extra_params to filter_all_points3D
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


def unproject_dense_points(
    depth_dict, image_paths, total_frames, all_interped_extrinsics, 
    intrinsics_original_resolution, all_interped_extra_params, 
    original_images, device
):
    """
    Unprojects dense points from depth maps to 3D space.

    Args:
        depth_dict (dict): Dictionary containing depth maps for each frame.
        image_paths (list): List of image paths.
        total_frames (int): Total number of frames.
        all_interped_extrinsics (torch.Tensor): Interpolated extrinsics for all frames.
        intrinsics_original_resolution (torch.Tensor): Intrinsics for all frames.
        all_interped_extra_params (torch.Tensor): Extra parameters for all frames.
        original_images (dict): Dictionary containing original images.
        device (torch.device): Device to perform computations on.

    Returns:
        dict: Dictionary containing unprojected 3D points and their RGB values for each frame.
    """
    # We assume all depth maps have the same shape
    ww, hh = depth_dict[image_paths[0]].shape
    x_coords = np.arange(hh)
    y_coords = np.arange(ww)
    xx, yy = np.meshgrid(x_coords, y_coords)

    grid_points2d = np.column_stack((xx.ravel(), yy.ravel()))
    grid_points2d = torch.from_numpy(grid_points2d).to(device)

    unproj_dense_points3D = {}
    for frame_idx in range(total_frames):
        frame_name = image_paths[frame_idx]
        cur_depth_map = depth_dict[frame_name]

        cur_extrinsics = all_interped_extrinsics[frame_idx]

        valid_depth_mask = cur_depth_map != 0
        valid_depth_mask = torch.from_numpy(valid_depth_mask.reshape(-1)).to(device)
        depth_values = cur_depth_map.reshape(-1)
        depth_values = torch.from_numpy(depth_values).to(device)

        cur_points2d = grid_points2d[valid_depth_mask]
        cur_depth_values = depth_values[valid_depth_mask]

        cur_intrinsics = intrinsics_original_resolution[frame_idx]
        cur_extra_params = all_interped_extra_params[frame_idx]
        unproject_points = cam_from_img(cur_points2d[None], cur_intrinsics[None], cur_extra_params[None])
        unproject_points = unproject_points.squeeze(0)
        unproject_points_homo = torch.cat((unproject_points, torch.ones_like(unproject_points[..., :1])), dim=-1)

        unproject_points_withz = (unproject_points_homo * cur_depth_values.reshape(-1, 1))

        unproject_points_withz_homo = torch.cat((unproject_points_withz, torch.ones_like(unproject_points_withz[..., :1])), dim=-1)

        extrinsics_4x4 = torch.eye(4).to(device).to(cur_extrinsics.dtype)
        extrinsics_4x4[:3, :] = cur_extrinsics
        extrinsics_4x4_inv = closed_form_inverse_OpenCV(extrinsics_4x4[None])[0]
        unproject_points_world = torch.mm(extrinsics_4x4_inv[:3], unproject_points_withz_homo.transpose(-1, -2))
        unproject_points_world = unproject_points_world.transpose(-1, -2)

        rgb_image = original_images[frame_name] / 255.0
        rgb = rgb_image.reshape(-1, 3)
        rgb = rgb[valid_depth_mask.cpu().numpy()]

        unproj_dense_points3D[frame_name] = np.array([unproject_points_world.cpu().numpy(), rgb])

    return unproj_dense_points3D




def compute_sparse_depth_and_points_for_interp(
    points3D, all_interped_extrinsics, intrinsics_original_resolution, 
    all_interped_extra_params, image_paths, total_frames
):
    """
    Compute sparse depth and points for each frame.

    Args:
        key_frame_predictions (dict): Predictions from key frames.
        all_interped_extrinsics (torch.Tensor): Interpolated extrinsics for all frames.
        intrinsics_original_resolution (torch.Tensor): Intrinsics for all frames.
        all_interped_extra_params (torch.Tensor): Extra parameters for all frames.
        image_paths (list): List of image paths.
        total_frames (int): Total number of frames.

    Returns:
        tuple: Dictionaries containing sparse depth and points for each frame.
    """
    sparse_depth = defaultdict(list)
    sparse_point = defaultdict(list)

    points3D_xyz = points3D.clone()
    point3D_idxes = np.arange(len(points3D_xyz))

    projected_points2D, projected_points_cam = project_3D_points(
        points3D_xyz, all_interped_extrinsics, intrinsics_original_resolution, 
        all_interped_extra_params, return_points_cam=True
    )
    depths = projected_points_cam[:, -1]
    for frame_idx in range(total_frames):
        cur_image_name = image_paths[frame_idx]
        cur_projected_points2D = projected_points2D[frame_idx]
        cur_depths = depths[frame_idx]
        valid_depth = cur_depths >= 0.01

        sparse_depth[cur_image_name] = np.hstack((
            cur_projected_points2D[valid_depth].cpu().numpy(), 
            cur_depths[valid_depth][:, None].cpu().numpy()
        ))
        sparse_point[cur_image_name] = np.hstack((
            points3D_xyz[valid_depth].cpu().numpy(), 
            point3D_idxes[valid_depth.cpu().numpy()][:, None]
        ))

    return sparse_depth, sparse_point


# python interp_demo.py save_to_disk=True camera_type=SIMPLE_RADIAL interp_extra_pt_pixel_interval=16  dense_depth=True subsample_fps=8 make_reproj_video=True

# python interp_demo.py SCENE_DIR=examples/1017086644/ save_to_disk=True camera_type=SIMPLE_RADIAL interp_extra_pt_pixel_interval=16  dense_depth=True subsample_fps=24 make_reproj_video=True