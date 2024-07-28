# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType


import torch
import torch.nn as nn
import torch.nn.functional as F


from .metric import closed_form_inverse

import os
import cv2
import struct


def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that we can switch [query_index] and [0]
    so that the content of query_index would be placed at [0]
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    """
    Switch the tensor among the specific dimension
    """
    return [torch.index_select(tensor, dim, order) if tensor is not None else None for tensor in tensors]


def set_seed_and_print(seed):
    accelerate_set_seed(seed, device_specific=True)
    print(f"----------Seed is set to {np.random.get_state()[1][0]} now----------")


def transform_camera_relative_to_first(pred_cameras, batch_size):
    pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()
    rel_transform = closed_form_inverse(pred_se3[0:1, :, :])
    rel_transform = rel_transform.expand(batch_size, -1, -1)

    pred_se3_rel = torch.bmm(rel_transform, pred_se3)
    pred_se3_rel[..., :3, 3] = 0.0
    pred_se3_rel[..., 3, 3] = 1.0

    pred_cameras.R = pred_se3_rel[:, :3, :3].clone()
    pred_cameras.T = pred_se3_rel[:, 3, :3].clone()
    return pred_cameras


def farthest_point_sampling(distance_matrix, num_samples, most_common_frame_index=0):
    # Number of points
    distance_matrix = distance_matrix.clamp(min=0)

    N = distance_matrix.size(0)

    # Initialize
    # Start from the first point (arbitrary choice)
    selected_indices = [most_common_frame_index]
    # Track the minimum distances to the selected set
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        # Find the farthest point from the current set of selected points
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())

        check_distances = distance_matrix[farthest_point]
        # the ones already selected would not selected any more
        check_distances[selected_indices] = 0

        # Break the loop if all points have been selected
        if len(selected_indices) == N:
            break

    return selected_indices


def generate_rank_by_interval(N, k):
    result = []
    for start in range(k):
        for multiple in range(0, N, k):
            number = start + multiple
            if number < N:
                result.append(number)
            else:
                break
    return result


def visual_query_points(images, query_index, query_points, save_name="image_cv2.png"):
    """
    Processes an image by converting it to BGR color space, drawing circles at specified points,
    and saving the image to a file.
    Args:
    images (torch.Tensor): A batch of images in the shape (N, C, H, W).
    query_index (int): The index of the image in the batch to process.
    query_points (list of tuples): List of (x, y) tuples where circles should be drawn.
    Returns:
    None
    """
    # Convert the image from RGB to BGR
    image_cv2 = cv2.cvtColor(
        (images[:, query_index].squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    # Draw circles at the specified query points
    for x, y in query_points[0]:
        image_cv2 = cv2.circle(image_cv2, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Save the processed image to a file
    cv2.imwrite(save_name, image_cv2)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)



def filter_invisible_reprojections(uvs_int, depths):
    """
    Filters out invisible 3D points when projecting them to 2D.

    When reprojecting 3D points to 2D, multiple 3D points may map to the same 2D pixel due to occlusion or close proximity. 
    This function filters out the reprojections of invisible 3D points based on their depths.

    Parameters:
    uvs_int (np.ndarray): Array of 2D points with shape (n, 2).
    depths (np.ndarray): Array of depths corresponding to the 3D points, with shape (n).

    Returns:
    np.ndarray: A boolean mask with shape (n). True indicates the point is kept, False means it is removed.
    """
    
    # Find unique rows and their indices
    _, inverse_indices, counts = np.unique(uvs_int, axis=0, return_inverse=True, return_counts=True)

    # Initialize mask with True (keep all points initially)
    mask = np.ones(uvs_int.shape[0], dtype=bool)

    # Set the mask to False for non-unique points and keep the one with the smallest depth
    for i in np.where(counts > 1)[0]:
        duplicate_indices = np.where(inverse_indices == i)[0]
        min_depth_index = duplicate_indices[np.argmin(depths[duplicate_indices])]
        mask[duplicate_indices] = False
        mask[min_depth_index] = True

    return mask


def create_video_with_reprojections(output_path, fname_prefix, video_size, reconstruction, 
                                    image_paths, 
                                    sparse_depth, sparse_point,
                                    draw_radius=3, cmap = "gist_rainbow", fps=1, color_mode="dis_to_center"):
    import matplotlib
    print("Generating reprojection video")
    
    video_size_rev = video_size[::-1]

    video_writer = cv2.VideoWriter(output_path, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)
    
    colormap = matplotlib.colormaps.get_cmap(cmap)
    
    if color_mode == "dis_to_center":
        points3D = np.array([point.xyz for point in reconstruction.points3D.values()])
        median_point = np.median(points3D, axis=0)
        distances = np.linalg.norm(points3D - median_point, axis=1)
        min_dis = distances.min()
        # max_dis = distances.max()
        max_dis = np.percentile(distances, 99)  # 99th percentile distance to avoid extreme values
    elif color_mode == "dis_to_origin":
        points3D = np.array([point.xyz for point in reconstruction.points3D.values()])
        distances = np.linalg.norm(points3D, axis=1)
        min_dis = distances.min()
        max_dis = distances.max()
    elif color_mode == "point_order":
        max_point3D_idx = max(reconstruction.point3D_ids())
    else:
        raise NotImplementedError
        
    for fname in sorted(image_paths):  
        img_with_circles = cv2.imread(os.path.join(fname_prefix, fname))
        
        uvds = np.array(sparse_depth[fname])
        uvs, uv_depth = uvds[:, :2], uvds[:, -1]
        uvs_int = np.round(uvs).astype(int)
    
        if color_mode == "dis_to_center":
            point3D_xyz = np.array(sparse_point[fname])[:, :3]
            dis = np.linalg.norm(point3D_xyz-median_point, axis=1)
            color_indices = (dis - min_dis) / (max_dis-min_dis) # 0-1
        elif color_mode == "dis_to_origin":
            point3D_xyz = np.array(sparse_point[fname])[:, :3]
            dis = np.linalg.norm(point3D_xyz, axis=1)
            color_indices = (dis - min_dis) / (max_dis-min_dis) # 0-1
        elif color_mode == "point_order":
            point3D_idxes = np.array(sparse_point[fname])[:, -1]
            color_indices = point3D_idxes / max_point3D_idx # 0-1
        else:
            raise NotImplementedError

        colors = (colormap(color_indices)[:, :3] * 255).astype(int)
        
        # Filter out occluded points
        vis_reproj_mask = filter_invisible_reprojections(uvs_int, uv_depth)

        for (x, y), color in zip(uvs_int[vis_reproj_mask], colors[vis_reproj_mask]): 
            cv2.circle(img_with_circles, (x, y), radius=draw_radius, 
                       color=(int(color[0]), int(color[1]), int(color[2])), 
                       thickness=-1, lineType=cv2.LINE_AA)

        if img_with_circles.shape[:2] != video_size_rev:
            # Center Pad
            target_h, target_w = video_size_rev
            h, w, c = img_with_circles.shape
            top_pad = (target_h - h) // 2
            bottom_pad = target_h - h - top_pad
            left_pad = (target_w - w) // 2
            right_pad = target_w - w - left_pad
            img_with_circles = cv2.copyMakeBorder(img_with_circles, top_pad, bottom_pad, left_pad, right_pad, 
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
        video_writer.write(img_with_circles)

    video_writer.release()
    print("Finished generating reprojection video")



def create_depth_map_visual(depth_map, raw_img, img_fname, outdir):
    import matplotlib

    # Normalize the depth map to the range 0-255
    depth_map_visual = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map_visual = depth_map_visual.astype(np.uint8)

    # Get the colormap
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    # Apply the colormap and convert to uint8
    depth_map_visual = (cmap(depth_map_visual)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # Create a white split region
    split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255

    # Combine the raw image, split region, and depth map visual
    combined_result = cv2.hconcat([raw_img, split_region, depth_map_visual])

    # Save the result to a file
    output_filename = outdir + f"/depth_{os.path.basename(img_fname)}.png"
    cv2.imwrite(output_filename, combined_result)

    return output_filename



def align_dense_depth_maps(reconstruction, sparse_depth, depth_dir, cfg):
    # For dense depth estimation
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression

    # Define disparity and depth limits
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    unproj_dense_points3D = []
    fname_to_id = {reconstruction.images[imgid].name: imgid for imgid in reconstruction.images}

    for img_name in sparse_depth:
        sparse_uvd = np.array(sparse_depth[img_name])
        disp_map = read_array(os.path.join(depth_dir, img_name + ".bin"))

        ww, hh = disp_map.shape
        # Filter out the projections outside the image
        int_uv = np.round(sparse_uvd[:, :2]).astype(int)
        maskhh = (int_uv[:, 0] >= 0) & (int_uv[:, 0] < hh)
        maskww = (int_uv[:, 1] >= 0) & (int_uv[:, 1] < ww)
        mask = maskhh & maskww
        sparse_uvd = sparse_uvd[mask]
        int_uv = int_uv[mask]

        # Nearest neighbour sampling
        sampled_disps = disp_map[int_uv[:, 1], int_uv[:, 0]]

        # Note that dense depth maps may have some invalid values such as sky
        # they are marked as 0, hence filter out 0 from the sampled depths
        positive_mask = sampled_disps > 0
        sampled_disps = sampled_disps[positive_mask]
        sfm_depths = sparse_uvd[:, -1][positive_mask]

        sfm_depths = np.clip(sfm_depths, depth_min, depth_max)

        thres_ratio = 30
        target_disps = 1 / sfm_depths

        # RANSAC
        X = sampled_disps.reshape(-1, 1)
        y = target_disps
        ransac_thres = np.median(y) / thres_ratio
        ransac = RANSACRegressor(
            LinearRegression(),
            min_samples=2,
            residual_threshold=ransac_thres,
            max_trials=20000,
            loss="squared_error",
        )
        ransac.fit(X, y)
        scale = ransac.estimator_.coef_[0]
        shift = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_

        nonzero_mask = disp_map != 0

        # Rescale the disparity map
        disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift

        valid_depth_mask = (disp_map > 0) & (disp_map <= disparity_max)
        disp_map[~valid_depth_mask] = 0

        # Convert the disparity map to depth map
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)

        write_array(depth_map, os.path.join(depth_dir, img_name) + ".bin")

        if cfg.visual_dense_point_cloud:
            # TODO: remove the dirty codes here
            pyimg = reconstruction.images[fname_to_id[img_name]]
            pycam = reconstruction.cameras[pyimg.camera_id]

            # Generate the x and y coordinates
            x_coords = np.arange(hh)  
            y_coords = np.arange(ww)  
            xx, yy = np.meshgrid(x_coords, y_coords)

            valid_depth_mask_hw = np.copy(valid_depth_mask)
            sampled_points2d = np.column_stack(( xx.ravel(), yy.ravel()))
            # sampled_points2d = sampled_points2d + 0.5 # TODO figure it out if we still need +0.5
                    
            depth_values = depth_map.reshape(-1)
            valid_depth_mask = valid_depth_mask.reshape(-1)
            
            sampled_points2d = sampled_points2d[valid_depth_mask] 
            depth_values = depth_values[valid_depth_mask]
            
            unproject_points = pycam.cam_from_img(sampled_points2d)
            unproject_points_homo = np.hstack((unproject_points,  np.ones((unproject_points.shape[0], 1))))
            unproject_points_withz = unproject_points_homo * depth_values.reshape(-1, 1)
            unproject_points_world = pyimg.cam_from_world.inverse() * unproject_points_withz
            
            bgr = cv2.imread(os.path.join(cfg.SCENE_DIR, "images", img_name))
            
            rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_image = rgb_image / 255
            rgb = rgb_image.reshape(-1, 3)
            rgb = rgb[valid_depth_mask]
                
            unproj_dense_points3D.append(np.array([unproject_points_world, rgb]))

    return unproj_dense_points3D
