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
    return [torch.index_select(tensor, dim, order) for tensor in tensors]


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


def visual_query_points(images, query_index, query_points):
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
    cv2.imwrite("image_cv2.png", image_cv2)


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
