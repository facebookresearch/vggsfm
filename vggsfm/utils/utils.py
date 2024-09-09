# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib
import numpy as np

import torch
import torch.nn.functional as F
import os
import cv2
import math
import random
import struct
from tqdm import tqdm
from .metric import closed_form_inverse, closed_form_inverse_OpenCV

from scipy.spatial.transform import Rotation as sciR
from minipytorch3d.cameras import CamerasBase, PerspectiveCameras


def average_camera_prediction(
    camera_predictor,
    reshaped_image,
    batch_size,
    repeat_times=5,
    query_indices=None,
):
    # Use different frames as query for camera prediction
    # since camera_predictor is super fast,
    # this is almost a free-lunch

    # Ensure function is only used for inference with batch_size 1
    assert (
        batch_size == 1
    ), "This function is designed for inference with batch_size=1."

    # Determine the number of frames in the input image
    num_frames = len(reshaped_image)
    device = reshaped_image.device

    if query_indices is None:
        # Adjust repeat_times if there are fewer frames than requested repeats
        repeat_times = min(repeat_times, num_frames)

        # Randomly select query indices, ensuring the first frame is always included
        query_indices = random.sample(range(num_frames), repeat_times)
        if 0 not in query_indices:
            query_indices.insert(0, 0)

    # Initialize lists to store predictions
    rotation_matrices = []
    translations = []
    focal_lengths = []

    for query_index in query_indices:
        # Create a new order to place the query frame at the first position
        new_order = calculate_index_mappings(
            query_index, num_frames, device=device
        )
        reshaped_image_ordered = switch_tensor_order(
            [reshaped_image], new_order, dim=0
        )[0]

        # Predict camera parameters using the reordered image
        # NOTE the output has been in OPENCV format instead of PyTorch3D
        pred_cameras = camera_predictor(
            reshaped_image_ordered, batch_size=batch_size
        )["pred_cameras"]

        R = pred_cameras.R
        abs_T = pred_cameras.T

        extrinsics_4x4 = torch.eye(4, 4).to(R.dtype).to(R.device)
        extrinsics_4x4 = extrinsics_4x4[None].repeat(len(R), 1, 1)

        extrinsics_4x4[:, :3, :3] = R.clone()
        extrinsics_4x4[:, :3, 3] = abs_T.clone()

        # Get rotation and focal length from predictions
        extrinsics_4x4, focal_length = switch_tensor_order(
            [extrinsics_4x4, pred_cameras.focal_length], new_order, dim=0
        )

        rel_transform = closed_form_inverse_OpenCV(extrinsics_4x4[0:1])
        rel_transform = rel_transform.expand(len(extrinsics_4x4), -1, -1)

        # relative to the first camera
        # NOTE it is extrinsics_4x4 x rel_transform instead of rel_transform x extrinsics_4x4
        # this is different in opencv / pytorch3d convention
        extrinsics_4x4 = torch.bmm(extrinsics_4x4, rel_transform)

        R = extrinsics_4x4[:, :3, :3].clone()
        abs_T = extrinsics_4x4[:, :3, 3].clone()

        # Collect the relative rotation and translation matrices
        rotation_matrices.append(R[None])
        translations.append(abs_T[None])
        focal_lengths.append(focal_length[None])

    # Concatenate the predictions across sampled frames
    rotation_matrices = torch.concat(rotation_matrices)
    translations = torch.concat(translations)
    focal_lengths = torch.concat(focal_lengths)

    # Average the rotation matrices using a helper function
    avg_rotation_matrices = average_batch_rotation_matrices(
        rotation_matrices.cpu().numpy()
    )
    avg_rotation_matrices = torch.from_numpy(avg_rotation_matrices).to(device)

    # Compute the average translation and focal length
    avg_translation = translations.mean(0)
    avg_focal_length = focal_lengths.mean(0)

    # Create and return the averaged camera prediction
    avg_predicted_camera = PerspectiveCameras(
        focal_length=avg_focal_length,
        R=avg_rotation_matrices,
        T=avg_translation,
        device=device,
    )

    return avg_predicted_camera


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def average_batch_rotation_matrices(batch_rotation_matrices):
    """
    Average a batch of rotation matrices across the batch dimension.

    :param batch_rotation_matrices: Array of shape (B, N, 3, 3) containing rotation matrices.
    :return: Averaged rotation matrices of shape (N, 3, 3).
    """
    B, N, _, _ = batch_rotation_matrices.shape

    # Reshape batch matrices to a single array for vectorized operations
    reshaped_matrices = batch_rotation_matrices.reshape(B * N, 3, 3)

    # Convert matrices to quaternions using vectorized operation
    quaternions = sciR.from_matrix(reshaped_matrices).as_quat()

    # Reshape quaternions to (B, N, 4) for easier averaging
    quaternions = quaternions.reshape(B, N, 4)

    # Compute the mean quaternion for each set of N matrices
    mean_quaternions = np.mean(quaternions, axis=0)

    # import pdb;pdb.set_trace()
    # Normalize the resulting quaternions
    mean_quaternions /= np.linalg.norm(mean_quaternions, axis=1, keepdims=True)

    # Convert back to rotation matrices using vectorized operation
    average_rotation_matrices = sciR.from_quat(mean_quaternions).as_matrix()

    return average_rotation_matrices


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
    return [
        torch.index_select(tensor, dim, order) if tensor is not None else None
        for tensor in tensors
    ]


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


def farthest_point_sampling(
    distance_matrix, num_samples, most_common_frame_index=0
):
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


def generate_rank_by_midpoint(N):
    def mid(start, end):
        return start + (end - start) // 2

    # Start with the first midpoint, then add 0 and N-1
    sequence = [mid(0, N - 1), 0, N - 1]
    queue = [(0, mid(0, N - 1)), (mid(0, N - 1), N - 1)]  # Queue for BFS

    while queue:
        start, end = queue.pop(0)
        m = mid(start, end)
        if m not in sequence and start < m < end:
            sequence.append(m)
            queue.append((start, m))
            queue.append((m, end))

    return sequence


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


def generate_rank_by_dino(
    reshaped_image, camera_predictor, query_frame_num, image_size=336
):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = F.interpolate(
        reshaped_image,
        (image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )
    rgbs = camera_predictor._resnet_normalize_image(rgbs)

    # Get the image features (patch level)
    frame_feat = camera_predictor.backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(
        frame_feat_norm, frame_feat_norm.transpose(-1, -2)
    )
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
    fps_idx = farthest_point_sampling(
        distance_matrix, query_frame_num, most_common_frame_index
    )

    return fps_idx


def visual_query_points(
    images, query_index, query_points, save_name="image_cv2.png"
):
    """
    Processes an image by converting it to BGR color space, drawing circles at specified points,
    and saving the image to a file.
    Args:
    images (torch.Tensor): A batch of images in the shape (N, C, H, W).
    query_index (int): The index of the image in the batch to process.
    query_points (list of tuples): List of (x, y) tuples where circles should be drawn.
    Returns:
    """
    # Convert the image from RGB to BGR
    image_cv2 = cv2.cvtColor(
        (
            images[:, query_index].squeeze().permute(1, 2, 0).cpu().numpy()
            * 255
        ).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    # Draw circles at the specified query points
    for x, y in query_points[0]:
        image_cv2 = cv2.circle(image_cv2, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Save the processed image to a file
    cv2.imwrite(save_name, image_cv2)


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
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
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
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
    _, inverse_indices, counts = np.unique(
        uvs_int, axis=0, return_inverse=True, return_counts=True
    )

    # Initialize mask with True (keep all points initially)
    mask = np.ones(uvs_int.shape[0], dtype=bool)

    # Set the mask to False for non-unique points and keep the one with the smallest depth
    for i in np.where(counts > 1)[0]:
        duplicate_indices = np.where(inverse_indices == i)[0]
        min_depth_index = duplicate_indices[
            np.argmin(depths[duplicate_indices])
        ]
        mask[duplicate_indices] = False
        mask[min_depth_index] = True

    return mask


def create_video_with_reprojections(
    fname_prefix,
    video_size,
    reconstruction,
    image_paths,
    sparse_depth,
    sparse_point,
    original_images=None,
    draw_radius=3,
    cmap="gist_rainbow",
    color_mode="dis_to_center",
):
    """
    Generates a list of images with reprojections of 3D points onto 2D images.

    Args:
        fname_prefix (str): Prefix for image file names.
        video_size (tuple): Size of the video (width, height).
        reconstruction (object): 3D reconstruction object containing points3D.
        image_paths (list): List of image file paths.
        sparse_depth (dict): Dictionary of sparse depth information for each image.
        sparse_point (dict): Dictionary of sparse 3D points for each image.
        original_images (dict, optional): Dictionary of original images. Defaults to None.
        draw_radius (int, optional): Radius of the circles to draw. Defaults to 3.
        cmap (str, optional): Colormap to use for drawing. Defaults to "gist_rainbow".
        color_mode (str, optional): Mode for coloring points. Defaults to "dis_to_center".

    Returns:
        list: List of images with drawn circles.
    """
    print("Generating reprojection images")

    video_size_rev = video_size[::-1]
    colormap = matplotlib.colormaps.get_cmap(cmap)

    points3D = np.array(
        [point.xyz for point in reconstruction.points3D.values()]
    )

    if color_mode == "dis_to_center":
        median_point = np.median(points3D, axis=0)
        distances = np.linalg.norm(points3D - median_point, axis=1)
        min_dis, max_dis = distances.min(), np.percentile(distances, 95)
    elif color_mode == "dis_to_origin":
        distances = np.linalg.norm(points3D, axis=1)
        min_dis, max_dis = distances.min(), distances.max()
    elif color_mode == "point_order":
        max_point3D_idx = max(reconstruction.point3D_ids())
    else:
        raise NotImplementedError(
            f"Color mode '{color_mode}' is not implemented."
        )

    img_with_circles_list = []

    for img_basename in sorted(image_paths):
        if original_images is not None:
            img_with_circles = original_images[img_basename]
            img_with_circles = cv2.cvtColor(img_with_circles, cv2.COLOR_RGB2BGR)
        else:
            img_with_circles = cv2.imread(
                os.path.join(fname_prefix, img_basename)
            )

        uvds = np.array(sparse_depth[img_basename])
        uvs, uv_depth = uvds[:, :2], uvds[:, -1]
        uvs_int = np.round(uvs).astype(int)

        if color_mode == "dis_to_center":
            point3D_xyz = np.array(sparse_point[img_basename])[:, :3]
            dis = np.linalg.norm(point3D_xyz - median_point, axis=1)
            color_indices = (dis - min_dis) / (max_dis - min_dis)  # 0-1
        elif color_mode == "dis_to_origin":
            point3D_xyz = np.array(sparse_point[img_basename])[:, :3]
            dis = np.linalg.norm(point3D_xyz, axis=1)
            color_indices = (dis - min_dis) / (max_dis - min_dis)  # 0-1
        elif color_mode == "point_order":
            point3D_idxes = np.array(sparse_point[img_basename])[:, -1]
            color_indices = point3D_idxes / max_point3D_idx

        colors = (colormap(color_indices)[:, :3] * 255).astype(int)

        # Filter out occluded points
        vis_reproj_mask = filter_invisible_reprojections(uvs_int, uv_depth)

        for (x, y), color in zip(
            uvs_int[vis_reproj_mask], colors[vis_reproj_mask]
        ):
            cv2.circle(
                img_with_circles,
                (x, y),
                radius=draw_radius,
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if img_with_circles.shape[:2] != video_size_rev:
            # Center Pad
            target_h, target_w = video_size_rev
            h, w, c = img_with_circles.shape
            top_pad = (target_h - h) // 2
            bottom_pad = target_h - h - top_pad
            left_pad = (target_w - w) // 2
            right_pad = target_w - w - left_pad
            img_with_circles = cv2.copyMakeBorder(
                img_with_circles,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        img_with_circles_list.append(img_with_circles)

    print("Finished generating reprojection images")
    return img_with_circles_list


def save_video_with_reprojections(
    output_path, img_with_circles_list, video_size, fps=1
):
    """
    Saves a list of images as a video.

    Args:
        output_path (str): Path to save the output video.
        img_with_circles_list (list): List of images with drawn circles.
        video_size (tuple): Size of the video (width, height).
        fps (int, optional): Frames per second for the video. Defaults to 1.
    """
    print("Saving video with reprojections")

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size
    )

    for img_with_circles in img_with_circles_list:
        video_writer.write(img_with_circles)

    video_writer.release()
    print("Finished saving video with reprojections")


def create_depth_map_visual(depth_map, raw_img, output_filename):
    # Normalize the depth map to the range 0-255
    depth_map_visual = (
        (depth_map - depth_map.min())
        / (depth_map.max() - depth_map.min())
        * 255.0
    )
    depth_map_visual = depth_map_visual.astype(np.uint8)

    # Get the colormap
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    # Apply the colormap and convert to uint8
    depth_map_visual = (cmap(depth_map_visual)[:, :, :3] * 255)[
        :, :, ::-1
    ].astype(np.uint8)

    # Create a white split region
    split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255

    # Combine the raw image, split region, and depth map visual
    combined_result = cv2.hconcat([raw_img, split_region, depth_map_visual])

    # Save the result to a file
    cv2.imwrite(output_filename, combined_result)

    return output_filename


def extract_dense_depth_maps(depth_model, image_paths, original_images=None):
    """
    Extract dense depth maps from a list of image paths
    Note that the monocular depth model outputs disp instead of real depth map
    """

    print("Extracting dense depth maps")
    disp_dict = {}

    for idx in tqdm(
        range(len(image_paths)), desc="Predicting monocular depth maps"
    ):
        img_fname = image_paths[idx]
        basename = os.path.basename(img_fname)

        if original_images is not None:
            raw_img = original_images[basename]
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        else:
            raw_img = cv2.imread(img_fname)

        # raw resolution
        disp_map = depth_model.infer_image(
            raw_img, min(1024, max(raw_img.shape[:2]))
        )

        disp_dict[basename] = disp_map

    print("Monocular depth maps complete. Depth alignment to be conducted.")
    return disp_dict


def align_dense_depth_maps(
    reconstruction,
    sparse_depth,
    disp_dict,
    original_images,
    visual_dense_point_cloud=False,
):
    # For dense depth estimation
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression

    # Define disparity and depth limits
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    depth_dict = {}
    unproj_dense_points3D = {}
    fname_to_id = {
        reconstruction.images[imgid].name: imgid
        for imgid in reconstruction.images
    }

    for img_basename in tqdm(
        sparse_depth, desc="Load monocular depth and Align"
    ):
        sparse_uvd = np.array(sparse_depth[img_basename])

        if len(sparse_uvd) <= 0:
            raise ValueError("Too few points for depth alignment")

        disp_map = disp_dict[img_basename]

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

        if ransac_thres <= 0:
            raise ValueError("Ill-posed scene for depth alignment")

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
        # inlier_mask = ransac.inlier_mask_

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

        depth_dict[img_basename] = depth_map

        if visual_dense_point_cloud:
            # TODO: remove the dirty codes here
            pyimg = reconstruction.images[fname_to_id[img_basename]]
            pycam = reconstruction.cameras[pyimg.camera_id]

            # Generate the x and y coordinates
            x_coords = np.arange(hh)
            y_coords = np.arange(ww)
            xx, yy = np.meshgrid(x_coords, y_coords)

            # valid_depth_mask_hw = np.copy(valid_depth_mask)
            sampled_points2d = np.column_stack((xx.ravel(), yy.ravel()))
            # sampled_points2d = sampled_points2d + 0.5 # TODO figure it out if we still need +0.5

            depth_values = depth_map.reshape(-1)
            valid_depth_mask = valid_depth_mask.reshape(-1)

            sampled_points2d = sampled_points2d[valid_depth_mask]
            depth_values = depth_values[valid_depth_mask]

            unproject_points = pycam.cam_from_img(sampled_points2d)
            unproject_points_homo = np.hstack(
                (unproject_points, np.ones((unproject_points.shape[0], 1)))
            )
            unproject_points_withz = (
                unproject_points_homo * depth_values.reshape(-1, 1)
            )
            unproject_points_world = (
                pyimg.cam_from_world.inverse() * unproject_points_withz
            )

            rgb_image = original_images[img_basename] / 255.0
            rgb = rgb_image.reshape(-1, 3)
            rgb = rgb[valid_depth_mask]

            unproj_dense_points3D[img_basename] = np.array(
                [unproject_points_world, rgb]
            )

    if not visual_dense_point_cloud:
        unproj_dense_points3D = None

    return depth_dict, unproj_dense_points3D


def generate_grid_samples(rect, N=None, pixel_interval=None):
    """
    Generate a tensor with shape (N, 2) representing grid-sampled points inside a rectangle.

    Parameters:
    rect (torch.Tensor): Tensor of shape (1, 4) indicating the rectangle [topleftx, toplefty, bottomrightx, bottomrighty].
    N (int): Number of points to sample within the rectangle.

    Returns:
    torch.Tensor: Tensor of shape (N, 2) containing sampled points.
    """
    # Extract coordinates from the rectangle
    topleft_x, topleft_y, bottomright_x, bottomright_y = rect[0]

    # Calculate the width and height of the rectangle
    width = bottomright_x - topleft_x
    height = bottomright_y - topleft_y

    # Determine the number of points along each dimension
    if pixel_interval is not None:
        num_samples_x = max(1, int(width // pixel_interval))
        num_samples_y = max(1, int(height // pixel_interval))
        N = num_samples_x * num_samples_y
    else:
        aspect_ratio = width / height
        num_samples_x = int(math.sqrt(N * aspect_ratio))
        num_samples_y = int(N / num_samples_x)

    # Generate linspace for x and y coordinates
    x_coords = torch.linspace(
        topleft_x, bottomright_x, num_samples_x, device=rect.device
    )
    y_coords = torch.linspace(
        topleft_y, bottomright_y, num_samples_y, device=rect.device
    )

    # Create a meshgrid of x and y coordinates
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")

    # Flatten the grids and stack them to create the final tensor of shape (N, 2)
    sampled_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    return sampled_points


def sample_subrange(N, idx, L):
    start = idx - L // 2
    end = start + L

    # Adjust start and end to ensure they are within bounds and cover L frames
    if start < 0:
        end -= start  # Increase end by the negative amount of start
        start = 0
    if end > N:
        start -= end - N  # Decrease start to adjust for end overshoot
        end = N
        if start < 0:  # In case the initial adjustment made start negative
            start = 0

    # Ensure the range is exactly L long
    if (end - start) < L:
        if end < N:
            end = min(N, start + L)  # Extend end if possible
        elif start > 0:
            start = max(0, end - L)  # Extend start backward if possible

    return start, end
