import gzip
import json
import os.path as osp
import random
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional, Tuple, Union
from torchvision.transforms.functional import _get_perspective_coeffs

from util.relpose_utils.bbox import square_bbox
from util.relpose_utils.misc import get_permutations
from util.relpose_utils.normalize_cameras import first_camera_transform, normalize_cameras, oneT_normalize_cameras
import h5py
from io import BytesIO

# from pytorch3d.ops import sample_farthest_points
from torch import Tensor

from multiprocessing import Pool
import tqdm
from util.camera_transform import (
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
)  # , adjust_camera_to_bbox_crop_np, adjust_camera_to_image_scale_np
from pytorch3d.renderer.utils import ndc_grid_sample


import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Optional
from dataclasses import asdict
import torchvision.transforms.functional as F
import math

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def batch_perspective_images_and_points(images, points_batch, distortion_scale=0.5, p=0.15):
    """perspective a batch of images and points by random angles within angle_range."""
    perspectived_images = []
    perspectived_points_list = []

    # import pdb;pdb.set_trace()

    for i in range(images.size(0)):
        if random.random() < p:
            perspectived_image, perspectived_points = perspective_image_and_points(
                images[i], points_batch[i], distortion_scale
            )
        else:
            perspectived_image = images[i]
            perspectived_points = points_batch[i]
        perspectived_images.append(perspectived_image.unsqueeze(0))  # Add batch dimension
        perspectived_points_list.append(perspectived_points.unsqueeze(0))  # Add batch dimension

    # Concatenate results to get back into Bx3xHxW and BxNx2 formats
    perspectived_images_batch = torch.cat(perspectived_images, dim=0)
    perspectived_points_batch = torch.cat(perspectived_points_list, dim=0)

    # NOTE It assumes all the images have the same shape here
    h, w = images.shape[2], images.shape[3]  # Use the common height and width
    inside_image_flag = (
        (perspectived_points_batch[..., 0] >= 0)
        & (perspectived_points_batch[..., 0] < w)
        & (perspectived_points_batch[..., 1] >= 0)
        & (perspectived_points_batch[..., 1] < h)
    )

    return perspectived_images_batch, perspectived_points_batch, inside_image_flag


def perspective_image_and_points(image, points, distortion_scale=0.3):
    oh, ow = image.shape[-2:]  # Corrected line
    dtype = points.dtype
    device = points.device
    startpoints, endpoints = transforms.RandomPerspective(distortion_scale=distortion_scale).get_params(
        ow, oh, distortion_scale
    )
    transformed_img = F.perspective(image, startpoints, endpoints, interpolation=transforms.InterpolationMode.BILINEAR)

    coeffs = _get_perspective_coeffs(startpoints, endpoints)

    transformation_matrix = torch.tensor(
        [[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]], [coeffs[6], coeffs[7], 1]],
        dtype=dtype,
        device=device,
    )

    transformation_matrix = torch.inverse(transformation_matrix)

    ones = torch.ones(points.shape[0], 1, dtype=dtype, device=device)
    points_homogeneous = torch.cat([points, ones], dim=1)
    transformed_points_homogeneous = torch.mm(points_homogeneous, transformation_matrix.t())
    transformed_points = transformed_points_homogeneous[:, :2] / (
        transformed_points_homogeneous[:, 2].unsqueeze(1) + 1e-8
    )

    return transformed_img, transformed_points


#################################################################################################


def batch_rotate_images_and_points(images, points_batch, angle_range=30, p=0.15):
    """Rotate a batch of images and points by random angles within angle_range."""
    angle_range = (-angle_range, angle_range)
    
    rotated_images = []
    rotated_points_list = []

    for i in range(images.size(0)):
        if random.random() < p:
            rotated_image, rotated_points = rotate_image_and_points(images[i], points_batch[i], angle_range)
        else:
            rotated_image = images[i]
            rotated_points = points_batch[i]
        rotated_images.append(rotated_image.unsqueeze(0))  # Add batch dimension
        rotated_points_list.append(rotated_points.unsqueeze(0))  # Add batch dimension

    # Concatenate results to get back into Bx3xHxW and BxNx2 formats
    rotated_images_batch = torch.cat(rotated_images, dim=0)
    rotated_points_batch = torch.cat(rotated_points_list, dim=0)

    # NOTE It assumes all the images have the same shape here
    h, w = images.shape[2], images.shape[3]  # Use the common height and width
    inside_image_flag = (
        (rotated_points_batch[..., 0] >= 0)
        & (rotated_points_batch[..., 0] < w)
        & (rotated_points_batch[..., 1] >= 0)
        & (rotated_points_batch[..., 1] < h)
    )

    return rotated_images_batch, rotated_points_batch, inside_image_flag


def rotate_image_and_points(image, points, angle_range=(-60, 60), p=0.2):
    """Rotate an image and points by a random angle within angle_range."""
    # Generate a random angle for rotation
    angle = torch.FloatTensor(1).uniform_(*angle_range).item()
    # Rotate image
    rotated_image = F.rotate(image, angle)

    # Calculate rotation matrix
    angle_rad = math.radians(angle)
    rotation_matrix = torch.tensor(
        [[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]
    )

    # Adjust points to have the center of image as origin
    h, w = image.shape[1:]  # Assuming image shape is CxHxW
    center = torch.tensor([w / 2, h / 2])
    points_centered = points - center

    # Rotate points
    # rotated_points = torch.mm(points_centered, rotation_matrix.t()) + center

    rotated_points = torch.mm(points_centered, rotation_matrix) + center

    return rotated_image, rotated_points


@dataclass(eq=False)
class SfMData:
    """
    Dataclass for storing video tracks data.
    """

    frames: Optional[torch.Tensor] = None  # B, N, C, H, W
    rot: Optional[torch.Tensor] = None
    trans: Optional[torch.Tensor] = None
    fl: Optional[torch.Tensor] = None
    pp: Optional[torch.Tensor] = None
    tracks: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    visibility: Optional[torch.Tensor] = None
    seq_name: Optional[str] = None
    frame_num: Optional[int] = None
    frame_idx: Optional[torch.Tensor] = None
    crop_params: Optional[torch.Tensor] = None


# def collate_fn_sfm(batch):
#     frames = torch.stack([b.frames for b in batch if b.frames is not None], dim=0) if any(b.frames is not None for b in batch) else None
#     rot = torch.stack([b.rot for b in batch if b.rot is not None], dim=0) if any(b.rot is not None for b in batch) else None
#     trans = torch.stack([b.trans for b in batch if b.trans is not None], dim=0) if any(b.trans is not None for b in batch) else None
#     fl = torch.stack([b.fl for b in batch if b.fl is not None], dim=0) if any(b.fl is not None for b in batch) else None
#     pp = torch.stack([b.pp for b in batch if b.pp is not None], dim=0) if any(b.pp is not None for b in batch) else None
#     tracks = torch.stack([b.tracks for b in batch if b.tracks is not None], dim=0) if any(b.tracks is not None for b in batch) else None
#     points = torch.stack([b.points for b in batch if b.points is not None], dim=0) if any(b.points is not None for b in batch) else None
#     visibility = torch.stack([b.visibility for b in batch if b.visibility is not None], dim=0) if any(b.visibility is not None for b in batch) else None
#     seq_name = [b.seq_name for b in batch if b.seq_name is not None]
#     frame_num = [b.frame_num for b in batch if b.frame_num is not None]
#     frame_idx = torch.stack([b.frame_idx for b in batch if b.frame_idx is not None], dim=0) if any(b.frame_idx is not None for b in batch) else None
#     crop_params = torch.stack([b.crop_params for b in batch if b.crop_params is not None], dim=0) if any(b.crop_params is not None for b in batch) else None

#     # Handling lists or single values for non-tensor fields
#     seq_name = seq_name[0] if len(seq_name) == 1 else seq_name
#     frame_num = frame_num[0] if len(frame_num) == 1 else frame_num

#     return SfMData(
#         frames=frames,
#         rot=rot,
#         trans=trans,
#         fl=fl,
#         pp=pp,
#         tracks=tracks,
#         points=points,
#         visibility=visibility,
#         seq_name=seq_name,
#         frame_num=frame_num,
#         frame_idx=frame_idx,
#         crop_params=crop_params,
#     )


def collate_fn_sfm(batch):
    """
    Collate function for SfMData during training.
    """
    sfm_data_list = batch

    # Convert the first SfMData instance to a dictionary to get the keys
    keys = asdict(sfm_data_list[0]).keys()

    # Initialize an empty dictionary to store the stacked tensors
    stacked_tensors = {}

    # Iterate over the keys and stack the tensors
    for key in keys:
        # Check if any attribute is None for the given key
        if any(getattr(b, key) is None for b in sfm_data_list):
            # Handle the None case here (skip, provide a default value, etc.)
            # For this example, we'll simply skip this key
            # print(f"Skipping {key} due to None values")
            continue

        attr = getattr(sfm_data_list[0], key)
        if isinstance(attr, torch.Tensor):
            # Stack tensor attributes
            try:
                stacked_tensors[key] = torch.stack([getattr(b, key) for b in sfm_data_list], dim=0)
            except:
                for b in sfm_data_list:
                    print(b["seq_name"])
                # print(batch["seq_name"])
        elif isinstance(attr, str):
            # Handle string attributes
            stacked_tensors[key] = [getattr(b, key) for b in sfm_data_list]
        else:
            pass

    # Convert the dictionary back to an SfMData instance
    sfm_data_stacked = SfMData(**stacked_tensors)

    return sfm_data_stacked


def select_top_and_sample(points_tensor, visibility, visibility_num, top_percent, sample_count, name=None):
    """
    Select the top `top_percent` of points based on `visibility_num` and then randomly sample `sample_count` points.

    Args:
    - points_tensor (torch.Tensor): The tensor containing the points. Shape: [batch_size, total_points, dimensions]
    - visibility_num (torch.Tensor): The tensor containing the visibility numbers. Shape: [total_points]
    - top_percent (float): The percentage of top points to consider (between 0 and 1).
    - sample_count (int): The number of points to randomly sample from the top points.

    Returns:
    - selected_points (torch.Tensor): The selected points. Shape: [batch_size, sample_count, dimensions]
    - selected_visibility (torch.Tensor): The selected visibility numbers. Shape: [sample_count]
    """
    # Calculate the number of samples corresponding to the top percentage
    top_count = int(top_percent * visibility_num.shape[0])
    # Get the top `top_count` indices based on visibility_num
    top_indices = torch.topk(visibility_num, top_count, largest=True)[1]

    filtered_top_indices = top_indices[visibility[0, top_indices]]

    if len(filtered_top_indices) == 0:
        # print(f"{name} fails")

        # Step 1: Filter all points based on visibility at the first frame
        all_visible_indices = torch.nonzero(visibility[0, :], as_tuple=True)[0]

        if len(all_visible_indices) == 0:
            # print(f"{name} STILL fails")
            top_indices = top_indices
        else:
            top_indices = all_visible_indices
    else:
        top_indices = filtered_top_indices

    # Check if there are enough unique indices to meet sample_count
    if len(top_indices) == 0:
        # for _ in range(100):
        print(f"Scene {name} so wrong")
        print(points_tensor.shape)
        print(visibility.shape)
            
        return None, None, None
        top_indices = torch.arange(top_count)
    elif top_indices.shape[0] < sample_count:
        avai_point_num = len(top_indices)
        repeats = sample_count // avai_point_num
        remainder = sample_count % avai_point_num

        top_indices = torch.cat([top_indices] * repeats + [top_indices[:remainder]], dim=0)

    
    # Randomly permute these top indices
    random_permuted_indices = torch.randperm(top_indices.shape[0])

    # Select `sample_count` indices randomly from the top indices
    random_sample_indices = top_indices[random_permuted_indices[:sample_count]]

    # Select the corresponding points_tensor entries
    selected_points = points_tensor[:, random_sample_indices, :]

    # Get the corresponding visibility_num values
    # selected_visibility = visibility_num[random_sample_indices]

    selected_visibility = visibility[:, random_sample_indices]

    return selected_points, selected_visibility, random_sample_indices


def stratum_sample(first_depth_tensor, ratio, first_depth, grid_size):
    H, W = first_depth.shape[0:2]
    stratum_height, stratum_width = H // grid_size, W // grid_size

    sampled_indices = []

    for i in range(grid_size):
        for j in range(grid_size):
            y_min, y_max = i * stratum_height, (i + 1) * stratum_height
            x_min, x_max = j * stratum_width, (j + 1) * stratum_width

            # Get the indices of points in the current stratum
            stratum_indices = (
                (first_depth_tensor[:, 1] >= y_min)
                & (first_depth_tensor[:, 1] < y_max)
                & (first_depth_tensor[:, 0] >= x_min)
                & (first_depth_tensor[:, 0] < x_max)
            )

            stratum_points_indices = torch.where(stratum_indices)[0]

            # Randomly sample a fixed percentage of points from the current stratum
            num_sampled_points = int(len(stratum_points_indices) * ratio)
            sampled_stratum_indices = stratum_points_indices[
                torch.randperm(len(stratum_points_indices))[:num_sampled_points]
            ]

            sampled_indices.append(sampled_stratum_indices)

    # Concatenate the list of sampled indices into a single tensor and use it to index first_depth_tensor
    sampled_indices_tensor = torch.cat(sampled_indices)
    sampled_tensor = first_depth_tensor[sampled_indices_tensor]

    return sampled_tensor


def sanitize_coordinates(coordinates, crop_dims):
    # Ensure coordinates are within the valid range
    coordinates = coordinates.round().long()
    return coordinates


def safe_indexing(cur_depth, cur_screen):
    # Get the dimensions of cur_depth
    crop_dims = torch.tensor(cur_depth.shape, dtype=torch.float32)
    # Split cur_screen into its two coordinate components
    y_coords, x_coords = cur_screen[:, 1], cur_screen[:, 0]
    # Sanitize the coordinates
    y_coords = sanitize_coordinates(y_coords, crop_dims[0])
    x_coords = sanitize_coordinates(x_coords, crop_dims[1])

    # Create a mask of valid indices
    valid_mask = (y_coords >= 0) & (y_coords < crop_dims[0]) & (x_coords >= 0) & (x_coords < crop_dims[1])

    # Prepare the result tensor, initialize with inf
    result = torch.full_like(y_coords, float("inf"), dtype=torch.float32)

    # Use the mask to index into cur_depth or set values to inf
    result[valid_mask] = cur_depth[y_coords[valid_mask], x_coords[valid_mask]]

    return result, valid_mask


def rawcamera_to_track(
    focal_lengths_raw,
    principal_points_raw,
    rotations,
    translations,
    original_sizes,
    depths,
    sequence_name,
    maxdepth=80,
    cfg=None,
    in_ndc=False,
    depth_thres=None,
    per_scene=False,
    return_track=False,
    images=None,
    depth_vis_thres = 0.05,
):
    
    original_sizes = np.array(original_sizes)

    rawcameras = PerspectiveCameras(
        focal_length=torch.stack(focal_lengths_raw).float(),
        principal_point=torch.stack(principal_points_raw).float(),
        R=torch.stack(rotations).float(),
        T=torch.stack(translations).float(),
        image_size=original_sizes,
        in_ndc=in_ndc,
    )

    # left and right mat
    rawcameras.R = rawcameras.R.clone().permute(0, 2, 1)

    # first_depth = depths[0], to make size happy
    first_depth = depths[0].squeeze()

    valid_mask = (first_depth != 0) & (first_depth < maxdepth)
    non_zero_indices = torch.nonzero(valid_mask, as_tuple=False)
    y_coords, x_coords = non_zero_indices[:, 0], non_zero_indices[:, 1]
    non_zero_values = first_depth[y_coords, x_coords]
    first_depth_tensor = torch.stack((x_coords, y_coords, non_zero_values), dim=1)

    rgb = None
    if images is not None:
        sampled_xyd = first_depth_tensor
        transform_to_tensor = transforms.ToTensor()
        rgb = transform_to_tensor(images[0])
        rgb = rgb[:, valid_mask]
    else:
        sampled_xyd = stratum_sample(
            first_depth_tensor, 0.1, first_depth, 10
        )  # first_depth_tensor, ratio, first_depth, grid_size

    if in_ndc:
        original_sizes_t = torch.from_numpy(np.array(original_sizes))
        scale = original_sizes_t.min(dim=-1)[0]

        sampled_xyd_ndc = sampled_xyd.clone()
        sampled_xyd_ndc[:, 0] = -(sampled_xyd_ndc[:, 0] - original_sizes_t[0, 1] / 2) * 2.0 / scale[0]
        sampled_xyd_ndc[:, 1] = -(sampled_xyd_ndc[:, 1] - original_sizes_t[0, 0] / 2) * 2.0 / scale[0]

        xyz_unproj_world = rawcameras[0].unproject_points(sampled_xyd_ndc, world_coordinates=True)
        proj_depth = rawcameras.get_world_to_view_transform().transform_points(xyz_unproj_world, eps=1e-6)[:, :, 2:]
        xy_ndc = rawcameras.transform_points(xyz_unproj_world)[:, :, :2]
        xy_screen = xy_ndc.clone()

        xy_screen[..., 0] = -xy_screen[..., 0] * scale[:, None] / 2 + original_sizes_t[..., 1:2] / 2
        xy_screen[..., 1] = -xy_screen[..., 1] * scale[:, None] / 2 + original_sizes_t[..., 0:1] / 2
    else:
        xyz_unproj_world = rawcameras[0].unproject_points(sampled_xyd, world_coordinates=True)
        proj_depth = rawcameras.get_world_to_view_transform().transform_points(xyz_unproj_world, eps=1e-6)[:, :, 2:]
        xy_screen = rawcameras.transform_points(xyz_unproj_world)[:, :, :2]

    depth_by_img = []
    inbound_mask = []
    depth_all = []

    for tmpidx in range(len(xy_screen)):
        cur_screen = xy_screen[tmpidx]
        # cur_depth = depths[tmpidx], to make size happy
        cur_depth = depths[tmpidx].squeeze()
        if per_scene:
            depth_all.append(cur_depth[cur_depth > 0])
        cur_depth_sampled, cur_depth_mask = safe_indexing(cur_depth, cur_screen)
        depth_by_img.append(cur_depth_sampled[None])
        inbound_mask.append(cur_depth_mask[None])

    depth_by_img = torch.cat(depth_by_img)

    if per_scene:
        depth_all = torch.cat(depth_all, dim=0)
        depth_visibility = (proj_depth[:, :, 0] - depth_by_img).abs() < (depth_all.median() * depth_vis_thres)
    elif depth_thres is not None:
        import pdb
        pdb.set_trace()
        depth_visibility = (proj_depth[:, :, 0] - depth_by_img).abs() < (proj_depth[:, :, 0] * depth_vis_thres)
    else:
        depth_visibility = (proj_depth[:, :, 0] - depth_by_img).abs() < (proj_depth[:, :, 0] * depth_vis_thres)

    inside_flag = torch.cat(inbound_mask)

    visibility = depth_visibility & inside_flag
    
    if return_track:
        visibility_num = visibility.sum(dim=0)
        selected_new_xy_screen, selected_visibility, indices = select_top_and_sample(
            xy_screen, visibility, visibility_num, top_percent=0.5, sample_count=cfg.train.track_num, name=sequence_name
        )
            
        return visibility, xyz_unproj_world, selected_new_xy_screen, selected_visibility, rawcameras, rgb

    return visibility, xyz_unproj_world


def visual_track_fn(debugidx, images, xy_screen, visibility, to_tensor_transform=None):
    # Set a default transform if none is provided
    if to_tensor_transform is None:
        to_tensor_transform = transforms.Compose([transforms.ToTensor()])

    xy_screen_toshow = xy_screen[:, debugidx]
    vis_toshow = visibility[:, debugidx]

    list_of_arrays = []

    psize = 20
    for iii, (img, point) in enumerate(zip(images, xy_screen_toshow)):
        # Get the x, y coordinates from the tensor
        x, y = point[0].item(), point[1].item()

        if isinstance(img, torch.Tensor):
            img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        # Get image dimensions
        img_width, img_height = img.size

        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        ax.imshow(img)
        plt.axis("off")

        if vis_toshow[iii]:
            ax.scatter(x, y, c="skyblue", s=psize)
        else:
            ax.scatter(x, y, c="red", s=psize)

        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        # Create a PIL image from the BytesIO stream
        pil_image = Image.open(buf)

        list_of_arrays.append(to_tensor_transform(pil_image))

        plt.close()
    toshowimg = torch.stack(list_of_arrays)
    return toshowimg.clamp(0, 1)
