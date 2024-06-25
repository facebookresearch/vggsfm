# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import random

import glob
import torch
import numpy as np

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras

import pycolmap

from ..utils.tensor_to_pycolmap import pycolmap_to_batch_matrix


from .camera_transform import (
    normalize_cameras,
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
)


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DemoLoader(Dataset):
    def __init__(
        self,
        SCENE_DIR,
        transform=None,
        img_size=1024,
        eval_time=True,
        normalize_cameras=True,
        sort_by_filename=True,
        load_gt=False,
        cfg=None,
    ):
        self.cfg = cfg

        self.sequences = {}

        if SCENE_DIR == None:
            raise NotImplementedError

        print(f"SCENE_DIR is {SCENE_DIR}")

        self.SCENE_DIR = SCENE_DIR
        self.crop_longest = True
        self.load_gt = load_gt
        self.sort_by_filename = sort_by_filename

        bag_name = os.path.basename(SCENE_DIR)
        img_filenames = glob.glob(os.path.join(SCENE_DIR, "images/*"))

        if self.sort_by_filename:
            img_filenames = sorted(img_filenames)

        filtered_data = []

        if self.load_gt:
            """
            We assume the ground truth cameras exist in the format of colmap
            """
            reconstruction = pycolmap.Reconstruction(os.path.join(SCENE_DIR, "sparse", "0"))

            calib_dict = {}
            for image_id, image in reconstruction.images.items():
                extrinsic = reconstruction.images[image_id].cam_from_world.matrix
                camera_id = image.camera_id
                intrinsic = reconstruction.cameras[camera_id].calibration_matrix()

                R = torch.from_numpy(extrinsic[:, :3])
                T = torch.from_numpy(extrinsic[:, 3])
                fl = torch.from_numpy(intrinsic[[0, 1], [0, 1]])
                pp = torch.from_numpy(intrinsic[[0, 1], [2, 2]])

                calib_dict[image.name] = {"R": R, "T": T, "focal_length": fl, "principal_point": pp}

        for img_name in img_filenames:
            frame_dict = {}
            frame_dict["filepath"] = img_name

            if self.load_gt:
                anno_dict = calib_dict[os.path.basename(img_name)]
                frame_dict.update(anno_dict)

            filtered_data.append(frame_dict)
        self.sequences[bag_name] = filtered_data

        self.sequence_list = sorted(self.sequences.keys())

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])
        else:
            self.transform = transform

        self.jitter_scale = [1, 1]
        self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time

        self.normalize_cameras = normalize_cameras

        print(f"Data size of Sequence: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
            )
        return image_crop

    def __getitem__(self, idx_N):
        if self.eval_time:
            return self.get_data(index=idx_N, ids=None)
        else:
            raise NotImplementedError("Do not train on Sequence.")

    def get_data(self, index=None, sequence_name=None, ids=None, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]

        metadata = self.sequences[sequence_name]

        if ids is None:
            ids = np.arange(len(metadata))

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images = []
        image_paths = []

        if self.load_gt:
            rotations = []
            translations = []
            focal_lengths = []
            principal_points = []

        for anno in annos:
            image_path = anno["filepath"]

            image = Image.open(image_path).convert("RGB")

            images.append(image)
            image_paths.append(image_path)

            if self.load_gt:
                rotations.append(anno["R"])
                translations.append(anno["T"])

                # focal length and principal point
                # from OPENCV to PT3D
                original_size_wh = np.array(image.size)
                scale = min(original_size_wh) / 2
                c0 = original_size_wh / 2.0
                focal_pytorch3d = anno["focal_length"] / scale

                # mirrored principal point
                p0_pytorch3d = -(anno["principal_point"] - c0) / scale
                focal_lengths.append(focal_pytorch3d)
                principal_points.append(p0_pytorch3d)

        batch = {"seq_name": sequence_name, "frame_num": len(metadata)}

        crop_parameters = []
        images_transformed = []

        if self.load_gt:
            new_fls = []
            new_pps = []

        for i, (anno, image) in enumerate(zip(annos, images)):
            w, h = image.width, image.height

            if self.crop_longest:
                crop_dim = max(h, w)
                top = (h - crop_dim) // 2
                left = (w - crop_dim) // 2
                bbox = np.array([left, top, left + crop_dim, top + crop_dim])
            else:
                bbox = np.array(anno["bbox"])

            crop_paras = calculate_crop_parameters(image, bbox, crop_dim, self.img_size)
            crop_parameters.append(crop_paras)

            # Crop image by bbox
            image = self._crop_image(image, bbox)

            images_transformed.append(self.transform(image))

            if self.load_gt:
                bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox))

                # Cropping images
                focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(
                    focal_lengths[i], principal_points[i], torch.FloatTensor(image.size), bbox_xywh
                )

                # Resizing images
                new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
                    focal_length_cropped,
                    principal_point_cropped,
                    torch.FloatTensor(image.size),
                    torch.FloatTensor([self.img_size, self.img_size]),
                )

                new_fls.append(new_focal_length)
                new_pps.append(new_principal_point)

        images = images_transformed

        if self.load_gt:
            new_fls = torch.stack(new_fls)
            new_pps = torch.stack(new_pps)

            batchR = torch.cat([data["R"][None] for data in annos])
            batchT = torch.cat([data["T"][None] for data in annos])

            batch["rawR"] = batchR.clone()
            batch["rawT"] = batchT.clone()

            # From OPENCV/COLMAP to PT3D
            batchR = batchR.clone().permute(0, 2, 1)
            batchT = batchT.clone()
            batchR[:, :, :2] *= -1
            batchT[:, :2] *= -1

            cameras = PerspectiveCameras(
                focal_length=new_fls.float(), principal_point=new_pps.float(), R=batchR.float(), T=batchT.float()
            )

            if self.normalize_cameras:
                # Move the cameras so that they stay in a better coordinate
                # This will not affect the evaluation result
                normalized_cameras, points = normalize_cameras(cameras, points=None)

                if normalized_cameras == -1:
                    print("Error in normalizing cameras: camera scale was 0")
                    raise RuntimeError

                batch["R"] = normalized_cameras.R
                batch["T"] = normalized_cameras.T

                batch["fl"] = normalized_cameras.focal_length
                batch["pp"] = normalized_cameras.principal_point

                if torch.any(torch.isnan(batch["T"])):
                    print(ids)
                    # print(category)
                    print(sequence_name)
                    raise RuntimeError
            else:
                batch["R"] = cameras.R
                batch["T"] = cameras.T
                batch["fl"] = cameras.focal_length
                batch["pp"] = cameras.principal_point

        batch["crop_params"] = torch.stack(crop_parameters)

        # Add images
        if self.transform is not None:
            images = torch.stack(images)

        if not self.eval_time:
            raise ValueError("color aug should not happen for Sequence")

        batch["image"] = images.clamp(0, 1)

        if return_path:
            return batch, image_paths

        return batch


def calculate_crop_parameters(image, bbox, crop_dim, img_size):
    crop_center = (bbox[:2] + bbox[2:]) / 2
    # convert crop center to correspond to a "square" image
    width, height = image.size
    length = max(width, height)
    s = length / min(width, height)
    crop_center = crop_center + (length - np.array([width, height])) / 2
    # convert to NDC
    cc = s - 2 * s * crop_center / length
    crop_width = 2 * s * (bbox[2] - bbox[0]) / length
    bbox_after = bbox / crop_dim * img_size
    crop_parameters = torch.tensor(
        [width, height, crop_width, s, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]
    ).float()
    return crop_parameters
