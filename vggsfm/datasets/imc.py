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

from minipytorch3d.cameras import PerspectiveCameras

from .camera_transform import (
    normalize_cameras,
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
)

from .imc_helper import parse_file_to_list, load_calib


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class IMCDataset(Dataset):
    def __init__(
        self,
        IMC_DIR,
        split="train",
        transform=None,
        img_size=1024,
        eval_time=True,
        normalize_cameras=True,
        sort_by_filename=True,
        cfg=None,
    ):
        self.cfg = cfg

        self.sequences = {}

        if IMC_DIR == None:
            raise NotImplementedError

        print(f"IMC_DIR is {IMC_DIR}")

        if split == "train":
            raise ValueError("We don't want to train on IMC")
        elif split == "test":
            bag_names = glob.glob(
                os.path.join(IMC_DIR, "*/set_100/sub_set/*.txt")
            )

            if cfg.imc_scene_eight:
                # In some settings, the scene london_bridge is removed from IMC
                bag_names = [
                    name for name in bag_names if "london_bridge" not in name
                ]

            for bag_name in bag_names:
                parts = bag_name.split(
                    "/"
                )  # Split the string into parts by '/'
                location = parts[-4]  # The location part is at index 5
                bag_info = parts[-1].split(".")[
                    0
                ]  # The bag info part is the last part, and remove '.txt'
                new_bag_name = (
                    f"{bag_info}_{location}"  # Format the new bag name
                )

                img_filenames = parse_file_to_list(
                    bag_name, "/".join(parts[:-2])
                )
                filtered_data = []

                for img_name in img_filenames:
                    calib_file = img_name.replace(
                        "images", "calibration"
                    ).replace("jpg", "h5")
                    calib_file = "/".join(
                        calib_file.rsplit("/", 1)[:-1]
                        + ["calibration_" + calib_file.rsplit("/", 1)[-1]]
                    )
                    calib_dict = load_calib([calib_file])

                    calib = calib_dict[os.path.basename(img_name).split(".")[0]]
                    intri = torch.from_numpy(np.copy(calib["K"]))

                    R = torch.from_numpy(np.copy(calib["R"]))

                    tvec = torch.from_numpy(np.copy(calib["T"]).reshape((3,)))

                    fl = torch.from_numpy(
                        np.stack([intri[0, 0], intri[1, 1]], axis=0)
                    )
                    pp = torch.from_numpy(
                        np.stack([intri[0, 2], intri[1, 2]], axis=0)
                    )

                    filtered_data.append(
                        {
                            "filepath": img_name,
                            "R": R,
                            "T": tvec,
                            "focal_length": fl,
                            "principal_point": pp,
                            "calib": calib,
                        }
                    )
                self.sequences[new_bag_name] = filtered_data
        else:
            raise ValueError("please specify correct set")

        self.IMC_DIR = IMC_DIR
        self.crop_longest = True

        self.sequence_list = sorted(self.sequences.keys())

        self.split = split
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size, antialias=True),
                ]
            )
        else:
            self.transform = transform

        random_aug = False  # do not use random_aug for IMC

        if random_aug and not eval_time:
            self.jitter_scale = cfg.jitter_scale
            self.jitter_trans = cfg.jitter_trans
        else:
            self.jitter_scale = [1, 1]
            self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time

        self.normalize_cameras = normalize_cameras

        print(f"Data size of IMC: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self, idx_N):
        if self.eval_time:
            return self.get_data(index=idx_N, ids=None)
        else:
            raise NotImplementedError("Do not train on IMC.")

    def get_data(
        self, index=None, sequence_name=None, ids=None, return_path=False
    ):
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
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = os.path.join(self.IMC_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            images.append(image)
            image_paths.append(image_path)
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

            if self.eval_time:
                bbox_jitter = bbox
            else:
                # No you should not go here for IMC
                # because we never use IMC for training
                bbox_jitter = self._jitter_bbox(bbox)

            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox_jitter))

            # Cropping images
            (focal_length_cropped, principal_point_cropped) = (
                adjust_camera_to_bbox_crop_(
                    focal_lengths[i],
                    principal_points[i],
                    torch.FloatTensor(image.size),
                    bbox_xywh,
                )
            )

            crop_paras = calculate_crop_parameters(
                image, bbox_jitter, crop_dim, self.img_size
            )
            crop_parameters.append(crop_paras)

            # Crop image by bbox_jitter
            image = self._crop_image(image, bbox_jitter)

            # Resizing images
            (new_focal_length, new_principal_point) = (
                adjust_camera_to_image_scale_(
                    focal_length_cropped,
                    principal_point_cropped,
                    torch.FloatTensor(image.size),
                    torch.FloatTensor([self.img_size, self.img_size]),
                )
            )

            images_transformed.append(self.transform(image))
            new_fls.append(new_focal_length)
            new_pps.append(new_principal_point)

        images = images_transformed

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
            focal_length=new_fls.float(),
            principal_point=new_pps.float(),
            R=batchR.float(),
            T=batchT.float(),
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
            raise ValueError("color aug should not happen for IMC")

        batch["image"] = images.clamp(0, 1)

        if return_path:
            return batch, image_paths

        return batch


def calculate_crop_parameters(image, bbox_jitter, crop_dim, img_size):
    crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
    # convert crop center to correspond to a "square" image
    width, height = image.size
    length = max(width, height)
    s = length / min(width, height)
    crop_center = crop_center + (length - np.array([width, height])) / 2
    # convert to NDC
    cc = s - 2 * s * crop_center / length
    crop_width = 2 * s * (bbox_jitter[2] - bbox_jitter[0]) / length
    bbox_after = bbox_jitter / crop_dim * img_size
    crop_parameters = torch.tensor(
        [
            -cc[0],
            -cc[1],
            crop_width,
            s,
            bbox_after[0],
            bbox_after[1],
            bbox_after[2],
            bbox_after[3],
        ]
    ).float()
    return crop_parameters
