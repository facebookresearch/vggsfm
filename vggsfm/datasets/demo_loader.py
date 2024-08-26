# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import torch
import pycolmap
import numpy as np


from typing import Optional

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


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DemoLoader(Dataset):
    def __init__(
        self,
        SCENE_DIR: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 1024,
        eval_time: bool = True,
        normalize_cameras: bool = True,
        sort_by_filename: bool = True,
        load_gt: bool = False,
        prefix: str = "images",
    ):
        """
        Initialize the DemoLoader dataset.

        Args:
            SCENE_DIR (str): Directory containing the scene data. Assumes the following structure:
                - images/: Contains all the images in the scene.
                - (optional) masks/: Contains masks for the images with the same name.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
            img_size (int): Size to resize images to.
            eval_time (bool): Flag to indicate if it's evaluation time.
            normalize_cameras (bool): Flag to indicate if cameras should be normalized.
            sort_by_filename (bool): Flag to indicate if images should be sorted by filename.
            load_gt (bool): Flag to indicate if ground truth data should be loaded.
                    If True, the gt is assumed to be in the sparse/0 directory, in a colmap format.
        """
        if not SCENE_DIR:
            raise ValueError("SCENE_DIR cannot be None")

        self.SCENE_DIR = SCENE_DIR
        self.crop_longest = True
        self.load_gt = load_gt
        self.sort_by_filename = sort_by_filename
        self.sequences = {}
        self.prefix = prefix

        bag_name = os.path.basename(os.path.normpath(SCENE_DIR))
        self.have_mask = os.path.exists(os.path.join(SCENE_DIR, "masks"))

        img_filenames = glob.glob(os.path.join(SCENE_DIR, f"{self.prefix}/*"))

        if self.sort_by_filename:
            img_filenames = sorted(img_filenames)

        self.sequences[bag_name] = self._load_images(img_filenames)
        self.sequence_list = sorted(self.sequences.keys())

        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size, antialias=True)]
        )

        self.jitter_scale = [1, 1]
        self.jitter_trans = [0, 0]
        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras

        print(f"Data size of Sequence: {len(self)}")

    def _load_images(self, img_filenames: list) -> list:
        """
        Load images and optionally their annotations.

        Args:
            img_filenames (list): List of image file paths.

        Returns:
            list: List of dictionaries containing image paths and annotations.
        """
        filtered_data = []
        calib_dict = self._load_calibration_data() if self.load_gt else {}

        for img_name in img_filenames:
            frame_dict = {"img_path": img_name}
            if self.load_gt:
                anno_dict = calib_dict[os.path.basename(img_name)]
                frame_dict.update(anno_dict)
            filtered_data.append(frame_dict)
        return filtered_data

    def _load_calibration_data(self) -> dict:
        """
        Load calibration data from the colmap reconstruction.

        Returns:
            dict: Dictionary containing calibration data for each image.
        """
        reconstruction = pycolmap.Reconstruction(
            os.path.join(self.SCENE_DIR, "sparse", "0")
        )
        calib_dict = {}
        for image_id, image in reconstruction.images.items():
            extrinsic = image.cam_from_world.matrix
            intrinsic = reconstruction.cameras[
                image.camera_id
            ].calibration_matrix()

            R = torch.from_numpy(extrinsic[:, :3])
            T = torch.from_numpy(extrinsic[:, 3])
            fl = torch.from_numpy(intrinsic[[0, 1], [0, 1]])
            pp = torch.from_numpy(intrinsic[[0, 1], [2, 2]])

            calib_dict[image.name] = {
                "R": R,
                "T": T,
                "focal_length": fl,
                "principal_point": pp,
            }
        return calib_dict

    def __len__(self) -> int:
        return len(self.sequence_list)

    def __getitem__(self, idx_N: int):
        """
        Get data for a specific index.

        Args:
            idx_N (int): Index of the data to retrieve.

        Returns:
            dict: Data for the specified index.
        """
        if self.eval_time:
            return self.get_data(index=idx_N, ids=None)
        else:
            raise NotImplementedError("Do not train on Sequence.")

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Optional[np.ndarray] = None,
        return_path: bool = False,
    ) -> dict:
        """
        Get data for a specific sequence or index.

        Args:
            index (Optional[int]): Index of the sequence.
            sequence_name (Optional[str]): Name of the sequence.
            ids (Optional[np.ndarray]): Array of indices to retrieve.
            return_path (bool): Flag to indicate if image paths should be returned.

        Returns:
            dict: Batch of data.
        """
        if sequence_name is None:
            sequence_name = self.sequence_list[index]

        metadata = self.sequences[sequence_name]

        if ids is None:
            ids = np.arange(len(metadata))

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["img_path"])

        images, masks, image_paths = self._load_images_and_masks(annos)
        batch = self._prepare_batch(
            sequence_name, metadata, annos, images, masks, image_paths
        )

        if return_path:
            return batch, image_paths

        return batch

    def _load_images_and_masks(self, annos: list) -> tuple:
        """
        Load images and masks from annotations.

        Args:
            annos (list): List of annotations.

        Returns:
            tuple: Tuple containing lists of images, masks, and image paths.
        """
        images, masks, image_paths = [], [], []
        for anno in annos:
            image_path = anno["img_path"]
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            image_paths.append(image_path)

            if self.have_mask:
                mask_path = image_path.replace(f"/{self.prefix}", "/masks")
                mask = Image.open(mask_path).convert("L")
                masks.append(mask)
        return images, masks, image_paths

    def _prepare_batch(
        self,
        sequence_name: str,
        metadata: list,
        annos: list,
        images: list,
        masks: list,
        image_paths: list,
    ) -> dict:
        """
        Prepare a batch of data for a given sequence.

        This function processes the provided sequence name, metadata, annotations, images, masks, and image paths
        to create a batch of data. It handles the transformation of images and masks, the adjustment of camera parameters,
        and the preparation of ground truth camera data if required.

        Args:
            sequence_name (str): Name of the sequence.
            metadata (list): List of metadata for the sequence.
            annos (list): List of annotations for the sequence.
            images (list): List of images for the sequence.
            masks (list): List of masks for the sequence.
            image_paths (list): List of image paths for the sequence.

        Returns:
            dict: Batch of data containing transformed images, masks, crop parameters, original images, and other relevant information.
        """
        batch = {"seq_name": sequence_name, "frame_num": len(metadata)}
        crop_parameters, images_transformed, masks_transformed = [], [], []
        original_images = (
            {}
        )  # Dictionary to store original images before any transformations

        if self.load_gt:
            new_fls, new_pps = [], []

        for i, (anno, image) in enumerate(zip(annos, images)):
            mask = masks[i] if self.have_mask else None

            # Store the original image in the dictionary with the basename of the image path as the key
            original_images[os.path.basename(image_paths[i])] = np.array(image)

            # Transform the image and mask, and get crop parameters and bounding box
            (image_transformed, mask_transformed, crop_paras, bbox) = (
                pad_and_resize_image(
                    image,
                    self.crop_longest,
                    self.img_size,
                    mask=mask,
                    transform=self.transform,
                )
            )
            images_transformed.append(image_transformed)
            if mask_transformed is not None:
                masks_transformed.append(mask_transformed)
            crop_parameters.append(crop_paras)

            if self.load_gt:
                bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox))
                (focal_length_cropped, principal_point_cropped) = (
                    adjust_camera_to_bbox_crop_(
                        anno["focal_length"],
                        anno["principal_point"],
                        torch.FloatTensor(image.size),
                        bbox_xywh,
                    )
                )
                (new_focal_length, new_principal_point) = (
                    adjust_camera_to_image_scale_(
                        focal_length_cropped,
                        principal_point_cropped,
                        torch.FloatTensor(image.size),
                        torch.FloatTensor([self.img_size, self.img_size]),
                    )
                )
                new_fls.append(new_focal_length)
                new_pps.append(new_principal_point)

        images = torch.stack(images_transformed)
        masks = torch.stack(masks_transformed) if self.have_mask else None

        if self.load_gt:
            batch.update(self._prepare_gt_camera_batch(annos, new_fls, new_pps))

        batch.update(
            {
                "image": images.clamp(0, 1),
                "crop_params": torch.stack(crop_parameters),
                "scene_dir": os.path.dirname(os.path.dirname(image_paths[0])),
                "masks": masks.clamp(0, 1) if self.have_mask else None,
                "original_images": original_images,  # A dict with the image path as the key and the original np image as the value
            }
        )

        return batch

    def _prepare_gt_camera_batch(
        self, annos: list, new_fls: list, new_pps: list
    ) -> dict:
        """

        Prepare a batch of ground truth camera data from annotations and adjusted camera parameters.

        This function processes the provided annotations and adjusted camera parameters (focal lengths and principal points)
        to create a batch of ground truth camera data. It also handles the conversion of camera parameters from the
        OpenCV/COLMAP format to the PyTorch3D format. If normalization is enabled, the cameras are normalized, and the
        resulting parameters are included in the batch.

        Args:
            annos (list): List of annotations, where each annotation is a dictionary containing camera parameters.
            new_fls (list): List of new focal lengths after adjustment.
            new_pps (list): List of new principal points after adjustment.

        Returns:
            dict: A dictionary containing the batch of ground truth camera data, including raw and processed camera
                  parameters such as rotation matrices (R), translation vectors (T), focal lengths (fl), and principal
                  points (pp). If normalization is enabled, the normalized camera parameters are included.
        """
        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([data["R"][None] for data in annos])
        batchT = torch.cat([data["T"][None] for data in annos])

        batch = {"rawR": batchR.clone(), "rawT": batchT.clone()}

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
            normalized_cameras, _ = normalize_cameras(cameras, points=None)
            if normalized_cameras == -1:
                raise RuntimeError(
                    "Error in normalizing cameras: camera scale was 0"
                )

            batch.update(
                {
                    "R": normalized_cameras.R,
                    "T": normalized_cameras.T,
                    "fl": normalized_cameras.focal_length,
                    "pp": normalized_cameras.principal_point,
                }
            )

            if torch.any(torch.isnan(batch["T"])):
                raise RuntimeError("NaN values found in camera translations")
        else:
            batch.update(
                {
                    "R": cameras.R,
                    "T": cameras.T,
                    "fl": cameras.focal_length,
                    "pp": cameras.principal_point,
                }
            )

        return batch


def calculate_crop_parameters(image, bbox, crop_dim, img_size):
    """
    Calculate the parameters needed to crop an image based on a bounding box.

    Args:
        image (PIL.Image.Image): The input image.
        bbox (np.array): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        crop_dim (int): The dimension to which the image will be cropped.
        img_size (int): The size to which the cropped image will be resized.

    Returns:
        torch.Tensor: A tensor containing the crop parameters, including width, height, crop width, scale, and adjusted bounding box coordinates.
    """
    crop_center = (bbox[:2] + bbox[2:]) / 2
    # convert crop center to correspond to a "square" image
    width, height = image.size
    length = max(width, height)
    s = length / min(width, height)
    crop_center = crop_center + (length - np.array([width, height])) / 2
    # convert to NDC
    # cc = s - 2 * s * crop_center / length
    crop_width = 2 * s * (bbox[2] - bbox[0]) / length
    bbox_after = bbox / crop_dim * img_size
    crop_parameters = torch.tensor(
        [
            width,
            height,
            crop_width,
            s,
            bbox_after[0],
            bbox_after[1],
            bbox_after[2],
            bbox_after[3],
        ]
    ).float()
    return crop_parameters


def pad_and_resize_image(
    image: Image.Image,
    crop_longest: bool,
    img_size,
    mask: Optional[Image.Image] = None,
    bbox_anno: Optional[np.array] = None,
    transform=None,
):
    """
    Pad (through cropping) and resize an image, optionally with a mask.

    Args:
        image (PIL.Image.Image): Image to be processed.
        crop_longest (bool): Flag to indicate if the longest side should be cropped.
        img_size (int): Size to resize the image to.
        mask (Optional[PIL.Image.Image]): Mask to be processed.
        bbox_anno (Optional[np.array]): Bounding box annotations.
        transform (Optional[transforms.Compose]): Transformations to apply.
    """
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size, antialias=True)]
        )

    w, h = image.width, image.height
    if crop_longest:
        crop_dim = max(h, w)
        top = (h - crop_dim) // 2
        left = (w - crop_dim) // 2
        bbox = np.array([left, top, left + crop_dim, top + crop_dim])
    else:
        assert bbox_anno is not None
        bbox = np.array(bbox_anno)

    crop_paras = calculate_crop_parameters(image, bbox, crop_dim, img_size)

    # Crop image by bbox
    image = _crop_image(image, bbox)
    image_transformed = transform(image).clamp(0.0, 1.0)

    if mask is not None:
        mask = _crop_image(mask, bbox)
        mask_transformed = transform(mask).clamp(0.0, 1.0)
    else:
        mask_transformed = None

    return image_transformed, mask_transformed, crop_paras, bbox


def _crop_image(image, bbox, white_bg=False):
    """
    Crop an image to a bounding box. When bbox is larger than the image, the image is padded.

    Args:
        image (PIL.Image.Image): Image to be cropped.
        bbox (np.array): Bounding box for the crop.
        white_bg (bool): Flag to indicate if the background should be white.

    Returns:
        PIL.Image.Image: Cropped image.
    """
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
