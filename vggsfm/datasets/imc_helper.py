# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Adapted from https://github.com/ubc-vision/image-matching-benchmark
# and
# https://github.com/colmap/colmap

import os
import numpy as np
import json

from copy import deepcopy
from datetime import datetime


import cv2
import h5py


import collections
import struct
import argparse


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


#################################################################


def build_composite_image(image_path1, image_path2, axis=1, margin=0, background=1):
    """
    Load two images and returns a composite image.

    Parameters
    ----------
    image_path1: Fullpath to image 1.
    image_path2: Fullpath to image 2.
    in: Space between images
    ite)

    Returns
    -------
    (Composite image,
        (vertical_offset1, vertical_offset2),
        (horizontal_offset1, horizontal_offset2))
    """

    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError("Axis must be 0 (vertical) or 1 (horizontal")

    im1 = cv2.imread(image_path1)
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    elif im1.ndim == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError("invalid image format")

    im2 = cv2.imread(image_path2)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    elif im2.ndim == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError("invalid image format")

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3), dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1 : voff1 + h1, hoff1 : hoff1 + w1, :] = im1
    composite[voff2 : voff2 + h2, hoff2 : hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def save_h5(dict_to_save, filename):
    """Saves dictionary to HDF5 file"""

    with h5py.File(filename, "w") as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def load_h5(filename):
    """Loads dictionary from hdf5 file"""

    dict_to_load = {}
    try:
        with h5py.File(filename, "r") as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except Exception as e:
        print("Following error occured when loading h5 file {}: {}".format(filename, e))
    return dict_to_load


##########################################


def load_image(image_path, use_color_image=False, input_width=512, crop_center=True, force_rgb=False):
    """
    Loads image and do preprocessing.

    Parameters
    ----------
    image_path: Fullpath to the image.
    use_color_image: Flag to read color/gray image
    input_width: Width of the image for scaling
    crop_center: Flag to crop while scaling
    force_rgb: Flag to convert color image from BGR to RGB

    Returns
    -------
    Tuple of (Color/Gray image, scale_factor)
    """

    # Assuming all images in the directory are color images
    image = cv2.imread(image_path)
    if not use_color_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif force_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop center and resize image into something reasonable
    scale_factor = 1.0
    if crop_center:
        rows, cols = image.shape[:2]
        if rows > cols:
            cut = (rows - cols) // 2
            img_cropped = image[cut : cut + cols, :]
        else:
            cut = (cols - rows) // 2
            img_cropped = image[:, cut : cut + rows]
        scale_factor = float(input_width) / float(img_cropped.shape[0])
        image = cv2.resize(img_cropped, (input_width, input_width))

    return (image, scale_factor)


def load_depth(depth_path):
    return load_h5(depth_path)["depth"]


def load_vis(vis_fullpath_list, subset_index=None):
    """
    Given fullpath_list load all visibility ranges
    """
    vis = []
    if subset_index is None:
        for vis_file in vis_fullpath_list:
            # Load visibility
            vis.append(np.loadtxt(vis_file).flatten().astype("float32"))
    else:
        for idx in subset_index:
            tmp_vis = np.loadtxt(vis_fullpath_list[idx]).flatten().astype("float32")
            tmp_vis = tmp_vis[subset_index]
            vis.append(tmp_vis)
    return vis


def load_calib(calib_fullpath_list, subset_index=None):
    """Load all calibration files and create a dictionary."""

    calib = {}
    if subset_index is None:
        for _calib_file in calib_fullpath_list:
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace("calibration_", "")
            # _calib_file.split(
            #     '/')[-1].replace('calibration_', '')[:-3]
            # # Don't know why, but rstrip .h5 also strips
            # # more then necssary sometimes!
            # #
            # # img_name = _calib_file.split(
            # #     '/')[-1].replace('calibration_', '').rstrip('.h5')
            calib[img_name] = load_h5(_calib_file)
    else:
        for idx in subset_index:
            _calib_file = calib_fullpath_list[idx]
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace("calibration_", "")
            calib[img_name] = load_h5(_calib_file)
    return calib


def load_h5_valid_image(path, deprecated_images):
    return remove_keys(load_h5(path), deprecated_images)


def remove_keys(d, key_list):
    for key in key_list:
        del_key_list = [tmp_key for tmp_key in d.keys() if key in tmp_key]
        for del_key in del_key_list:
            del d[del_key]
    return d


#####################################################################################
def get_uuid(cfg):
    return cfg.method_dict["config_common"]["json_label"].split("-")[0]


def get_eval_path(mode, cfg):
    if mode == "feature":
        return get_feature_path(cfg)
    elif mode == "match":
        return get_match_path(cfg)
    elif mode == "filter":
        return get_filter_path(cfg)
    elif mode == "model":
        return get_geom_path(cfg)
    elif mode == "stereo":
        return get_stereo_path(cfg)
    elif mode == "multiview":
        return get_multiview_path(cfg)
    else:
        raise ValueError("Unknown job type")


def get_eval_file(mode, cfg, job_id=None):
    if job_id:
        return os.path.join(get_eval_path(mode, cfg), "{}.{}".format(job_id, mode))
    else:
        try:
            file_list = os.listdir(get_eval_path(mode, cfg))
            valid_file = [file for file in file_list if file.split(".")[-1] == mode]
            if len(valid_file) == 0:
                return None
            elif len(valid_file) == 1:
                return os.path.join(get_eval_path(mode, cfg), valid_file[0])
            else:
                print("Should never be here")
                import IPython

                IPython.embed()
                return None
        except FileNotFoundError:
            os.makedirs(get_eval_path(mode, cfg))
            return None


def get_data_path(cfg):
    """Returns where the per-dataset results folder is stored.

    TODO: This probably should be done in a neater way.
    """

    # Get data directory for 'set_100'
    return os.path.join(cfg.path_data, cfg.dataset, cfg.scene, "set_{}".format(cfg.num_max_set))


def get_base_path(cfg):
    """Returns where the per-dataset results folder is stored."""

    if cfg.is_challenge:
        cur_date = "{:%Y%m%d}".format(datetime.now())
        return os.path.join(cfg.path_results, "challenge", get_uuid(cfg), cfg.dataset, cfg.scene)
    else:
        return os.path.join(cfg.path_results, cfg.dataset, cfg.scene)


def get_feature_path(cfg):
    """Returns where the keypoints and descriptor results folder is stored.

    Method names converted to lower-case."""

    common = cfg.method_dict["config_common"]
    return os.path.join(
        get_base_path(cfg),
        "{}_{}_{}".format(common["keypoint"].lower(), common["num_keypoints"], common["descriptor"].lower()),
    )


def get_kp_file(cfg):
    """Returns the path to the keypoint file."""

    return os.path.join(get_feature_path(cfg), "keypoints.h5")


def get_scale_file(cfg):
    """Returns the path to the scale file."""

    return os.path.join(get_feature_path(cfg), "scales.h5")


def get_score_file(cfg):
    """Returns the path to the score file."""

    return os.path.join(get_feature_path(cfg), "scores.h5")


def get_angle_file(cfg):
    """Returns the path to the angle file."""

    return os.path.join(get_feature_path(cfg), "angles.h5")


def get_affine_file(cfg):
    """Returns the path to the angle file."""

    return os.path.join(get_feature_path(cfg), "affine.h5")


def get_desc_file(cfg):
    """Returns the path to the descriptor file."""

    return os.path.join(get_feature_path(cfg), "descriptors.h5")


def get_match_name(cfg):
    """Return folder name for the matching model.

    Converted to lower-case to avoid conflicts."""
    cur_key = "config_{}_{}".format(cfg.dataset, cfg.task)

    # simply return 'custom_matcher' if it is provided
    if cfg.method_dict[cur_key]["use_custom_matches"]:
        return cfg.method_dict[cur_key]["custom_matches_name"]

    # consturct matcher name
    matcher = cfg.method_dict[cur_key]["matcher"]

    # Make a custom string for the matching folder
    label = []

    # Base name
    label += [matcher["method"]]

    # flann/bf
    if matcher["flann"]:
        label += ["flann"]
    else:
        label += ["bf"]

    # number of neighbours
    label += ["numnn-{}".format(matcher["num_nn"])]

    # distance
    label += ["dist-{}".format(matcher["distance"])]

    # 2-way matching
    if not matcher["symmetric"]["enabled"]:
        label += ["nosym"]
    else:
        label += ["sym-{}".format(matcher["symmetric"]["reduce"])]

    # filtering
    if matcher["filtering"]["type"] == "none":
        label += ["nofilter"]
    elif matcher["filtering"]["type"].lower() in ["snn_ratio_pairwise", "snn_ratio_vs_last"]:
        # Threshold == 1 means no ratio test
        # It just makes writing the config files easier
        if matcher["filtering"]["threshold"] == 1:
            label += ["nofilter"]
        else:
            label += ["filter-{}-{}".format(matcher["filtering"]["type"], matcher["filtering"]["threshold"])]
    elif matcher["filtering"]["type"].lower() == "fginn_ratio_pairwise":
        label += [
            "filter-fginn-pairwise-{}-{}".format(
                matcher["filtering"]["threshold"], matcher["filtering"]["fginn_radius"]
            )
        ]
    else:
        raise ValueError("Unknown filtering type")

    # distance filtering
    if "descriptor_distance_filter" in matcher:
        if "threshold" in matcher["descriptor_distance_filter"]:
            max_dist = matcher["descriptor_distance_filter"]["threshold"]
            label += ["maxdist-{:.03f}".format(max_dist)]

    return "_".join(label).lower()


def get_filter_path(cfg):
    """Returns folder location for the outlier filter results."""

    cur_key = "config_{}_{}".format(cfg.dataset, cfg.task)

    # Bypass this when using custom matches
    if cfg.method_dict[cur_key]["use_custom_matches"]:
        return os.path.join(get_match_path(cfg), "no_filter")

    # Otherwise, depends on the filter method
    outlier_filter = cfg.method_dict[cur_key]["outlier_filter"]
    if outlier_filter["method"] in ["cne-bp-nd"]:
        return os.path.join(get_match_path(cfg), outlier_filter["method"])
    elif outlier_filter["method"] == "none":
        return os.path.join(get_match_path(cfg), "no_filter")
    else:
        raise ValueError("Unknown outlier_filter type")


def get_match_path(cfg):
    """Returns where the match results folder is stored."""
    return os.path.join(get_feature_path(cfg), get_match_name(cfg))


def get_match_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_match_path(cfg), "matches.h5")


def get_filter_match_file(cfg):
    """Returns the path to the match file after pre-filtering."""

    return os.path.join(get_filter_path(cfg), "matches.h5")


def get_match_cost_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_match_path(cfg), "matching_cost.h5")


def get_geom_name(cfg):
    """Return folder name for the geometry model.

    Converted to lower-case to avoid conflicts."""

    geom = cfg.method_dict["config_{}_{}".format(cfg.dataset, cfg.task)]["geom"]
    method = geom["method"].lower()

    # This entry is a temporary fix
    if method in ["cv2-ransac-f", "cv2-usacdef-f", "cv2-usacmagsac-f", "cv2-usacfast-f", "cv2-usacaccurate-f"]:
        label = "_".join(
            [method, "th", str(geom["threshold"]), "conf", str(geom["confidence"]), "maxiter", str(geom["max_iter"])]
        )
    elif method in ["cv2-ransac-e"]:
        label = "_".join([method, "th", str(geom["threshold"]), "conf", str(geom["confidence"])])
    elif method in ["cmp-degensac-f", "cmp-degensac-f-laf", "cmp-gc-ransac-e"]:
        label = "_".join(
            [
                method,
                "th",
                str(geom["threshold"]),
                "conf",
                str(geom["confidence"]),
                "max_iter",
                str(geom["max_iter"]),
                "error",
                str(geom["error_type"]),
                "degencheck",
                str(geom["degeneracy_check"]),
            ]
        )
    elif method in ["cmp-gc-ransac-f", "skimage-ransac-f", "cmp-magsac-f"]:
        label = "_".join(
            [method, "th", str(geom["threshold"]), "conf", str(geom["confidence"]), "max_iter", str(geom["max_iter"])]
        )
    elif method in ["cv2-lmeds-e", "cv2-lmeds-f"]:
        label = "_".join([method, "conf", str(geom["confidence"])])
    elif method in ["intel-dfe-f"]:
        label = "_".join([method, "th", str(geom["threshold"]), "postprocess", str(geom["postprocess"])])
    elif method in ["cv2-7pt", "cv2-8pt"]:
        label = method
    else:
        raise ValueError("Unknown method for E/F estimation")

    return label.lower()


def get_geom_path(cfg):
    """Returns where the match results folder is stored."""

    geom_name = get_geom_name(cfg)
    return os.path.join(get_filter_path(cfg), "stereo-fold-{}".format(cfg.run), geom_name)


def get_geom_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_geom_path(cfg), "essential.h5")


def get_geom_inl_file(cfg):
    """Returns the path to the match file."""
    return os.path.join(get_geom_path(cfg), "essential_inliers.h5")


def get_geom_cost_file(cfg):
    """Returns the path to the geom cost file."""
    return os.path.join(get_geom_path(cfg), "geom_cost.h5")


def get_cne_temp_path(cfg):
    return os.path.join(get_filter_path(cfg), "temp_cne")


def get_filter_match_file_for_computing_model(cfg):
    filter_match_file = os.path.join(get_filter_path(cfg), "matches_imported_stereo_{}.h5".format(cfg.run))
    if os.path.isfile(filter_match_file):
        return filter_match_file
    else:
        return get_filter_match_file(cfg)


def get_filter_match_file(cfg):
    return os.path.join(get_filter_path(cfg), "matches_inlier.h5")


def get_filter_cost_file(cfg):
    return os.path.join(get_filter_path(cfg), "matches_inlier_cost.h5")


def get_cne_data_dump_path(cfg):
    return os.path.join(get_cne_temp_path(cfg), "data_dump")


def get_stereo_path(cfg):
    """Returns the path to where the stereo results are stored."""

    return os.path.join(get_geom_path(cfg), "set_{}".format(cfg.num_max_set))


def get_stereo_pose_file(cfg, th=None):
    """Returns the path to where the stereo pose file."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_pose_errors{}.h5".format(label))


def get_repeatability_score_file(cfg, th=None):
    """Returns the path to the repeatability file."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "repeatability_score_file{}.h5".format(label))


def get_stereo_epipolar_pre_match_file(cfg, th=None):
    """Returns the path to the match file."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_epipolar_pre_match_errors{}.h5".format(label))


def get_stereo_epipolar_refined_match_file(cfg, th=None):
    """Returns the path to the filtered match file."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_epipolar_refined_match_errors{}.h5".format(label))


def get_stereo_epipolar_final_match_file(cfg, th=None):
    """Returns the path to the match file after RANSAC."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_epipolar_final_match_errors{}.h5".format(label))


def get_stereo_depth_projection_pre_match_file(cfg, th=None):
    """Returns the path to the errors file for input matches."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_projection_errors_pre_match{}.h5".format(label))


def get_stereo_depth_projection_refined_match_file(cfg, th=None):
    """Returns the path to the errors file for filtered matches."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_projection_errors_refined_match{}.h5".format(label))


def get_stereo_depth_projection_final_match_file(cfg, th=None):
    """Returns the path to the errors file for final matches."""

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg), "stereo_projection_errors_final_match{}.h5".format(label))


def get_colmap_path(cfg):
    """Returns the path to colmap results for individual bag."""

    return os.path.join(get_multiview_path(cfg), "bag_size_{}".format(cfg.bag_size), "bag_id_{:05d}".format(cfg.bag_id))


def get_multiview_path(cfg):
    """Returns the path to multiview folder."""

    return os.path.join(get_filter_path(cfg), "multiview-fold-{}".format(cfg.run))


def get_colmap_mark_file(cfg):
    """Returns the path to colmap flag."""

    return os.path.join(get_colmap_path(cfg), "colmap_has_run")


def get_colmap_pose_file(cfg):
    """Returns the path to colmap pose files."""

    return os.path.join(get_colmap_path(cfg), "colmap_pose_errors.h5")


def get_colmap_output_path(cfg):
    """Returns the path to colmap outputs."""

    return os.path.join(get_colmap_path(cfg), "colmap")


def get_colmap_temp_path(cfg):
    """Returns the path to colmap working path."""

    # TODO: Do we want to use slurm temp directory?
    return os.path.join(get_colmap_path(cfg), "temp_colmap")


def parse_file_to_list(file_name, data_dir):
    """
    Parses filenames from the given text file using the `data_dir`

    :param file_name: File with list of file names
    :param data_dir: Full path location appended to the filename

    :return: List of full paths to the file names
    """

    fullpath_list = []
    with open(file_name, "r") as f:
        while True:
            # Read a single line
            line = f.readline()
            # Check the `line` type
            if not isinstance(line, str):
                line = line.decode("utf-8")
            if not line:
                break
            # Strip `\n` at the end and append to the `fullpath_list`
            fullpath_list.append(os.path.join(data_dir, line.rstrip("\n")))
    return fullpath_list


def get_fullpath_list(data_dir, key):
    """
    Returns the full-path lists to image info in `data_dir`

    :param data_dir: Path to the location of dataset
    :param key: Which item to retrieve from

    :return: Tuple containing fullpath lists for the key item
    """
    # Read the list of images, homography and geometry
    list_file = os.path.join(data_dir, "{}.txt".format(key))

    # Parse files to fullpath lists
    fullpath_list = parse_file_to_list(list_file, data_dir)

    return fullpath_list


def get_item_name_list(fullpath_list):
    """Returns each item name in the full path list, excluding the extension"""

    return [os.path.splitext(os.path.basename(_s))[0] for _s in fullpath_list]


def get_stereo_viz_folder(cfg):
    """Returns the path to the stereo visualizations folder."""

    base = os.path.join(cfg.method_dict["config_common"]["json_label"].lower(), cfg.dataset, cfg.scene, "stereo")

    return os.path.join(cfg.path_visualization, "png", base), os.path.join(cfg.path_visualization, "jpg", base)


def get_colmap_viz_folder(cfg):
    """Returns the path to the multiview visualizations folder."""

    base = os.path.join(cfg.method_dict["config_common"]["json_label"].lower(), cfg.dataset, cfg.scene, "multiview")

    return os.path.join(cfg.path_visualization, "png", base), os.path.join(cfg.path_visualization, "jpg", base)


def get_pairs_per_threshold(data_dir):
    pairs = {}
    for th in np.arange(0, 1, 0.1):
        pairs["{:0.1f}".format(th)] = np.load("{}/new-vis-pairs/keys-th-{:0.1f}.npy".format(data_dir, th))
    return pairs
