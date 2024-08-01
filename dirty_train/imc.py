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

from util.relpose_utils.bbox import square_bbox
from util.relpose_utils.misc import get_permutations
from util.relpose_utils.normalize_cameras import first_camera_transform, normalize_cameras, oneT_normalize_cameras
import h5py
from util.metric import closed_form_inverse

from multiprocessing import Pool
import tqdm
from util.camera_transform import (
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
    _convert_ndc_to_pixels,
    _convert_pixels_to_ndc,
)  # , adjust_camera_to_bbox_crop_np, adjust_camera_to_image_scale_np
from pytorch3d.renderer.utils import ndc_grid_sample

from pytorch3d.implicitron.dataset.utils import load_pointcloud, load_depth, load_depth_mask
import matplotlib.pyplot as plt
import glob
from .imc_helper import read_model, qvec2rotmat, parse_file_to_list, load_calib
from datasets.dataset_util import *


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class IMCDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        # jitter_scale=[0.8, 1.0],
        # jitter_trans=[-0.07, 0.07],
        # num_images=2,
        min_num_images=50,
        img_size=224,
        # random_num_images=True,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=True,
        first_camera_rotation_only=False,
        mask_images=False,
        IMC_DIR=None,
        preload_image=False,
        center_box=True,
        sort_by_filename=True,
        compute_optical=False,
        normalize_T=False,
        color_aug=True,
        erase_aug=False,
        load_point=False,
        load_track=False,
        max_3dpoints=10240,
        close_box_aug=False,
        cfg=None,
    ):
        self.cfg = cfg

        self.wholedata = {}
        if split == "train":
            raise ValueError("we don't want to train on IMC")
        elif split == "test":
            IMC_DIR = os.path.join(IMC_DIR, "test")
            """
            british_museum  florence_cathedral_side  lincoln_memorial_statue  
            milan_cathedral  mount_rushmore  piazza_san_marco  
            sagrada_familia  st_pauls_cathedral
            """
            bag_names = glob.glob(os.path.join(IMC_DIR, "*/set_100/sub_set/*.txt"))

            # Hard cases:

            # 10bag_035_piazza_san_marco
            # 10bag_047_florence_cathedral_side
            # 25bag_021_mount_rushmore
            # 25bag_023_mount_rushmore
            # 5bag_045_florence_cathedral_side
            # 5bag_088_mount_rushmore

            if False:
                bag_names = [
                    # "/fsx-repligen/shared/datasets/IMC/test/piazza_san_marco/set_100/sub_set/10bag_012.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/mount_rushmore/set_100/sub_set/25bag_021.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/mount_rushmore/set_100/sub_set/25bag_023.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/mount_rushmore/set_100/sub_set/5bag_088.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/piazza_san_marco/set_100/sub_set/10bag_035.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/piazza_san_marco/set_100/sub_set/10bag_040.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/piazza_san_marco/set_100/sub_set/5bag_038.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/florence_cathedral_side/set_100/sub_set/10bag_047.txt",
                    "/fsx-repligen/shared/datasets/IMC/test/florence_cathedral_side/set_100/sub_set/5bag_045.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/florence_cathedral_side/set_100/sub_set/25bag_012.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/st_pauls_cathedral/set_100/sub_set/25bag_012.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/london_bridge/set_100/sub_set/25bag_023.txt",
                    # "/fsx-repligen/shared/datasets/IMC/test/lincoln_memorial_statue/set_100/sub_set/5bag_087.txt",
                ]

            # bag_names = ["10bag_009_sagrada_familia", "10bag_009_sagrada_familia"]
            # bag_names = ['/fsx-repligen/shared/datasets/IMC/test/sagrada_familia/set_100/sub_set/10bag_009.txt',
            #              '/fsx-repligen/shared/datasets/IMC/test/sagrada_familia/set_100/sub_set/10bag_008.txt',
            #              '/fsx-repligen/shared/datasets/IMC/test/sagrada_familia/set_100/sub_set/10bag_007.txt',
            #              '/fsx-repligen/shared/datasets/IMC/test/sagrada_familia/set_100/sub_set/10bag_006.txt',
            #              '/fsx-repligen/shared/datasets/IMC/test/sagrada_familia/set_100/sub_set/10bag_005.txt',]
            if cfg.debug:
                # bag_names = bag_names[:10]
                m=1
            # bag_names = [name for name in bag_names if "25bag_" in name]
            # bag_names = [name for name in bag_names if "10bag_" in name]
            # bag_names = bag_names[:10]
            # print("shorter datasets")

            for bag_name in bag_names:
                # print(bag_name)
                parts = bag_name.split("/")  # Split the string into parts by '/'
                location = parts[-4]  # The location part is at index 5
                bag_info = parts[-1].split(".")[0]  # The bag info part is the last part, and remove '.txt'
                new_bag_name = f"{bag_info}_{location}"  # Format the new bag name

                imgfilenames = parse_file_to_list(bag_name, "/".join(parts[:-2]))
                filtered_data = []

                for imgname in imgfilenames:
                    calib_file = imgname.replace("images", "calibration").replace("jpg", "h5")
                    calib_file = "/".join(
                        calib_file.rsplit("/", 1)[:-1] + ["calibration_" + calib_file.rsplit("/", 1)[-1]]
                    )
                    calib_dict = load_calib([calib_file])

                    calib = calib_dict[os.path.basename(imgname).split(".")[0]]
                    intri = torch.from_numpy(np.copy(calib["K"]))

                    R = torch.from_numpy(np.copy(calib["R"]))

                    tvec = torch.from_numpy(np.copy(calib["T"]).reshape((3,)))

                    fl = torch.from_numpy(np.stack([intri[0, 0], intri[1, 1]], axis=0))
                    pp = torch.from_numpy(np.stack([intri[0, 2], intri[1, 2]], axis=0))
                    """
                    # if self.cfg.debug:
                    #     R = R[None].permute(0,2,1)
                    #     tmpcamera = PerspectiveCameras(R=R,T=tvec[None],)
                    #     rawse3 = tmpcamera.get_world_to_view_transform().get_matrix()
                    #     invse3 = closed_form_inverse(rawse3)
                    #     R = invse3[:,:3,:3]
                    #     tvec = invse3[0,3,:3]
                    #     R = R.permute(0,2,1)[0]
                    #     # import pdb;pdb.set_trace()
                    """

                    filtered_data.append(
                        {
                            "filepath": imgname,
                            "R": R,
                            "T": tvec,
                            "focal_length": fl,
                            "principal_point": pp,
                            "calib": calib,
                        }
                    )
                self.wholedata[new_bag_name] = filtered_data
        else:
            raise ValueError("please specify correct set")

        if IMC_DIR == None:
            raise NotImplementedError

        print(f"IMC_DIR is {IMC_DIR}")

        self.IMC_DIR = IMC_DIR
        self.center_box = center_box
        self.crop_longest = cfg.crop_longest

        self.min_num_images = min_num_images

        if cfg.debug:
            from pytorch3d.implicitron.tools import model_io, vis_utils

            self.viz = vis_utils.get_visdom_connection(
                server="http://10.201.30.39", port=int(os.environ.get("VISDOM_PORT", 10088))
            )
        else:
            self.viz = None

        self.sequence_list = sorted(self.wholedata.keys())

        self.split = split
        self.debug = debug
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])
        else:
            self.transform = transform

        if random_aug and not eval_time:
            self.jitter_scale = cfg.jitter_scale
            self.jitter_trans = cfg.jitter_trans
            # self.jitter_scale = jitter_scale
            # self.jitter_trans = [-0.02, 0.02]
        else:
            self.jitter_scale = [1, 1]
            self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.first_camera_rotation_only = first_camera_rotation_only
        self.mask_images = mask_images
        self.compute_optical = compute_optical
        self.normalize_T = normalize_T
        self.color_aug = color_aug
        self.erase_aug = erase_aug
        self.load_point = load_point
        self.max_3dpoints = max_3dpoints
        self.load_track = load_track

        self.close_box_aug = close_box_aug
        self.hdimg = cfg.hdimg

        if self.color_aug:
            self.color_jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1)], p=0.75
                    ),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                    # transforms.RandomApply([transforms.RandomPosterize(bits=2)], p=0.05),
                ]
            )

        if self.erase_aug:
            self.rand_erase = transforms.RandomErasing(
                p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False
            )

        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        raise ValueError("We should never jitter box in IMC")
        bbox = square_bbox(bbox.astype(np.float32))

        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

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
            raise ValueError("we should not use getitem for ")
            index, n_per_seq = idx_N
            sequence_name = self.sequence_list[index]
            metadata = self.wholedata[sequence_name]
            ids = np.random.choice(len(metadata), n_per_seq, replace=False)
            return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]

        metadata = self.wholedata[sequence_name]

        if ids is None:
            ids = np.arange(len(metadata))

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images = []
        depths = []
        image_paths = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        focal_lengths_raw = []
        principal_points_raw = []
        original_sizes = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.IMC_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            if self.load_track:
                # filepath
                depth_filepath = image_path.replace("images", "depth_maps")
                depth_filepath = os.path.splitext(depth_filepath)[0] + ".h5"

                depth_map = load_h5(depth_filepath)["depth"]
                depths.append(torch.tensor(depth_map).float())
                if depth_map.shape[::-1] != image.size:
                    image = image.resize(depth_map.shape[::-1])
                assert depth_map.shape[::-1] == image.size

            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            original_sizes.append(np.array(image.size[::-1]))

            # Raw FL PP
            focal_lengths_raw.append(torch.tensor(anno["focal_length"]))
            principal_points_raw.append(torch.tensor(anno["principal_point"]))

            # PT3D FL PP
            original_size_wh = np.array(image.size)
            scale = min(original_size_wh) / 2
            c0 = original_size_wh / 2.0
            focal_pytorch3d = anno["focal_length"] / scale
            # mirrored principal point
            p0_pytorch3d = -(anno["principal_point"] - c0) / scale
            focal_lengths.append(torch.tensor(focal_pytorch3d))
            principal_points.append(torch.tensor(p0_pytorch3d))

            image_paths.append(image_path)

        # print(sequence_name)
        # print(focal_lengths)
        # print(focal_lengths_raw)
        # print("*****************************************")
        # print(sequence_name)
        # print(f"focal_lengths_raw: {focal_lengths_raw[0]}")
        # print(f"principal_points_raw: {principal_points_raw[0]}")
        # print(f"original_sizes: {original_sizes[0]}")

        batch = {"seq_name": sequence_name, "frame_num": len(metadata)}

        if False:
            (
                visibility,
                xyz_unproj_world,
                selected_new_xy_screen,
                selected_visibility,
                rawcameras,
                pointrgb,
            ) = rawcamera_to_track(
                focal_lengths_raw,
                principal_points_raw,
                rotations,
                translations,
                original_sizes,
                depths,
                sequence_name,
                cfg=self.cfg,
                return_track=True,
                images=images,
            )

            transform_to_tensor = transforms.Compose(
                [transforms.ToTensor(), transforms.CenterCrop(1200), transforms.Resize(256, antialias=True)]
            )
            image_subset = [transform_to_tensor(image) for image in images]
            image_subset = torch.stack(image_subset, dim=0)
            image_subset = image_subset[None]
            env_name = f"visual_{self.cfg.exp_name}"

            # self.viz.images((res_video_gt[0] / 255).clamp(0, 1), env="hey", win="tmp")

            from pytorch3d.structures import Pointclouds
            from pytorch3d.vis.plotly_vis import plot_scene

            point_cloud = Pointclouds(points=xyz_unproj_world[None], features=pointrgb.permute(1, 0)[None])

            pcl_dict = {"scenes": {"points": point_cloud, "camera": rawcameras}}

            fig = plot_scene(pcl_dict)

            self.viz.plotlyplot(fig, env=f"finally", win="points")
            self.viz.images(image_subset[0].clamp(0, 1), env=f"finally", win="imgs")
            import pdb

            pdb.set_trace()

        if self.load_track:
            # visibility, xyz_unproj_world = rawcamera_to_track(focal_lengths_raw, principal_points_raw, rotations,
            #                                                                           translations, original_sizes, depths, sequence_name, images=images, cfg=self.cfg)
            visibility, xyz_unproj_world = rawcamera_to_track(
                focal_lengths_raw,
                principal_points_raw,
                rotations,
                translations,
                original_sizes,
                depths,
                sequence_name,
                cfg=self.cfg,
            )

        crop_parameters = []
        images_transformed = []
        images_hd = []
        new_fls = []
        new_pps = []

        for i, (anno, image) in enumerate(zip(annos, images)):
            w, h = image.width, image.height

            if self.crop_longest:
                crop_dim = max(h, w)
                top = (h - crop_dim) // 2
                left = (w - crop_dim) // 2
                bbox = np.array([left, top, left + crop_dim, top + crop_dim])
            elif self.center_box:
                crop_dim = min(h, w)
                top = (h - crop_dim) // 2
                left = (w - crop_dim) // 2
                bbox = np.array([left, top, left + crop_dim, top + crop_dim])
            else:
                bbox = np.array(anno["bbox"])

            if self.eval_time or self.close_box_aug:
                bbox_jitter = bbox
            else:
                bbox_jitter = self._jitter_bbox(bbox)

            bbox_xywh = torch.FloatTensor(bbox_xyxy_to_xywh(bbox_jitter))

            ### cropping images
            focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(
                focal_lengths[i], principal_points[i], torch.FloatTensor(image.size), bbox_xywh
            )

            # focal_length_px, principal_point_px = _convert_ndc_to_pixels(focal_lengths[i], principal_points[i], torch.FloatTensor(image.size))

            # principal_point_px_cropped = principal_point_px - bbox_xywh[:2]

            # focal_length_tmp, principal_point_cropped_tmp = _convert_pixels_to_ndc(focal_length_px, principal_point_px_cropped, bbox_xywh[2:])

            # import pdb;pdb.set_trace()

            #############################################################################

            # Crop parameters
            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            # convert crop center to correspond to a "square" image
            width, height = image.size
            length = max(width, height)
            s = length / min(width, height)
            crop_center = crop_center + (length - np.array([width, height])) / 2
            # convert to NDC
            cc = s - 2 * s * crop_center / length
            crop_width = 2 * s * (bbox_jitter[2] - bbox_jitter[0]) / length

            # crop_parameters.append(torch.tensor([-cc[0], -cc[1], crop_width, s]).float())
            bbox_after = bbox_jitter / crop_dim * self.img_size
            crop_parameters.append(
                torch.tensor(
                    [-cc[0], -cc[1], crop_width, s, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]
                ).float()
            )

            #############################################################################

            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)

            ### resizing images
            new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
                focal_length_cropped,
                principal_point_cropped,
                torch.FloatTensor(image.size),
                torch.FloatTensor([self.img_size, self.img_size]),
            )

            # focal_lengths_raw[i]
            # 1024/1063

            images_transformed.append(self.transform(image))

            new_fls.append(new_focal_length)
            new_pps.append(new_principal_point)

            # crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            # cc = (2 * crop_center / crop_dim) - 1
            # ndcfl = 2 * (bbox_jitter[2] - bbox_jitter[0]) / crop_dim

            # bbox_after = bbox_jitter / crop_dim * self.img_size
            # # crop_parameters.append(torch.tensor([-cc[0], -cc[1], ndcfl, w/self.img_size, h/self.img_size, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())
            # crop_parameters.append(torch.tensor([bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

        images = images_transformed

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([data["R"][None] for data in annos])
        batchT = torch.cat([data["T"][None] for data in annos])

        batch["rawR"] = batchR.clone()
        batch["rawT"] = batchT.clone()
        # batch["rawFL"] = torch.stack(focal_lengths_raw)
        # batch["new_fls"] = new_fls
        # batch["rawFL_adjust"] = focal_length_cropped

        # From COLMAP to PT3D
        batchR = batchR.clone().permute(0, 2, 1)
        batchT = batchT.clone()
        batchR[:, :, :2] *= -1
        batchT[:, :2] *= -1

        cameras = PerspectiveCameras(
            focal_length=new_fls.float(), principal_point=new_pps.float(), R=batchR.float(), T=batchT.float()
        )

        if self.load_track:
            new_image_size = torch.Size([self.img_size, self.img_size])
            new_xy_ndc = cameras.transform_points(xyz_unproj_world)[:, :, :2]
            new_xy_screen = cameras.transform_points_screen(xyz_unproj_world, image_size=new_image_size)[:, :, :2]

            inside_flag = (new_xy_ndc <= 1) & (new_xy_ndc >= -1)

            inside_flag = inside_flag.sum(dim=-1) > 1
            visibility = visibility & inside_flag
            visibility_num = visibility.sum(dim=0)

            selected_new_xy_screen, selected_visibility, indices = select_top_and_sample(
                new_xy_screen,
                visibility,
                visibility_num,
                top_percent=0.5,
                sample_count=self.cfg.train.track_num,
                name=sequence_name,
            )

        if self.load_point:
            points = xyz_unproj_world[indices]
        else:
            points = None

        if self.normalize_cameras:
            ################################################################################################################
            # if self.cfg.onet_norm:
            #     normalized_cameras, points = oneT_normalize_cameras(cameras, points=points)
            # else:
            normalized_cameras, points = normalize_cameras(
                cameras,
                points=points,
                compute_optical=self.compute_optical,
                first_camera=self.first_camera_transform,
                normalize_T=self.normalize_T,
                max_norm=self.cfg.max_normT,
            )

            if normalized_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                raise RuntimeError

            batch["R"] = normalized_cameras.R
            batch["T"] = normalized_cameras.T
            # batch["R_original"] = torch.stack([torch.tensor(anno["R"]) for anno in annos])
            # batch["T_original"] = torch.stack([torch.tensor(anno["T"]) for anno in annos])

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
        if self.hdimg:
            images_hd = torch.stack(images_hd)

        if not self.eval_time:
            raise ValueError("color aug should not happen for IMC")

        if self.color_aug and (not self.eval_time):
            raise ValueError("color aug should not happen for IMC")
            for augidx in range(len(images)):
                images[augidx] = self.color_jitter(images[augidx])
                if self.erase_aug:
                    if random.random() < 0.15:
                        ex, ey, eh, ew, ev = self.rand_erase.get_params(
                            images[augidx],
                            scale=self.rand_erase.scale,
                            ratio=self.rand_erase.ratio,
                            value=[self.rand_erase.value],
                        )
                        images[augidx] = transforms.functional.erase(
                            images[augidx], ex, ey, eh, ew, ev, self.rand_erase.inplace
                        )

                        # (N, 2)
                        cur_track = selected_new_xy_screen[augidx]

                        valid_track_by_erase = (
                            (cur_track[:, 1] < ex)
                            | (cur_track[:, 1] > (ex + eh))
                            | (cur_track[:, 0] < ey)
                            | (cur_track[:, 0] > (ey + ew))
                        )
                        invalid_track_by_erase = ~valid_track_by_erase
                        if invalid_track_by_erase.sum() > 0:
                            selected_visibility[augidx][invalid_track_by_erase] = False

        if self.load_track:
            batch["tracks"] = selected_new_xy_screen
            batch["tracks_visibility"] = selected_visibility

        batch["image"] = images.clamp(0, 1)

        if points is not None:
            batch["points"] = points

        if return_path:
            return batch, image_paths

        return batch


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
