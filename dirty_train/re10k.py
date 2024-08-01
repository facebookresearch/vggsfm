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
import pickle
from util.relpose_utils.bbox import square_bbox
from util.relpose_utils.misc import get_permutations
from util.relpose_utils.normalize_cameras import first_camera_transform, normalize_cameras, oneT_normalize_cameras
import h5py
from io import BytesIO

# from pytorch3d.ops import sample_farthest_points

from multiprocessing import Pool
import tqdm
from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh
from pytorch3d.renderer.utils import ndc_grid_sample

from pytorch3d.implicitron.dataset.utils import load_pointcloud, load_depth, load_depth_mask
import matplotlib.pyplot as plt
from datasets.dataset_util import *
from util.metric import closed_form_inverse
import glob

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Re10KDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        # jitter_scale=[0.8, 1.0],
        # jitter_trans=[-0.07, 0.07],
        min_num_images=50,
        img_size=224,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=True,
        first_camera_rotation_only=False,
        mask_images=False,
        Re10K_DIR=None,
        preload_image=False,
        center_box=True,
        sort_by_filename=False,
        compute_optical=False,
        normalize_T=False,
        color_aug=True,
        erase_aug=False,
        load_point=False,
        load_track=False,
        max_3dpoints=10240,
        close_box_aug=False,
        scene_info_path="/data/home/jianyuan/Re10K_anno/RealEstate10K",
        cfg=None,
    ):
        self.cfg = cfg
        self.root_dir = Re10K_DIR

        if Re10K_DIR == None:
            raise NotImplementedError

        if split == "train":
            self.train_dir = os.path.join(self.root_dir, "frames/train")
            video_loc = os.path.join(self.root_dir, "frames/train/video_loc.txt")
            scenes = np.loadtxt(video_loc, dtype=np.str_)
            self.scene_info_dir = os.path.join(scene_info_path, "train")

            if cfg.debug:
                scenes = scenes[:50]
                
            self.scenes = scenes
        elif split == "test":
            self.train_dir = os.path.join(self.root_dir, "all_frames/test")
            scenes = glob.glob(self.train_dir+"/*")

            if cfg.debug:
                scenes = scenes[:50]
                
            self.scenes = scenes
            self.scene_info_dir = os.path.join(scene_info_path, "test")
            # import pdb;pdb.set_trace()
            
        else:
            raise ValueError("only implemneted training at this stage")

        print(f"Re10K_DIR is {Re10K_DIR}")

        self.Re10K_DIR = Re10K_DIR

        self.center_box = center_box
        self.crop_longest = cfg.crop_longest
        self.min_num_images = min_num_images

        if cfg.debug:
            from pytorch3d.implicitron.tools import model_io, vis_utils

            self.viz = vis_utils.get_visdom_connection(
                server="http://10.201.5.215", port=int(os.environ.get("VISDOM_PORT", 10088))
            )
        else:
            self.viz = None

        self.build_dataset(split = split)


        for key in list(self.wholedata.keys()):
            frame_num = len(self.wholedata[key])
            if frame_num < self.min_num_images:
                del self.wholedata[key]

        
        self.valid_scene = False

        self.sequence_list = sorted(list(self.wholedata.keys()))
        self.sequence_list_len = len(self.sequence_list)

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
        self.overlap_thres = cfg.overlap_thres
        self.to_filter_list = [1, 2, 3, 4, 6, 7, 8]

        self.close_box_aug = close_box_aug
        self.seqlen = cfg.seqlen

        assert not self.load_track

        if self.color_aug:
            self.color_jitter = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1)], p=0.75
                    ),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05),
                ]
            )

        if self.erase_aug:
            self.rand_erase = transforms.RandomErasing(
                p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0, inplace=False
            )

        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def build_dataset(self, split):
        self.wholedata = {}

        if self.cfg.inverse_RT:
            # False here
            pkl_path = "/data/home/jianyuan/Re10K_anno/processed_inv.pkl"
        else:
            pkl_path = "/data/home/jianyuan/Re10K_anno/processed.pkl"

        if os.path.exists(pkl_path) and split=="train":
            with open(pkl_path, "rb") as file:
                self.wholedata = pickle.load(file)
        else:
            for scene in self.scenes:
                print(scene)
                scene_name = "re10k" + scene

                scene_info_name = os.path.join(self.scene_info_dir, os.path.basename(scene) + ".txt")
                scene_info = np.loadtxt(scene_info_name, delimiter=" ", dtype=np.float64, skiprows=1)

                filtered_data = []

                for frame_idx in range(len(scene_info)):
                    try:
                        raw_line = scene_info[frame_idx]
                        timestamp = raw_line[0]
                        intrinsics = raw_line[1:7]
                        extrinsics = raw_line[7:]

                        imgpath = os.path.join(self.train_dir, scene, "%s" % int(timestamp) + ".png")
                        image_size = Image.open(imgpath).size
                        Posemat = extrinsics.reshape(3, 4).astype("float64")
                        focal_length = intrinsics[:2] * image_size
                        principal_point = intrinsics[2:4] * image_size
                        # focal_length = torch.from_numpy(intrinsics[:2]* image_size)
                        # principal_point = torch.from_numpy(intrinsics[2:4]* image_size),
                        data = {
                            "filepath": imgpath,
                            # "depth_filepath": None,
                            "R": Posemat[:3, :3],
                            "T": Posemat[:3, -1],
                            "focal_length": focal_length,
                            "principal_point": principal_point,
                        }

                        filtered_data.append(data)
                    except:
                        print("one image is missing")

                if len(filtered_data) > self.min_num_images:
                    self.wholedata[scene_name] = filtered_data
                else:
                    print(f"scene {scene_name} does not have enough image nums")

            print("finished")

    def _jitter_bbox(self, bbox):
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
            image_crop = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
            )

        return image_crop

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.wholedata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_valid_ids(self, ids, sequence_name, full_seq_len=None):
        whole_num = len(ids)
        start_idx = ids[0]

        import pdb

        pdb.set_trace()

        overlap = self.overlapmat[sequence_name]
        scalemat = self.scaleratiomat[sequence_name]  # Note: 'scalemat' is not used in the provided code

        valid_indexes = np.where(overlap[start_idx] > self.overlap_thres)[0]

        if len(valid_indexes) < whole_num:
            valid_indexes = np.where(overlap[start_idx] > self.overlap_thres / 2)[0]

        if len(valid_indexes) >= whole_num:
            ids = np.random.choice(valid_indexes, size=whole_num, replace=False)
        else:
            sorted_indexes = np.argsort(-overlap[start_idx])  # sorts in descending order
            ids = sorted_indexes[:whole_num]  # takes the first 'whole_num' indices

        return ids

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]

        metadata = self.wholedata[sequence_name]

        if self.valid_scene:
            ids = self.get_valid_ids(ids, sequence_name, full_seq_len=len(metadata))

        assert len(np.unique(ids)) == len(ids)

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

            image_path = osp.join(self.Re10K_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            # if self.load_track:
            #     depth_filepath = image_path.replace("Undistorted_SfM", "depth_undistorted").replace("/images","")
            #     depth_filepath = os.path.splitext(depth_filepath)[0] + ".h5"

            #     with h5py.File(depth_filepath, "r") as f:
            #         depth_map = f["/depth"].__array__().astype(np.float32, copy=False)
            #     depths.append(torch.tensor(depth_map))

            #     assert depth_map.shape[::-1] == image.size

            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            original_sizes.append(np.array(image.size[::-1]))

            ############################### for pytorch 3D
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

            ###############################

        # batch = {
        #     "seq_id": sequence_name,
        #     "n": len(metadata),
        #     "ind": torch.tensor(ids),
        # }

        # if self.load_track:
        #     visibility, xyz_unproj_world = rawcamera_to_track(focal_lengths_raw, principal_points_raw, rotations,
        #                                                         translations, original_sizes, depths, sequence_name, cfg=self.cfg, per_scene=self.cfg.mega_persceneD)

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

            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)

            ### resizing images
            new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
                focal_length_cropped,
                principal_point_cropped,
                torch.FloatTensor(image.size),
                torch.FloatTensor([self.img_size, self.img_size]),
            )

            images_transformed.append(self.transform(image))

            new_fls.append(new_focal_length)
            new_pps.append(new_principal_point)

            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            cc = (2 * crop_center / crop_dim) - 1
            ndcfl = 2 * (bbox_jitter[2] - bbox_jitter[0]) / crop_dim

            bbox_after = bbox_jitter / crop_dim * self.img_size
            # crop_parameters.append(torch.tensor([-cc[0], -cc[1], ndcfl, w/self.img_size, h/self.img_size, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

            crop_parameters.append(torch.tensor([bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

        ################################################################
        images = images_transformed

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([torch.tensor(data["R"][None]) for data in annos])
        batchT = torch.cat([torch.tensor(data["T"][None]) for data in annos])

        # From COLMAP to PT3D
        batchR = batchR.clone().permute(0, 2, 1)
        batchR[:, :, :2] *= -1
        batchT[:, :2] *= -1

        cameras = PerspectiveCameras(
            focal_length=new_fls.float(), principal_point=new_pps.float(), R=batchR.float(), T=batchT.float()
        )

        # if self.cfg.inverse_RT:
        #     rawse3 = cameras.get_world_to_view_transform().get_matrix()
        #     invse3 = closed_form_inverse(rawse3)
        #     cameras = PerspectiveCameras(focal_length=new_fls.float(),
        #                                 principal_point=new_pps.float(),
        #                                 R=invse3[:,:3,:3].clone(),
        #                                 T=invse3[:,3,:3].clone(),)

        # if self.load_track:
        #     new_image_size = torch.Size([self.img_size, self.img_size])
        #     new_xy_ndc = cameras.transform_points(xyz_unproj_world)[:, :, :2]

        #     new_xy_screen = cameras.transform_points_screen(xyz_unproj_world, image_size=new_image_size)[:, :, :2]

        #     inside_flag = (new_xy_ndc <= 1) & (new_xy_ndc >= -1)

        #     inside_flag = inside_flag.sum(dim=-1) > 1
        #     visibility = visibility & inside_flag
        #     visibility_num = visibility.sum(dim=0)

        #     selected_new_xy_screen, selected_visibility, indices = select_top_and_sample(
        #         new_xy_screen, visibility, visibility_num, top_percent=0.5, sample_count=self.cfg.train.track_num, name=sequence_name
        #     )

        if self.load_point:
            points = xyz_unproj_world[indices]
        else:
            points = None

        # datait
        # dataitem = SfMData()

        if self.normalize_cameras:
            ################################################################################################################
            norm_cameras, points = normalize_cameras(
                cameras,
                points=points,
                compute_optical=self.compute_optical,
                first_camera=self.first_camera_transform,
                normalize_T=self.normalize_T,
                max_norm=self.cfg.max_normT,
            )

            if norm_cameras == -1:
                print("Error in normalizing cameras: camera scale was 0")
                raise RuntimeError
        else:
            raise NotImplementedError("please normalize cameras")

        crop_params = torch.stack(crop_parameters)

        # batch = SfMData(seq_name = sequence_name, frame_num = len(metadata), frame_idx = torch.tensor(ids),
        #                 rot = norm_cameras.R, trans = norm_cameras.T,
        #                 fl = norm_cameras.focal_length, pp = norm_cameras.principal_point, crop_params = crop_params)

        batch = {
            "seq_name": sequence_name,
            "frame_num": len(metadata),
            # "ind": torch.tensor(ids),
        }

        # batch = SfMData(seq_name = sequence_name, frame_num = len(metadata), )

        # Add images
        if self.transform is not None:
            images = torch.stack(images)

        if self.color_aug and (not self.eval_time):
            for augidx in range(len(images)):
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
                        # if self.load_track:
                        #     cur_track = selected_new_xy_screen[augidx]

                        #     valid_track_by_erase = ( (cur_track[:,1] < ex) | (cur_track[:,1] > (ex + eh)) | (cur_track[:,0] < ey) | (cur_track[:,0] > (ey + ew)) )
                        #     invalid_track_by_erase = ~valid_track_by_erase
                        #     if invalid_track_by_erase.sum()>0:
                        #         selected_visibility[augidx][invalid_track_by_erase] = False

                images[augidx] = self.color_jitter(images[augidx])

        if False:
            print("hhhhhhhhhhh")
            debugidx = 32
            from pytorch3d.structures import Pointclouds
            from pytorch3d.vis.plotly_vis import plot_scene

            pcl = Pointclouds(points=points[None])
            combined_dict = {"scenes": {"points": pcl, "cameras": norm_cameras}}
            fig = plot_scene(combined_dict)
            self.viz.plotlyplot(fig, env=f"haha", win="cams")
            self.viz.images(images.clamp(0, 1), env=f"haha", win="imgs")
            import pdb

            pdb.set_trace()

        # if self.load_track:
        #     batch["tracks"] = selected_new_xy_screen
        #     batch["tracks_visibility"] = selected_visibility

        # batch.tracks = selected_new_xy_screen
        # batch.visibility = selected_visibility

        batch["image"] = images.clamp(0, 1)

        # batch.frames = images

        # if points is not None:
        # batch.points = points

        if self.cfg.train.load_camera:
            batch["R"] = norm_cameras.R
            batch["T"] = norm_cameras.T

            batch["fl"] = norm_cameras.focal_length
            batch["pp"] = norm_cameras.principal_point
            batch["crop_params"] = torch.stack(crop_parameters)

        if return_path:
            return batch, image_paths

        return batch


class Re10KDatasetFix(Re10KDataset):
    def __init__(self, *args, **kwargs):
        # use trainsetscale to control the size of co3d during training
        # because its sequence length is much higher than others
        self.trainsetscale = 1

        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.sequence_list_len // self.trainsetscale

    def __getitem__(self, idx_N):
        # index = idx_N
        # start = idx_N * self.trainsetscale
        # # end = (idx_N + 1) * self.trainsetscale - 1  # Subtract 1 because the end limit is exclusive
        # end = min((idx_N + 1) * self.trainsetscale - 1, self.sequence_list_len - 1)

        # index = random.randint(start, end)  # Get a random integer from the range
        index = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.wholedata[sequence_name]
        ids = np.random.choice(len(metadata), self.seqlen, replace=False)
        return self.get_data(index=index, ids=ids)
