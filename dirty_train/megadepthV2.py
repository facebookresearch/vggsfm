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

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MegaDepthDatasetV2(Dataset):
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
        MegaDepth_DIR=None,
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
        scene_info_path="/fsx-repligen/shared/datasets/megadepth/scene_info",
        cfg=None,
    ):
        self.undist_names = undist_names = os.path.join(MegaDepth_DIR, "Undistorted_SfM")
        self.cfg = cfg

        if split == "train":
            # TODO: train clean

            if cfg.mega_clean:
                # file_path = 'megadepth_train_scenes_clean.txt'

                # Open the file and read the lines
                with open("/data/home/jianyuan/vggsfm/VGGSfM/datasets/megadepth_train_scenes_clean.txt", "r") as file:
                    lines = file.readlines()

                # Strip newline characters and collect lines in a list
                formatted_lines = [line.strip() for line in lines]
                self.scenes = formatted_lines
            else:
                exclude_folders = [
                    "0000",
                    "0002",
                    "0011",
                    "0020",
                    "0033",
                    "0050",
                    "0103",
                    "0105",
                    "0143",
                    "0176",
                    "0177",
                    "0265",
                    "0366",
                    "0474",
                    "0860",
                    "4541",
                    "0024",
                    "0021",
                    "0025",
                    "1589",
                    "0019",
                    "0008",
                    "0032",
                    "0063",
                    # "0331",
                ]

                scenes = [d for d in os.listdir(undist_names) if os.path.isdir(os.path.join(undist_names, d))]
                filtered_scenes = [folder for folder in scenes if folder not in exclude_folders]
                if cfg.debug:
                    filtered_scenes = ["0286", "0023", "0102", "0240", "0027"]
                self.scenes = filtered_scenes
        elif split == "test":
            self.scenes = ["0024", "0021", "0025", "1589", "0019", "0008", "0032", "0063"]
        else:
            raise ValueError("please specify correct set")

        if MegaDepth_DIR == None:
            raise NotImplementedError

        self.scene_info_path = scene_info_path
        print(f"MegaDepth_DIR is {MegaDepth_DIR}")

        self.MegaDepth_DIR = MegaDepth_DIR

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

        self.valid_scene = True

        self.build_dataset()

        self.sequence_list = sorted(list(self.wholedata.keys()))
        self.sequence_list_num = len(self.sequence_list)

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

        if self.cfg.rot_aug:
            assert not self.cfg.enable_pose
            print("augment rotation")

        if self.cfg.pers_aug:
            assert not self.cfg.enable_pose
            print("augment perspective")

        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

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

    def build_dataset(self):
        self.overlapmat = {}
        self.scaleratiomat = {}
        # self.points3D_id_to_2D = {}
        # self.points3D_id_to_ndepth = {}
        self.wholedata = {}

        for scene in self.scenes:
            # print(scene)
            scene_name = "mega" + scene
            scene_info_path = os.path.join(self.scene_info_path, f"{scene}.npz")
            if not os.path.exists(scene_info_path):
                print(f"scene {scene} is omitted")
                continue

            scene_info = np.load(scene_info_path, allow_pickle=True)
            exist_mask = scene_info["image_paths"] != None
            true_indices = np.where(exist_mask)[0]

            self.overlapmat[scene_name] = scene_info["overlap_matrix"][np.ix_(true_indices, true_indices)]
            self.scaleratiomat[scene_name] = scene_info["scale_ratio_matrix"][np.ix_(true_indices, true_indices)]
            # self.points3D_id_to_2D[scene_name] = scene_info['points3D_id_to_2D'][exist_mask]
            # self.points3D_id_to_ndepth[scene_name] = scene_info['points3D_id_to_ndepth'][exist_mask]

            filtered_info = {
                key: scene_info[key][exist_mask] for key in ["depth_paths", "intrinsics", "poses", "image_paths"]
            }
            poses_mat = torch.from_numpy(np.stack(filtered_info["poses"], axis=0))
            R, tvec = poses_mat[:, :3, :3], poses_mat[:, :3, 3]

            intrinsics_mat = np.stack(filtered_info["intrinsics"], axis=0)
            # intrinsics_mat = torch.from_numpy(intrinsics_mat).clone()
            # assert len(intrinsics_mat) == len(poses_mat)
            # assert len(intrinsics_mat) == len(filtered_info['image_paths'])
            # assert len(intrinsics_mat) == len(tvec)

            fl = np.stack([intrinsics_mat[:, 0, 0], intrinsics_mat[:, 1, 1]], axis=1)
            pp = np.stack([intrinsics_mat[:, 0, 2], intrinsics_mat[:, 1, 2]], axis=1)

            filtered_data = [
                {
                    "filepath": filtered_info["image_paths"][idx],
                    "depth_filepath": filtered_info["depth_paths"][idx],
                    "R": R[idx],
                    "T": tvec[idx],
                    # "intri": intrinsics_mat[idx],
                    "focal_length": fl[idx],
                    "principal_point": pp[idx],
                }
                for idx in range(len(filtered_info["image_paths"]))
            ]

            if len(filtered_info["image_paths"]) > self.min_num_images:
                self.wholedata[scene_name] = filtered_data
            else:
                print(f"scene {scene_name} does not have enough image nums")

        print("finished")

    def get_valid_ids(self, ids, sequence_name):
        whole_num = len(ids)
        start_idx = ids[0]

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
            ids = self.get_valid_ids(ids, sequence_name)

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

            image_path = osp.join(self.MegaDepth_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            if self.load_track:
                depth_filepath = image_path.replace("Undistorted_SfM", "depth_undistorted").replace("/images", "")
                depth_filepath = os.path.splitext(depth_filepath)[0] + ".h5"

                with h5py.File(depth_filepath, "r") as f:
                    depth_map = f["/depth"].__array__().astype(np.float32, copy=False)
                depths.append(torch.tensor(depth_map))

                assert depth_map.shape[::-1] == image.size

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

        # if False:

        #     visibility, xyz_unproj_world, selected_new_xy_screen, selected_visibility, rawcameras, pointrgb = rawcamera_to_track(focal_lengths_raw, principal_points_raw, rotations,
        #                                                                               translations, original_sizes, depths, sequence_name, cfg=self.cfg, return_track=True, images=images)

        #     transform_to_tensor = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.CenterCrop(1200),
        #             transforms.Resize(256, antialias=True),
        #         ]
        #     )
        #     image_subset = [transform_to_tensor(image) for image in images]
        #     image_subset = torch.stack(image_subset, dim=0)
        #     image_subset=image_subset[None]
        #     env_name = f"visual_{self.cfg.exp_name}"

        #     # self.viz.images((res_video_gt[0] / 255).clamp(0, 1), env="hey", win="tmp")

        #     from pytorch3d.structures import Pointclouds
        #     from pytorch3d.vis.plotly_vis import plot_scene

        #     point_cloud = Pointclouds(points = xyz_unproj_world[None], features = pointrgb.permute(1,0)[None])

        #     pcl_dict = {"scenes": {"points": point_cloud,"camera": rawcameras,}}

        #     fig = plot_scene(pcl_dict)

        #     self.viz.plotlyplot(fig, env=f"finally", win="points")
        #     self.viz.images(image_subset[0].clamp(0,1), env=f"finally", win="imgs")
        #     import pdb;pdb.set_trace()

        if self.load_track:
            visibility, xyz_unproj_world = rawcamera_to_track(
                focal_lengths_raw,
                principal_points_raw,
                rotations,
                translations,
                original_sizes,
                depths,
                sequence_name,
                cfg=self.cfg,
                per_scene=self.cfg.mega_persceneD,
            )

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
            crop_parameters.append(torch.tensor([-cc[0], -cc[1], crop_width, s]).float())

            #############################################################################

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

            # import pdb;pdb.set_trace()
            # crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            # cc = (2 * crop_center / crop_dim) - 1
            # ndcfl = 2 * (bbox_jitter[2] - bbox_jitter[0]) / crop_dim

            # bbox_after = bbox_jitter / crop_dim * self.img_size
            # # crop_parameters.append(torch.tensor([-cc[0], -cc[1], ndcfl, w/self.img_size, h/self.img_size, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

            # crop_parameters.append(torch.tensor([bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

        ################################################################
        images = images_transformed

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        batchR = torch.cat([data["R"][None] for data in annos])
        batchT = torch.cat([data["T"][None] for data in annos])

        # From COLMAP to PT3D
        batchR = batchR.clone().permute(0, 2, 1)
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


            if selected_new_xy_screen is None:
                # selected_new_xy_screen = new_xy_screen[:, :self.cfg.train.track_num]
                # selected_visibility = visibility[:, :self.cfg.train.track_num]
                # if selected_new_xy_screen.shape[1] == 0:
                selected_new_xy_screen = torch.zeros(len(batchR), self.cfg.train.track_num, 2)
                # set as bool, all False
                selected_visibility = torch.zeros(len(batchR), self.cfg.train.track_num) >1

                

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

        batch = {"seq_name": sequence_name, "frame_num": len(metadata)}

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
                        if self.load_track:
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

                images[augidx] = self.color_jitter(images[augidx])

        if self.cfg.rot_aug:
            images, selected_new_xy_screen, inside_after_rot = batch_rotate_images_and_points(
                images, selected_new_xy_screen, angle_range=self.cfg.rot_aug_range
            )
            selected_visibility = selected_visibility & inside_after_rot

        if self.cfg.pers_aug:
            images, selected_new_xy_screen, inside_after_pers = batch_perspective_images_and_points(
                images, selected_new_xy_screen
            )
            selected_visibility = selected_visibility & inside_after_pers

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

        if self.load_track:
            if selected_new_xy_screen.shape[1] == 0:
                for _ in range(100):
                    print(sequence_name)

            batch["tracks"] = selected_new_xy_screen
            batch["tracks_visibility"] = selected_visibility

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


class MegaDepthDatasetV2Fix(MegaDepthDatasetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx_N):
        if self.cfg.inside_shuffle:
            index = np.random.randint(0, self.sequence_list_num)
        else:
            index = idx_N

        sequence_name = self.sequence_list[index]
        metadata = self.wholedata[sequence_name]
        ids = np.random.choice(len(metadata), self.seqlen, replace=False)
        return self.get_data(index=index, ids=ids)
