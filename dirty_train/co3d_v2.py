import gzip
import json
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras

from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
from torch.utils.data import Dataset
from torchvision import transforms

from util.relpose_utils.bbox import square_bbox
from util.relpose_utils.misc import get_permutations
from util.relpose_utils.normalize_cameras import first_camera_transform, normalize_cameras
from datasets.dataset_util import *

from multiprocessing import Pool
import tqdm
from util.camera_transform import adjust_camera_to_bbox_crop_, adjust_camera_to_image_scale_, bbox_xyxy_to_xywh
from pytorch3d.renderer.utils import ndc_grid_sample

from pytorch3d.implicitron.dataset.utils import load_pointcloud, load_depth, load_depth_mask
import matplotlib.pyplot as plt


TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = ["ball", "book", "couch", "frisbee", "hotdog", "kite", "remote", "sandwich", "skateboard", "suitcase"]

DEBUG_CATEGORIES = [
    "teddybear",
    # "hydrant",
]

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Co3dDataset(Dataset):
    def __init__(
        self,
        category=("all",),
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        # jitter_scale=[0.8, 1.2],
        # jitter_trans=[-0.07, 0.07],
        min_num_images=50,
        img_size=224,
        eval_time=False,
        normalize_cameras=False,
        first_camera_transform=True,
        first_camera_rotation_only=False,
        mask_images=False,
        CO3D_DIR=None,
        CO3D_ANNOTATION_DIR=None,
        foreground_crop=True,
        preload_image=False,
        center_box=True,
        sort_by_filename=False,
        compute_optical=False,
        normalize_T=False,
        color_aug=True,
        erase_aug=False,
        load_point=False,
        load_track=False,
        max_3dpoints=40960,
        close_box_aug=False,
        cfg=None,
    ):
        """
        Args:
            category (iterable): List of categories to use. If "all" is in the list,
                all training categories are used.
            num_images (int): Default number of images in each batch.
            normalize_cameras (bool): If True, normalizes cameras so that the
                intersection of the optical axes is placed at the origin and the norm
                of the first camera translation is 1.
            first_camera_transform (bool): If True, tranforms the cameras such that
                camera 1 has extrinsics [I | 0].
            first_camera_rotation_only (bool): If True, transforms the cameras such that
                camera 1 has identity rotation.
            mask_images (bool): If True, masks out the background of the images.
        """

        if "seen" in category:
            category = TRAINING_CATEGORIES

        if "unseen" in category:
            category = TEST_CATEGORIES

        if "all" in category:
            category = TRAINING_CATEGORIES + TEST_CATEGORIES

        if "debug" in category:
            category = DEBUG_CATEGORIES

        if cfg.debug:
            category = DEBUG_CATEGORIES

        category = sorted(category)

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"

        self.filter_sequence = ["613_98137_195475"]
        self.low_quality_translations = []
        self.wholedata = {}
        self.category_map = {}
        if CO3D_DIR == None:
            raise NotImplementedError

        print(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR
        self.center_box = center_box
        self.crop_longest = True

        self.split_name = split_name
        self.min_num_images = min_num_images
        self.foreground_crop = foreground_crop
        self.viz = None

        for c in category:
            annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")

            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            counter = 0
            for seq_name, seq_data in annotation.items():
                counter += 1
                if len(seq_data) < min_num_images:
                    continue

                if seq_name in self.filter_sequence:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False

                for data in seq_data:
                    # Make sure translations are not ridiculous
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    # import pdb;pdb.set_trace()
                    # Ignore all unnecessary information.
                    filtered_data.append(data)

                    # filtered_data.append(
                    #     {
                    #         "filepath": data["filepath"],
                    #         "bbox": data["bbox"],
                    #         "R": data["R"],
                    #         "T": data["T"],
                    #         "focal_length": data["focal_length"],
                    #         "principal_point": data["principal_point"],
                    #     },
                    # )

                if not bad_seq:
                    self.wholedata[seq_name] = filtered_data

            print(annotation_file)
            print(counter)

        self.sequence_list = list(self.wholedata.keys())
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
        self.close_box_aug = close_box_aug
        self.seqlen = cfg.seqlen
        self.cfg = cfg

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

        print(f"Low quality translation sequences, not used: {self.low_quality_translations}")
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
            index = idx_N
            sequence_name = self.sequence_list[index]
            metadata = self.wholedata[sequence_name]
            ids = np.random.choice(len(metadata), 30, replace=False)
            return self.get_data(index=index, ids=ids)
        else:
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

        category = self.category_map[sequence_name]

        if no_images:
            annos = [metadata[i] for i in ids]
            rotations = [torch.tensor(anno["R"]) for anno in annos]
            translations = [torch.tensor(anno["T"]) for anno in annos]
            batch = {}
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)
            return batch

        annos = [metadata[i] for i in ids]

        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images = []
        rotations = []
        translations = []
        focal_lengths = []
        principal_points = []
        original_sizes = []

        depths = []
        image_paths = []

        for anno in annos:
            filepath = anno["filepath"]

            image_path = osp.join(self.CO3D_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            if self.load_track:
                depth_filepath = filepath.replace("images", "depths") + ".geometric.png"
                depth_map = load_depth(osp.join(self.CO3D_DIR, depth_filepath), anno["depth_scale_adjustment"])
                depths.append(torch.tensor(depth_map))
                # depth_mask_filepath = filepath.replace("images", "depth_masks").replace("jpg", "png")
                # depth_mask = load_depth_mask(osp.join(self.CO3D_DIR, depth_mask_filepath))
                # depth_masks.append(torch.tensor(depth_mask))

            if self.mask_images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                mask_name = osp.basename(filepath.replace(".jpg", ".png"))

                mask_path = osp.join(self.CO3D_DIR, category, sequence_name, "masks", mask_name)
                mask = Image.open(mask_path).convert("L")

                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)

            # filepaths.append(filepath)
            images.append(image)
            rotations.append(torch.tensor(anno["R"]))
            translations.append(torch.tensor(anno["T"]))
            focal_lengths.append(torch.tensor(anno["focal_length"]))
            principal_points.append(torch.tensor(anno["principal_point"]))
            image_paths.append(image_path)
            original_sizes.append(np.array(image.size[::-1]))

        if self.load_point or self.load_track:
            img_file_path = annos[0]["filepath"]
            point_file_path = osp.dirname(img_file_path).replace("images", "pointcloud.ply")
            points = load_pointcloud(osp.join(self.CO3D_DIR, point_file_path), max_points=self.max_3dpoints)
            points = points.points_list()[0]

            if len(points) < self.max_3dpoints:
                avai_point_num = len(points)
                # Calculate the number of repetitions needed to reach max_3dpoints
                repeats = self.max_3dpoints // avai_point_num
                remainder = self.max_3dpoints % avai_point_num

                # Repeat the tensor and truncate or pad as necessary
                points = torch.cat([points] * repeats + [points[:remainder]], dim=0)
        else:
            points = None

        if self.load_track:
            # compute depth visibility from original-size image&depth

            if False:
                depth_visibility, points = rawcamera_to_track(
                    focal_lengths,
                    principal_points,
                    rotations,
                    translations,
                    original_sizes,
                    depths,
                    sequence_name,
                    cfg=self.cfg,
                    in_ndc=True,
                    per_scene=True,
                )
            else:
                rawcameras = PerspectiveCameras(
                    focal_length=torch.stack(focal_lengths),
                    principal_point=torch.stack(principal_points),
                    R=torch.stack(rotations),
                    T=torch.stack(translations),
                )

                # cameras.get_image_size
                xyz_cam = rawcameras.get_world_to_view_transform().transform_points(points, eps=1e-6)
                # extract the depth of each point as the 3rd coord of xyz_cam
                depth = xyz_cam[:, :, 2:]
                # project the points xyz to the camera
                xy_ndc = rawcameras.transform_points(points)[:, :, :2]

                # xy_screen = rawcameras.transform_points_screen(points, image_size=(1850, 1040))[:, :, :2]

                depth_batch = torch.stack(depths)
                proj_depth = ndc_grid_sample(depth_batch, xy_ndc, mode="nearest", align_corners=False)[:, 0]

                depth_batch_median = depth_batch[depth_batch > 0].median()

                if True:
                    depth_visibility = (depth[:, :, 0] - proj_depth).abs() < (depth_batch_median * 0.01)
                else:
                    # # depth_visibility = (depth[:, :, 0] - proj_depth).abs() < (depth[:, :, 0] * 0.02)
                    depth_visibility = (depth[:, :, 0] - proj_depth).abs() < (proj_depth * 0.01)

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

            # crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            # cc = (2 * crop_center / crop_dim) - 1
            # ndcfl = 2 * (bbox_jitter[2] - bbox_jitter[0]) / crop_dim

            bbox_after = bbox_jitter / crop_dim * self.img_size
            # crop_parameters.append(torch.tensor([-cc[0], -cc[1], ndcfl, w/self.img_size, h/self.img_size, bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())
            crop_parameters.append(torch.tensor([bbox_after[0], bbox_after[1], bbox_after[2], bbox_after[3]]).float())

        images = images_transformed

        new_fls = torch.stack(new_fls)
        new_pps = torch.stack(new_pps)

        cameras = PerspectiveCameras(
            focal_length=new_fls.numpy(),
            principal_point=new_pps.numpy(),
            R=[data["R"] for data in annos],
            T=[data["T"] for data in annos],
        )

        if self.load_track:
            new_image_size = torch.Size([self.img_size, self.img_size])

            new_xy_ndc = cameras.transform_points(points)[:, :, :2]
            # new_xy_screen: torch.Size([32, 20480, 2])
            new_xy_screen = cameras.transform_points_screen(points, image_size=new_image_size)[:, :, :2]

            # new_xy_screen

            # depth_batch
            ################################################################################################
            # cameras.principal_point *= 0
            # new_xy_screen = cameras.transform_points_screen(points, image_size=new_image_size)[:, :, :2]

            # points_clone = points.clone()   #
            # points_clone[:, :2] *= -1
            # cameras_clone = cameras.clone()
            # cameras_clone.R[:, :, :2] *= -1
            # cameras_clone.T[:, :2] *= -1
            # new_xy_screen_clone = cameras_clone.transform_points_screen(points_clone, image_size=new_image_size)[:, :, :2]
            # new_xy_screen_not_transed = cameras_clone.transform_points_screen(points, image_size=new_image_size)[:, :, :2]

            # points_2 = points.clone()
            # points_2[:,0]  *= -1
            # tttt = cameras_clone.transform_points_screen(points_2, image_size=new_image_size)[:, :, :2]

            # # cameras_clone.transform_points(points_clone)[:, :, :2]
            # import pdb;pdb.set_trace()
            ################################################################################################

            # ndc_to_screen

            inside_flag = (new_xy_ndc < 1) & (new_xy_ndc > -1)

            inside_flag = inside_flag.sum(dim=-1) > 1
            # visibility: torch.Size([32, 20480])
            visibility = depth_visibility & inside_flag
            # visibility_num: torch.Size([20480])
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


            # batch["tracks_ndc"] = new_xy_ndc[:, indices]

            # if self.cfg.dummy:
            #     batch["tracks"] = new_xy_screen
            #     batch["tracks_visibility"] = visibility
            # else:
            #     batch["tracks"] = selected_new_xy_screen
            #     batch["tracks_visibility"] = selected_visibility

            visual_track = False
            if visual_track:
                # valid_mask
                debugidx = 245
                # debugidx = valid_mask.sum(dim=0).argmax()
                from io import BytesIO

                xy_screen_toshow = new_xy_screen[:, debugidx]
                valid_mask_show = visibility[:, debugidx]
                iii = 0

                # for img, point in zip(images, xy_screen_toshow):
                #     # Get the x, y coordinates from the tensor
                #     x, y = point[0].item(), point[1].item()

                #     plt.imshow(img)
                #     if valid_mask_show[iii]:
                #         plt.scatter(x, y, c='green', s=10)
                #     else:
                #         plt.scatter(x, y, c='red', s=10)

                #     # plt.savefig(f"debug_womask/{iii}.png")
                #     plt.savefig(f"after/{iii}.png")
                #     plt.close()
                #     iii+=1

                list_of_arrays = []
                to_tensor_transform = transforms.Compose([transforms.ToTensor()])

                for img, point in zip(images, xy_screen_toshow):
                    # Get the x, y coordinates from the tensor
                    x, y = point[0].item(), point[1].item()

                    fig, ax = plt.subplots(figsize=(2.24, 2.24))
                    ax.imshow(img)
                    plt.axis("off")

                    if x < 0 or y < 0 or x > self.img_size or y > self.img_size:
                        print("WARNING! Outside Image now")
                        ax.scatter(x, y, c="blue", s=50)
                    else:
                        if valid_mask_show[iii]:
                            ax.scatter(x, y, c="green", s=50)
                        else:
                            ax.scatter(x, y, c="red", s=50)

                    # Save the figure to a BytesIO object
                    buf = BytesIO()
                    plt.savefig(buf, format="png", dpi=100)
                    buf.seek(0)

                    # Create a PIL image from the BytesIO stream
                    pil_image = Image.open(buf)
                    new_size = (self.img_size, self.img_size)
                    pil_image = pil_image.resize(new_size, Image.ANTIALIAS)

                    # plt.imshow(pil_image)
                    # plt.savefig(f"debug.png")
                    list_of_arrays.append(to_tensor_transform(pil_image))

                    plt.close()

                    # Convert PIL image to NumPy array
                    # numpy_image = np.array(pil_image)
                    # list_of_arrays.append(numpy_image)

                    # plt.close(fig)
                    iii += 1

                toshowimg = torch.stack(list_of_arrays)
                print(sequence_name)
                # self.viz.images(toshowimg.clamp(0,1), env=f"track_{sequence_name}", win="imgs")
                self.viz.images(toshowimg.clamp(0, 1), env=f"track_debug", win="imgs")

                import pdb

                pdb.set_trace()
                return torch.zeros(100)

        if self.normalize_cameras:
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

        batch = {"seq_name": sequence_name, "frame_num": len(metadata)}

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



        # if self.color_aug and (not self.eval_time):
        #     if self.erase_aug:
        #         images = self.rand_erase(images)

        #     images = self.color_jitter(images)

        if self.load_track:
            batch["tracks"] = selected_new_xy_screen
            batch["tracks_visibility"] = selected_visibility

        batch["image"] = images.clamp(0, 1)

        if self.cfg.train.load_camera:
            batch["R"] = norm_cameras.R
            batch["T"] = norm_cameras.T

            batch["fl"] = norm_cameras.focal_length
            batch["pp"] = norm_cameras.principal_point
            batch["crop_params"] = torch.stack(crop_parameters)

        # if points is not None:
        #     batch["points"] = points

        if return_path:
            return batch, image_paths

        return batch


class Co3dDatasetFix(Co3dDataset):
    def __init__(self, *args, **kwargs):
        # use trainsetscale to control the size of co3d during training
        # because its sequence length is much higher than others
        self.trainsetscale = 5

        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.sequence_list_len // self.trainsetscale

    def __getitem__(self, idx_N):
        # index = idx_N
        start = idx_N * self.trainsetscale
        # end = (idx_N + 1) * self.trainsetscale - 1  # Subtract 1 because the end limit is exclusive
        end = min((idx_N + 1) * self.trainsetscale - 1, self.sequence_list_len - 1)

        index = random.randint(start, end)  # Get a random integer from the range

        sequence_name = self.sequence_list[index]
        metadata = self.wholedata[sequence_name]

        # loose_co3d
        if self.cfg.loose_co3d:
            expand_range = int(self.seqlen * 2)
            start_idx = np.random.choice(len(metadata), self.seqlen, replace=False)[0]
            valid_range = np.arange(max(0, start_idx - expand_range), min(len(metadata), start_idx + expand_range))
            ids = np.random.choice(valid_range, self.seqlen, replace=False)
        else:
            ids = np.random.choice(len(metadata), self.seqlen, replace=False)
        return self.get_data(index=index, ids=ids)