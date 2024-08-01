# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

import imageio
import numpy as np

from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image
import cv2


import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional
from datasets.dataset_util import SfMData
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from datasets.dataset_util import *


class CoTrackerSfMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(CoTrackerSfMDataset, self).__init__()
        # np.random.seed(0)
        # torch.manual_seed(0)
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.use_augs = use_augs
        self.crop_size = crop_size

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    # def __getitem__(self, index):
    #     gotit = False

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        gotit = False

        if self.cfg.inside_shuffle:
            index = np.random.randint(0, len(self.seq_names))
        else:
            index = index

        sample, gotit = self.getitem_helper(index, n_per_seq)

        # sample, gotit = self.getitem_helper(index)

        if not gotit:
            for _ in range(1000):
                print("warning: KUBRIC sampling failed")

        #     raise NotImplementedError("warning: sampling failed for kubric CoTrackerDataset")

        #     # fake sample, so we can still collate
        #     sample = SfMData(
        #         frames=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1])),
        #         tracks=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
        #         visibility=torch.zeros((self.seq_len, self.traj_per_sample)),
        #     )

        return sample

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(np.random.randint(1, self.eraser_max + 1)):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max + 1)):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = scale_delta_x * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                scale_delta_y = scale_delta_y * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, self.crop_size[0] + 10, None)
            W_new = np.clip(W_new, self.crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = W_new / float(W)
            scale_y = H_new / float(H)

            ############
            haha = cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)

            # # let's use ANTIALIAS to hold the geometry structure
            # tmp = Image.fromarray(rgbs[s].astype(np.uint8)).resize((W_new, H_new), Image.ANTIALIAS)
            # tmp = np.array(tmp, dtype=np.float32)

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust the subplot grid as needed

            # # Display the original image
            # axs[0].imshow(rgbs[s].astype(np.uint8))
            # axs[0].set_title('Original Image')
            # axs[0].axis('off')  # Hide the axis for a cleaner look

            # # Display the resized image
            # axs[1].imshow(haha.astype(np.uint8))
            # axs[1].set_title('Resized Image')
            # axs[1].axis('off')  # Hide the axis for a cleaner look

            # # Display the resized image
            # axs[2].imshow(tmp.astype(np.uint8))
            # axs[2].set_title('Resized PIL Image')
            # axs[2].axis('off')  # Hide the axis for a cleaner look

            # # Save the figure to a file
            # plt.savefig('/data/home/jianyuan/vggsfm/VGGSfM/tmp/comparison_image.png', bbox_inches='tight')
            # plt.close()  # Close the figure to free up memory

            # ############
            # Should we change this to PIL?
            # import pdb;pdb.set_trace()

            rgbs_scaled.append(haha)

            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = self.crop_size[0]
            mid_x = self.crop_size[1]

        x0 = int(mid_x - self.crop_size[1] // 2)
        y0 = int(mid_y - self.crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
                offset_y = int(
                    offset_y * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == self.crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

            if W_new == self.crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        return rgbs, trajs

    def crop(self, rgbs, trajs):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else np.random.randint(0, H_new - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W_new else np.random.randint(0, W_new - self.crop_size[1])
        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs


class KubricDynamicDatasetV2(CoTrackerSfMDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
        cfg=None,
    ):
        super(KubricDynamicDatasetV2, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )

        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.seq_names = [fname for fname in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, fname))]
        self.seq_names = sorted(self.seq_names)

        # if cfg.debug:
        #     self.seq_names = self.seq_names[1:2]
        #     self.use_augs = False

        self.cfg = cfg
        # self.seq_names = self.seq_names[:20]
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    # def getitem_helper(self, index):
    def getitem_helper(self, index, seq_length):
        gotit = True
        seq_name = self.seq_names[index]

        npy_path = os.path.join(self.data_root, seq_name, seq_name + ".npy")
        rgb_path = os.path.join(self.data_root, seq_name, "frames")

        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []

        for i, img_path in enumerate(img_paths):
            rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))

        rgbs = np.stack(rgbs)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        traj_2d = annot_dict["coords"]
        visibility = annot_dict["visibility"]

        if self.cfg.train.load_camera:
            import pdb

            pdb.set_trace()

        # random crop
        
        assert seq_length <= len(rgbs)
        if seq_length < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - seq_length, 1)[0]

            if self.cfg.debug:
                start_ind = 1
                # for _ in range(10): print(start_ind)

            rgbs = rgbs[start_ind : start_ind + seq_length]
            traj_2d = traj_2d[:, start_ind : start_ind + seq_length]
            visibility = visibility[:, start_ind : start_ind + seq_length]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(np.logical_not(visibility), (1, 0))
        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(rgbs, traj_2d, visibility)
            rgbs, traj_2d = self.add_spatial_augs(rgbs, traj_2d, visibility)
        else:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)

        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[seq_length // 2]).nonzero(as_tuple=False)[:, 0]
            visibile_pts_inds = torch.cat((visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0)
        point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]

        if len(point_inds) < self.traj_per_sample:
            # gotit = False
            avai_point_num = len(point_inds)
            repeats = self.traj_per_sample // avai_point_num
            remainder = self.traj_per_sample % avai_point_num
            # Repeat the tensor and truncate or pad as necessary
            point_inds = torch.cat([point_inds] * repeats + [point_inds[:remainder]], dim=0)

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((seq_length, self.traj_per_sample))

        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        # segs = torch.ones((self.seq_len, 1, self.crop_size[0], self.crop_size[1]))
        # sample = CoTrackerData(video=rgbs, segmentation=segs, trajectory=trajs, visibility=visibles, valid=valids, seq_name=seq_name)

        rgbs = rgbs / 255.0

        if (self.crop_size[0] > rgbs.shape[-2]) or (self.crop_size[1] > rgbs.shape[-1]):
            assert self.crop_size[0] == self.crop_size[1]
            ratio = self.crop_size[0] // rgbs.shape[-1]
            trajs = trajs * ratio
            rgbs = F.interpolate(rgbs, scale_factor=ratio, mode="bilinear", align_corners=True)

        if self.cfg.rot_aug:
            rgbs, trajs, inside_after_rot = batch_rotate_images_and_points(rgbs, trajs, angle_range=self.cfg.rot_aug_range)
            visibles = visibles & inside_after_rot

        if self.cfg.pers_aug:
            rgbs, trajs, inside_after_pers = batch_perspective_images_and_points(rgbs, trajs)
            visibles = visibles & inside_after_pers

        if trajs.shape[1] == 0:
            for _ in range(100):
                print(seq_name)

        sample = {
            "seq_name": seq_name,
            "frame_num": seq_length,
            "tracks": trajs,
            "tracks_visibility": visibles,
            "image": rgbs.clamp(0, 1),
        }

        return sample, gotit

    def __len__(self):
        return len(self.seq_names)
