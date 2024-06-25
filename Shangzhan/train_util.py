import torch
import random
import numpy as np
import inspect
import logging
import os
from collections import defaultdict
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

import torch.optim
from collections import OrderedDict

from accelerate import Accelerator
import math

from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle

from pytorch3d.implicitron.tools.stats import Stats, AverageMeter
from pytorch3d.vis.plotly_vis import plot_scene

import gzip
import json
import warnings
from collections.abc import Iterable
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from pytorch3d.implicitron.tools.vis_utils import get_visdom_connection
from matplotlib import cm
from accelerate.utils import set_seed as accelerate_set_seed, PrecisionType
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
import glob
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.dataset.data_loader_map_provider import SequenceDataLoaderMapProvider
from util.track_visual import Visualizer
import torch.nn as nn

from pytorch3d.implicitron.tools.config import expand_args_fields, registry, run_auto_creation
from omegaconf import DictConfig
from datasets.co3d_v2 import Co3dDataset, TRAINING_CATEGORIES, Co3dDatasetFix
from torch.utils.data import BatchSampler
from griddle.utils import is_aws_cluster

# from datasets.megadepth import MegaDepthDataset
from datasets.megadepthV2 import MegaDepthDatasetV2, MegaDepthDatasetV2Fix
from datasets.tartanair import TartanAirDataset
from datasets.imc import IMCDataset
import kornia
from kornia.feature import *
from datasets.dataset_util import *
from torch.utils.data import ConcatDataset
from torch.utils.data import SequentialSampler
from torch.utils.data.dataset import Dataset
import psutil
import bisect

import cProfile
import pstats
import io
from .get_mix_dyna import get_mix_dataset_dyna, DynamicBatchSampler

logger = logging.getLogger(__name__)


TO_PLOT_METRICS = (
    "loss",
    "loss_re",
    "loss_point",
    "loss_track",
    "loss_pose",
    "loss_trackcolor",
    "loss_reproj",
    "loss_vis",
    "loss_tconf",
    "chamfer",
    "lr",
    "Track_m",
    "Track_mvis",
    "Track_a1",
    "Vis_acc",
    "Auc_30",
    "Auc_10",
    "Auc_5",
    "Auc_3",
    "PairAuc_10",
    "PairAuc_3",
    "Racc_5",
    "Racc_15",
    "Racc_30",
    "Tacc_5",
    "Tacc_15",
    "Tacc_30",
    "sec/it",
)


class ConcatDataset_inside_shuffle(ConcatDataset):
    def __getitem__(self, idx):
        idx = random.randint(0, self.cumulative_sizes[-1] - 1)

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def record_and_print_cpu_memory_and_usage():
    """
    Records and prints detailed CPU memory usage and CPU utilization percentage
    statistics.
    """
    # Recording CPU utilization percentage over a 1-second interval
    cpu_usage_percent = psutil.cpu_percent(interval=1)

    # Recording memory usage details
    memory_stats = psutil.virtual_memory()
    total_memory_GB = memory_stats.total / (1024**3)
    available_memory_GB = memory_stats.available / (1024**3)
    used_memory_GB = memory_stats.used / (1024**3)
    memory_utilization_percentage = memory_stats.percent

    # Printing the statistics
    print("------------------------------------------------------------------")
    print(f"CPU Utilization: {cpu_usage_percent}%")
    print(f"Total Memory: {total_memory_GB:.2f} GB")
    print(f"Available Memory: {available_memory_GB:.2f} GB")
    print(f"Used Memory: {used_memory_GB:.2f} GB")
    print(f"Memory Utilization: {memory_utilization_percentage}%")
    print("------------------------------------------------------------------")


def load_model_weights(model, weights_path, device, relax_load):
    """
    Load model weights from a .bin file onto the specified device, handling the .module prefix problem for DDP.

    Args:
    model (torch.nn.Module): The PyTorch model instance.
    weights_path (str): Path to the .bin file containing the model weights.
    is_ddp (bool): Flag indicating whether the model is being used in DDP training.
    device (str): The device to load the model on ('cuda' or 'cpu').

    Returns:
    model (torch.nn.Module): Model with loaded weights.
    """
    if not os.path.isfile(weights_path):
        raise ValueError(f"Weight file not found: {weights_path}")

    is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)

    # Load state dict from the file onto the specified device
    state_dict = torch.load(weights_path, map_location=device)

    # Handle .module prefix incompatibility for DDP models
    if is_ddp and not list(state_dict.keys())[0].startswith("module."):
        # Add 'module.' prefix to each key
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = "module." + k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    elif not is_ddp and list(state_dict.keys())[0].startswith("module."):
        # Remove 'module.' prefix from each key
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:]  # remove `module.`
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    # Load the adapted state_dict into the model

    if relax_load:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)

    # Ensure the model is on the correct device
    model = model.to(device)

    return model


def build_optimizer(cfg, model, dataloader):
    if cfg.train.warmup_sche:
        if cfg.var_lr:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.image_feature_extractor.parameters(), "lr": cfg.train.lr / 10},
                    {
                        "params": [p for name, p in model.named_parameters() if "image_feature_extractor" not in name],
                        "lr": cfg.train.lr,
                    },
                ]
            )
        else:
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)

        # warmup_ratio=0.1, warmup_lr_init
        lr_scheduler = WarmupCosineRestarts(
            optimizer=optimizer,
            T_0=cfg.train.restart_num,
            iters_per_epoch=len(dataloader),
            warmup_ratio=cfg.warmup_ratio,
            warmup_lr_init=cfg.warmup_lr_init,
        )
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0, total_iters=len(dataloader) * 100
        )

        # import pdb;pdb.set_trace()
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloader)*100, gamma=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=cfg.train.lr, epochs=cfg.train.restart_num, steps_per_epoch=len(dataloader))
    return optimizer, lr_scheduler


def build_dataset(cfg):
    if cfg.train.dataset == "mix":
        if cfg.dynamix:
            # from .get_mix_dyna import get_mix_dataset_dyna
            dataset, eval_dataset, dataloader, eval_dataloader = get_mix_dataset_dyna(cfg)
        else:
            dataset, eval_dataset, dataloader, eval_dataloader = get_mix_dataset(cfg)
    elif cfg.train.dataset == "co3d":
        dataset, eval_dataset, dataloader, eval_dataloader = get_co3d_dataset(cfg)
    elif cfg.train.dataset == "kubric":
        dataset, eval_dataset, dataloader, eval_dataloader = get_kubric_dataset(cfg)
    elif cfg.train.dataset == "megadepthV2":
        dataset, eval_dataset, dataloader, eval_dataloader = get_megadepthV2_dataset(cfg)
    elif cfg.train.dataset == "megadepth":
        dataset, eval_dataset, dataloader, eval_dataloader = get_megadepth_dataset(cfg)
    elif cfg.train.dataset == "tartan":
        dataset, eval_dataset, dataloader, eval_dataloader = get_tartan_dataset(cfg)
    elif cfg.train.dataset == "imc":
        dataset, eval_dataset, dataloader, eval_dataloader = get_imc_dataset(cfg)
    else:
        raise ValueError("Dataset Not Implemented")

    return dataset, eval_dataset, dataloader, eval_dataloader


def visualize_track(
    predictions,
    images,
    tracks,
    tracks_visibility,
    cfg,
    step,
    viz,
    n_points=1,
    selected_indices=None,
    total_points=256,
    save_dir=None,
    visual_gt = False,
):
    if "pred_tracks" in predictions:
        track_visualizer = Visualizer(save_dir=save_dir, fps=2, show_first_frame=0, linewidth=8, mode="rainbow")
        image_subset = images[0:1]

        # the selected track number
        if selected_indices is None:
            selected_indices = sorted(random.sample(range(total_points), min(n_points, total_points)))

        if cfg.debug:
            selected_indices = list(range(24))

        if visual_gt:
            tracks_subset = tracks[0:1][:, :, selected_indices]
            tracks_vis_subset = tracks_visibility[0:1][:, :, selected_indices]
            # tracks_vis_subset = tracks_vis_subset > 0.8            
        else:
            tracks_subset = predictions["pred_tracks"][-1][0:1][:, :, selected_indices]
            tracks_vis_subset = predictions["pred_vis"][0:1][:, :, selected_indices]
            tracks_vis_subset = tracks_vis_subset > 0.8

        res_video_gt = track_visualizer.visualize(
            255 * image_subset, tracks_subset, tracks_vis_subset, save_video=False
        )
        env_name = f"visual_{cfg.exp_name}"

        viz.images((res_video_gt[0] / 255).clamp(0, 1), env=env_name, win="tmp")

        # viz.images((res_video_gt[0] / 255).clamp(0, 1), env="debug", win="tmp")

        # viz.images(res_video_gt[0,1], env="tmp", win="tmp")

        # res_combined = res_video_gt
        # _, num_frames, channels, height, width = res_combined.shape
        # res_row = res_combined.squeeze(0).permute(1, 2, 0, 3).reshape(3, height, num_frames * width)
        # res_row_np = res_row.numpy()
        # res_row_np = ((res_row_np - res_row_np.min()) / (res_row_np.max() - res_row_np.min()) * 255).astype(np.uint8)
        # res_row_np = np.transpose(res_row_np, (1, 2, 0))
        # import cv2
        # cv2.imwrite('combined_frames.png', res_row_np)
        print(env_name)
        # import pdb

        # pdb.set_trace()

        return res_video_gt, save_dir

        """ TO BE CONTINUED 
        import pdb;pdb.set_trace()        
        m=1


        pred_vis_thresed = predictions["pred_vis"][0:1] > 0.5
        
        # total_points = predictions["pred_tracks"][-1].shape[2]
        pred_tracks_subset = predictions["pred_tracks"][-1][0:1][:, :, selected_indices]
        pred_sigsq_subset = predictions["sig_sq"][0:1][:, :, selected_indices]

        
        

        
        res_video_pred = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False)
        res_video_raw = track_visualizer.visualize(255 * image_subset, tracks_subset, tracks_vis_subset, save_video=False, skip_track= True)
        res_video_pixpoint = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True)
        res_video_pix = track_visualizer.visualize_withsig(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True, sigsq=pred_sigsq_subset)

        # res_video_pix = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True)


        # res_video_pred = add_colored_border(res_video_pred, 5, color=(0, 255, 255))
        # res_video_gt = add_colored_border(res_video_gt, 5, color=(255, 0, 0))
        res_combined = torch.cat([res_video_pred, res_video_gt, res_video_raw, res_video_pixpoint, res_video_pix], dim=-2)

        # env_name = f"perfect_{cfg.exp_name}_Single" if single_pt else f"perfect_{cfg.exp_name}"
        env_name = f"visual_{cfg.exp_name}"
        print(env_name)
        # 
        # viz.images((res_combined[0] / 255).clamp(0, 1), env=env_name, win="imgs")

        # track_visualizer.save_video(res_combined, filename=f"sample_{step}")

        # wide_list = list(video.unbind(1))
        # wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        # clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

        # # Write the video file
        # save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
        # clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

        

        return res_combined, save_dir
        """


def add_colored_border(video_tensor, border_width, color=(0, 0, 255)):
    """
    Adds a colored border to a video represented as a PyTorch tensor.

    Parameters:
    video_tensor (torch.Tensor): A tensor of shape [batch_size, num_frames, num_channels, height, width].
    border_width (int): The width of the border to add.
    color (tuple): RGB values of the border color as a tuple (R, G, B).

    Returns:
    torch.Tensor: A new tensor with the colored border added, shape [batch_size, num_frames, num_channels, new_height, new_width].
    """
    # Extract original dimensions
    batch_size, num_frames, num_channels, original_height, original_width = video_tensor.shape

    # Calculate new dimensions
    new_height, new_width = original_height + 2 * border_width, original_width + 2 * border_width

    # Create new tensor filled with the specified color
    new_video_tensor = torch.zeros([batch_size, num_frames, num_channels, new_height, new_width])
    for i, c in enumerate(color):
        new_video_tensor[:, :, i, :, :] = c

    # Copy original video frames into new tensor
    new_video_tensor[
        :, :, :, border_width : border_width + original_height, border_width : border_width + original_width
    ] = video_tensor

    return new_video_tensor


def gather_nodes_by_track(graph, track_labels, keypoints):
    # Create a dictionary where the keys are the track labels and the values are lists of nodes.
    track_to_nodes = defaultdict(list)
    track_to_kps = defaultdict(list)
    for node_idx, track_label in enumerate(track_labels):
        # graph.nodes[node_idx].track_idx = track_label
        # Append the node to the list of nodes for its track.
        feature_idx = graph.nodes[node_idx].feature_idx
        image_id = graph.nodes[node_idx].image_id
        image_name = graph.image_id_to_name[image_id]
        track_to_kps[track_label].append([image_id, keypoints[image_name][feature_idx]])
        track_to_nodes[track_label].append(graph.nodes[node_idx])
    return track_to_kps, track_to_nodes


def save_images_as_png(images, path_prefix=""):
    """
    Save images from a torch tensor to ordered PNG files.

    Parameters:
    - images (torch.Tensor): Tensor of images to be saved, e.g., torch.Size([10, 3, 512, 512])
    - path_prefix (str): Optional path prefix if you want to save in a specific directory or with a prefix.
    """
    # Convert tensor to PIL Image
    import torchvision.transforms as transforms
    from PIL import Image

    tensor_to_image = transforms.ToPILImage()

    for idx, img_tensor in enumerate(images):
        img = tensor_to_image(img_tensor)
        filename = f"{path_prefix}{idx:04}.png"
        img.save(filename)


def check_ni(tensor):
    return torch.isnan(tensor).any() or (~torch.isfinite(tensor)).any()


def save_visual_track(res_combined, filename):
    _, num_frames, channels, height, width = res_combined.shape
    res_row = res_combined.squeeze(0).permute(1, 2, 0, 3).reshape(channels, height, num_frames * width)
    res_row_np = res_row.numpy()
    res_row_np = np.transpose(res_row_np, (1, 2, 0))
    res_row_np_rgb = cv2.cvtColor(res_row_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, res_row_np_rgb)

    ratio = height // width
    res_row_single = res_combined.squeeze(0).permute(1, 2, 0, 3).reshape(channels, ratio, width, num_frames, width)
    res_row_single = res_row_single.numpy()

    os.makedirs(filename[:-4], exist_ok=True)

    for rrr in range(ratio):
        for nnn in range(num_frames):
            cur_to_save = res_row_single[:, rrr, :, nnn, :]
            cur_to_save = np.transpose(cur_to_save, (1, 2, 0))
            cur_to_save = cv2.cvtColor(cur_to_save, cv2.COLOR_RGB2BGR)

            cv2.imwrite(filename[:-4] + f"/row_{rrr}_num_{nnn}.png", cur_to_save)

    # res_combined
    # m=1


def prefix_with_module(checkpoint):
    prefixed_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        prefixed_key = "module." + key
        prefixed_checkpoint[prefixed_key] = value
    return prefixed_checkpoint


# def process_kubric_data(batch, device, kubric_rand = False):
#     batch, gotit = batch

#     images = batch.video.to(device)
#     tracks = batch.trajectory.to(device)
#     tracks_visibility = batch.visibility.to(device)

#     crop_params = None
#     translation = None
#     rotation = None
#     fl = pp = points = None
#     tracks_ndc = None
#     images =  (images / 255.0)

#     if kubric_rand:
#         B, S, N, _ = tracks.shape
#         rand_idx = torch.randperm(S)
#         images = images[:, rand_idx]
#         tracks = tracks[:, rand_idx]
#         tracks_visibility = tracks_visibility[:, rand_idx]

#     return images, crop_params, translation, rotation, fl, pp, points, tracks, tracks_visibility, tracks_ndc, gotit


def process_co3d_data(batch, device, cfg):
    if ("mix" in cfg.train.dataset) or ("kubric" in cfg.train.dataset):
        images = batch["image"].to(device)

        tracks = None
        tracks_visibility = None
        crop_params = None
        translation = None
        rotation = None
        fl = None
        pp = None
        points = None
        points_rgb = None
        tracks_ndc = None
        if cfg.train.load_track:
            tracks = batch["tracks"].to(device)
            tracks_visibility = batch["tracks_visibility"].to(device)

        # TODO
        if cfg.train.load_camera:
            crop_params = batch["crop_params"].to(device)
            translation = batch["T"].to(device)
            rotation = batch["R"].to(device)
            fl = batch["fl"].to(device)
            pp = batch["pp"].to(device)

    elif isinstance(batch, SfMData):
        images = batch.frames.to(device) if batch.frames is not None else None
        # images =  (images / 255.0)
        translation = batch.trans.to(device) if batch.trans is not None else None
        rotation = batch.rot.to(device) if batch.rot is not None else None
        fl = batch.fl.to(device) if batch.fl is not None else None
        pp = batch.pp.to(device) if batch.pp is not None else None
        points = batch.points.to(device) if batch.points is not None else None
        tracks = batch.tracks.to(device) if batch.tracks is not None else None
        tracks_visibility = batch.visibility.to(device) if batch.visibility is not None else None
        # Assuming that crop_params, points_rgb, and tracks_ndc are attributes of SfMData
        crop_params = batch.crop_params.to(device) if batch.crop_params is not None else None
        points_rgb = None
        tracks_ndc = None
    else:
        images = batch["image"].to(device)
        crop_params = batch["crop_params"].to(device)
        translation = batch["T"].to(device)
        rotation = batch["R"].to(device)
        fl = batch["fl"].to(device)
        pp = batch["pp"].to(device)

        if "points" in batch:
            points = batch["points"].to(device)
        else:
            points = None

        if "points_rgb" in batch:
            points_rgb = batch["points_rgb"].to(device)
        else:
            points_rgb = None

        if "tracks" in batch:
            tracks = batch["tracks"].to(device)
            tracks_visibility = batch["tracks_visibility"].to(device)
        else:
            tracks = None
            tracks_visibility = None

        if "tracks_ndc" in batch:
            tracks_ndc = batch["tracks_ndc"].to(device)
        else:
            tracks_ndc = None

    return images, crop_params, translation, rotation, fl, pp, points, points_rgb, tracks, tracks_visibility, tracks_ndc


def get_laf_mask(reshaped_image, _emb_xyscale=None):
    with torch.no_grad():
        lafs, resps, lafs_scale = get_laf(reshaped_image)
        laf_center = lafs[..., 2]

        if False:
            image_index = 0  # Can be any index from 0 to batch_size - 1

            image_to_show = reshaped_image[image_index].permute(1, 2, 0).cpu().detach().numpy()
            plt.imshow(image_to_show)

            # Extract the x, y coordinates of the keypoints
            x_coords = laf_center[image_index, :, 0].cpu().detach().numpy()
            y_coords = laf_center[image_index, :, 1].cpu().detach().numpy()

            plt.scatter(
                x_coords, y_coords, c="r", s=10
            )  # c='r' specifies the color red, and s=10 specifies the size of the scatter points
            plt.savefig("debug_laf_kp.png")

        # Get the height and width from reshaped_image
        height, width = reshaped_image.shape[2:4]

        # Round laf_center values to the nearest integers and clip them to be within the image boundaries
        laf_center_rounded = laf_center.round().long()
        laf_center_rounded[..., 0] = laf_center_rounded[..., 0].clamp(0, width - 1)
        laf_center_rounded[..., 1] = laf_center_rounded[..., 1].clamp(0, height - 1)

        _emb_xyscale = _emb_xyscale.to(lafs_scale.device)

        xyscale_emb = _emb_xyscale(torch.cat([lafs_scale[..., 0], laf_center], dim=-1))
        # Create a mask initialized with zeros

        feature_map = torch.zeros(
            reshaped_image.shape[0],
            xyscale_emb.shape[2],
            height,
            width,
            dtype=torch.float32,
            device=reshaped_image.device,
        )
        batch_indices = torch.arange(feature_map.shape[0], device=reshaped_image.device)[:, None, None]
        keypoint_indices = torch.arange(xyscale_emb.shape[1], device=reshaped_image.device)[None, :, None]
        extracted = xyscale_emb[batch_indices, keypoint_indices]
        feature_map[batch_indices[..., 0], :, laf_center_rounded[..., 1], laf_center_rounded[..., 0]] = extracted[
            :, :, 0, :
        ]

        if False:
            image_index = 0  # Can be any index from 0 to batch_size - 1

            # image_to_show = reshaped_image[image_index].permute(1, 2, 0).cpu().detach().numpy()
            # plt.imshow(image_to_show)

            sum_feature_map = feature_map.sum(dim=1)
            sum_feature_map = sum_feature_map.abs()
            plt.imshow(sum_feature_map[image_index, :, :].cpu().detach().numpy(), cmap="viridis")
            plt.savefig("debug_laf_fmap.png")

        # import pdb;pdb.set_trace()

    return feature_map


def get_laf(reshaped_image, PS=41):
    timg_gray = kornia.color.rgb_to_grayscale(reshaped_image)
    descriptor = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(reshaped_image.device)

    resp = kornia.feature.BlobDoG()
    scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

    nms = kornia.geometry.ConvQuadInterp3d(10)
    n_features = 1000
    detector = ScaleSpaceDetector(
        n_features,
        resp_module=resp,
        scale_space_response=True,
        nms_module=nms,
        scale_pyr_module=scale_pyr,
        ori_module=kornia.feature.LAFOrienter(19),
        mr_size=6.0,
        minima_are_also_good=True,
    ).to(
        reshaped_image.device
    )  # dark blobs are as good as bright.
    lafs, resps = detector(timg_gray)
    lafs_scale = get_laf_scale(lafs)
    return lafs, resps, lafs_scale


def get_co3d_dataset_test(cfg, category=None):
    # Common dataset parameters
    if category is None:
        category = cfg.test.category

    common_params = {
        "category": (category,),
        "debug": False,
        "mask_images": False,
        "img_size": cfg.test.img_size,
        "normalize_cameras": cfg.test.normalize_cameras,
        "normalize_T": cfg.test.normalize_T,
        "min_num_images": cfg.test.min_num_images,
        "CO3D_DIR": cfg.test.CO3D_DIR,
        "CO3D_ANNOTATION_DIR": cfg.test.CO3D_ANNOTATION_DIR,
        "first_camera_transform": cfg.test.first_camera_transform,
        "compute_optical": cfg.test.compute_optical,
        "sort_by_filename": True,  # to ensure images are aligned with extracted matches
        "load_point": cfg.test.load_point,
        "max_3dpoints": cfg.test.max_3dpoints,
        "load_track": cfg.test.load_track,
        "cfg": cfg,
    }

    # Create the test dataset
    test_dataset = Co3dDataset(**common_params, split="test", eval_time=True)

    return test_dataset


def get_kubric_dataset(cfg):
    from datasets.kubric_movif_SfM_dataset import KubricSfMDataset

    if cfg.more_kubric:
        data_root = os.path.join("/fsx-repligen/shared/datasets/kubric/", "kubric_combined")
    else:
        data_root = os.path.join("/fsx-repligen/shared/datasets/kubric/", "kubric_movi_f_tracks")

    train_dataset = KubricSfMDataset(
        data_root=data_root,
        crop_size=[cfg.train.img_size, cfg.train.img_size],
        seq_len=cfg.seqlen,
        traj_per_sample=cfg.train.track_num,
        sample_vis_1st_frame=True,
        use_augs=True,
        cfg=cfg,
    )

    train_dataset = ConcatDataset([train_dataset] * cfg.repeat_kub)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        # collate_fn=collate_fn_sfm,
        drop_last=True,
    )

    # eval_dataset, _, eval_dataloader = get_co3d_dataset_eval(cfg)
    eval_dataset = IMCDataset(
        split="test",
        eval_time=True,
        min_num_images=0,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        IMC_DIR="/fsx-repligen/shared/datasets/IMC",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=True,  # to compare with colmap
        cfg=cfg,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,  # Assuming you want to process one sample at a time
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    return train_dataset, eval_dataset, train_loader, eval_dataloader


def get_imc_dataset(cfg, category=None):
    eval_dataset = IMCDataset(
        split="test",
        eval_time=True,
        min_num_images=0,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        IMC_DIR="/fsx-repligen/shared/datasets/IMC",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=True,  # to compare with colmap
        cfg=cfg,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,  # Assuming you want to process one sample at a time
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    dataset = IMCDataset(
        split="train",
        min_num_images=0,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        IMC_DIR="/fsx-repligen/shared/datasets/IMC",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=True,  # to compare with colmap
        cfg=cfg,
    )

    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=cfg.train.len_train,
        max_images=cfg.train.max_images,
        images_per_seq=cfg.train.images_per_seq,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    dataloader.batch_sampler.sampler = dataloader.batch_sampler

    return dataset, eval_dataset, dataloader, eval_dataloader


def get_tartan_dataset(cfg, category=None):
    dataset = TartanAirDataset(
        split="train",
        min_num_images=cfg.train.min_num_images,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        TartanAir_DIR="/datasets01/TartanAirExtracted",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=cfg.train.sort_by_filename,
        cfg=cfg,
    )

    ####################################
    # for i in range(len(dataset)):
    #     for _ in range(3):
    #         dataset.__getitem__((i,random.randint(5, 25)))
    ####################################

    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=cfg.train.len_train,
        max_images=cfg.train.max_images,
        images_per_seq=cfg.train.images_per_seq,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    if cfg.eval_on_mega:
        eval_dataset = MegaDepthDataset(
            split="test",
            eval_time=True,
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            mask_images=False,
            MegaDepth_DIR="/fsx-repligen/shared/datasets/megadepth",
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            cfg=cfg,
        )

        eval_batch_sampler = DynamicBatchSampler(
            len(eval_dataset),
            dataset_len=cfg.train.len_eval,
            max_images=cfg.train.max_images // 2,
            images_per_seq=cfg.train.images_per_seq,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
        )

        eval_dataloader.batch_sampler.sampler = eval_dataloader.batch_sampler
    else:
        eval_dataset = IMCDataset(
            split="test",
            eval_time=True,
            min_num_images=0,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            IMC_DIR="/fsx-repligen/shared/datasets/IMC",
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=True,  # to compare with colmap
            cfg=cfg,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=1,  # Assuming you want to process one sample at a time
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
        )

    dataloader.batch_sampler.sampler = dataloader.batch_sampler

    return dataset, eval_dataset, dataloader, eval_dataloader


class CombinedDataset(Dataset):
    def __init__(self, datasets, ratios=None):
        """
        Initialize the Combined Dataset with a list of datasets and corresponding sampling ratios.
        Args:
            datasets (list of Dataset): The list of datasets to combine.
            ratios (list of int): The list of sampling ratios for each dataset. Defaults to equal ratio.
        """
        assert ratios is None or len(datasets) == len(ratios), "Ratios list must match datasets list in length."

        self.datasets = datasets
        self.ratios = ratios if ratios is not None else [1] * len(datasets)

        # Adjust dataset lengths according to ratios
        self.adjusted_lengths = [len(d) * r for d, r in zip(datasets, self.ratios)]

        # Compute cumulative lengths for indexing
        self.cumulative_lengths = np.cumsum(self.adjusted_lengths)

    def __len__(self):
        """Return the total adjusted number of items across all datasets."""
        return sum(self.adjusted_lengths)

    def __getitem__(self, idx_N):
        global_idx, n_per_seq = idx_N

        # Find the dataset index that the global index maps to
        dataset_idx = np.searchsorted(self.cumulative_lengths, global_idx, side="right")

        if dataset_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_lengths[dataset_idx - 1]

        # Adjust local_idx for datasets with increased sampling ratio
        local_idx = local_idx // self.ratios[dataset_idx]

        return self.datasets[dataset_idx].__getitem__((local_idx, n_per_seq))


def get_mix_dataset(cfg):
    mixset = cfg.train.mixset
    dataset = None

    if_shuffle = not cfg.inside_shuffle

    if if_shuffle:
        ConcatDataset_fn = ConcatDataset
    else:
        ConcatDataset_fn = ConcatDataset_inside_shuffle

    if "b" in mixset:
        from datasets.blendMVG import BlendMVGDatasetFix

        blendmvg_ds = BlendMVGDatasetFix(
            split="train",
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            BlendMVG_DIR="/fsx-repligen/shared/datasets/NeSF/BlendedMVG/combined",
            cfg=cfg,
        )

        blendmvg_scale = 20

        if dataset is None:
            dataset = ConcatDataset_fn([blendmvg_ds] * blendmvg_scale)
        else:
            dataset = ConcatDataset_fn([dataset] + [blendmvg_ds] * blendmvg_scale)

    if "t" in mixset:
        from datasets.tartanair import TartanAirDatasetFix

        tartan_ds = TartanAirDatasetFix(
            split="train",
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            TartanAir_DIR="/fsx-repligen/shared/datasets/TartanAirExtracted",
            cfg=cfg,
        )

        tar_scale = 20

        if dataset is None:
            dataset = ConcatDataset_fn([tartan_ds] * tar_scale)
        else:
            dataset = ConcatDataset_fn([dataset] + [tartan_ds] * tar_scale)

    if "r" in mixset:
        from datasets.re10k import Re10KDatasetFix

        re10k_ds = Re10KDatasetFix(
            split="train",
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            Re10K_DIR="/fsx-repligen/shared/datasets/RealEstate10K",
            # scene_info_path = "/fsx-repligen/shared/datasets/megadepth/scene_info",
            cfg=cfg,
        )

        re10k_scale = 3
        if dataset is None:
            dataset = ConcatDataset_fn([re10k_ds] * re10k_scale)
        else:
            dataset = ConcatDataset_fn([re10k_ds] + [dataset])

    if "k" in mixset:
        from datasets.kubric_movif_SfM_dataset import KubricSfMDataset

        # data_root = os.path.join("/fsx-repligen/shared/datasets/kubric/", "kubric_movi_f_tracks")
        # data_root = os.path.join("/fsx-repligen/shared/datasets/kubric/", "kubric_movi_f_24_frames_1024/")

        if cfg.new_kub:
            data_root = "/fsx-repligen/shared/datasets/kubric/kubric_movi_f_24_frames_1024/movi_f"
        else:
            data_root = os.path.join("/fsx-repligen/shared/datasets/kubric/", "kubric_movi_f_tracks")

        kubric_ds = KubricSfMDataset(
            data_root=data_root,
            crop_size=[cfg.train.img_size, cfg.train.img_size],
            seq_len=cfg.seqlen,
            traj_per_sample=cfg.train.track_num,
            sample_vis_1st_frame=True,
            use_augs=True,
            cfg=cfg,
        )

        # kubric_ds.seq_len
        # import pdb;pdb.set_trace()
        if dataset is None:
            dataset = kubric_ds
        else:
            dataset = ConcatDataset_fn([kubric_ds] + [dataset])

    if "c" in mixset:
        co3d_ds = Co3dDatasetFix(
            category=(cfg.train.category,),
            split="train",
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            CO3D_DIR=cfg.train.CO3D_DIR,
            CO3D_ANNOTATION_DIR=cfg.train.CO3D_ANNOTATION_DIR,
            preload_image=cfg.train.preload_image,
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            cfg=cfg,
        )
        if dataset is None:
            dataset = co3d_ds
        else:
            dataset = ConcatDataset_fn([co3d_ds] + [dataset])

    if "m" in mixset:
        megad_ds = MegaDepthDatasetV2Fix(
            split="train",
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            MegaDepth_DIR="/fsx-repligen/shared/datasets/megadepth",
            scene_info_path="/fsx-repligen/shared/datasets/megadepth/scene_info",
            cfg=cfg,
        )

        mega_ds_scale = 100

        if cfg.debug:
            mega_ds_scale = 1000

        if dataset is None:
            dataset = ConcatDataset_fn([megad_ds] * mega_ds_scale)
        else:
            dataset = ConcatDataset_fn([dataset] + [megad_ds] * mega_ds_scale)

    dataset = ConcatDataset_fn([dataset] * cfg.repeat_mix)

    # if cfg.dynamix:
    #     batch_sampler = DynamicBatchSampler(len(dataset), dataset_len=cfg.train.len_train, max_images=cfg.train.max_images, images_per_seq=cfg.train.images_per_seq)
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_sampler=batch_sampler,
    #         num_workers=cfg.train.num_workers,
    #         pin_memory=cfg.train.pin_memory,
    #         persistent_workers=cfg.train.persistent_workers,
    #         # collate_fn = collate_fn_sfm,
    #     )
    #     dataloader.batch_sampler.sampler = dataloader.batch_sampler
    #     import pdb;pdb.set_trace()
    # else:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=if_shuffle,
        prefetch_factor=cfg.pre_factor,
        batch_size=cfg.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    if cfg.debug:
        print("start frofile")

        # for i, batch in enumerate(dataloader):
        #     print(i)
        #     print(batch["seq_name"])
        # profile_dataloader(dataloader, 1000)

        # import pdb;pdb.set_trace()
        # cfg.train.clip_grad

    eval_dataset = IMCDataset(
        split="test",
        eval_time=True,
        min_num_images=0,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        IMC_DIR="/fsx-repligen/shared/datasets/IMC",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=True,  # to compare with colmap
        cfg=cfg,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,  # Assuming you want to process one sample at a time
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    return dataset, eval_dataset, dataloader, eval_dataloader


def get_megadepthV2_dataset(cfg):
    raise NotImplementedError("TODO here")

    dataset = MegaDepthDatasetV2(
        split="train",
        min_num_images=cfg.train.min_num_images,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=cfg.train.sort_by_filename,
        MegaDepth_DIR="/fsx-repligen/shared/datasets/megadepth",
        scene_info_path="/fsx-repligen/shared/datasets/megadepth/scene_info",
        cfg=cfg,
    )

    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=cfg.train.len_train,
        max_images=cfg.train.max_images,
        images_per_seq=cfg.train.images_per_seq,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        collate_fn=collate_fn_sfm,
    )

    eval_dataset = IMCDataset(
        split="test",
        eval_time=True,
        min_num_images=0,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        IMC_DIR="/fsx-repligen/shared/datasets/IMC",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=True,  # to compare with colmap
        cfg=cfg,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,  # Assuming you want to process one sample at a time
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    dataloader.batch_sampler.sampler = dataloader.batch_sampler

    return dataset, eval_dataset, dataloader, eval_dataloader


def profile_dataloader(dl, num_batches=1000):
    """
    Profile the dataloader by iterating through `num_batches` batches.
    """
    pr = cProfile.Profile()
    pr.enable()

    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        # Normally, you would process your batches here
        # For profiling, we're just iterating through to measure performance

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def get_megadepth_dataset(cfg, category=None):
    dataset = MegaDepthDataset(
        split="train",
        min_num_images=cfg.train.min_num_images,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        MegaDepth_DIR="/fsx-repligen/shared/datasets/megadepth",
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=cfg.train.sort_by_filename,
        cfg=cfg,
    )

    ####################################
    # for i in range(len(dataset)):
    #     for _ in range(3):
    #         dataset.__getitem__((i,random.randint(5, 25)))
    ####################################

    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=cfg.train.len_train,
        max_images=cfg.train.max_images,
        images_per_seq=cfg.train.images_per_seq,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    if cfg.eval_on_mega:
        eval_dataset = MegaDepthDataset(
            split="test",
            eval_time=True,
            min_num_images=cfg.train.min_num_images,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            mask_images=False,
            MegaDepth_DIR="/fsx-repligen/shared/datasets/megadepth",
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=cfg.train.sort_by_filename,
            cfg=cfg,
        )

        eval_batch_sampler = DynamicBatchSampler(
            len(eval_dataset),
            dataset_len=cfg.train.len_eval,
            max_images=cfg.train.max_images // 2,
            images_per_seq=cfg.train.images_per_seq,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
        )
        eval_dataloader.batch_sampler.sampler = eval_dataloader.batch_sampler
    else:
        eval_dataset = IMCDataset(
            split="test",
            eval_time=True,
            min_num_images=0,
            debug=False,
            img_size=cfg.train.img_size,
            normalize_cameras=cfg.train.normalize_cameras,
            normalize_T=cfg.train.normalize_T,
            mask_images=False,
            IMC_DIR="/fsx-repligen/shared/datasets/IMC",
            first_camera_transform=cfg.train.first_camera_transform,
            compute_optical=cfg.train.compute_optical,
            color_aug=cfg.train.color_aug,
            erase_aug=cfg.train.erase_aug,
            load_point=cfg.train.load_point,
            max_3dpoints=cfg.train.max_3dpoints,
            load_track=cfg.train.load_track,
            close_box_aug=cfg.train.close_box_aug,
            sort_by_filename=True,  # to compare with colmap
            cfg=cfg,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=1,  # Assuming you want to process one sample at a time
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
        )

    dataloader.batch_sampler.sampler = dataloader.batch_sampler

    return dataset, eval_dataset, dataloader, eval_dataloader


def get_co3d_dataset(cfg):
    if is_aws_cluster():
        CO3D_DIR = cfg.train.CO3D_DIR
        CO3D_ANNOTATION_DIR = cfg.train.CO3D_ANNOTATION_DIR
    else:
        CO3D_DIR = "/datasets01/co3dv2/080422/"
        CO3D_ANNOTATION_DIR = "/private/home/jianyuan/src/relpose/data/co3d_annotations"

    dataset = Co3dDataset(
        category=(cfg.train.category,),
        split="train",
        min_num_images=cfg.train.min_num_images,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
        preload_image=cfg.train.preload_image,
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=cfg.train.sort_by_filename,
        cfg=cfg,
    )

    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=cfg.train.len_train,
        max_images=cfg.train.max_images,
        images_per_seq=cfg.train.images_per_seq,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )  # collate_fn

    eval_dataset, eval_batch_sampler, eval_dataloader = get_co3d_dataset_eval(cfg)

    dataloader.batch_sampler.sampler = dataloader.batch_sampler
    # eval_dataloader.batch_sampler.sampler = eval_dataloader.batch_sampler

    return dataset, eval_dataset, dataloader, eval_dataloader


def get_co3d_dataset_eval(cfg):
    if is_aws_cluster():
        CO3D_DIR = cfg.train.CO3D_DIR
        CO3D_ANNOTATION_DIR = cfg.train.CO3D_ANNOTATION_DIR
    else:
        CO3D_DIR = "/datasets01/co3dv2/080422/"
        CO3D_ANNOTATION_DIR = "/private/home/jianyuan/src/relpose/data/co3d_annotations"

    eval_dataset = Co3dDataset(
        category=(cfg.train.category,),
        split="test",
        eval_time=True,
        min_num_images=cfg.train.min_num_images,
        debug=False,
        img_size=cfg.train.img_size,
        normalize_cameras=cfg.train.normalize_cameras,
        normalize_T=cfg.train.normalize_T,
        mask_images=False,
        CO3D_DIR=CO3D_DIR,
        CO3D_ANNOTATION_DIR=CO3D_ANNOTATION_DIR,
        preload_image=cfg.train.preload_image,
        first_camera_transform=cfg.train.first_camera_transform,
        compute_optical=cfg.train.compute_optical,
        color_aug=cfg.train.color_aug,
        erase_aug=cfg.train.erase_aug,
        load_point=cfg.train.load_point,
        max_3dpoints=cfg.train.max_3dpoints,
        load_track=cfg.train.load_track,
        close_box_aug=cfg.train.close_box_aug,
        sort_by_filename=cfg.train.sort_by_filename,
        cfg=cfg,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,  # Assuming you want to process one sample at a time
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )

    # import pdb;pdb.set_trace()
    # eval_batch_sampler = DynamicBatchSampler(
    #     len(eval_dataset), dataset_len=cfg.train.len_eval, max_images=cfg.train.max_images // 2, images_per_seq=cfg.train.images_per_seq
    # )

    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset,
    #     batch_sampler=eval_batch_sampler,
    #     num_workers=cfg.train.num_workers,
    #     pin_memory=cfg.train.pin_memory,
    #     persistent_workers=cfg.train.persistent_workers,
    # )

    return eval_dataset, None, eval_dataloader


def set_seed_and_print(seed):
    accelerate_set_seed(seed, device_specific=True)
    print(f"----------Seed is set to {np.random.get_state()[1][0]} now----------")


def find_last_checkpoint(exp_dir, all_checkpoints: bool = False):
    fls = sorted(glob.glob(os.path.join(glob.escape(exp_dir), "ckpt_" + "[0-9]" * 6)))

    if len(fls) == 0:
        return None
    elif all_checkpoints:
        return fls
    else:
        return fls[-1]


def view_color_coded_images_for_visdom(images):
    num_frames, _, height, width = images.shape
    cmap = cm.get_cmap("hsv")
    bordered_images = []

    for i in range(num_frames):
        img = images[i]
        color = torch.tensor(np.array(cmap(i / num_frames))[:3], dtype=img.dtype, device=img.device)

        # Create colored borders
        thickness = 5  # Border thickness
        # img[:, :, :thickness] = color  # Left border
        # img[:, :, -thickness:] = color  # Right border
        # img[:, :thickness, :] = color  # Top border
        # img[:, -thickness:, :] = color  # Bottom border
        # Left border
        img[:, :, :thickness] = color[:, None, None]

        # Right border
        img[:, :, -thickness:] = color[:, None, None]

        # Top border
        img[:, :thickness, :] = color[:, None, None]

        # Bottom border
        img[:, -thickness:, :] = color[:, None, None]

        bordered_images.append(img)

    return torch.stack(bordered_images)


def plotly_scene_visualization(camera_dict, batch_size):
    fig = plot_scene(camera_dict, camera_scale=0.03, ncols=2)
    fig.update_scenes(aspectmode="data")

    cmap = plt.get_cmap("hsv")

    for i in range(batch_size):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (batch_size)))
        fig.data[i + batch_size].line.color = matplotlib.colors.to_hex(cmap(i / (batch_size)))

    return fig


def _get_postfixed_filename(fl, postfix):
    return fl if fl.endswith(postfix) else fl + postfix


class VizStats(Stats):
    @staticmethod
    def from_json_str(json_str):
        self = VizStats([])
        # load the global state
        self.__dict__.update(json.loads(json_str))
        # recover the AverageMeters
        for stat_set in self.stats:
            self.stats[stat_set] = {
                log_var: AverageMeter.from_json_str(log_vals_json_str)
                for log_var, log_vals_json_str in self.stats[stat_set].items()
            }
        return self

    @staticmethod
    def load(flpath, postfix=".jgz"):
        flpath = _get_postfixed_filename(flpath, postfix)
        with gzip.open(flpath, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
        return VizStats.from_json_str(data)

    def plot_stats(self, viz=None, visdom_env=None, plot_file=None, visdom_server=None, visdom_port=None):
        # use the cached visdom env if none supplied
        if visdom_env is None:
            visdom_env = self.visdom_env
        if visdom_server is None:
            visdom_server = self.visdom_server
        if visdom_port is None:
            visdom_port = self.visdom_port
        if plot_file is None:
            plot_file = self.plot_file

        stat_sets = list(self.stats.keys())

        logger.debug(f"printing charts to visdom env '{visdom_env}' ({visdom_server}:{visdom_port})")

        novisdom = False

        if viz is None:
            viz = get_visdom_connection(server=visdom_server, port=visdom_port)

        if viz is None or not viz.check_connection():
            logger.info("no visdom server! -> skipping visdom plots")
            novisdom = True

        lines = []

        # plot metrics
        if not novisdom:
            viz.close(env=visdom_env, win=None)

        for stat in self.log_vars:
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                val = self.stats[stat_set][stat].get_epoch_averages()
                if val is None:
                    continue
                else:
                    val = np.array(val).reshape(-1)
                    stat_sets_now.append(stat_set)
                vals.append(val)

            if len(vals) == 0:
                continue

            lines.append((stat_sets_now, stat, vals))

        if not novisdom:
            for tmodes, stat, vals in lines:
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                for i, (tmode, val) in enumerate(zip(tmodes, vals)):
                    update = "append" if i > 0 else None
                    valid = np.where(np.isfinite(val))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(val))
                    viz.line(
                        Y=val[valid],
                        X=x[valid],
                        env=visdom_env,
                        opts=opts,
                        win=f"stat_plot_{title}",
                        name=tmode,
                        update=update,
                    )

        if plot_file:
            logger.info(f"plotting stats to {plot_file}")
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                plt.gca()
                for vali, vals_ in enumerate(vals):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    valid = np.where(np.isfinite(vals_))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(vals_))
                    plt.plot(x[valid], vals_[valid], c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                grid_params = {"visible": True, "color": gcolor}
                plt.grid(**grid_params, which="major", linestyle="-", linewidth=0.4)
                plt.grid(**grid_params, which="minor", linestyle="--", linewidth=0.2)
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            try:
                fig.savefig(plot_file)
            except PermissionError:
                warnings.warn("Cant dump stats due to insufficient permissions!")


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    # masks_flat: B, 1
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    rel_tangle_deg = evaluate_translation_batch(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def evaluate_translation_batch(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def closed_form_inverse(se3):
    # se3:    Nx4x4
    # return: Nx4x4
    # inverse each 4x4 matrix
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]
    R_trans = R.transpose(1, 2)

    left_down = -T.bmm(R_trans)
    left = torch.cat((R_trans, left_down), dim=1)
    right = se3[:, :, 3:].detach().clone()
    inversed = torch.cat((left, right), dim=-1)
    return inversed


class WarmupCosineRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, T_0, iters_per_epoch, T_mult=1, eta_min=0, warmup_ratio=0.1, warmup_lr_init=1e-7, last_epoch=-1
    ):
        self.T_0 = T_0 * iters_per_epoch
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_iters = int(T_0 * warmup_ratio * iters_per_epoch)
        self.warmup_lr_init = warmup_lr_init
        super(WarmupCosineRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restart = self.last_epoch // self.T_0
            T_cur = self.last_epoch - i_restart * self.T_0
        else:
            n = int(math.log((self.last_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            T_cur = self.last_epoch - self.T_0 * (self.T_mult**n - 1) // (self.T_mult - 1)

        if T_cur < self.warmup_iters:
            warmup_ratio = T_cur / self.warmup_iters
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * warmup_ratio for base_lr in self.base_lrs]
        else:
            T_cur_adjusted = T_cur - self.warmup_iters
            T_i = self.T_0 - self.warmup_iters
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur_adjusted / T_i)) / 2
                for base_lr in self.base_lrs
            ]


class FixBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, dataset_len=1024, batch_size=64, max_images=128, images_per_seq=(3, 20)):
        self.dataset = dataset
        self.max_images = max_images
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))
        self.num_sequences = len(self.dataset)
        self.dataset_len = dataset_len
        self.batch_size = 64
        self.fix_images_per_seq = True

    def _capped_random_choice(self, x, size, replace: bool = True):
        len_x = x if isinstance(x, int) else len(x)
        if replace:
            return np.random.choice(x, size=size, replace=len_x < size)
        else:
            return np.random.choice(x, size=min(size, len_x), replace=False)

    def __iter__(self):
        for batch_idx in range(self.dataset_len):
            # NOTE batch_idx is never used later
            # print(f"process {batch_idx}")
            if self.fix_images_per_seq:
                n_per_seq = 10
            else:
                n_per_seq = np.random.choice(self.images_per_seq)

            n_seqs = self.batch_size

            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)
            # print(f"get the chosen_seq for {batch_idx}")

            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            # print(f"yield the batches for {batch_idx}")
            yield batches

    def __len__(self):
        return self.dataset_len