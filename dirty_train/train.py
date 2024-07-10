# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Standard library imports


import argparse
import cProfile
import datetime
import glob
import io
import json
import os
import pickle
import pstats
import re
import time
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union

# Related third-party imports
from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from hydra.utils import instantiate, get_original_cwd

import cv2
import hydra
import models
import numpy as np
import psutil
import torch
import tqdm
import visdom
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from griddle.utils import is_aws_cluster
from test_category import test_co3d, test_imc
from util.load_img_folder import load_and_preprocess_images
from util.metric import camera_to_rel_deg, calculate_auc, calculate_auc_np
from util.triangulation import intersect_skew_line_groups
from util.train_util import *
from train_eval_func import train_or_eval_fn

# Local application/library specific imports
# from gluefactory.models.extractors.disk_kornia import DISK
# from gluefactory.models.extractors.superpoint_open import SuperPoint


def get_thread_count(var_name):
    return os.environ.get(var_name)


def train_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision=cfg.mixed_precision)

    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))

    accelerator.print(accelerator.state)

    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark

    set_seed_and_print(cfg.seed)

    if accelerator.is_main_process:
        viz = vis_utils.get_visdom_connection(
            server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
        )

    # Building datasets
    dataset, eval_dataset, dataloader, eval_dataloader = build_dataset(cfg)

    # to make accelerator happy
    dataloader.batch_sampler.drop_last = True
    eval_dataloader.batch_sampler.drop_last = True

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)
    model = model.to(accelerator.device)

    num_epochs = cfg.train.epochs

    # Building optimizer
    optimizer, lr_scheduler = build_optimizer(cfg, model, dataloader)

    ########################################################################################################################
    if cfg.train.resume_ckpt:
        accelerator.print(f"Loading ckpt from {cfg.train.resume_ckpt}")
        model = load_model_weights(model, cfg.train.resume_ckpt, accelerator.device, cfg.relax_load)

    # accelerator preparation
    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))

    accelerator.print(f"dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 0

    stats = VizStats(TO_PLOT_METRICS)

    # CHECKPOINT RESUMING
    if cfg.train.auto_resume:
        # if cfg.debug:
        #     import pdb;pdb.set_trace()

        last_checkpoint = find_last_checkpoint(cfg.exp_dir)

        try:
            resume_epoch = int(os.path.basename(last_checkpoint)[5:])
        except:
            resume_epoch = -1

        if last_checkpoint is not None and resume_epoch > 0:
            accelerator.print(f"Loading ckpt from {last_checkpoint}")

            accelerator.load_state(last_checkpoint)

            try:
                loaded_tdict = pickle.load(open(os.path.join(last_checkpoint, "tdict.pkl"), "rb"))
                start_epoch = loaded_tdict["epoch"] - 1  # + 1
            except:
                start_epoch = resume_epoch - 1  # + 1

            try:
                stats = stats.load(os.path.join(last_checkpoint, "train_stats.jgz"))
            except:
                stats.hard_reset(epoch=start_epoch)
                accelerator.print(f"No stats to load from {last_checkpoint}")
        else:
            accelerator.print(f"Starting from scratch")

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()
        set_seed_and_print(cfg.seed + epoch * 1000)

        if (epoch != 0) and epoch % cfg.train.ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")

            if accelerator.is_main_process:
                accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)
                pickle.dump({"epoch": epoch, "cfg": cfg}, open(os.path.join(ckpt_path, "tdict.pkl"), "wb"))
                stats.save(os.path.join(ckpt_path, "train_stats.jgz"))

        if cfg.force_eval:
            train_or_eval_fn(
                model,
                eval_dataloader,
                cfg,
                optimizer,
                stats,
                accelerator,
                lr_scheduler,
                training=False,
                epoch=epoch,
                viz=viz,
            )  # viz=viz)
            raise NotImplementedError

        if cfg.debug:
            print("debugging")
            # dataset.__getitem__((0))
            # eval_dataset.__getitem__((0))
            # for hahaha in range(10):
            #    dataset.__getitem__((hahaha))
            if cfg.debugeval:
                train_or_eval_fn(
                    model,
                    eval_dataloader,
                    cfg,
                    optimizer,
                    stats,
                    accelerator,
                    lr_scheduler,
                    training=False,
                    epoch=epoch,
                    viz=viz,
                )  # viz=viz)

        if cfg.force_test:
            test_imc(model, cfg, accelerator, epoch=epoch, print_detail=False)
            test_co3d(model, cfg, accelerator, epoch=epoch, print_detail=False)

        # Testing
        if (epoch != 0) and (epoch % cfg.test.test_interval == 0):
            accelerator.print(f"----------Start to test at epoch {epoch}----------")
            test_co3d(model, cfg, accelerator, epoch=epoch, print_detail=False)
        elif (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            train_or_eval_fn(
                model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=False, epoch=epoch
            )  # viz=viz)
        else:
            accelerator.print(f"----------Skip the test/eval at epoch {epoch}----------")

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, epoch=epoch
        )  # viz=viz)

        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------")
            stats.update({"lr": lr}, stat_set="train")
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name)
            accelerator.print(f"----------Done----------")
            # viz.save([cfg.exp_name])

    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"), safe_serialization=False)
    return True