# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader
from vggsfm.utils.utils import seed_all_random_engines



@hydra.main(config_path="cfgs/", config_name="demo")
def demo_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed
    seed_all_random_engines(cfg.seed)

    # Build VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR, img_size=cfg.img_size, 
        normalize_cameras=False, load_gt=cfg.load_gt)   
    
    sequence_list = test_dataset.sequence_list
    
    seq_name = sequence_list[0] # Run on one Scene
    
    # Load the data
    batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)

    output_dir = batch["scene_dir"] # which is also cfg.SCENE_DIR for DemoLoader

    images = batch["image"]
    crop_params = batch["crop_params"]    
    if batch["masks"] is not None:
        masks = batch["masks"]
    else:
        masks = None 
    
    # both visualization and output writing are performed inside VGGSfMRunner
    predictions = vggsfm_runner.run(
        images,
        masks,
        crop_params,
        image_paths,
        seq_name=seq_name,
        output_dir=output_dir,
    )

    print("Demo Finished")

    return True


if __name__ == "__main__":
    with torch.no_grad():
        demo_fn()
