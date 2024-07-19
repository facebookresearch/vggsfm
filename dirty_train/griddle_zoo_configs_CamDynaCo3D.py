import copy
import os

from griddle.griddle_experiment_config import ExperimentConfigGrid, GriddleMode
from griddle.utils import param_grid
from griddle.utils import is_aws_cluster

# -----
from train_experiment import ExperimentConfig
from omegaconf import OmegaConf


# -----

DEFAULT_STATS = ["Racc_5", "Racc_15", "Racc_30", "Tacc_5", "Tacc_15", "Tacc_30"]


if is_aws_cluster():
    EXPERIMENT_ROOT = os.path.expandvars("/fsx-repligen/jianyuan/gridexp")
    # EXPERIMENT_ROOT = os.path.expandvars("/data/home/jianyuan/gridexp_home")
else:
    EXPERIMENT_ROOT = os.path.expandvars("/checkpoint/$USER/exps/griddle/")


def griddle_zoo_configs(griddle_mode: GriddleMode, experiment_mode: str):
    # ----
    cfgs = {}
    # ----

    node_num = 1
    gpu_num = 8
    JOB_PARAMS = {
        "slurm_time": 4000,
        "slurm_gpus_per_node": gpu_num,
        "slurm_partition": "learn",
        "slurm_account": "repligen",
        "slurm_qos": "repligen",
        "slurm_cpus_per_gpu": 12,
        "slurm_mem": "0",
        "slurm_requeue": True,
        "slurm_nodes": node_num,  # This line requests 4 nodes
    }

    accelerate_args = {
        "num_machines": node_num,
        "multi_gpu": True,
        "debug": True,  # if met error, will give sth; if set to False, give nothing with error
        "mixed_precision": "no",
        "num_cpu_threads_per_process": 4,
        "num_processes": gpu_num * node_num,  # 8 gpus requested
    }

    is_eval = experiment_mode == "eval"
    is_debug = griddle_mode == GriddleMode.DEBUG

    cfg_name = "CamDynaCo3D"
    hydra_config = "../cfgs/vggsfm_v5.yaml"
    base_conf = OmegaConf.load(hydra_config)

    # griddle_run griddle_zoo_configs_folder=./griddle_zoo_configs cfg="CamDyna100" griddle_mode=DEBUG 

    # griddle_run griddle_zoo_configs_folder=./griddle_zoo_configs cfg="CamMC" griddle_mode=DISPATCH

    # Common params
    base_conf.experiment_mode = experiment_mode
    base_conf.cfg_name = cfg_name

    if is_debug:
        base_conf = base_conf
        # base_conf.visualize = True
        base_conf.debug = True
        base_conf.debugeval = False
        JOB_PARAMS = {
            "slurm_time": 4000,
            "slurm_gpus_per_node": 1,
            "slurm_partition": "learngenai",
            "slurm_cpus_per_task": 10,
            "slurm_mem": "0",
        }

        accelerate_args = {
            "num_machines": 1,
            "multi_gpu": True,
            "debug": True,  # if met error, will give sth; if set to False, give nothing with error
            "mixed_precision": "no",
            "num_cpu_threads_per_process": 8,
            "num_processes": 1,  # 4 gpus requested
        }

    # base_conf.train.resume_ckpt = "/data/home/jianyuan/gridexp_home/CamPreV1T2DynaCNP/img_size_336_down_size_336_mixset_mtbcr_load_camera_True_enable_pose_True_lr_0.0001_enable_track_False_normalize_T_True_also_trunk_True_track_num_2048_repeat_mix_1_restart_num_50_dynamix_True_max_images_128_len_train_4096_prelimi_cam_False/ckpt_000250/pytorch_model.bin"

    base_conf.train.resume_ckpt = "/data/home/jianyuan/gridexp_home/CamDyna50/close_valid_True_mixset_mtbcrud_lr_0.0001_repeat_mix_1_restart_num_50_dynamix_True_max_images_120_len_train_4096_prelimi_cam_False/ckpt_000145/pytorch_model.bin"
    base_conf.train.load_track = False

    base_conf.train.load_camera = True
    base_conf.train.img_size = 336
    
    base_conf.MODEL.CAMERAPRED.down_size = 336
    
    base_conf.enable_pose = True
    base_conf.enable_track = False
    base_conf.train.normalize_T = True
    base_conf.also_trunk = True
    
    base_conf.train.images_per_seq = [24, 51]
    base_conf.train.min_num_images = 51
    
    grid_param = {
        ############################################################
        "close_valid": [True],
        "train.mixset": ["c"],
        ############################################################
        # "train.img_size": [336],
        # "MODEL.CAMERAPRED.down_size": [336],
        # "mixed_precision": [ "bf16"],
        # kmtbcud
        # "train.load_track": [True],
        # "train.load_camera": [True],
        # "enable_pose": [True],
        "train.lr": [0.0001],
        # "enable_track": [False],
        # "train.normalize_T": [True],
        # "also_trunk": [True],
        # "train.mixset": ["mt"],
        # "train.track_num": [2048],
        # "batch_size": [6],
        "repeat_mix": [1],
        "train.restart_num": [50],
        # "train.restart_num": [50],
        "dynamix": [True],
        "train.max_images": [112],
        # "train.lr": [0.0001, 0.00005,],
        "train.len_train": [4096],
        "prelimi_cam": [False],
        "testimc2024": [True],
    }

    # prelimi_cam
    
    grid, exp_names = param_grid(grid_param, common_params=base_conf, return_name=True)
    exp_names = [cfg_name + "/" + name for name in exp_names]

    cfgs[cfg_name] = ExperimentConfigGrid(
        griddle_experiment_configurable=ExperimentConfig(),
        cfg_dicts=grid,
        exp_names=exp_names,
        experiment_root=EXPERIMENT_ROOT,
        experiment_name_prefix=cfg_name,
        experiment_mode=experiment_mode,
        autogenerate_exp_dirs=True,
        stats_analyze=copy.deepcopy(DEFAULT_STATS_EVAL if is_eval else DEFAULT_STATS),
        slurm_job_params=copy.deepcopy(JOB_PARAMS),
        accelerate_job_params=copy.deepcopy(accelerate_args),
    )
    #################################

    return cfgs
