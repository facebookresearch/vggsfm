# VGGSfM: Visual Geometry Grounded Deep Structure From Motion


![Teaser](https://raw.githubusercontent.com/vggsfm/vggsfm.github.io/main/resources/vggsfm_teaser.gif)

**[Meta AI Research, GenAI](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**


[Jianyuan Wang](https://jytime.github.io/), [Nikita Karaev](https://nikitakaraevv.github.io/), [Christian Rupprecht](https://chrirupp.github.io/), [David Novotny](https://d-novotny.github.io/)



<p 
dir="auto">[<a href="https://arxiv.org/pdf/2312.04563.pdf" rel="nofollow">Paper</a>]
[<a href="https://vggsfm.github.io/" rel="nofollow">Project Page</a>]   
[<a href="https://huggingface.co/spaces/facebook/vggsfm" rel="nofollow">🤗 Demo</a>] 
[Version 2.0]
</p> 


**Updates:**


- [Sep 9, 2024] Allow to export a dense point cloud!

- [Sep 5, 2024] Added the instruction on how to train a Gaussian splatting model with our results! 

- [Aug 26, 2024]
  - We have introduced a video runner that can process sequential frames, such as those in videos. It supports the reconstruction of over ```1000``` input frames. By using masks to filter out moving objects, it can also effectively recover camera poses and point clouds from dynamic video sequences.

- [Jul 28, 2024] Added support for filtering out dynamic objects using ```masks```.

- [Jul 10, 2024] Now we support exporting dense depth maps!

- Happy to share we were ranked 1st 🥇 in the CVPR24 IMC Challenge regarding camera pose (Rot&Trans) estimation.

## Table of Contents
- [Installation](#installation)
- [Reconstruction](#reconstruction)
  - [Download Pre-trained Model](#1-download-pre-trained-model)
  - [Run a Demo](#2-run-a-demo)
  - [Visualization Options](#3-visualization-options)
  - [Use Your Own Data](#4-use-your-own-data)
  - [Generate Denser Point Cloud](#5-generate-denser-point-cloud)
  - [Dense Depth Prediction (Beta)](#6-dense-depth-prediction-beta)
  - [Sequential Input](#7-sequential-input)
  - [Dynamic/Moving Objects](#8-dynamicmoving-objects)
  - [Train a Gaussian Splatting Model](#9-train-a-gaussian-splatting-model)
  - [FAQs](#10-faqs)

## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.10, PyTorch 2.1, and CUDA 12.1.

```.bash
source install.sh
python -m pip install -e .
```

This script installs official ```pytorch3d```, ```lightglue```, ```pycolmap```, ```poselib```, and ```visdom```. If you cannot install ```pytorch3d``` on your machine, feel free to skip it, because now we only use it for visdom visualization (i.e., ```cfg.viz_visualize=True```). 


## RECONSTRUCTION  

### 1. Download Pre-trained Model
The checkpoint will be automatically downloaded from [Hugging Face](https://huggingface.co/facebook/VGGSfM/tree/main) during the first run. Alternatively, you can manually download it from [Hugging Face](https://huggingface.co/facebook/VGGSfM/blob/main/vggsfm_v2_0_0.bin) or [Google Drive](https://drive.google.com/file/d/163bHiqeTJhQ2_UnihRNPRA4Y9X8-gZ1-/view?usp=sharing). If you prefer to specify the checkpoint path manually, set `auto_download_ckpt` to `False` and update `resume_ckpt` to your path in the Hydra config.


### 2. Run a Demo 

Now it's time to enjoy your 3D reconstruction! You can start with our provided examples:

```bash
# Use default settings
python demo.py SCENE_DIR=examples/kitchen 

# Specify query method: sp+sift (default: aliked)
python demo.py SCENE_DIR=examples/statue query_method=sp+sift

# Increase query number to 4096 (default: 2048)
python demo.py SCENE_DIR=examples/british_museum max_query_pts=4096 

# Assume a shared camera model for all frames, and
# Use SIMPLE_RADIAL camera model instead of the default SIMPLE_PINHOLE
# Increase the number of query frames from the default value of 3 to 6
python demo.py shared_camera=True camera_type=SIMPLE_RADIAL query_frame_num=6

# If you want a fast reconstruction without fine tracking
python demo.py SCENE_DIR=examples/kitchen fine_tracking=False
```

All default settings for the flags are specified in `cfgs/demo.yaml`. You can adjust these flags as needed, such as reducing the number of query points by ```max_query_pts```, or increase ```query_frame_num``` to use more frames as query. To enforce a shared camera model for a scene, set ```shared_camera=True```. To use query points from different methods, set ```query_method``` to ```sp```, ```sift```, ```aliked```, or any combination like ```sp+sift```. Additionally, ```fine_tracking``` is enabled by default, but you can set it to False to switch to coarse matching only, which speeds up inference a lot. 

The reconstruction result (camera parameters and 3D points) will be automatically saved under ```SCENE_DIR/sparse``` in the COLMAP format, as ```cameras.bin```, ```images.bin```, and ```points3D.bin```. This format is widely supported by the recent NeRF/Gaussian Splatting codebases. You can use [COLMAP GUI](https://colmap.github.io/gui.html) or [viser](https://github.com/nerfstudio-project/viser) to view the reconstruction. 

This sparse reconstruction mode can process up to ```400``` frames at a time. To handle more frames (e.g., over ```1,000``` frames), please refer to the ```Sequential Input``` section below for guidance.


### 3. Visualization Options

If you want to visualize it more easily, we also provide visualization options using [Gradio](https://github.com/gradio-app/gradio) and [Visdom](https://github.com/fossasia/visdom).



#### 3.1 Gradio Visualization (recommended)

<details>
<summary>Click to expand</summary>

The easiest way to visualize your results is by using Gradio. Simply set the `gr_visualize` flag to `True`, and a link will be generated that you can open in any web browser in any machine. This is especially useful when running the program on a remote Linux server without a GUI, allowing you to view the results on your local computer. Please note it may take a few seconds to load.


```bash
python demo.py gr_visualize=True ...(other flags)
```
</details>


#### 3.2 Visdom Visualization

<details>
<summary>Click to expand</summary>

To begin using Visdom, start the server by entering ```visdom``` in the command line. Once the server is running, access Visdom by navigating to ```http://localhost:8097``` in your web browser. Now every reconstruction can be visualized and saved to the Visdom server by enabling ```viz_visualize=True```:

```bash
python demo.py viz_visualize=True ...(other flags)
```

You should see an interface like this:

![UI](assets/ui.png)
</details>



#### 3.3 Additional Visualizations
- **Visualizing 2D Reprojections:**
  <!-- <details> -->
  <!-- <summary>Click to expand</summary> -->
  - To visualize the 2D reprojections of reconstructed 3D points, set the `make_reproj_video` flag to `True`. This will generate a video named `reproj.mp4` in the `SCENE_DIR/visuals` directory. For example:
  <p align="center">
    <img src="https://github.com/vggsfm/vggsfm.github.io/blob/main/resources/reproj.gif" width="500" alt="reproj">
  </p>
  <!-- </details> -->


- **Visualizing Track Predictions:**
  <details>
  <summary>Click to expand</summary>

  - To visualize the raw predictions from our track predictor, enable ```visual_tracks=True``` to generate ```track.mp4```. In this video, transparent points indicate low visibility or confidence. 
  </details>


### 4. Use Your Own Data

You only need to specify the address of your data. For example, I would recommend to start from 

```bash
python demo.py SCENE_DIR=/YOUR_FOLDER camera_type=SIMPLE_RADIAL gr_visualize=True make_reproj_video=True
```

Please ensure that the images are stored in ```YOUR_FOLDER/images```. This folder should contain only the images. Check the ```examples``` folder for the desired data structure.


### 5. Generate Denser Point Cloud

To generate a denser point cloud, you can triangulate additional 3D points by setting the `extra_pt_pixel_interval` parameter. For each frame, a 2D grid is sampled with a pixel interval defined by `extra_pt_pixel_interval`. This grid is used as query points to estimate tracks, which are then triangulated into 3D points. After filtering out noisy 3D points, they are added to the existing point cloud. Since these extra 3D points are not optimized in the bundle adjustment process, this method is quite fast while maintaining reasonable quality. You can generally start from ```python demo.py extra_pt_pixel_interval=10 concat_extra_points=True```. The additional points will be concatenated with the existing points if `concat_extra_points=True`, allowing direct use for Gaussian Splatting. The details of these additional points are saved in `additional/additional_points_dict.pt`. You can also set `extra_by_neighbor` to control the number of neighboring frames for each additional point, ensuring efficient inference when dealing with a large number of frames.


### 6. Dense Depth Prediction (Beta)

<details>
<summary>Click to expand</summary>
We support extracting dense depth maps with the help of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Basically, we align the dense depth prediction from Depth-Anything-V2 using the sparse SfM point cloud predicted by VGGSfM. To enable this, please first git clone Depth-Anything-V2 and install scikit-learn:

```bash
pip install scikit-learn
git clone git@github.com:DepthAnything/Depth-Anything-V2.git dependency/depth_any_v2
python -m pip install -e .
```

To enable dense depth prediction, set ```dense_depth=True``` when running `demo.py`. The generated depth maps will be saved in the ```depths``` folder under ```cfg.SCENE_DIR```, using the COLMAP format (e.g., ```*.bin```). To visualize the dense point cloud (unprojected dense depth maps) in Visdom, set ```visual_dense_point_cloud=True```. Note that due to the large number of points, we only support ```visual_dense_point_cloud``` in Visdom, not in Gradio.
</details>


### 7. Sequential Input

Given ordered frames as input (e.g., videos), we support running reconstruction in a sliding window manner, which allows for the reconstruction of over ```1,000``` frames.

Its launcher ```video_demo.py``` follows the same convention to ```demo.py``` above. You need to put images under  ```YOUR_VIDEO_FOLDER/images``` with the their image names ordered, e.g., ```0000.png```, ```0001.png```, ```0002.png```... Then, simply run the following command:


```bash
python video_demo.py SCENE_DIR=/YOUR_VIDEO_FOLDER
```

Please note that the configuration for `video_demo.py` is initialized in `cfgs/video_demo.yaml`. You can adjust ```init_window_size``` and ```window_size``` to control the number of frames for each window. The flag ```joint_BA_interval``` is used to control the frequency of joint bundle adjustment over the whole sequence. Other flags, such as thoses regarding output saving or visualization, are exactly the same as in `demo.py`.


### 8. Dynamic/Moving Objects

Sometimes, the input frames may contain dynamic or moving objects. Our method can generally handle these when the dynamic objects are relatively small. However, if the dynamic objects occupy a significant portion of the frames, especially when the camera motion is minimal, we recommend filtering out these dynamic pixels.


This codebase supports the use of masks to filter out dynamic pixels. The masks should be placed in the `masks` folder under `cfg.SCENE_DIR`, with filenames matching the corresponding images (e.g., `images/0000.png` and `masks/0000.png`). These masks should be binary, with 1 indicating pixels to be filtered out (i.e., dynamic pixels) and 0 for pixels that should remain. You can refer to the masks from the DAVIS dataset as an example. 

Masks can be generated using object detection, video segmentation, or manual labeling. Here is an [instruction](https://github.com/vye16/shape-of-motion/blob/main/preproc/README.md) on how to build such masks using SAM and Track-Anything. [SAM2](https://github.com/facebookresearch/segment-anything-2) is also a good option for generating these masks.


### 9. Train a Gaussian Splatting model

If you have successfully reconstructed a scene using the commands above, you should have a folder named `sparse` under your `SCENE_DIR`, with the following structure:
``` 
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

You can now use [gsplat](https://github.com/nerfstudio-project/gsplat) to train your own Gaussian Splatting model. Please install `gsplat` according to their official instructions. We assume you are using `gsplat==1.3.0`.

An example command to train the model is:
```
cd gsplat
python examples/simple_trainer.py  default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```

### 10. FAQs
<details>
<summary><strong>What should I do if I encounter an out-of-memory error?</strong></summary>

We may encounter an out-of-memory error when the number of input frames or query points is too high. In v2.0, we address this by splitting the points into several chunks and running the prediction separately. This involves two hardcoded hyperparameters: ```max_points_num=163840``` in [predict_tracks](https://github.com/facebookresearch/vggsfm/blob/cfbc06e2f30639073b52d65828e6a6d27087c4f4/vggsfm/runners/runner.py#L894C20-L894C26) and ```max_tri_points_num=819200``` in [triangulate_tracks](https://github.com/facebookresearch/vggsfm/blob/cfbc06e2f30639073b52d65828e6a6d27087c4f4/vggsfm/utils/triangulation.py#L712). These values are set for a ```32GB``` GPU. If your GPU has less or more memory, reduce or increase these values ​​accordingly.

</details>


<details>
<summary><strong>How to handle sparse data with minimal view overlap?</strong></summary>

For scenarios with sparse views and minimal overlap, the simplest solution is to set ```query_frame_num``` to the total number of your images and use a ```max_query_pts``` of 4096 or more. This ensures all frames are registered. Since we only have sparse views, the inference process remains very fast. For example, the following command took around 20 seconds to reconstruct an 8-frame scene:
```
python demo.py SCENE_DIR=a_scene_with_8_frames query_frame_num=8 max_query_pts=4096 query_method=aliked
```
</details>


<details>
<summary><strong>When should I set shared_camera to True?</strong></summary>

Set ```shared_camera``` to True when you know that the input frames were captured by the same camera and the camera focal length did not significantly change during the capture. This assumption is usually valid for images extracted from a video.
</details>


## Testing 

We are still preparing the testing script for VGGSfM v2. However, you can use our code for VGGSfM v1.1 to reproduce our benchmark results in the paper. Please refer to the branch ```v1.1```.


## Acknowledgement

We are highly inspired by [colmap](https://github.com/colmap/colmap), [pycolmap](https://github.com/colmap/pycolmap), [posediffusion](https://github.com/facebookresearch/PoseDiffusion), [cotracker](https://github.com/facebookresearch/co-tracker), and [kornia](https://github.com/kornia/kornia).


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.


## Citing VGGSfM
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:


```bibtex
@inproceedings{wang2024vggsfm,
  title={VGGSfM: Visual Geometry Grounded Deep Structure From Motion},
  author={Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21686--21697},
  year={2024}
}