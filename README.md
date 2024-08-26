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

- [Aug 26, 2024]
  - We provided a video runner that can run on sequential frames, e.g., videos!


- [Aug 6, 2024]
  - VGGSfM is now available as a Python package, making it easier to integrate into other codebases!
  - Introduced a new `VGGSfMRunner` class that serves as a central controller for all functionalities.
  - Added support for gradio visualization, controlled by ```gr_visualize```.


- [Jul 28, 2024]
  - Added support for filtering out dynamic objects using ```masks```. We will add an example soon but you can check `demo_loader.py` for a quick view.
  - Added support for `visual_dense_point_cloud`.

- [Jul 10, 2024] Now we support exporting dense depth maps!

- Happy to share we were ranked 1st 🥇 in the CVPR24 IMC Challenge regarding camera pose (Rot&Trans) estimation.



## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.10, PyTorch 2.1, and CUDA 12.1.

```.bash
source install.sh
python -m pip install -e .
```

This script installs official ```pytorch3d```, ```lightglue```, ```pycolmap```, ```poselib```, and ```visdom```. If you cannot install ```pytorch3d``` on your machine, feel free to comment the line, because now we only use it for visdom visualization (i.e., ```cfg.viz_visualize=True```). 


## Demo 

### 1. Download Model
The checkpoint will be automatically downloaded from [Hugging Face](https://huggingface.co/facebook/VGGSfM/tree/main). You can also manually download it from [Hugging Face](https://huggingface.co/facebook/VGGSfM/blob/main/vggsfm_v2_0_0.bin) and [Google Drive](https://drive.google.com/file/d/163bHiqeTJhQ2_UnihRNPRA4Y9X8-gZ1-/view?usp=sharing). If you prefer to specify the checkpoint path manually, set the path in ```resume_ckpt``` and set ```auto_download_ckpt``` to False.


### 2. Run the Demo 

Now it's time to enjoy your 3D reconstruction! You can start with our provided examples:

```bash
# Use default settings
python demo.py SCENE_DIR=examples/kitchen 

# Specify query method: sp+sift (default: aliked)
python demo.py SCENE_DIR=examples/statue query_method=sp+sift

# Increase query number to 4096 (default: 2048)
python demo.py SCENE_DIR=examples/british_museum max_query_pts=4096 

# Assume a shared camera model for all frames, and
# use SIMPLE_RADIAL camera model instead of the default SIMPLE_PINHOLE
python demo.py shared_camera=True camera_type=SIMPLE_RADIAL

# If you want a fast reconstruction without fine tracking
python demo.py SCENE_DIR=examples/kitchen fine_tracking=False
```

All default settings for the flags are specified in `cfgs/demo.yaml`. You can adjust these flags as needed, such as reducing ```max_query_pts``` to lower GPU memory usage. To enforce a shared camera model for a scene, set ```shared_camera=True```. To use query points from different methods, set ```query_method``` to ```sp```, ```sift```, ```aliked```, or any combination like ```sp+sift```. ```fine_tracking``` is set to True by default. Set it to False to switch to coarse mode for faster inference. 


To run reconstruction on a scene with ```100``` frames on a ```32 GB``` GPU, you can start from the setting below:

```bash
python demo.py SCENE_DIR=TO/YOUR/PATH max_query_pts=1024 query_frame_num=6
```


The reconstruction result (camera parameters and 3D points) will be automatically saved under ```SCENE_DIR``` in the COLMAP format, as ```cameras.bin```, ```images.bin```, and ```points3D.bin```. This format is widely supported by the recent NeRF/Gaussian Splatting codebases. You can use [COLMAP GUI](https://colmap.github.io/gui.html) or [viser](https://github.com/nerfstudio-project/viser) to view the reconstruction. 


### 3. Visualization

If you want to visualize it more easily, we also provide visualization options using [Visdom](https://github.com/fossasia/visdom) and [Gradio](https://github.com/gradio-app/gradio).


#### 3.1 Visdom Visualization

To begin using Visdom, start the server by entering ```visdom``` in the command line. Once the server is running, access Visdom by navigating to ```http://localhost:8097``` in your web browser. Now every reconstruction can be visualized and saved to the Visdom server by enabling ```viz_visualize=True```:

```bash
python demo.py viz_visualize=True ...(other flags)
```

You should see an interface like this:

![UI](assets/ui.png)



#### 3.2 Gradio Visualization

For a serverless option, use Gradio by setting `gr_visualize` to True. This will generate a link accessible from any browser (but it may take seconds to load).


```bash
python demo.py gr_visualize=True ...(other flags)
```

#### 3.3 Additional Visualizations

- **2D Reprojections:**
  - To visualize the 2D reprojections of reconstructed 3D points, set the `make_reproj_video` flag to `True`. This will generate a video named `reproj.mp4` in the `SCENE_DIR/visuals` directory. For example:

<!-- <img src="https://github.com/vggsfm/vggsfm.github.io/blob/main/resources/reproj.gif" width="500" alt="reproj"> -->

  <p align="center">
    <img src="https://github.com/vggsfm/vggsfm.github.io/blob/main/resources/reproj.gif" width="500" alt="reproj">
  </p>


- **Track Predictions:**
  - To visualize the raw predictions from our track predictor, enable ```visual_tracks=True``` to generate ```track.mp4```. In this video, transparent points indicate low visibility or confidence. 


### 4. Try your own data

You only need to specify the address of your data, such as:

```bash
python demo.py SCENE_DIR=examples/YOUR_FOLDER ...(other flags)
```

Please ensure that the images are stored in ```YOUR_FOLDER/images```. This folder should contain only the images. Check the ```examples``` folder for the desired data structure.


Have fun and feel free to create an issue if you meet any problem. SfM is always about corner/hard cases. I am happy to help. If you prefer not to share your images publicly, please send them to me by email.


### 5. Dense depth maps (Beta)

We support extracting dense depth maps with the help of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Basically, we align the dense depth prediction from Depth-Anything-V2 using the sparse SfM point cloud predicted by VGGSfM. To enable this, please first git clone Depth-Anything-V2 and install scikit-learn:

```bash
pip install scikit-learn
git clone git@github.com:DepthAnything/Depth-Anything-V2.git dependency/depth_any_v2
python -m pip install -e .
```

Then, you just need to set ```dense_depth=True``` when running demo.py. Depth maps will be saved in the ```depths``` folder under ```cfg.SCENE_DIR```, using the COLMAP format (e.g., ```*.bin```). To visualize the dense point cloud (unprojected dense depth maps) in Visdom, set ```visual_dense_point_cloud=True``` (note it may take seconds to open the Visdom page when there are too many points).


### 6. Video Runner (Beta)

You can choose to run on a video by: 

```bash
python video_demo.py SCENE_DIR=/PATH/TO/YOUR/VIDEO/FOLDER
```

This script will call a VideoRunner to reconstruct the input frames in a sliding window manner. The input format is the same as the demo script, but just ensure the frames under the ```SCENE_DIR/images``` are ordered. 



### FAQ

* __What should I do if I encounter an out-of-memory error?__

To resolve an out-of-memory error, you can simply try reducing the number of ```max_query_pts``` to  a lower value. Be aware that this may result in a sparser point cloud and could potentially impact the accuracy of the reconstruction. Please note that in the latest commit, the value of ```query_frame_num``` will not affect the GPU memory consumption any more. Feel free to increase ```query_frame_num```.


* __How to handle sparse data with minimal view overlap?__

For scenarios with sparse views and minimal overlap, the simplest solution is to set ```query_frame_num``` to the total number of your images and use a ```max_query_pts``` of 4096 or more. This ensures all frames are registered. Since we only have sparse views, the inference process remains very fast. For example, the following command took around 20 seconds to reconstruct an 8-frame scene:
```
python demo.py SCENE_DIR=a_scene_with_8_frames query_frame_num=8 max_query_pts=4096 query_method=aliked
```


* __When should I set ```shared_camera``` to True?__

Set ```shared_camera``` to True when you know that the input frames were captured by the same camera and the camera focal length did not significantly change during the capture. This assumption is usually valid for images extracted from a video.



## Testing 

We are still preparing the testing script for VGGSfM v2. However, you can use our code for VGGSfM v1.1 to reproduce our benchmark results in the paper. Please refer to the branch ```v1.1```.


## Acknowledgement

We are highly inspired by [colmap](https://github.com/colmap/colmap), [pycolmap](https://github.com/colmap/pycolmap), [posediffusion](https://github.com/facebookresearch/PoseDiffusion), [cotracker](https://github.com/facebookresearch/co-tracker), and [kornia](https://github.com/kornia/kornia).


## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.


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