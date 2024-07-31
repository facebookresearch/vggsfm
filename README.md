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

- [Jul 28, 2024]
  - Added support for filtering out dynamic objects using masks. We will add an example soon but you can check `demo_loader.py` for a quick view.
  - Added support for `visual_dense_point_cloud`.
  - Added support for `visual_query_points`.
  - Made `visual_depths` optional.

- [Jul 10, 2024] Now we support exporting dense depth maps!

- Happy to share we were ranked 1st 🥇 in the CVPR24 IMC Challenge regarding camera pose (Rot&Trans) estimation.



## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.10, PyTorch 2.1, and CUDA 12.1.

```.bash
source install.sh
```

This script installs official ```pytorch3d```, ```accelerate```, ```lightglue```, ```pycolmap```, ```poselib```, and ```visdom```. If you cannot install ```pytorch3d``` on your machine, feel free to comment the line, because now we only use it during visualization (i.e., ```cfg.visualize=True```). 


## Demo 

### 1. Download Model
The checkpoint will be automatically downloaded from [Hugging Face](https://huggingface.co/facebook/VGGSfM/tree/main). You can also manually download it from [Hugging Face](https://huggingface.co/facebook/VGGSfM/blob/main/vggsfm_v2_0_0.bin) and [Google Drive](https://drive.google.com/file/d/163bHiqeTJhQ2_UnihRNPRA4Y9X8-gZ1-/view?usp=sharing). If you prefer to specify the checkpoint path manually, set the path in ```resume_ckpt``` and set ```auto_download_ckpt``` to False.


### 2. Run the Demo 

Now time to enjoy your 3D reconstruction! You can start by our provided examples, such as:

```bash

python demo.py SCENE_DIR=examples/statue shared_camera=True query_method=sp+sift

python demo.py SCENE_DIR=examples/kitchen query_method=aliked

python demo.py SCENE_DIR=examples/british_museum max_query_pts=4096 
```

All default settings for the flags are specified in `cfgs/demo.yaml`. You can adjust these flags as needed, such as reducing ```max_query_pts``` to lower GPU memory usage. To enforce a shared camera model for a scene, set ```shared_camera=True```. To use query points from different methods, set ```query_method``` to ```sp```, ```sift```, ```aliked```, or any combination like ```sp+sift```.

To run reconstruction on a scene with ```100``` frames on a ```32 GB``` GPU, you can start from the setting below:

```bash
python demo.py SCENE_DIR=TO/YOUR/PATH max_query_pts=1024 query_frame_num=6
```


The reconstruction result (camera parameters and 3D points) will be automatically saved in the COLMAP format at ```output/seq_name```. You can use the [COLMAP GUI](https://colmap.github.io/gui.html) to view them. 

If you want to visualize it more easily, we provide an approach supported by [visdom](https://github.com/fossasia/visdom). To begin using Visdom, start the server by entering ```visdom``` in the command line. Once the server is running, access Visdom by navigating to ```http://localhost:8097``` in your web browser. Now every reconstruction will be visualized and saved to the Visdom server by enabling ```visualize=True```:

```bash
python demo.py visualize=True ...(other flags)
```

By doing so, you should see an interface such as:

![UI](assets/ui.png)

[Beta] If you want to visualize the 2D reprojections of the reconstructed 3D points, set ```make_reproj_video``` to True. This will generate a video named ```reproj.mp4``` under ```SCENE_DIR```. For example:

<img src="https://github.com/vggsfm/vggsfm.github.io/blob/main/resources/reproj.gif" width="500" alt="reproj">



### 3. Try your own data

You only need to specify the address of your data, such as:

```bash
python demo.py SCENE_DIR=examples/YOUR_FOLDER ...(other flags)
```

Please ensure that the images are stored in ```YOUR_FOLDER/images```. This folder should contain only the images. Check the ```examples``` folder for the desired data structure.


Have fun and feel free to create an issue if you meet any problem. SfM is always about corner/hard cases. I am happy to help. If you prefer not to share your images publicly, please send them to me by email.


### 4. Dense depth maps (Beta)

We support extracting dense depth maps with the help of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Bascially, we align the dense depth prediction from Depth-Anything-V2 using the sparse SfM point cloud predicted by VGGSfM. To enable this, please first git clone Depth-Anything-V2 and install scikit-learn:

```bash
pip install scikit-learn
git clone git@github.com:DepthAnything/Depth-Anything-V2.git dependency/depth_any_v2
```

Then, you just need to set ```dense_depth=True``` when running demo.py. Depth maps will be saved in the ```depths``` folder under ```cfg.SCENE_DIR```, using the COLMAP format (e.g., ```*.bin```). To visualize 2D depth maps, set ```visual_depths=True```. To visualize the dense point cloud (unprojected dense depth maps) in Visdom, set ```visual_dense_point_cloud=True``` (note it may take seconds to open the Visdom page when there are too many points).



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
@article{wang2023vggsfm,
  title={VGGSfM: Visual Geometry Grounded Deep Structure From Motion},
  author={Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
  journal={arXiv preprint arXiv:2312.04563},
  year={2023}
}
