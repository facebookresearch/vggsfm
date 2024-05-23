# VGGSfM: Visual Geometry Grounded Deep Structure From Motion


![Teaser](https://raw.githubusercontent.com/vggsfm/vggsfm.github.io/main/resources/vggsfm_teaser.gif)

**[Meta AI Research, GenAI](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**


[Jianyuan Wang](https://jytime.github.io/), [Nikita Karaev](https://nikitakaraevv.github.io/), [Christian Rupprecht](https://chrirupp.github.io/), [David Novotny](https://d-novotny.github.io/)



<p 
dir="auto">[<a href="https://arxiv.org/pdf/2312.04563.pdf" rel="nofollow">Paper</a>]
[<a href="https://vggsfm.github.io/" rel="nofollow">Project Page</a>] 
[Version 1.1]
</p> 


**Updates:**

- [Apr 23, 2024] Release the code and model weight for VGGSfM v1.1.




## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.10, PyTorch 2.1, and CUDA 12.1.

```.bash
source install.sh
```

## Testing on IMC

### 1. Download Dataset and Model

To get started, you'll need to download the IMC dataset. You can do this by running the following commands in your terminal:

```bash
wget https://www.cs.ubc.ca/research/kmyi_data/imc2021-public/imc-2021-test-gt-phototourism.tar.gz

tar -xzvf imc-2021-test-gt-phototourism.tar.gz
```

Once the dataset is downloaded and extracted, you'll need to specify its path in the ```IMC_DIR``` field in the ./cfgs/test.yaml configuration file or give it as an input such as ```python test.py IMC_DIR=YOUR/PATH```.

Next, you'll need to download the model checkpoint of [v1.1](https://drive.google.com/file/d/1eSJDMj7tWsM2FzVZAiWYSpvm5bSUIZwq/view?usp=sharing) for testing or [v1.2](https://drive.google.com/file/d/1WEGN0RpDqynOnxI18hXlxzeQTqdX5lkA/view?usp=sharing) for demo. If you are interested in comparing different methods following the standard setting (as in our paper), v1.1 should be the one for fair comparison. If you want to apply our method for demo/downstream applications, it is quite likely v1.2 should be the better choice.


After downloading the model checkpoint, specify its path in the ```resume_ckpt``` field in ./cfgs/test.yaml.


### 2. Run Testing

```bash
python test.py
```

When it finishes (it would take several hours to complete the testing on the whole IMC dataset), you should see something like:

```bash
----------------------------------------------------------------------------------------------------
On the IMC dataset (query_frame_num=3)
Auc_3  (%): 64.74418604651163
Auc_5  (%): 72.20720930232558
Auc_10 (%): 80.98441860465115
----------------------------------------------------------------------------------------------------
```

If your machine support ```torch.bfloat16```, you are welcome to enable the ```use_bf16``` option in the configuration file or by ```python test.py use_bf16=True```. Our model was trained using bf16 and the testing performance is nearly identical when using bf16.

Typically, running our model on a 25-frame IMC scene takes approximately 40 seconds. If you're looking to save time, you can adjust the ```query_frame_num``` to 1. This adjustment reduces the inference time to roughly 15 seconds, while maintaining a comparable performance.


```bash
----------------------------------------------------------------------------------------------------
On the IMC dataset (query_frame_num=1)
Auc_3  (%): 61.99207579672695
Auc_5  (%): 69.78997416020671
Auc_10 (%): 78.88826873385013
----------------------------------------------------------------------------------------------------
```



If want to run the model on your own data, please check the ```run_one_scene``` function in ```test.py```. We are also going to provide a demo file for it very soon. The default output cameras of ```run_one_scene``` follows the PyTorch3D convention. You can set ```return_in_pt3d=False``` to let it return in COLMAP convention. 


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
