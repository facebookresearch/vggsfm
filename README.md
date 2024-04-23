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

Once the dataset is downloaded and extracted, you'll need to specify its path in the ```IMC_DIR``` field in the ./cfgs/test.yaml configuration file.

Next, you'll need to download the model checkpoint from [Google Drive](https://drive.google.com/file/d/1eSJDMj7tWsM2FzVZAiWYSpvm5bSUIZwq/view?usp=sharing).


After downloading the model checkpoint, specify its path in the ```resume_ckpt``` field in ./cfgs/test.yaml.


### 2. Run Testing

```bash
python test.py
```

You should see something like:

```bash
----------------------------------------------------------------------------------------------------
On the IMC dataset
Auc_5  (%): 71.90304909560726
Auc_10 (%): 80.58532299741603
Auc_30 (%): 90.00782084409991
----------------------------------------------------------------------------------------------------
```

If your machine support ```torch.bfloat16```, you are welcome to enable the ```use_bf16``` option in the configuration file. Our model was trained using bf16 and the testing performance is nearly identical when using bf16.

Typically, running our model on a 25-frame IMC scene takes approximately 40 seconds. If you're looking to save time, you can adjust the ```query_frame_num``` to 1. This adjustment reduces the inference time to roughly 15 seconds, while maintaining a comparable performance.



## Acknowledgement

We are highly inspired by [colmap](https://github.com/colmap/colmap), [pycolmap](https://github.com/colmap/pycolmap), [cotracker](https://github.com/facebookresearch/co-tracker), and [kornia](https://github.com/kornia/kornia).


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
