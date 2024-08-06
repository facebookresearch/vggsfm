from setuptools import setup, find_packages

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vggsfm",
    version="2.0.0",
    author="Jianyuan Wang",
    description="A package for the VGGSfM project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/vggsfm.git",
    packages=find_packages(),
    python_requires=">=3.10",
    package_data={"vggsfm": ["cfgs/*.yaml"]},
    install_requires=[
        "torch==2.1.0",
        "torchvision",
        "fvcore",
        "iopath",
        "pytorch3d",
        "hydra-core==1.3.2",
        "omegaconf",
        "opencv-python",
        "einops",
        "visdom",
        "tqdm",
        "accelerate==0.24.0",
        "numpy==1.26.3",
        "pycolmap==0.6.1",
        "poselib==2.0.2",
    ],
    entry_points={"console_scripts": ["vggsfm-demo=vggsfm_demo:demo_fn"]},
)
