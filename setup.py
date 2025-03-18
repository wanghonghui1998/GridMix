from setuptools import find_packages, setup

setup(
    name="marble",
    version="1.0.0",
    description="Package for EXPLORING SPATIAL MODULATION FOR NEURAL FIELDS IN PDE MODELING",
    author="LeapLabTHU",
    author_email="wanghh20@mails.tsinghua.edu.cn",
    install_requires=[
        "einops",
        "hydra-core",
        "wandb",
        "torch",
        "pandas",
        "matplotlib",
        "xarray",
        "scipy",
        "h5py",
        "timm",
        "torchdiffeq",
    ],
    package_dir={"coral": "coral"},
    packages=find_packages(),
)
