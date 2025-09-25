# STAIR: Spatial Transcriptomic Alignment, Integration, and 3D Reconstruction.

STAIR is a deep learning-based algorithm for Spatial Transcriptomic Alignment, Integration, and 3D Reconstruction. STAIR utilizes heterogeneous graph attention network with spot-level and slice-level attention mechanisms to learn and integrate spatial features and obtain consistent spatial region divisions across slices. These results are then used to completes the 2D alignment. Unlike previous methods relying on known slice distance or known 3D coordinates, STAIR requires only ST data as input and infers the relative positioning of slices along z-axis in a completely unsupervised manner. In addition, STAIR seamlessly integrates new slices into the existing 3D atlas, effectively expanding the reference 3D atlas. 


## Installation

First, create and activate a virtual environment named STAIR-env:

```bash
conda env create -f environment.yaml
conda activate STAIR-env
```
Then, install the appropriate versions of PyTorch and PyG for your device. We will follow the [PyG tutorial](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and use CUDA 11.7 as an example:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Next, install the corresponding PyG and its related packages:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch_geometric
```

Finally, install the latest version of STAIR-tools via pip:

```bash
pip install https://github.com/yuyuanyuana/STAIR/releases/download/1.3.1/STAIR_tools-1.3.1-py3-none-any.whl 
```

## Tutorial
For the specific usage of STAIR, please refer to the [Tutorial](https://stair-tutorial.readthedocs.io/en/latest/STAIR-Tutorial.html). The datasets in the tutorial can be downloaded from [zenodo](https://zenodo.org/records/11084262).

## Reference

If you use STAIR-tools in your work, please cite:

Yuanyuan Yu, Zhi Xie. Spatial Transcriptomic Alignment, Integration, and 3D Reconstruction by STAIR, 08 February 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3939678/v1]
