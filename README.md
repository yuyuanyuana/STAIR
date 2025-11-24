# Code for STAIR: spatial transcriptomic alignment, integration, and 3D reconstruction.

STAIR is a deep learning-based algorithm for spatial transcriptomic alignment, integration, and 3D reconstruction. STAIR utilizes heterogeneous graph attention network with spot-level and slice-level attention mechanisms to learn and integrate spatial features and obtain consistent spatial region divisions across slices. These results are then used to completes the 2D alignment. Unlike previous methods relying on known slice distance or known 3D coordinates, STAIR requires only ST data as input and infers the relative positioning of slices along z-axis in a completely unsupervised manner. In addition, STAIR seamlessly integrates new slices into the existing 3D atlas, effectively expanding the reference 3D atlas. 


## Installation

We provide two ready-to-use conda environment configuration files for different Python and CUDA versions:

	•	Python 3.10 + CUDA 11.7: environment-python3.10.yaml
	•	Python 3.12 + CUDA 12.4: environment-python3.12.yaml

You can create and activate the desired environment as follows:

```bash
# Example: using Python 3.10 + CUDA 11.7
# conda env create -f environment-python3.10.yaml

# Example: using Python 3.12 + CUDA 12.4
conda env create -f environment-python3.12.yaml
conda activate STAIR-env
```

### 1. Install PyTorch

For CUDA 12.4 (Python 3.12):
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
For CUDA 11.7 (Python 3.10):
```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
### 2. Install PyTorch Geometric (PyG)

For CUDA 12.4 (Torch 2.6.0):
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric
```
For CUDA 11.7 (Torch 1.13.1):
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch_geometric
```

### 3. Install STAIR-tools
Finally, install the latest version of STAIR-tools via pip:

```bash
pip install https://github.com/yuyuanyuana/STAIR/releases/download/1.3.1/STAIR_tools-1.3.1-py3-none-any.whl 
```

## Tutorial
For the specific usage of STAIR, please refer to the [Tutorial](https://stair-tutorial.readthedocs.io/en/latest/STAIR-Tutorial.html). The datasets in the tutorial can be downloaded from [zenodo](https://zenodo.org/records/11084262).

## Reference

If you use STAIR-tools in your work, please cite:

Yuanyuan Yu, Zhi Xie. Spatial Transcriptomic Alignment, Integration, and 3D Reconstruction by STAIR, 08 February 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3939678/v1]

## License

This project is released under the MIT License. See the `LICENSE` file for details.

