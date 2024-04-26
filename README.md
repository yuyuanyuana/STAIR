![image](https://github.com/yuyuanyuana/STAIR/assets/53144397/10c9b388-6870-4b58-b12c-7db313362a0f)# STAIR
Spatial Transcriptomic Alignment, Integration, and de novo 3D Reconstruction.

STAIR is a deep learning-based algorithm for Spatial Transcriptomic Alignment, Integration, and de novo 3D Reconstruction. STAIR utilizes heterogeneous graph attention network with spot-level and slice-level attention mechanisms to learn and integrate spatial features and obtain consistent spatial region divisions across slices. These results are then used to completes the 2D alignment. Unlike previous methods relying on known slice distance or known 3D coordinates, STAIR requires only ST data as input and infers the relative positioning of slices along z-axis in a completely unsupervised manner. In addition, STAIR seamlessly integrates new slices into the existing 3D atlas, effectively expanding the reference 3D atlas. 




## Installation

First, create and activate a virtual environment named STAIR-test:

```
conda create -n STAIR-test python=3.9.13
conda activate STAIR-test
```
Next, install the appropriate versions of PyTorch and PyG for your device. We will follow the PyG tutorial and use CUDA 11.3 as an example:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Confirm that the installed PyTorch version is correct:
```
(STAIR-test)[yuyuanyuan@mu03 ~]$ python -c "import torch; print(torch.version)"
1.12.1+cu113
(STAIR-test)[yuyuanyuan@mu03 ~]$ python -c "import torch; print(torch.version.cuda)"
11.3
```
Then, install the corresponding PyG and its related packages:
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
```

Finally, install the latest version of STAIR-tools via pip:

```python
   pip install STAIR-tools
```
STAIR can be used in python:

```python
   import STAIR
```
## Tutorial
