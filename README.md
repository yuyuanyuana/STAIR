# STAIR
Spatial Transcriptomic Alignment, Integration, and de novo 3D Reconstruction.

STAIR is a deep learning-based algorithm for Spatial Transcriptomic Alignment, Integration, and de novo 3D Reconstruction. STAIR employs a heterogeneous graph attention network16 to learn adaptive spatial embeddings across slices. It constructs a unified graph with all spots as nodes, attributed by their original slices. The spot-level and slice-level attention mechanism then enable flexible aggregation of context from neighboring spots based on data-driven measures of both spot and slice similarity. The slice-level attention also captures high-order semantic information crucial for reconstructing z-axis coordinates of slices. Subsequently, STAIR utilizes these spatial embeddings to calibrate the x- and y- axis guided by the reconstructed slice order, thereby completing the 3D reconstruction. Additionally, STAIR integrates new slices into the existing 3D atlas, expanding and updating the reference 3D atlas.



## Installation

Before installing STAIR, please ensure that the software dependencies are installed.

```python
   scanpy
   pytorch
   torch-geometric
```
STAIR can be downloaded via pip:

```python
   pip install STAIR-tools
```
Then STAIR can be used in python:

```python
   import STAIR
```
## Tutorial
