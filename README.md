# geodesic-shooting - surface registration in PyTorch using auto differentiation

## Overview
This package performs geodesic shooting [1] between triangulated surfaces using a currents cost function [2] (not requiring corresponding surface vertices) using PyTorch's auto differentiation, for the purpose of easily modifying the algorithm. The currents matching cost function is implemented in PyTorch based on https://github.com/saurabh-jain/registration/blob/master/py-lddmm/surfaces.py.

See ./examples/ directory for example Jupyter notebooks.

## Features
This package performs the following:
* Rigid/affine surface-to-surface matching
* Geodesic shooting between surfaces

## Quick-start guide
Dependencies: See requirements.txt
Download the package with: git clone github.com/brianlee324/geodesic-shooting.git

In a Python session, generate a pair of Nx3 and Mx3 arrays representing faces and vertices of a triangulated surface. The second pair does not require corresponding numbers of vertices.

### Example Images: Transport of points
<p float="left">
  <img src="/data/cube_arrows.png" height="200" />
  <img src="/data/cube_expand.gif" height="200" /> 
</p>

### Example Images: Transport of vectors
<p float="left">
  <img src="/data/cube_to_sphere.gif" height="200" />
  <img src="/data/cube_arrows_transport.gif" height="200" /> 
  <img src="/data/transport_target.png" height="200" /> 
</p>

### Example Images: Surface expansion
<p float="left">
  <img src="/data/pyramid_vectors.gif" height="200" />
  <img src="/data/pyramid_jacobian.gif" height="200" /> 
</p>

### Example: Nonlinear Surface Matching
```python
niter = 750 # number of iterations
nT = [0,0.5,1] # time steps
sigma = 0.4 # Gaussian variance, representing influence of cost function kernel
reg_weight = 0.4 # regularization weight
vertices_deformed, params = register_nonlinear(faces_template,vertices_template,faces_target,vertices_target,niter=niter,nT=nT,sigma=sigma,reg_weight=reg_weight)
```

## References
1. Miller MI, Trouvé A, Younes L. Geodesic Shooting for Computational Anatomy. J Math Imaging Vis. 2006 Jan 31;24(2):209-228. doi: 10.1007/s10851-005-3624-0. PMID: 20613972; PMCID: PMC2897162.
2. Vaillant M, Glaunès J. Surface matching via currents. Inf Process Med Imaging. 2005;19:381-92. doi: 10.1007/11505730_32. PMID: 17354711.

