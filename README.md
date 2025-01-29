# DANS
# Deep Attention Network for Single Image Super Resolution
This repository is for DANS introduced in the following paper "Deep Attention Network for Single Image Super Resolution", IEEE Access, [[Link]](https://ieeexplore.ieee.org/document/10210219) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch >= 1.1.0) with V100 GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

This repository contains the implementation of DANS, a novel Deep Attention Network for Single Image Super-Resolution (SISR). DANS leverages a U-Net-based encoder-decoder structure, non-local sparse attention mechanisms, and inception blocks to enhance the reconstruction of high-resolution images from low-resolution inputs. The model achieves state-of-the-art performance in terms of quantitative metrics (PSNR, SSIM) and computational efficiency across multiple benchmark datasets.

![DANS](DANS_architecture.png)

Key features:

Non-local Sparse Attention: Captures long-range dependencies to improve contextual learning.
Inception Blocks: Enables multi-scale feature representation for enhanced detail reconstruction.
Efficient Architecture: Reduces computational costs with depth-wise separable convolutions and skip connections.
This work has applications in security, medical imaging, remote sensing, and more. 
