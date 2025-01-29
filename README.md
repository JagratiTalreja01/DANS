git add Figures/*
git commit -m "Add missing images"
git push

# DANS
# Deep Attention Network for Single Image Super Resolution
This repository is for DANS introduced in the following paper "Deep Attention Network for Single Image Super Resolution", IEEE Access, [[Link]](https://ieeexplore.ieee.org/document/10210219) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch >= 1.1.0) with NVIDIA GeForce GTX 2080ti GPU. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

This repository contains the implementation of DANS, a novel Deep Attention Network for Single Image Super-Resolution (SISR). DANS leverages a U-Net-based encoder-decoder structure, non-local sparse attention mechanisms, and inception blocks to enhance the reconstruction of high-resolution images from low-resolution inputs. The model achieves state-of-the-art performance in terms of quantitative metrics (PSNR, SSIM) and computational efficiency across multiple benchmark datasets.

![DANS](./Figures/DANS_architecture.png)

Key features:

Non-local Sparse Attention: Captures long-range dependencies to improve contextual learning.
Inception Blocks: Enables multi-scale feature representation for enhanced detail reconstruction.
Efficient Architecture: Reduces computational costs with depth-wise separable convolutions and skip connections.

Deep Attention Network for Single Image Super Resolution.

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. 

For more information, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

Cd to 'src', run the following script to train models.

 **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model dans --scale 2 --patch_size 96 --save Dans_x2 --data_train DIV2K
    ```
## Test
### Quick start
1. Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)


Cd to 'src', run the following scripts.

 **Example command is in the file 'demo.sh'.**

    ```bash
    
    # Example X2 SR
    python main.py --dir_data ../../ --model dans  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only 
    ```

## Results
### Visual Patches

![BSD100x4](./Figures/BSDx4.png)

![BSD100x8](./Figures/BSDx8.png)

![Urnan100x8](./Figures/Urbanx8.png)

![Manga109x8](./Figures/Mangax8.png)

### Quantitative Results

![Number of Parameters](./Figures/Parameters.png)

![Execution Time](./Figures/Execution_Time.png)

![Space Complexity](./Figures/space_complexity.png)

![Time Complexity](./Figures/Time_complexity.png)

![Flops](./Figures/Flops.png)

For more Quantitative Results please read the paper [[Link]](https://ieeexplore.ieee.org/document/10210219)

## Citation
If you find the code helpful in your research or work, please cite the following papers.
```
@ARTICLE{10210219,
  author={Talreja, Jagrati and Aramvith, Supavadee and Onoye, Takao},
  journal={IEEE Access}, 
  title={DANS: Deep Attention Network for Single Image Super-Resolution}, 
  year={2023},
  volume={11},
  number={},
  pages={84379-84397},
  keywords={Superresolution;Computational modeling;Image reconstruction;Network architecture;Image resolution;Neural networks;Representation learning;Image super-resolution;inception blocks;non-local sparse attention;U-Net},
  doi={10.1109/ACCESS.2023.3302692}}

```

## Acknowledgements
This code is built on [NLSN (PyTorch)](https://github.com/HarukiYqM/Non-Local-Sparse-Attention) and [edsr-pytorch](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
