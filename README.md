# Magnetic Resonance Arbitrary Density Compensation Function (MRArbDcf, MAD)

## Introduction
This is the code repository for paper "Sampling Density Compensation using Fast Fourier Deconvolution" [1]. To fill the gap that the previous iterative DCF methods are slow (especially in 3D), this package provides a non-iterative method based on fast Fourier deconvolution. This package supports both CPU and GPU and can derive DCF for a trajectory designed for a 256³ matrix size in 30 seconds on a multi-core CPU or 10 seconds on a GPU.

## Installation
You can install this package either via pip:
```bash
$ pip install mrarbdcf
```
or offline
```bash
$ bash install.bash
```

Optionally, to enable CUDA acceleration in this package, you need to install `cufinufft` and `cupy`. It's recommended to read their official installation guides before installing - pip is not the best source for installation because some dependencies are only available from conda.

## Usage
For tutorials, you can find examples in the `example` folder. Most functions are well-commented in the Google style. We plan to release the documents on `readthedocs.org` in the future.

## Acknowledgement
FINUFFT [2,3] and CUFINUFFT [4] are used as NUFFT operators in this package. We thank the authors for their contributions to create such fast NUFFT libraries.

## Reference
[1] Luo R, Hu P, Qi H. Sampling Density Compensation using Fast Fourier Deconvolution [Internet]. arXiv; 2025 [cited 2025 Oct 17]. Available from: http://arxiv.org/abs/2510.14873

[2] Barnett AH, Magland J, af Klinteberg L. A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel. SIAM J Sci Comput. 2019 Jan;41(5):C479–504. 

[3] Barnett AH. Aliasing error of the exp(β√(1-z²)) kernel in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis. 2021 Mar 1;51:1–16. 

[4] Shih Y hsuan, Wright G, Anden J, Blaschke J, Barnett AH. cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 2021 June;688–97. 
