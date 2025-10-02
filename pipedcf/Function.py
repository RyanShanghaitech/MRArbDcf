from .Type import OpDir
from .Utility import grid
from numpy import *
from scipy.special import iv
from copy import deepcopy
from scipy.special import j1

from matplotlib.pyplot import figure, plot, show

def zwarts_kernel(r_squared, rfp_sq):
    """
    Python implementation of the _poly_sdc_kern_0lobes function from the C code.
    Accepts squared radius to avoid sqrt.
    """
    # Polynomial coefficients from the C code
    poly = [
        -1.1469041640943728E-13, 
         8.5313956268885989E-11, 
         1.3282009203652969E-08, 
        -1.7986635886194154E-05, 
         3.4511129626832091E-05, 
         0.99992359966186584
    ]
    
    # Scaling constants from the C code
    SPECTRAL_LEN = 25600.0
    FOV = 63.0
    
    # The polynomial is only valid up to the radius-fov-product
    if r_squared >= rfp_sq:
        return 0.0

    r = sqrt(r_squared)
    x = SPECTRAL_LEN * r / FOV
    
    # Evaluate the polynomial
    out = poly[5] # Zeroth order
    out += poly[4] * x
    out += poly[3] * x**2
    out += poly[2] * x**3
    out += poly[1] * x**4
    out += poly[0] * x**5
    
    return out if out > 0.0 else 0.0

def pipe(_arrK:ndarray, nPix:int, nIt:int=25, ovKer:int=101, ovKsp=2.1, fOptKer:bool=True) -> ndarray:
    # avoid modifying the passed-in value
    arrK = deepcopy(_arrK)

    # normalize `arrK`
    arrK = arrK*(0.5/abs(arrK).max())

    nK, nDim = arrK.shape

    if fOptKer:
        # if nDim==2:
        #     nPix_Ker = int(nPix*0.00954) # cut off at main-lobe
        #     arrRho = linspace(0, 0.5*nPix_Ker/nPix, 100000)
        #     arrRho[0] += 1e-3
        #     arrHalf1dWind = (nPix*j1(nPix*pi*arrRho)) / (2*arrRho)
        #     arrHalf1dWind **= 2
        #     arrHalf1dWind[0] = (nPix**2 * pi / 4)**2
        # elif nDim==3:
        #     nPix_Ker = int(nPix*0.011175) # cut off at main-lobe
        #     arrRho = linspace(0, 0.5*nPix_Ker/nPix, 500)
        #     arrRho[0] += 1e-3
        #     arrHalf1dWind = (sin(pi*nPix*arrRho) - pi*nPix*arrRho*cos(pi*nPix*arrRho)) / (2 * pi**2 * arrRho**3)
        #     arrHalf1dWind **= 2
        #     arrHalf1dWind[0] = (pi * nPix**3 / 6)**2
            
        # kwart's magic kernel, good but I did't figure out why
        RADIUS_FOV_PRODUCT = 0.96960938
        nPix_Ker = RADIUS_FOV_PRODUCT * 2
        rfp_sq = RADIUS_FOV_PRODUCT**2
        arrR_sq = linspace(0, rfp_sq, 5000)
        arrHalf1dWind = array([zwarts_kernel(r_sq, rfp_sq) for r_sq in arrR_sq])
    
    else: # parameter from Pipe's
        ordBessel = 8
        nPix_Ker = 5
        beta = 16
        arrHalf1dWind = iv(ordBessel, beta*sqrt(1-linspace(0,1,nPix_Ker*100,1)**2))/iv(ordBessel, beta)
    
    # arrHalf1dWind = sqrt(arrHalf1dWind) # Zwart recommend to do this because dual-step Cartesian gridding is used instead of a single non-Cartesian method

    nPix_Ksp = int(nPix*ovKsp)
    nPix_Ker = int(nPix_Ker*ovKsp)

    arrDcf = ones([arrK.shape[0]], dtype=complex128)
    arrKsp = ones([(nPix_Ksp+nPix_Ker//2*2+2) for _ in range(nDim)], dtype=complex128)
    for iIt in range(nIt):
        print(f"{iIt}/{nIt}")
        arrKsp = grid(OpDir.FOR, arrDcf, arrK, arrKsp, nPix_Ksp, nPix_Ker, ovKer, arrHalf1dWind)
        arrDcf /= grid(OpDir.BAC, arrDcf, arrK, arrKsp, nPix_Ksp, nPix_Ker, ovKer, arrHalf1dWind)

    return abs(arrDcf)