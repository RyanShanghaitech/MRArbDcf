from numpy import *
from numpy.typing import *
from numpy.linalg import norm

def cropDcf(arrDcf:NDArray, arrK:NDArray, nPix:int, nNyq:int=2) -> NDArray: 
    '''
    Null the outmost DCF points, useful for some methods including Voronoi, iterative methods.
    
    Args:
        arrDcf: array of DCF, shape: `[Nk,]`
        arrK: array of trajectory, shape: `[Nk,Nd]`, range: `[-0.5,0.5]`
        nPix: number of pixel, for calc the Nyquist Interval
        nNqy: DCF data within how long distance to the edge will be removed
    
    Returns:
        Cropped DCF.
    '''
    arrRho = norm(arrK, axis=-1)
    arrDcf[arrRho>0.5-nNyq/nPix] = 0
    return arrDcf


def normDcf(arrDcf:NDArray, nAx:int) -> NDArray:
    """
    Deprecated. Make the sum of DCF equivalent to the area of the inner circle of k-space in 2D, or a inner ball of k-space in 3D. Inspired by Voronoi method.

    Args:
        arrDcf: Input DCF.
        nAx: Number of dimensions.

    Returns:
        Normalized DCF.
    """
    arrDcf = arrDcf/abs(arrDcf).sum()
    if nAx == 2: arrDcf *= pi/4
    if nAx == 3: arrDcf *= pi/6
    return arrDcf

def normImg(arrData:NDArray, method:str="mean0_std1", mskFov:NDArray|None=None) -> NDArray:
    """
    Normalize a image using different methods.

    Args:
        arrData: Input image.
        method: Normalization strategy. Support values:
            "mean0_std1": Zero mean and unit standard deviation.
            "mean1_std1": Unit standard deviation with mean shifted by 1.
            "mean": Unit mean.
            "std": Unit deviation.
            "max": Unit max magnitude.
            "ene": Unit L2 norm (energy).
        mskFov: Optional mask for the Field of View. If provided, statistics (mean, std, etc.) are calculated only using pixels within the mask.

    Returns:
        Normalized image.
    """
    arrData = arrData.copy()
    
    if mskFov is None:
        _arrData = arrData
    else:
        _arrData = arrData[mskFov]
    vmean = _arrData.mean()
    vstd = _arrData.std()
    vmax = abs(_arrData).max()
    vene = norm(_arrData.flatten())
        
    if method=="mean0_std1":
        arrData -= vmean
        arrData /= vstd
    elif method=="mean1_std1":
        arrData -= vmean
        arrData /= vstd
        arrData += vmean/abs(vmean)
    elif method=="mean":
        arrData /= abs(vmean)
    elif method=="std":
        arrData /= vstd
    elif method=="max":
        arrData /= vmax
    elif method=="ene":
        arrData /= vene
    else:
        raise NotImplementedError("")
    return arrData