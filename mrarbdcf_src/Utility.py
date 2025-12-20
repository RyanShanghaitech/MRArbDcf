from numpy import *
from numpy.typing import *
from numpy.linalg import norm

# null the outmost DCF data, useful for both Voronoi, Iterative, FFD method
def cropDcf(arrDcf:NDArray, arrK:NDArray, nPix:int, nNyq:int=2) -> NDArray: 
    '''
    `arrDcf`: array of DCF, shape: `[Nk,]`
    `arrK`: array of trajectory, shape: `[Nk,Nd]`, range: `[-0.5,0.5]`
    `nPix`: number of pixel, for calc the Nyquist Interval
    `nNqy`: DCF data within how long distance to the edge will be removed
    '''
    arrRho = norm(arrK, axis=-1)
    arrDcf[arrRho>0.5-nNyq/nPix] = 0
    return arrDcf


def normDcf(arrDcf:NDArray, nAx:int) -> NDArray:
    arrDcf = arrDcf/abs(arrDcf).sum()
    if nAx == 2: arrDcf *= pi/4
    if nAx == 3: arrDcf *= pi/6
    return arrDcf

def normImg(arrData:NDArray, method:str="mean0_std1", mskFov:NDArray|None=None) -> NDArray:
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