from numpy import *
from numpy.linalg import norm

# null the outmost DCF data, useful for both Voronoi, Iterative, FFD method
def cropDcf(arrDcf:ndarray, arrK:ndarray, nPix:int, nNyq:int=2) -> ndarray: 
    '''
    `arrDcf`: array of DCF, shape: `[Nk,]`
    `arrK`: array of trajectory, shape: `[Nk,Nd]`, range: `[-0.5,0.5]`
    `nPix`: number of pixel, for calc the Nyquist Interval
    `nNqy`: DCF data within how long distance to the edge will be removed
    '''
    arrRho = norm(arrK, axis=-1)
    arrDcf[arrRho>0.5-nNyq/nPix] = 0
    return arrDcf


def normDcf(arrDcf:ndarray, nAx:int) -> ndarray:
    # arrDcf = abs(arrDcf).astype(arrDcf.dtype)
    arrDcf = arrDcf/arrDcf.sum()
    if nAx == 2: arrDcf *= pi/4
    if nAx == 3: arrDcf *= pi/6
    return arrDcf

def normImg(arrData:ndarray, method:str="mean_std") -> ndarray:
    if method=="mean_std":
        vmean = arrData.mean()
        arrData -= vmean
        arrData /= arrData.std()
        arrData += vmean/abs(vmean)
    elif method=="mean":
        arrData /= abs(arrData.mean())
    elif method=="std":
        arrData /= arrData.std()
    elif method=="max":
        arrData /= abs(arrData).max()
    elif method=="pow":
        arrData /= norm(arrData.flatten())
    else:
        raise NotImplementedError("")
    return arrData