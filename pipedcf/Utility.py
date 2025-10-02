from numpy import *
from matplotlib.pyplot import *
from .Type import OpDir
from copy import deepcopy

class Grid:
    def __init__(self):
        self.arrS = None
        self.arrK = None
        self.arrGrid = None
        self.arrHalf1dWind = None
        
        pass

    def init(eOpDir:OpDir, _arrS:ndarray, _arrK:ndarray, _arrGrid:ndarray, nPix_Ksp:int, nPix_Ker:int, ov:int, _arrHalf1dWind:ndarray):
        pass

    def run():
        pass
    
def grid(eOpDir:OpDir, _arrS:ndarray, _arrK:ndarray, _arrGrid:ndarray, nPix_Ksp:int, nPix_Ker:int, ov:int, _arrHalf1dWind:ndarray) -> ndarray:
    """
    # note
    this function accept `arrK` in range [-0.5,0.5), but scaled to [-nPix/2,nPix/2) internally
    """
    arrS = deepcopy(_arrS)
    arrK = deepcopy(_arrK)
    arrGrid = deepcopy(_arrGrid)
    arrHalf1dWind = deepcopy(_arrHalf1dWind)

    if arrGrid.shape[0] < int(nPix_Ksp + nPix_Ker//2*2 + 1): # +1: in case k=0.5
        raise ValueError("size of `arrGrid` is not big enough.")

    if ov%2==0: # ensure oversampling factor is odd
        ov+=1
        raise Warning(f"ov changed to {ov} to ensure odd.")

    nPix_Grid = arrGrid.shape[0]
    nPix_Ksp = int(nPix_Ksp)
    nPix_Ker = int(nPix_Ker) # kernel width
    nPix_KerOv = int(nPix_Ker*ov) # kernel width after ov
    nPix_Wind = arrHalf1dWind.size
    nK, nDim = arrK.shape

    # convert /pix to /fov
    arrK *= nPix_Ksp # -Npix/2 ~ Npix/2

    # generate oversampled kernel
    tupKerK = meshgrid\
    (
        *(linspace(-1, 1, nPix_KerOv, 1) for _ in range(nDim)),
        indexing="ij"
    )[::-1]
    arrKerK = array(tupKerK).transpose(*arange(1,nDim+1), 0)
    arrKerR = sqrt(sum((arrKerK)**2, axis=-1))
    arrKer = interp(arrKerR, linspace(0,1,nPix_Wind,1), arrHalf1dWind)

    # generate idx in kernel when on-grid
    tupKerI0 = meshgrid\
    (
        *(arange(nPix_Ker) for _ in range(nDim)),
        indexing="ij"
    )[::-1]
    arrKerI0 = array(tupKerI0).transpose(*arange(1,nDim+1), 0)
    arrKerI0 *= ov
    arrKerI0 += ov//2

    # grid
    if eOpDir==OpDir.FOR:
        arrGrid[:] = 0 # = zeros([nPix_Grid for _ in range(nDim)], dtype=complex128)
    else:
        arrS[:] = 0

    for iK in range(nK):
        # coord of kernel
        if nPix_Ker%2==0:
            arrK_Ker = floor(arrK[iK,:]) + 0.5
        else:
            arrK_Ker = around(arrK[iK,:])
        
        # index of kernel
        biasIdx = around(ov*(arrK_Ker - arrK[iK,:])).astype(int)
        arrKerI = arrKerI0 + biasIdx

        # calculate range of K
        minK = ceil(arrK[iK,:] - nPix_Ker/2).astype(int64)
        maxK = minK + nPix_Ker - 1

        # perform grid
        if eOpDir==OpDir.FOR:
            arrGrid[*(slice(minK[-iDim-1]+nPix_Grid//2, maxK[-iDim-1]+1+nPix_Grid//2) for iDim in range(nDim))] +=\
            arrS[iK]*arrKer[*(arrKerI[*(slice(None) for _ in range(nDim)), -iDim-1] for iDim in range(nDim))]
        elif eOpDir==OpDir.BAC:
            arrS[iK] += sum\
            (
                arrGrid[*(slice(minK[-iDim-1]+nPix_Grid//2, maxK[-iDim-1]+1+nPix_Grid//2) for iDim in range(nDim))]*\
                arrKer[*(arrKerI[*(slice(None) for _ in range(nDim)), -iDim-1] for iDim in range(nDim))]
            )
        else:
            raise NotImplementedError("")

    if eOpDir==OpDir.FOR:
        return arrGrid
    elif eOpDir==OpDir.BAC:
        return arrS
    else:
        raise NotImplementedError("")
