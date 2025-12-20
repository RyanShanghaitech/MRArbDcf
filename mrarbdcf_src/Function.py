py_sum = sum

from numpy import *
from numpy.linalg import norm
import finufft as fn
# import cufinufft as cufn
from time import time
from itertools import product

fDbgInfo = False
def setDbgInfo(x:bool):
    global fDbgInfo
    fDbgInfo = bool(x)
    
fDtypeCheck = True
def setDtypeCheck(x:bool):
    global fDtypeCheck
    fDtypeCheck = bool(x)
   
def calDcf(nPix:int, arrK:ndarray, arrI0:ndarray|None=None, sWind:str="poly", pShape:float=None, fInit:bool=True) -> ndarray:
    t0 = time()
    
    if fDtypeCheck:
        # dtype check
        if not issubdtype(type(nPix), integer): raise RuntimeError("nPix not a integer")
            
        if arrK.ndim!=2: raise RuntimeError("")
        if arrK.shape[1] not in (2,3): raise RuntimeError("`arrK.shape` should be `[nK,nDim]`")
        arrRho = norm(arrK, axis=-1)
        if abs(arrRho.max()-0.5)>0.1: raise UserWarning("k-range: [-0.5,0.5]")
        
    nPix = int(nPix)
    nK, nAx = arrK.shape
    if arrK.dtype==float64:
        sdtypeC = "complex128"
        dtypeC = complex128
        sdtypeF = "float64"
        dtypeF = float64
    elif arrK.dtype==float32:
        sdtypeC = "complex64"
        dtypeC = complex64
        sdtypeF = "float32"
        dtypeF = float32
    else:
        raise NotImplementedError("")
    
    if fDbgInfo: print(f"# dtype check: {time() - t0:.3f}s"); t0 = time()
    
    # data initialize
    arrDcf = ones((nK,), dtype=dtypeC)
    
    # radial DCF
    if fInit:
        arrRho = sum(arrK**2, axis=-1, dtype=dtypeC)
        sqrt(arrRho, out=arrRho)
        arrDcf *= (arrRho+1/nPix)**(nAx-1) # Initialize weights
        
        # 1D DCF
        if arrI0 is not None:
            arrDcf1D = empty((nK,), dtype=dtypeC)
            arrDcf1D[:-1] = sqrt(sum(diff(arrK, axis=0)**2, axis=-1)) # this step takes 1.2s?
            arrDcf1D[-1] = arrDcf1D[-2]
            arrDcf1D[arrI0[1:]-1] = arrDcf1D[arrI0[1:]-2] # fix the error at seam of two trajectories
            arrDcf *= arrDcf1D
    
    # # see how initial DCF be like
    # return arrDcf 
    
    if fDbgInfo: print(f"# data initialize: {time() - t0:.3f}s"); t0 = time()

    # grid of rho
    coords = ogrid[tuple(slice(0, 1, nPix*1j) for _ in range(nAx))]
    arrGridRho = sqrt(py_sum(c.astype(dtypeF)**2 for c in coords))
    if fDbgInfo: print(f"# grid of rho: {time() - t0:.3f}s"); t0 = time()

    # generate Nd PSF window
    if sWind=="poly": arrWindNd = 1 - arrGridRho.clip(0,1)**(2.4 if pShape is None else pShape)
    elif sWind=="cos": arrWindNd = cos(arrGridRho*pi/2).clip(0,1)**(0.7 if pShape is None else pShape)
    elif sWind=="es": beta=2.0 if pShape is None else pShape; arrWindNd = exp(beta*sqrt(1-arrGridRho.clip(0,1)**2))/exp(beta)
    else: raise NotImplementedError("")
    
    arrWindNd0 = None
    if not fInit: arrWindNd0 = exp(8.6*sqrt(1-arrGridRho.clip(0,1)**2))/exp(8.6)
    
    arrWindNd[arrGridRho>1]=0
    if arrWindNd0 is not None:
        arrWindNd0[arrGridRho>1]=0
    
    del arrGridRho
    
    for iAx in range(nAx):
        tupSli = tuple(0 if iAx==_iAx else slice(None) for _iAx in range(nAx))
        sqrt(arrWindNd[tupSli], out=arrWindNd[tupSli])
        if arrWindNd0 is not None:
            sqrt(arrWindNd0[tupSli], out=arrWindNd0[tupSli])
    if fDbgInfo: print(f"# Nd window: {time() - t0:.3f}s"); t0 = time()
    
    # deconvolve
    if nAx==2:
        nuift = fn.nufft2d1
        nufft = fn.nufft2d2
    elif nAx==3:
        nuift = fn.nufft3d1
        nufft = fn.nufft3d2
    else:
        raise NotImplementedError("")
    n_modes = tuple(2*nPix-1 for _ in range(nAx))
    arrOm_T = (2*pi)*arrK.T.copy() # [::-1]
    eps = 1e-6
    
    for i in ([1,] if fInit else [0,1]):
        arrPsf = nuift(*arrOm_T, arrDcf, n_modes=n_modes, eps=eps)
        
        # suppress alias outside of PSF
        sliNeg = slice(nPix-1,None,-1)
        sliPos = slice(nPix-1,None,1)
        for iCorner in product(range(2), repeat=nAx):
            tupSli = tuple(sliNeg if i else sliPos for i in iCorner)
            if i==0: arrPsf[tupSli] *= arrWindNd0
            else: arrPsf[tupSli] *= arrWindNd
        
        arrDcf /= nufft(*arrOm_T, arrPsf, eps=eps)
    
    if fDbgInfo: print(f"# deconvolve: {time() - t0:.3f}s"); t0 = time()
    
    return arrDcf