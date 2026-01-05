py_sum = sum

# import library
import numpy
from numpy.typing import NDArray
import finufft
from time import time
from itertools import product
from psutil import cpu_count

# gpu library
try:
    import cufinufft, cupy
    useCuda = bool(1)
except ImportError:
    useCuda = bool(0)
xp = cupy if useCuda else numpy

pi = numpy.pi

# interface function
nStep = 2 # 1
def setNumStep(x:int):
    """
    While this method can converge in 1 step in most cases, it's possible to set the number of steps via this interface.

    Args:
        x: Number of steps to be set.
    """
    global nStep
    nStep = int(x)
    
fDbgInfo = False
def setDbgInfo(x:bool):
    """
    Set whether to print debug input. Default to False.

    Args:
        x: True for to print. False for not to.
    """
    global fDbgInfo
    fDbgInfo = bool(x)
    
enInputCheck = True
def setInputCheck(x:bool):
    """
    Set whether to check the input type. Recommend and default to True. Can be set to False for benchmark reason.

    Args:
        x: True for to use. False for not to.
    """
    global enInputCheck
    enInputCheck = bool(x)
    
def setUseCuda(x:bool):
    """
    Set whether to use cuda for acceleration.
    By default cuda would be used if cufinufft and cupy are both installed, this function is to let you turn off it.

    Args:
        x: True for to use. False for not to.
    """
    global useCuda, xp
    useCuda = bool(x)
    xp = cupy if useCuda else numpy
    
def _getArrKArrI0(lstArrK:list[NDArray]) -> tuple[NDArray,NDArray]:
    _xp = numpy if isinstance(lstArrK[0], numpy.ndarray) else cupy
    arrK = _xp.concatenate(lstArrK, axis=0).astype(_xp.float32)
    arrNRO = _xp.array([_.shape[0] for _ in lstArrK])
    arrI0 = _xp.zeros((len(arrNRO) + 1,), dtype=int)
    arrI0[1:] = _xp.cumsum(arrNRO)
    return arrK, arrI0
    
def sovDcf(nPix:int, lstArrK:list[NDArray], sWind:str="poly", pShape:float=None) -> list[NDArray]:
    """
    Solve density compensation function.
    
    Args:
        nPix: Designed number of pixels of the trajectory.
        lstArrK: List of trajectorys, element shape [nK,nAx].
        sWind: Window function type, can be "poly", "cos", "es".
        pShape: Window function shape parameter.
        
    Returns:
        list of density compensation function, w.r.t. lstArrK.
    """
    arrK, arrI0 = _getArrKArrI0(lstArrK)
    arrDcf = calDcf(nPix, arrK, arrI0, sWind, pShape)
    lstArrDcf = [arrDcf[i0:i1] for i0,i1 in zip(arrI0[:-1],arrI0[1:])]
    lstArrDcf.append(arrDcf[arrI0[-1]:])
    return lstArrDcf
   
def calDcf(nPix:int, arrK:NDArray, arrI0:NDArray|None=None, sWind:str="poly", pShape:float=None) -> NDArray:
    """
    Deprecated. Calculate density compensation function.
        
    Args:
        nPix: Designed number of pixels of the trajectory.
        arrK: Concatenated trajectories, shape [nK,nAx].
        arrI0: Start index of each interleaves, shape [nIntLea,].
        sWind: Window function type, can be "poly", "cos", "es".
        pShape: Window function shape parameter.
        
    Returns:
        Density compensation function, shape [nK,].
    """
    t0 = time()
    isInputNumpy = isinstance(arrK, numpy.ndarray)
    arrK = xp.asanyarray(arrK)
    arrI0 = xp.asanyarray(arrI0)
    
    if enInputCheck: # check
        # unfixable
        if not isInputNumpy and not useCuda:
            raise RuntimeError(f"cupy input must be processed via cuda")
        if arrK.ndim!=2 or arrK.shape[1] not in (2,3): 
            raise RuntimeError(f"`arrK.shape` should be `[nK,nDim]`, got `{arrK.shape}`")
        arrRho:NDArray = xp.linalg.norm(arrK, axis=-1)
        if abs(arrRho.max()-0.5)>0.1: raise UserWarning("k-range: [-0.5,0.5]")
        # fixable
        if arrK.shape[1]==3 and (arrK[:,2]==0).all(): arrK = arrK[:,:2]
        if useCuda: arrK = arrK.astype("float32")
    
    # basic parameter
    nPix = int(nPix)
    nK, nAx = arrK.shape
    if arrK.dtype==xp.float64:
        sdtypeC = "complex128"
        dtypeC = xp.complex128
        sdtypeF = "float64"
        dtypeF = xp.float64
    elif arrK.dtype==xp.float32:
        sdtypeC = "complex64"
        dtypeC = xp.complex64
        sdtypeF = "float32"
        dtypeF = xp.float32
    else:
        raise NotImplementedError("")
    
    if fDbgInfo: print(f"# input check: {time() - t0:.3f}s"); t0 = time()
    
    # data initialize
    arrDcf = xp.ones((nK,), dtype=dtypeC)
    
    # radial DCF
    arrRho = xp.sum(arrK**2, axis=-1, dtype=dtypeC)
    xp.sqrt(arrRho, out=arrRho)
    arrDcf *= (arrRho+1/nPix)**(nAx-1)
    
    # 1D DCF
    if arrI0 is not None:
        arrDcf1D = xp.empty((nK,), dtype=dtypeC)
        arrDcf1D[:-1] = xp.sqrt(xp.sum(xp.diff(arrK, axis=0)**2, axis=-1)) # this step takes 1.2s?
        arrDcf1D[-1] = arrDcf1D[-2]
        arrDcf1D[arrI0[1:]-1] = arrDcf1D[arrI0[1:]-2] # fix the error at seam of two trajectories
        arrDcf *= arrDcf1D
    
    # return arrDcf # see how initial DCF be like
    
    if fDbgInfo: print(f"# data initialize: {time() - t0:.3f}s"); t0 = time()

    # grid of rho
    coords = xp.ogrid[tuple(slice(0, 1, nPix*1j) for _ in range(nAx))]
    arrGridRho = xp.sqrt(py_sum(c.astype(dtypeF)**2 for c in coords))
    if fDbgInfo: print(f"# grid of rho: {time() - t0:.3f}s"); t0 = time()

    # Nd window
    if sWind=="poly": arrWindNd = 1 - arrGridRho.clip(0,1)**(2.0 if pShape is None else pShape) # 2.4
    elif sWind=="cos": arrWindNd = xp.cos(arrGridRho*pi/2).clip(0,1)**(1.0 if pShape is None else pShape) # 0.7
    elif sWind=="es": beta=2.0 if pShape is None else pShape; arrWindNd = xp.exp(beta*xp.sqrt(1-arrGridRho.clip(0,1)**2))/xp.exp(beta)
    else: raise NotImplementedError("")
    
    arrWindNd[arrGridRho>1]=0
    del arrGridRho
    
    for iAx in range(nAx):
        tupSli = tuple(0 if iAx==_iAx else slice(None) for _iAx in range(nAx))
        xp.sqrt(arrWindNd[tupSli], out=arrWindNd[tupSli])
    if fDbgInfo: print(f"# Nd window: {time() - t0:.3f}s"); t0 = time()
    
    # deconvolve
    nufftpara = {"upsampfac":1.25} if useCuda else {"debug":0, "spread_debug":0, "showwarn":0, "upsampfac":1.25, "nthreads":cpu_count(logical=True), "spread_sort":1, "fftw":64}
    fn = cufinufft if useCuda else finufft
    
    n_modes = tuple(2*nPix-1 for _ in range(nAx))
    arr2PiKT = xp.array(arrK.T, order='C', dtype=dtypeF)
    arr2PiKT *= 2*pi
    eps = 1e-3
    
    pNuift = fn.Plan(1, n_modes, eps=eps, dtype=sdtypeC, **nufftpara)
    pNufft = fn.Plan(2, n_modes, eps=eps, dtype=sdtypeC, **nufftpara)
    pNuift.setpts(*arr2PiKT)
    pNufft.setpts(*arr2PiKT)
    for i in range(nStep):
        arrPsf = pNuift.execute(arrDcf)
        if fDbgInfo: print(f"# nuift: {time() - t0:.3f}s"); t0 = time()
        
        # suppress alias outside of PSF
        sliNeg = slice(nPix-1,None,-1)
        sliPos = slice(nPix-1,None,1)
        for iCorner in product(range(2), repeat=nAx):
            tupSli = tuple(sliNeg if i else sliPos for i in iCorner)
            arrPsf[tupSli] *= arrWindNd
        
        arrDcfApo = pNufft.execute(arrPsf)
        arrDcf /= arrDcfApo
        if fDbgInfo: print(f"# nufft: {time() - t0:.3f}s"); t0 = time()
    
    if isInputNumpy and xp!=numpy:
        return arrDcf.get()
    else:
        return arrDcf
        