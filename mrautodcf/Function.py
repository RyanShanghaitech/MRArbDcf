from numpy import *
from numpy.fft import *
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal.windows import *
from matplotlib.pyplot import *
import finufft as fn
# import cufinufft as finufft
from time import time

def OptWind2d(nPix:int): # equivalent to Eq.4 in Johnson 2009, rho \in [0,1)]
    arrRho = linspace(-1,1,nPix,1)
    arrRho = abs(arrRho)
    return (2/pi) * (arccos(arrRho) - arrRho*sqrt(1-arrRho**2))

def OptWind3d(nPix:int): # Eq.6 in Johnson 2009
    arrRho = linspace(-1,1,nPix,1)
    arrRho = abs(arrRho)
    return (1-arrRho/2) * (1-arrRho)**2

fDbgInfo = False
def setDbgInfo(x:bool):
    global fDbgInfo
    fDbgInfo = bool(x)
    
def calDcf(nPix:int, arrK:ndarray) -> ndarray:
    t0 = time()
    
    # dtype check
    if not issubdtype(type(nPix), np.integer): raise RuntimeError("nPix not a integer")
    if arrK.ndim!=2: raise RuntimeError("arrK.ndim!=2")
    if abs(arrK).max()>0.5: raise RuntimeError("abs(arrK).max()>0.5")
    
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
    
    # initialize DCF
    arrDcf = zeros((nK,), dtype=dtypeC)
    arrDcf.real = norm(arrK, axis=-1)*2 + 1/nPix**nAx # Initialize weights
    if fDbgInfo: print(f"# initialize DCF: {time() - t0:.3f}s"); t0 = time()

    # grid of rho
    arrGridRho = norm(array(meshgrid\
    (
        *(linspace(-1,1,nPix*2-1,1, dtype=dtypeF) for _ in range(nAx)),
        indexing="ij"
    )), axis=0)
    if fDbgInfo: print(f"# grid of rho: {time() - t0:.3f}s"); t0 = time()

    # generate Nd PSF window
    arrWindProf_X = linspace(0,1,nPix,0, dtype=dtypeF)
    
    n_modes = tuple(2*nPix-1 for _ in range(nAx))
    arrOm_T = (2*pi)*arrK.T.copy() # [::-1]
    eps = 1e-6

    # deconvolve
    if nAx==2:
        nuift = fn.nufft2d1
        nufft = fn.nufft2d2
    elif nAx==3:
        nuift = fn.nufft3d1
        nufft = fn.nufft3d2
    
    arrWindProf = kaiser(2*nPix, 6)[-nPix:]
    interper = interp1d(arrWindProf_X, arrWindProf, kind="linear", fill_value=0, bounds_error=False)
    arrWindNd = interper(arrGridRho)
    if fDbgInfo: print(f"# Nd window: {time() - t0:.3f}s"); t0 = time()
    
    arrPsf = nuift(*arrOm_T, arrDcf, n_modes=n_modes, eps=eps)
    arrPsf *= arrWindNd; # suppress alias outside of PSF
    arrDcf /= nufft(*arrOm_T, arrPsf, eps=eps)
    
    if fDbgInfo: print(f"# deconvolve: {time() - t0:.3f}s"); t0 = time()
    
    # break

    return arrDcf