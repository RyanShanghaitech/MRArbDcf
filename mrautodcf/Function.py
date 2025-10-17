psum = sum

from numpy import *
from numpy.fft import *
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.signal.windows import *
from matplotlib.pyplot import *
import finufft as fn
# import cufinufft as finufft
from time import time
from itertools import product

fft = lambda x: fftshift(fftn(ifftshift(x)))
ifft = lambda x: fftshift(ifftn(ifftshift(x)))

fDbgInfo = False
def setDbgInfo(x:bool):
    global fDbgInfo
    fDbgInfo = bool(x)
    
fDtypeCheck = True
def setDtypeCheck(x:bool):
    global fDtypeCheck
    fDtypeCheck = bool(x)
   
def calDcf(nPix:int, arrK:ndarray, arrI0:ndarray|None=None, ovPsf:float=1.0, ovLkUpTb:int=1, sWind:str="poly", pShape:float=None) -> ndarray: # ovPsf seems useless (I though it can resolve the kernel wrap problem)
    t0 = time()
    
    if fDtypeCheck:
        # dtype check
        if not issubdtype(type(nPix), np.integer): raise RuntimeError("nPix not a integer")
            
        if arrK.ndim!=2: raise RuntimeError("")
        if arrK.shape[1] not in (2,3): raise RuntimeError("`arrK.shape` should be `[nK,nDim]`")
        arrRho = norm(arrK, axis=-1)
        if abs(arrRho.max()-0.5)>0.1: raise UserWarning("k-range: [-0.5,0.5]")
        
    nPix = int(nPix*ovPsf)
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
    
    
    # radial DCF
    arrRho = sum(arrK**2, axis=-1, dtype=dtypeC)
    sqrt(arrRho, out=arrRho)
    arrDcf = (arrRho+1/nPix)**(nAx-1) # Initialize weights
    
    # 1D DCF
    if arrI0 is not None:
        arrDcf1D = empty((nK,), dtype=dtypeC)
        arrDcf1D[:-1] = sqrt(sum(diff(arrK, axis=0)**2, axis=-1)) # this step takes 1.2s?
        arrDcf1D[-1] = arrDcf1D[-2]
        arrDcf1D[arrI0[1:]-1] = arrDcf1D[arrI0[1:]-2] # fix the error at seam of two trajectories
        arrDcf *= arrDcf1D
    
    # # init by unity vector
    # arrDcf = ones((nK,), dtype=dtypeC)
    
    # # see how initial DCF be like
    # return arrDcf 
    
    if fDbgInfo: print(f"# data initialize: {time() - t0:.3f}s"); t0 = time()

    # grid of rho
    coords = ogrid[tuple(slice(0, 1, nPix*1j) for _ in range(nAx))]
    arrGridRho = sqrt(psum(c.astype(dtypeF)**2 for c in coords))
    if fDbgInfo: print(f"# grid of rho: {time() - t0:.3f}s"); t0 = time()

    # generate Nd PSF window

    # Nd Window
    # arrWindProf_X = linspace(0,1,nPix*ovLkUpTb,1)
    
    # arrWindProf_Y = -arrWindProf_X**2 + 1 # good, unexpectedly
    # arrWindProf_Y = kaiser_bessel_derived(2*nPix, 5 if pShape is None else pShape)
    # arrWindProf_Y = chebwin(2*nPix, nPix/2) # [1:-1]
    # arrWindProf_Y = dpss(2*nPix, NW=10 if pShape is None else pShape, Kmax=1).squeeze()
    # arrWindProf_Y = kaiser(2*nPix, beta=5 if pShape is None else pShape)
    # arrWindProf_Y = cosine(2*nPix) # perfect
    # arrWindProf_Y = cos(linspace(0,pi,nPix*ovLkUpTb))+1 # must use float64 or there will be error
    # arrWindProf_Y = raised_cosine(2*nPix-1, alpha=1.0 if pShape is None else pShape, samples_per_symbol=nPix)
    # arrWindProf_Y = hamming(2*nPix)
    # arrWindProf_Y = hanning(2*nPix)
    # arrWindProf_Y = gaussian(2*nPix, nPix/3)
    # arrWindProf_Y = ones((nPix,), dtype=float64) # awful
    # arrWindProf_Y[-nPix//2:] = 0
    
    
    # print(arrWindProf_Y[0], arrWindProf_Y[-1])
    # figure()
    # subplot(211)
    # plot(arrWindProf_Y, ".-")
    # subplot(212)
    # plot(fft(arrWindProf_Y), ".-")
    # show()
    # exit()
    
    # arrWindProf_Y = arrWindProf_Y[-nPix*ovLkUpTb:]
    # arrWindProf_Y /= arrWindProf_Y[0]
    
    # arrWindNd = interp\
    # (
    #     arrGridRho.ravel(),
    #     arrWindProf_X,
    #     arrWindProf_Y,
    #     left=0.0,
    #     right=0.0,
    # ).reshape(arrGridRho.shape).astype(dtypeC)
  
    if sWind=="poly": arrWindNd = 1 - arrGridRho.clip(0,1)**(2.4 if pShape is None else pShape)
    elif sWind=="cos": arrWindNd = cos(arrGridRho*pi/2).clip(0,1)**(0.7 if pShape is None else pShape)
    else: raise NotImplementedError("")
    del arrGridRho
    for iAx in range(nAx):
        tupSli = tuple(0 if iAx==_iAx else slice(None) for _iAx in range(nAx))
        sqrt(arrWindNd[tupSli], out=arrWindNd[tupSli])
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
    arrOm_T = (2*pi/ovPsf)*arrK.T.copy() # [::-1]
    eps = 1e-6
        
    arrPsf = nuift(*arrOm_T, arrDcf, n_modes=n_modes, eps=eps)/(ovPsf**nAx)
    
    # suppress alias outside of PSF
    sliNeg = slice(nPix-1,None,-1)
    sliPos = slice(nPix-1,None,1)
    for iCorner in product(range(2), repeat=nAx):
        tupSli = tuple(sliNeg if i else sliPos for i in iCorner)
        arrPsf[tupSli] *= arrWindNd
    
    arrDcf /= nufft(*arrOm_T, arrPsf, eps=eps)
    
    if fDbgInfo: print(f"# deconvolve: {time() - t0:.3f}s"); t0 = time()
    
    return arrDcf