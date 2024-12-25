from numpy import *
from scipy.signal.windows import gaussian
from scipy.signal import fftconvolve
from .Function import calDcf
import finufft as fn

def smooth(arrDcf:ndarray, sigma:float=3):
    arrGaus = gaussian(6*sigma+1, std=sigma)
    arrGaus -= arrGaus[0]

    # arrDcf = concatenate([arrDcf[0]*ones([sigma*3]), arrDcf, arrDcf[-1]*ones([sigma*3])], axis=0)
    arrDcf = concatenate([zeros([sigma*3]), arrDcf, zeros([sigma*3])], axis=0)
    arrDcf = fftconvolve(arrDcf, arrGaus, mode="same")
    arrDcf = arrDcf[sigma*3:-sigma*3]

    return arrDcf

def calCrcMask\
(
    nPix:int,
    arrK:ndarray,
    nIt:int=1,
):
    assert arrK.ndim==2
    nDim = arrK.shape[1]

    arrImgTest = ones([nPix for _ in range(nDim)], dtype=complex128)

    # convert ones to S
    planFn = fn.Plan(2, tuple(nPix for _ in range(nDim)), dtype="complex128")
    planFn.setpts(*(2*pi)*arrK[:,::-1].T.copy())
    arrS = planFn.execute(arrImgTest.copy())
    
    # calculate DCF
    arrDcf = calDcf(nPix, arrK, nIt)
    
    # simulate reconstruction
    planFn = fn.Plan(1, tuple(nPix for _ in range(nDim)), dtype="complex128")
    planFn.setpts(*(2*pi)*arrK[:,::-1].T.copy())
    arrImgReco = planFn.execute((arrS*arrDcf).copy()).real

    return 1/arrImgReco