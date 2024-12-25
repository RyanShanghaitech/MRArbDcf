from numpy import *
from matplotlib.pyplot import *
import finufft
# import cufinufft as finufft
from time import time
from scipy.signal.windows import kaiser, cosine, hamming, hann, gaussian, kaiser_bessel_derived
from scipy.ndimage import gaussian_filter


def calDcf\
(
    nPix:int,
    arrK:ndarray
) -> ndarray:
    t = time()

    # dtype check
    nPix = int(nPix)
    arrK = arrK.astype(float64)
    arrK = arrK*(0.5/abs(arrK).max())
    nK, nDim = arrK.shape
    print(f"# dtype check: {time()-t:.3f}s"); t = time()

    # grid of rho
    tupGridCoor = meshgrid(
        *(arange(-nPix,nPix) for _ in range(nDim)),
        indexing="ij"
    )
    arrGridRho = sqrt(sum(array(tupGridCoor)**2, axis=0))
    arrGridRho = asarray(arrGridRho, dtype=float64)
    print(f"# grid of rho: {time()-t:.3f}s"); t = time()
    
    # generate Nd window
    arrWind1d = kaiser(2*nPix-1, 5)[-nPix:]
    arrWind1d -= arrWind1d[-1]
    arrWind1d /= arrWind1d[0]
    arrWindNd_Kb = interp(arrGridRho, linspace(0,nPix,nPix,0), arrWind1d).astype(float64)
    
    arrWind1d = cosine(2*nPix)[-nPix:]
    arrWind1d -= arrWind1d[-1]
    arrWind1d /= arrWind1d[0]
    arrWindNd_Cos = interp(arrGridRho, linspace(0,nPix,nPix,0), arrWind1d).astype(float64)
    
    print(f"# Nd window: {time()-t:.3f}s"); t = time()
    del arrGridRho, tupGridCoor

    # prepare Nufft plan
    planNuift = finufft.Plan(1, [2*nPix for _ in range(nDim)], 1, 1e-3, dtype="complex128")
    planNuift.setpts(*((2*pi)*arrK.T[::-1].copy()))
    planNufft = finufft.Plan(2, [2*nPix for _ in range(nDim)], 1, 1e-3, dtype="complex128")
    planNufft.setpts(*((2*pi)*arrK.T[::-1].copy()))
    print(f"# prepare Nufft plan: {time()-t:.3f}s"); t = time()

    # prepare variables for optimization
    arrDcf = (1/nK)*ones([nK], dtype=complex128)  # Initialize weights to 1
    print(f"# prepare variables for optimization: {time()-t:.3f}s"); t = time()

    # use a high-beta KB window to pre-compute an inaccurate but stable DCF
    if nDim==3: 
        arrPsf = planNuift.execute(arrDcf)
        arrDcf /= abs(planNufft.execute(arrWindNd_Kb*arrPsf)) # multiplied with window to suppress value out of FOV
        print(f"# pre-compute: {time()-t:.3f}s"); t = time()

    # use a cosine window to derive an accurate DCF (cosine window is tested to be the most accurate one, but unstable in 3D case, so a pre-compute is needed)
    arrPsf = planNuift.execute(arrDcf)
    arrDcf /= abs(planNufft.execute(arrWindNd_Cos*arrPsf)) # multiplied with window to suppress value out of FOV
    print(f"# derive DCF: {time()-t:.3f}s"); t = time()
    
    del planNuift, planNufft, arrPsf
    
    # array of Rho w.r.t. Trajectory, and cut-off window
    arrRho = sqrt(sum(arrK**2, axis=-1)); arrRho = asarray(arrRho)
    rhoMax = arrRho.max()
    arrWindCutOff = kaiser_bessel_derived(nPix*2,8)[-nPix:]
    arrWindCutOff -= arrWindCutOff[-1]
    arrWindCutOff /= arrWindCutOff[0]
    arrCutOffWind= interp(arrRho, linspace((1-2/nPix)*rhoMax,rhoMax,nPix), arrWindCutOff)
    print(f"# cut-off window: {time()-t:.3f}s"); t = time()
    del arrRho
    
    # apply cut-off window
    arrDcf *= arrCutOffWind
    print(f"# apply cut-off window: {time()-t:.3f}s"); t = time()
    del arrCutOffWind

    # normalize
    arrDcf = abs(arrDcf)
    arrDcf = arrDcf/arrDcf.sum()
    if nDim == 2: arrDcf *= pi/4
    if nDim == 3: arrDcf *= pi/6
    print(f"# normalize: {time()-t:.3f}s"); t = time()

    return arrDcf