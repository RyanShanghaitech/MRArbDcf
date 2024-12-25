from numpy import *
from matplotlib.pyplot import *
import slime
import mrtrjgen
import finufft
import fars as fars
from time import time
from skimage.metrics import structural_similarity as ssim

sr = 100 # desired slew rate
fov = 0.5
nPix = 256
nSp = 24
dtGrad = 10e-6 # temporal resolution of gradient coil
dtADC = 1e-6 # temporal resolution of ADC
gamma = 42.5756e6 # UIH

# generate phantom
arrM0 = slime.genPhan(nDim=2, nPix=nPix)["M0"]
arrM0 = asarray(arrM0, dtype=complex128).squeeze()

# calculate trajectory
lstArrK = []
scale = 1/gamma*nPix/fov
goldang = (3-sqrt(5))*pi
for iTht in range(nSp):
    if 0: tht0 = iTht*goldang
    else: tht0 = (2*pi)*(iTht/nSp)
    arrK, arrG = mrtrjgen.genSpiral2D(nPix, nSp/sqrt(2), 1, tht0, 0.5, sr*gamma*fov/nPix, dtGrad, 1e2)
    arrK, _ = mrtrjgen.intpTraj(arrG, dtGrad, dtADC)
    arrSR = (arrG[1:,:] - arrG[:-1,:])/dtGrad
    lstArrK.append(arrK)
arrK = concatenate(lstArrK, axis=0, dtype=float64)
nK = arrK.shape[0]
nPE = len(lstArrK)
nRO = nK//nPE
    
# simulate kspace
arrS = finufft.nufft2d2((2*pi)*arrK[:,0], (2*pi)*arrK[:,1], arrM0)

# calculate Dcf
t = time()
arrDcf = fars.calDcf(nPix, arrK)
arrDcf = arrDcf.astype(complex128)
t = time() - t
print(f"elapsed time: {t:.3f}s")

# calculate PSF and M0_Recon
arrM0Rec = finufft.nufft2d1((2*pi)*arrK[:,0], (2*pi)*arrK[:,1], arrS*arrDcf, (nPix,nPix))
arrPsf = finufft.nufft2d1((2*pi)*arrK[:,0], (2*pi)*arrK[:,1], arrDcf, (2*nPix,2*nPix))

# normalize, evaluate
arrM0 /= abs(arrM0).max()
arrM0Rec /= abs(arrM0Rec).max()
arrPsf /= abs(arrPsf).max()

arrM0_2Ch = arrM0[:,:,newaxis]
arrM0_2Ch = concatenate([arrM0_2Ch.real,arrM0_2Ch.imag], axis=-1)
arrM0Rec_2Ch = arrM0Rec[:,:,newaxis]
arrM0Rec_2Ch = concatenate([arrM0Rec_2Ch.real,arrM0Rec_2Ch.imag], axis=-1)
print(f"SSIM: {ssim(arrM0_2Ch, arrM0Rec_2Ch, data_range=1, channel_axis=-1):.3f}")
print(f"MAE: {abs(arrM0Rec-arrM0).mean():.3f}")

# plot
iK0 = 0
iK1 = lstArrK[0].shape[0]
figure()
subplot(311)
plot(sqrt((arrK[iK0:iK1]**2).sum(axis=-1)), ".-")
subplot(312)
plot(abs(arrDcf[iK0:iK1]), ".-")
subplot(313)
plot(abs(arrDcf[iK0:iK1]), ".-")

figure()
imshow(abs(arrPsf), norm="log")
title("PSF")

figure()
plot(abs(arrPsf)[nPix,:], ".-")
title("PSF")

figure(figsize=(9,3))

subplot(121)
mea = abs(arrM0).mean()
std = abs(arrM0).std()
vmin = 0 # mea - 1*std
vmax = 1 # mea + 3*std
imshow(abs(arrM0), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
title("Phantom")

subplot(122)
mea = abs(arrM0Rec).mean()
std = abs(arrM0Rec).std()
vmin = 0 # mea - 1*std
vmax = 1 # mea + 3*std
imshow(abs(arrM0Rec), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
title("FARS")

figure()
for iPE in range(nPE):
    plot(lstArrK[iPE][:,0], lstArrK[iPE][:,1], ".-")
    axis("equal")
    grid("on")

show()