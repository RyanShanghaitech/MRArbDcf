from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
import slime
import mrtrjgen
import finufft
import fars as fars
from time import time
from skimage.metrics import structural_similarity as ssim

nPix = 256
nTht = 64
nPhi = 64
sr = 100
fov = 0.5
dtGrad = 10e-6
dtADC = 2e-6

# generate phantom
arrM0 = slime.genPhan(nDim=3, nPix=nPix)["M0"]
arrM0 = asarray(arrM0, dtype=complex128).squeeze()

# load trajectory
lstArrK = []
dTht0 = (2*pi)/nTht
dPhi0 = (2*pi)/nPhi
goldang = (3-sqrt(5))*pi
for iPhi in range(nPhi):
    for iTht in range(nTht):
        tht0 = (2*pi)*(iTht/nTht)# + (2*pi/nTht)*(iPhi/nPhi)
        phi0 = (2*pi)*(iPhi/nPhi)# + (2*pi/nPhi)*(iTht/nTht)
        arrK, arrG = mrtrjgen.genSpiral3DTypeA(nPix, nTht, nPhi, tht0, phi0, 0.5, sr*(42.58e6)*(fov/nPix), dtGrad)
        # arrK, arrG = mrtrjgen.genSpiral3DTypeB(nPix, nTht, nPhi, tht0, phi0, 0.5, sr*(42.58e6)*(fov/nPix), dtGrad)
        arrK, _ = mrtrjgen.intpTraj(arrG, dtGrad, dtADC)
        lstArrK.append(arrK.astype(float64))
arrK = concatenate(lstArrK, axis=0)
nPE = len(lstArrK)
nK = arrK.shape[0]

# plan nufft
planNufft = finufft.Plan(2, (nPix,nPix,nPix), 1, 1e-6, dtype="complex128")
planNufft.setpts((2*pi)*arrK[:,2].copy(), (2*pi)*arrK[:,1].copy(), (2*pi)*arrK[:,0].copy())
planNuift = finufft.Plan(1, (nPix,nPix,nPix), 1, 1e-6, dtype="complex128")
planNuift.setpts((2*pi)*arrK[:,2].copy(), (2*pi)*arrK[:,1].copy(), (2*pi)*arrK[:,0].copy())

# perform nufft
arrS = planNufft.execute(arrM0)

# calculate Dcf
t = time()
arrDcf = fars.calDcf(nPix, arrK)
arrDcf = arrDcf.astype(complex128)
t = time() - t
print(f"elapsed time: {t:.3f}s")

# perform nuifft
arrM0Rec = planNuift.execute(arrS*arrDcf)
arrPsf = finufft.nufft3d1((2*pi)*arrK[:,2].copy(), (2*pi)*arrK[:,1].copy(), (2*pi)*arrK[:,0].copy(), arrDcf.astype(complex128), (2*nPix,2*nPix,2*nPix))

# normalize, evaluate
arrM0 /= abs(arrM0).max()
arrM0Rec /= abs(arrM0Rec).max()
arrPsf /= abs(arrPsf).max()

arrM0_2Ch = arrM0[:,:,:,newaxis]
arrM0_2Ch = concatenate([arrM0_2Ch.real,arrM0_2Ch.imag], axis=-1)
arrM0Rec_2Ch = arrM0Rec[:,:,:,newaxis]
arrM0Rec_2Ch = concatenate([arrM0Rec_2Ch.real,arrM0Rec_2Ch.imag], axis=-1)
print(f"SSIM: {ssim(arrM0Rec_2Ch, arrM0_2Ch, data_range=1, channel_axis=-1):.3f}")
print(f"MAE: {abs(arrM0Rec-arrM0).mean():.3f}")

# plot
figure()
iK = 0
iK_Max = argmax(arrDcf.real)
for arrK in lstArrK:
    if iK_Max >= iK and iK_Max < iK+arrK.shape[0]:
        plot(real(arrDcf[iK:iK+arrK.shape[0]]), ".-")
        break
    iK += arrK.shape[0]
grid(True, "both", "both")

figure()
mea = abs(arrM0).mean()
std = abs(arrM0).std()
vmin = 0 # mea - 1*std
vmax = 1 # mea + 3*std
subplot(231)
imshow(abs(arrM0[nPix//2,:,:]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(232)
imshow(abs(arrM0[:,nPix//2,:]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(233)
imshow(abs(arrM0[:,:,nPix//2]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(234)
imshow(angle(arrM0[nPix//2,:,:]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()
subplot(235)
imshow(angle(arrM0[:,nPix//2,:]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()
subplot(236)
imshow(angle(arrM0[:,:,nPix//2]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()

figure()
mea = abs(arrM0Rec).mean()
std = abs(arrM0Rec).std()
vmin = 0 # mea - 1*std
vmax = 1 # mea + 3*std
subplot(231)
imshow(abs(arrM0Rec[nPix//2,:,:]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(232)
imshow(abs(arrM0Rec[:,nPix//2,:]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(233)
imshow(abs(arrM0Rec[:,:,nPix//2]), cmap="gray", vmin=vmin, vmax=vmax); colorbar()
subplot(234)
imshow(angle(arrM0Rec[nPix//2,:,:]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()
subplot(235)
imshow(angle(arrM0Rec[:,nPix//2,:]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()
subplot(236)
imshow(angle(arrM0Rec[:,:,nPix//2]), cmap="hsv", vmin=-pi, vmax=pi); colorbar()

figure()
subplot(131)
imshow(abs(arrPsf[nPix,:,:]), norm="log"); colorbar()
subplot(132)
imshow(abs(arrPsf[:,nPix,:]), norm="log"); colorbar()
subplot(133)
imshow(abs(arrPsf[:,:,nPix]), norm="log"); colorbar()

show()