from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
from os.path import exists
import slime
import finufft as fn
import mrautograd as mag
import mrautodcf as mad
from sigpy.mri.dcf import pipe_menon_dcf
from time import time
from skimage.metrics import structural_similarity as ssim
from scipy.io import savemat, loadmat
import h5py

gamma = 42.5756e6 # UIH
goldrat = (1+sqrt(5))/2
random.seed(0)

nPix = 256
fov = 0.5
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 2e-6
sPathRes = "/mnt/d/LLibrary_250427/MrAutoDcf/resource/"

sTraj = ["VdSpiral", "Rosette", "Yarnball", "Cones"][2] # <- select trajectory
if sTraj in ["VdSpiral", "Rosette"]:
    nAx = 2
    mag.setGoldAng(1)
elif sTraj in ["Yarnball", "Cones"]:
    nAx = 3
    mag.setGoldAng(0)
else:
    raise NotImplementedError("")

# generate phantom
arrM0 = slime.genPhan(nDim=nAx, nPix=nPix)["M0"]
arrM0 = asarray(arrM0, dtype=complex64).squeeze()

# calculate trajectory
if 0:
    if sTraj=="VdSpiral": lstArrK0, lstArrGrad = mag.getG_VarDenSpiral(dFov=fov*sqrt(2)*goldrat, lNPix=nPix*sqrt(2)*goldrat, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
    elif sTraj=="Rosette": lstArrK0, lstArrGrad = mag.getG_Rosette(dFov=fov, lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
    elif sTraj=="Yarnball": lstArrK0, lstArrGrad = mag.getG_Yarnball(dFov=fov*sqrt(3), lNPix=nPix*sqrt(3), dSLim=sLim, dGLim=gLim, dDt=dtGrad); 
    elif sTraj=="Cones": lstArrK0, lstArrGrad = mag.getG_Cones(dFov=fov*sqrt(3), lNPix=nPix*sqrt(3), dSLim=sLim, dGLim=gLim, dDt=dtGrad)
    else: raise NotImplementedError("")
    
    lstArrK = []
    for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
        arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
        arrK += arrK0
        lstArrK.append(arrK[:,:nAx])
    arrK = concatenate(lstArrK, axis=0, dtype=float32)[:,:nAx]
    arrNk = array([arrK.shape[0] for arrK in lstArrK])
    savemat(f"{sPathRes}arrK_{sTraj}.mat", {"arrK":arrK, "arrNk":arrNk, "sTraj":sTraj})
    exit()
else:
    dicArchive = loadmat(f"{sPathRes}arrK_{sTraj}.mat")
    arrK = asarray(dicArchive["arrK"]).astype(float32)
    arrNk = asarray(dicArchive["arrNk"]).astype(int)
    sTraj = str(dicArchive["sTraj"].item())

sMethod = ""
if 0:
    # propsoed
    t = time()
    arrDcf = mad.calDcf(nPix, arrK).astype(complex64)
    t = time() - t
    print(f"Texe: {t:.3f}s")
    sMethod = "Proposed"
elif 0:
    # sigpy baseline
    t = time()
    arrDcf = pipe_menon_dcf(arrK*nPix, [nPix for _ in range(nAx)], max_iter=40, show_pbar=1).astype(complex64)
    t = time() - t
    print(f"Texe: {t:.3f}s")
    sMethod = "Sigpy"
else:
    # external baseline
    with h5py.File(f"{sPathRes}arrDcf_{sTraj}.mat") as f:
        arrDcf = asarray(f["arrDcf"])[-1,:].astype(complex64)
    sMethod = "Zwart"
    
arrDcf = mad.normDcf(arrDcf, nAx)
    
# simulate kspace
arrOm = (2*pi)*arrK.T.copy()
if nAx==2:
    nufft = fn.nufft2d2
    nuift = fn.nufft2d1
elif nAx==3:
    nufft = fn.nufft3d2
    nuift = fn.nufft3d1
else:
    raise NotImplementedError("")

arrS = nufft(*arrOm, arrM0)

# calculate PSF and M0_Recon
arrM0Rec = nuift(*arrOm, arrS*arrDcf, tuple(nPix for _ in range(nAx)))
arrPsf = nuift(*arrOm, arrDcf, tuple(2*nPix for _ in range(nAx)))

# normalize, evaluate
arrM0 = mad.normImg(arrM0)
arrM0Rec = mad.normImg(arrM0Rec)

# arrM0Gt_2Ch = arrM0[...,newaxis]
# arrM0Gt_2Ch = concatenate([arrM0Gt_2Ch.real, arrM0Gt_2Ch.imag], axis=-1)
# arrM0Rec_2Ch = arrM0Rec[...,newaxis]
# arrM0Rec_2Ch = concatenate([arrM0Rec_2Ch.real, arrM0Rec_2Ch.imag], axis=-1)
# print(f"SSIM: {ssim(arrM0Gt_2Ch, arrM0Rec_2Ch, data_range=4, channel_axis=-1):.3f}")
print(f"{sMethod} {sTraj} NRMSE: {sqrt(mean(abs(arrM0Rec-arrM0)**2))/3:.3f}")

# plot
if sTraj!="Yarnball": exit(0)

figure(dpi=150)
iK0 = 0
iK1 = arrNk[0]
print(arrNk.shape)
plot(real(arrDcf[iK0:iK1]), ".-")
grid(True, "both", "both")

figure(dpi=150)
vmean = abs(arrM0).mean()
vstd = abs(arrM0).std()
vmin = vmean - 1*vstd
vmax = vmean + 3*vstd
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

figure(dpi=150)
vmean = abs(arrM0Rec).mean()
vstd = abs(arrM0Rec).std()
vmin = vmean - 1*vstd
vmax = vmean + 3*vstd
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

figure(dpi=150)
subplot(131)
imshow(abs(arrPsf[nPix,:,:]), norm="log"); colorbar()
subplot(132)
imshow(abs(arrPsf[:,nPix,:]), norm="log"); colorbar()
subplot(133)
imshow(abs(arrPsf[:,:,nPix]), norm="log"); colorbar()

show()