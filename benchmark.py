# export OMP_NUM_THREAD=16
psum = sum

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

nPix = 256
fov = 0.5
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 5e-6

sPathRes = "/mnt/d/LProject/DcfBenchmark/resource/"
sMethod = ["Baseline", "Proposed"][1]
fDiagFov = 1

# generate phantom
sPathPhant2D = f"{sPathRes[:-1]}/arrM0_2d.npy"
sPathPhant3D = f"{sPathRes[:-1]}/arrM0_3d.npy"
if exists(sPathPhant2D) and exists(sPathPhant3D): # calculate phantom
    arrM0_2d = load(sPathPhant2D)
    arrM0_2d = asarray(arrM0_2d)
    arrM0_3d = load(sPathPhant3D)
    arrM0_3d = asarray(arrM0_3d)
else:
    random.seed(0)
    arrM0_2d = slime.genPhan(nDim=2, nPix=nPix)["M0"]
    arrM0_2d = asarray(arrM0_2d, dtype=complex64).squeeze()

    random.seed(0)
    arrM0_3d = slime.genPhan(nDim=3, nPix=nPix)["M0"]
    arrM0_3d = asarray(arrM0_3d, dtype=complex64).squeeze()

    save(sPathPhant2D, arrM0_2d)
    save(sPathPhant3D, arrM0_3d)

for sTraj in ["VdSpiral", "VdSpiral", "Rosette", "Yarnball", "Cones"]:
# for sTraj in ["Yarnball"]:
    if sTraj in ["VdSpiral", "Rosette"]:
        nAx = 2
        mag.setGoldAng(1)
    elif sTraj in ["Yarnball", "Cones"]:
        nAx = 3
        mag.setGoldAng(0)
    else:
        raise NotImplementedError("")
        
    ovTraj = sqrt(nAx) if fDiagFov else 1
    gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)]) # gLim*dtADC < 1/nPix

    # generate phantom
    if nAx==2: arrM0 = arrM0_2d.copy()
    if nAx==3: arrM0 = arrM0_3d.copy()
    
    # calculate trajectory
    if 0:
        if sTraj=="VdSpiral": lstArrK0, lstArrGrad = mag.getG_VarDenSpiral(dFov=fov*goldrat*ovTraj, lNPix=nPix*goldrat*ovTraj, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
        elif sTraj=="Rosette": lstArrK0, lstArrGrad = mag.getG_Rosette(dFov=fov*goldrat*ovTraj/5, lNPix=nPix*goldrat*ovTraj/5, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
        elif sTraj=="Yarnball": lstArrK0, lstArrGrad = mag.getG_Yarnball(dFov=fov*ovTraj, lNPix=nPix*ovTraj, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
        elif sTraj=="Cones": lstArrK0, lstArrGrad = mag.getG_Cones(dFov=fov*ovTraj, lNPix=nPix*ovTraj, dSLim=sLim, dGLim=gLim, dDt=dtGrad)
        else: raise NotImplementedError("")
        
        lstArrK = []
        for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
            arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
            arrK += arrK0
            lstArrK.append(arrK[:,:nAx])
        arrK = concatenate(lstArrK, axis=0, dtype=float32)[:,:nAx]
        arrNRO = array([arrK.shape[0] for arrK in lstArrK])
        savemat(f"{sPathRes}arrK_{sTraj}.mat", {"arrK":arrK, "arrNRO":arrNRO, "sTraj":sTraj})
        continue # test
    else:
        dicArchive = loadmat(f"{sPathRes}arrK_{sTraj}.mat")
        arrK = asarray(dicArchive["arrK"]).astype(float32)
        arrNRO = asarray(dicArchive["arrNRO"]).astype(int).squeeze()
        sTraj = str(dicArchive["sTraj"].item())
    arrI0 = zeros((arrNRO.size+1,), dtype=int)
    arrI0[1:] = cumsum(arrNRO)
    lstArrK = [arrK[arrI0[i]:arrI0[i+1]] for i in range(arrI0.size-1)]

    if sMethod == "Proposed":
        # propsoed
        mad.setDbgInfo(0)
        mad.setDtypeCheck(0)
        
        t0 = time()
        arrDcf = mad.calDcf(int(nPix), arrK, arrI0, sWind="cos") # *ovTraj
        tExe = time() - t0
        
        arrDcf = mad.normDcf(arrDcf, nAx)
        print(f"Texe: {tExe:.3f}s")
        # continue
    elif sMethod == "Sigpy":
        # sigpy baseline
        t0 = time()
        arrDcf = pipe_menon_dcf(arrK*nPix, [nPix for _ in range(nAx)], max_iter=40, show_pbar=1).astype(complex64)
        tExe = time() - t0
        print(f"Texe: {tExe:.3f}s")
    elif sMethod == "Baseline":
        # external baseline
        with h5py.File(f"{sPathRes}arrDcf_{sTraj}.mat") as f:
            arrDcf = asarray(f["arrDcf"]).astype(complex64).squeeze()
            tExe = asarray(f["Texe"]).astype(float32).squeeze()
        print(f"Texe: {tExe:.3f}s")
    else:
        raise NotImplementedError("")
        
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
    if sMethod=="Baseline": arrM0Rec *= norm(arrM0.flatten()) / norm(arrM0Rec.flatten())

    ovPsf = 10
    arrPsf = nuift(*arrOm/ovPsf, arrDcf, tuple(2*nPix for _ in range(nAx))) # must oversamp the PSF so that it can be properly ploted in surface plot view

    # FOV nulling, normalize, evaluate
    coords = ogrid[tuple(slice(-nPix/2, nPix/2, nPix*1j) for _ in range(nAx))]
    arrGridRho = sqrt(psum(float32(c)**2 for c in coords))
    mskFov = ones_like(arrGridRho, dtype=bool) if fDiagFov else arrGridRho < nPix/2 
    
    arrM0_Norm = mad.normImg(arrM0, mskFov=mskFov)
    arrM0Rec_Norm = mad.normImg(arrM0Rec, mskFov=mskFov)
    # SSIM is sensitive to `Luminance`, `Contrast` and `Structure`, so mean and std are normalized to let `Structure` speak
    
    vRange = abs(arrM0_Norm[mskFov]).max()*2
    metNrmse = sqrt(mean(abs(arrM0Rec_Norm-arrM0_Norm)[mskFov]**2))/vRange
    metSsim = ssim(abs(arrM0_Norm)*mskFov, abs(arrM0Rec_Norm)*mskFov, data_range=vRange)
    
    print(f"{sMethod} {sTraj} NRMSE: {metNrmse:.3f}")
    print(f"SSIM: {metSsim:.3f}")

    # save data
    arrDcf = mad.normDcf(arrDcf, nAx)
    savez(f"{sPathRes[:-1]}/{sTraj}_{sMethod}.npz", arrDcf=arrDcf, arrPsf=arrPsf, arrM0Rec=arrM0Rec, tExe=tExe, metSsim=metSsim, metNrmse=metNrmse)
    
    print("")

save(f"{sPathRes[:-1]}/mskFov_{nAx}d.npy", mskFov)

exit()
# plot
wFPage = 4.77
figure(dpi=150, figsize=(wFPage,wFPage/2))
subplot(111)
iK0 = arrI0[len(lstArrK)//16]
iK1 = arrI0[len(lstArrK)//16+1]
arrDcf = mad.normDcf(arrDcf, nAx)
plot(abs(arrDcf[iK0:iK1]), ".-", markersize=4, linewidth=1, label=sMethod)
# ylim(-0.1e-8,1.2e-8)
xlabel("index [a.u.]")
ylabel("DCF [a.u.]")
legend(loc=2)
grid(True, "both", "both")
subplots_adjust(0.15,0.2,0.99,0.9)

show()
exit()

figure()
for i in range(nAx):
    plot(lstArrGrad[0][:,i], ".-")
    plot(lstArrGrad[0][:,i], ".-")
    plot(lstArrGrad[0][:,i], ".-")

show()