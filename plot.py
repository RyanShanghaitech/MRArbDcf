from numpy import *
from matplotlib.pyplot import *
from scipy.io import loadmat
import mrautodcf as mad

nPix = 256
sPathRes = "/mnt/d/LProject/DcfBenchmark/resource/"

# show metrics table
for sTraj in ["VdSpiral", "Rosette", "Yarnball", "Cones"]:
    dicResult = dict()
    for sMethod in ["Baseline", "Proposed"]:
        dicResult[sMethod] = load(f"{sPathRes[:-1]}/{sTraj}_{sMethod}.npz")
    print(f"{dicResult["Baseline"]["metNrmse"]:.3f} & {dicResult["Proposed"]["metNrmse"]:.3f} & {dicResult["Baseline"]["metSsim"]:.3f} & {dicResult["Proposed"]["metSsim"]:.3f}")
print()
for sTraj in ["VdSpiral", "Rosette", "Yarnball", "Cones"]:
    dicResult = dict()
    for sMethod in ["Baseline", "Proposed"]:
        dicResult[sMethod] = load(f"{sPathRes[:-1]}/{sTraj}_{sMethod}.npz")
    print(f"{dicResult["Baseline"]["tExe"]:.3f} & {dicResult["Proposed"]["tExe"]:.3f} & {around(dicResult["Baseline"]["tExe"]/dicResult["Proposed"]["tExe"]).astype(int)}")
print()

# exit()

# load trajectory
sTraj = "Yarnball"; nAx = 3
dicTraj = loadmat(f"{sPathRes}arrK_{sTraj}.mat")
arrK = asarray(dicTraj["arrK"]).astype(float32)
arrNRO = asarray(dicTraj["arrNRO"]).astype(int).squeeze()
sTraj = str(dicTraj["sTraj"].item())
arrI0 = zeros((arrNRO.size+1,), dtype=int)
arrI0[1:] = cumsum(arrNRO)
lstArrK = [arrK[arrI0[i]:arrI0[i+1]] for i in range(arrI0.size-1)]

# load results
arrM0 = load(f"{sPathRes[:-1]}/arrM0_{nAx}d.npy")
arrM0 = asarray(arrM0)
mskFov = load(f"{sPathRes[:-1]}/mskFov_{nAx}d.npy")
mskFov = asarray(mskFov)
mskFov = ones_like(mskFov) # test

dicResult = dict()
for sMethod in ["Baseline", "Proposed"]:
    dicResult[sMethod] = load(f"{sPathRes[:-1]}/{sTraj}_{sMethod}.npz")
    

# plot
lstTitle = [*"ABCDEFGHIJKL"]
wFPage = 4.77
dpi = 300

# DCF
# iDcfMax = argmax(dicResult["Proposed"]["arrDcf"])
# iK0 = arrI0[arrI0<=iDcfMax][-1]
# iK1 = arrI0[arrI0>iDcfMax][0]
iK0 = arrI0[0]
iK1 = arrI0[1]

figure(dpi=150, figsize=(wFPage,wFPage*2/3))
subplot(111)
for sMethod in ["Baseline", "Proposed"]:
    arrDcf = dicResult[sMethod]["arrDcf"]
    plot(abs(arrDcf[iK0:iK1]), ".-", markersize=4, linewidth=1, label=sMethod)
ylim(-0.1*abs(arrDcf).max(), 1.5*abs(arrDcf).max())
xlabel("index [a.u.]")
ylabel("DCF [a.u.]")
legend(loc=2)
grid(True, "both", "both")
subplots_adjust(0.15,0.2,0.99,0.9)
savefig(f"temp/figure/DCF.png", dpi=dpi)

# Phantom
fig = figure(dpi=150, figsize=(wFPage,wFPage*2/3))
wFig = 20
lstAx = [None for _ in range(8)]
gsAx = fig.add_gridspec(2, 3*wFig+1)
lstAx[0] = fig.add_subplot(gsAx[0,0*wFig:1*wFig])
lstAx[1] = fig.add_subplot(gsAx[0,1*wFig:2*wFig])
lstAx[2] = fig.add_subplot(gsAx[0,2*wFig:3*wFig])
lstAx[3] = fig.add_subplot(gsAx[0,3*wFig])
lstAx[4] = fig.add_subplot(gsAx[1,0*wFig:1*wFig])
lstAx[5] = fig.add_subplot(gsAx[1,1*wFig:2*wFig])
lstAx[6] = fig.add_subplot(gsAx[1,2*wFig:3*wFig])
lstAx[7] = fig.add_subplot(gsAx[1,3*wFig])

vmean = abs(arrM0).mean()
vstd = abs(arrM0).std()
vmin = 0 # vmean - 1*vstd
vmax = 1 # vmean + 3*vstd

lstAx[0].imshow(abs(arrM0[nPix//2,:,:]), cmap="gray", vmin=vmin, vmax=vmax)
lstAx[1].imshow(abs(arrM0[:,nPix//2,:]), cmap="gray", vmin=vmin, vmax=vmax)
m = lstAx[2].imshow(abs(arrM0[:,:,nPix//2]), cmap="gray", vmin=vmin, vmax=vmax)
fig.colorbar(m, cax=lstAx[3])

lstAx[4].imshow(angle(arrM0[nPix//2,:,:]), cmap="hsv", vmin=-pi, vmax=pi)
lstAx[5].imshow(angle(arrM0[:,nPix//2,:]), cmap="hsv", vmin=-pi, vmax=pi)
m = lstAx[6].imshow(angle(arrM0[:,:,nPix//2]), cmap="hsv", vmin=-pi, vmax=pi)
fig.colorbar(m, cax=lstAx[7])
lstAx[7].set_yticks([-pi,pi], [r"-$\pi$",r"$\pi$"])

lstIAx = [0,1,2,4,5,6]
for i in range(6):
    iAx = lstIAx[i]
    lstAx[iAx].set_title(f"({lstTitle[i]})", loc="left")
    lstAx[iAx].set_xticklabels([])
    lstAx[iAx].set_yticklabels([])

subplots_adjust(0.01,0.05,0.9,0.9, wspace=0.01, hspace=0.2)
savefig(f"temp/figure/Phantom.png", dpi=dpi)

# show()
# exit()

# recon result
ovPsf = 10
fig = figure(dpi=150, figsize=(wFPage,wFPage*4/3))
gsFig = fig.add_gridspec(2,1)
lstMethod = ["Baseline", "Proposed"]
for i in range(2):
    sMethod = lstMethod[i]
    arrImg = dicResult[sMethod]["arrM0Rec"]
    arrErr = mad.normImg(arrImg, mskFov=mskFov) - mad.normImg(arrM0, mskFov=mskFov)
    
    arrImg *= mskFov
    arrErr *= mskFov
    
    subfig = fig.add_subfigure(gsFig[i])

    wFig = 20
    gsAx = subfig.add_gridspec(2, 3*wFig+1)
    lstAx = [None for _ in range(8)]
    lstAx[0] = subfig.add_subplot(gsAx[0,0*wFig:1*wFig])
    lstAx[1] = subfig.add_subplot(gsAx[0,1*wFig:2*wFig])
    lstAx[2] = subfig.add_subplot(gsAx[0,2*wFig:3*wFig])
    lstAx[3] = subfig.add_subplot(gsAx[0,3*wFig])
    lstAx[4] = subfig.add_subplot(gsAx[1,0*wFig:1*wFig])
    lstAx[5] = subfig.add_subplot(gsAx[1,1*wFig:2*wFig])
    lstAx[6] = subfig.add_subplot(gsAx[1,2*wFig:3*wFig])
    lstAx[7] = subfig.add_subplot(gsAx[1,3*wFig])

    vmean = abs(arrImg).mean()
    vstd = abs(arrImg).std()
    vmin = 0 # vmean - 1*vstd
    vmax = 1 # vmean + 3*vstd

    lstAx[0].imshow(abs(arrImg[nPix//2,:,:]), cmap="gray", vmin=vmin, vmax=vmax)
    lstAx[1].imshow(abs(arrImg[:,nPix//2,:]), cmap="gray", vmin=vmin, vmax=vmax)
    m = lstAx[2].imshow(abs(arrImg[:,:,nPix//2]), cmap="gray", vmin=vmin, vmax=vmax)
    subfig.colorbar(m, cax=lstAx[3])
    
    vmean = abs(arrErr).mean()
    vstd = abs(arrErr).std()
    vmin = 0
    vmax = 1
    # scale = 1/vmax

    lstAx[4].imshow(abs(arrErr[nPix//2,:,:]), cmap="viridis", vmin=vmin, vmax=vmax)
    lstAx[5].imshow(abs(arrErr[:,nPix//2,:]), cmap="viridis", vmin=vmin, vmax=vmax)
    m = lstAx[6].imshow(abs(arrErr[:,:,nPix//2]), cmap="viridis", vmin=vmin, vmax=vmax)
    subfig.colorbar(m, cax=lstAx[7])
    
    # for j in range(4,7):
    #     lstAx[j].text(nPix/2, nPix/2, f"{scale}x", color='white')

    lstIAx = [0,1,2,4,5,6]
    for j in range(6):
        iAx = lstIAx[j]
        lstAx[iAx].set_title(f"({lstTitle[j+i*6]})", loc="left")
        lstAx[iAx].set_xticklabels([])
        lstAx[iAx].set_yticklabels([])

    subplots_adjust(0.01,0.05,0.9,0.9, wspace=0.01, hspace=0.2)

fig.savefig(f"temp/figure/Recon.png", dpi=dpi)

# FWHM
for sMethod in ["Baseline", "Proposed"]:
    arrPsf = dicResult[sMethod]["arrPsf"]
    FWHM = zeros((nAx,), dtype=float32)
    FWHM[0] = count_nonzero(arrPsf[:,nPix,nPix] > abs(arrPsf).max()/2) / ovPsf
    FWHM[1] = count_nonzero(arrPsf[nPix,:,nPix] > abs(arrPsf).max()/2) / ovPsf
    FWHM[2] = count_nonzero(arrPsf[nPix,nPix,:] > abs(arrPsf).max()/2) / ovPsf
    print(f"mean FWHM: {FWHM.mean():.3f}")

# PSF
fig = figure(figsize=(wFPage,wFPage*2/3), dpi=150)
wFig = 20
gsAx = fig.add_gridspec(2,3*wFig+1)
lstAx = [None for _ in range(8)]
lstAx[0] = fig.add_subplot(gsAx[0,0:1*wFig], projection='3d')
lstAx[1] = fig.add_subplot(gsAx[0,1*wFig:2*wFig], projection='3d')
lstAx[2] = fig.add_subplot(gsAx[0,2*wFig:3*wFig], projection='3d')
lstAx[3] = fig.add_subplot(gsAx[1,0:1*wFig], projection='3d')
lstAx[4] = fig.add_subplot(gsAx[1,1*wFig:2*wFig], projection='3d')
lstAx[5] = fig.add_subplot(gsAx[1,2*wFig:3*wFig], projection='3d')
lstAx[6] = fig.add_subplot(gsAx[:,3*wFig])

X, Y = meshgrid(arange(2*nPix), arange(2*nPix))

for i in range(7):
    if i!=6:
        if i//4==0: arrPsf = dicResult["Baseline"]["arrPsf"]
        else: arrPsf = dicResult["Proposed"]["arrPsf"]
        if i%4==0: Z = abs(arrPsf[:,:,nPix])
        if i%4==1: Z = abs(arrPsf[:,nPix,:])
        if i%4==2: Z = abs(arrPsf[nPix,:,:])
        Z /= abs(arrPsf).max()
        m = lstAx[i].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', vmin=0, vmax=log10(1+1), norm="log")
        lstAx[i].set_title(f"({lstTitle[i]})", loc="left")
        lstAx[i].set_axis_off()
    elif i==6:
        fig.colorbar(m, cax=lstAx[i])

fig.subplots_adjust(0.01,0.1,0.9,0.8, wspace=0.01)
fig.savefig(f"temp/figure/PSF.png", dpi=dpi)

show()