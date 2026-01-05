from numpy import *
from numpy.typing import NDArray
from matplotlib.pyplot import *
import mrarbgrad as mag
import mrarbdcf as mad
from time import time

sTraj = ["VdSpiral", "Rosette", "Yarnball", "Cones"][2]
gamma = 42.5756e6

nPix = 256
fov = 0.5
sLim = 100 * gamma * fov / nPix
gLim = 120e-3 * gamma * fov / nPix
dtGrad = 10e-6
dtADC = 5e-6

if sTraj=="Yarnball":
    nAx = 3; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Yarnball(nAx==3, fov*sqrt(nAx), nPix*sqrt(nAx), sLim, gLim, dtGrad)
elif sTraj=="VdSpiral":
    nAx = 2; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_VarDenSpiral(nAx==3, fov*sqrt(nAx), nPix*sqrt(nAx), sLim, gLim, dtGrad)
elif sTraj=="Rosette":
    nAx = 2; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Rosette(nAx==3, fov*sqrt(nAx), nPix*sqrt(nAx), sLim, gLim, dtGrad)
elif sTraj=="Cones":
    nAx = 3; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Cones(nAx==3, fov*sqrt(nAx), nPix*sqrt(nAx), sLim, gLim, dtGrad)

# Convert gradients to k-space coordinates
lstArrK:list[NDArray] = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    # Keep all 3 dimensions for the 3D case
    lstArrK.append(arrK[:, :nAx])

mad.setDbgInfo(1)
t = time()
lstArrDcf = mad.sovDcf(nPix, lstArrK, sWind="poly")
t = time()-t
print(f"time: {t:.3f}")

# Normalize for 3D (nAx=3)
lstArrDcf = [mad.normDcf(arrDcf, nAx=nAx) for arrDcf in lstArrDcf]

# 4. Visualization
fig = figure(figsize=(12, 5))
idx = len(lstArrK)//4

# 3D Trajectory Plot (Subsampling for performance)
ax = fig.add_subplot(121, projection='3d' if nAx==3 else None)
ax.plot(*lstArrK[idx].T, '.-')
ax.set_title(sTraj)

# DCF Profile Plot
ax = fig.add_subplot(122)
ax.plot(abs(lstArrDcf[idx]), ".-")
ax.set_xlabel("Index")
ax.set_ylabel("DCF")
ax.grid("both")

show()