from numpy import *
from matplotlib.pyplot import *
import mrarbgrad as mag
import mrarbdcf as mad
from time import time

sTraj = ["VdSpiral", "Rosette", "Yarnball", "Cones"][3]
gamma = 42.5756e6

nPix = 256
fov = 0.5
sLim = 100 * gamma * fov / nPix
gLim = 120e-3 * gamma * fov / nPix
dtGrad = 10e-6
dtADC = 5e-6


if sTraj=="Yarnball":
    nAx = 3; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Yarnball(dFov=fov*sqrt(nAx), lNPix=nPix*sqrt(nAx), dSLim=sLim, dGLim=gLim, dDt=dtGrad)
elif sTraj=="VdSpiral":
    nAx = 2; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_VarDenSpiral(dFov=fov*sqrt(nAx), lNPix=nPix*sqrt(nAx), dSLim=sLim, dGLim=gLim, dDt=dtGrad)
elif sTraj=="Rosette":
    nAx = 2; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Rosette(dFov=fov*sqrt(nAx), lNPix=nPix*sqrt(nAx), dSLim=sLim, dGLim=gLim, dDt=dtGrad)
elif sTraj=="Cones":
    nAx = 3; mag.setGoldAng(nAx==2); ovTraj = sqrt(nAx); gLim = amin([gLim, 1/(dtADC*nPix*ovTraj)])
    lstArrK0, lstArrGrad = mag.getG_Cones(dFov=fov*sqrt(nAx), lNPix=nPix*sqrt(nAx), dSLim=sLim, dGLim=gLim, dDt=dtGrad)

# Convert gradients to k-space coordinates
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    # Keep all 3 dimensions for the 3D case
    lstArrK.append(arrK[:, :nAx])

arrK = concatenate(lstArrK, axis=0).astype(float32)
arrNRO = array([k.shape[0] for k in lstArrK])
arrI0 = zeros((len(arrNRO) + 1,), dtype=int)
arrI0[1:] = cumsum(arrNRO)

mad.setDbgInfo(1)
mad.setDtypeCheck(0)
t = time()
arrDcf = mad.calDcf(nPix, arrK, arrI0, sWind="poly", fInit=1)
t = time()-t
print(f"time: {t:.3f}")

# Normalize for 3D (nAx=3)
arrDcf = mad.normDcf(arrDcf, nAx=nAx)

# 4. Visualization
fig = figure(figsize=(12, 5))

# 3D Trajectory Plot (Subsampling for performance)
ax = fig.add_subplot(121, projection='3d' if nAx==3 else None)
ax.plot(*lstArrK[0].T, '.-')
ax.set_title(sTraj)

# DCF Profile Plot
ax = fig.add_subplot(122)
iStart, iEnd = arrI0[0], arrI0[1]
ax.plot(abs(arrDcf[iStart:iEnd]), ".-")
ax.set_title("DCF of first interleave (3D)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Density Weight")
ax.grid(True)

show()