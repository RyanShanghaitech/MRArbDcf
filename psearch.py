import finufft as fn
from numpy import *
from numpy.fft import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import matplotlib.ticker as ticker

nPix = 256
goldrat = (1+sqrt(5))/2

fft = lambda x: fftshift(fftn(ifftshift(x)))
ift = lambda x: fftshift(ifftn(ifftshift(x)))
nufft = lambda k, x: fn.nufft1d2((2*pi)*k, x, eps=1e-8)
nuift = lambda k, x, n: fn.nufft1d1((2*pi)*k, x, n, eps=1e-8)

arrP = arange(0.1,4.0,0.1) # arange(1.80, 2.00, 0.01)
arrQuaErr = None
for iTest in range(100):
    print(f"iTest: {iTest}")
    random.seed(iTest)
    arrK = random.randn(2*nPix)*0.5/3
    # arrK = random.rand(2*nPix)-0.5
    # arrK = (iTest/nPix + arange(nPix*goldrat)/goldrat)%1-0.5
    arrK[0] = 0
    arrK = arrK[abs(arrK)<0.5]
    print(f"Nk: {arrK.size}, range: {arrK.min():.3f}, {arrK.max():.3f}")

    arrDcf = ones_like(arrK, dtype=complex128) # E
    # arrDcf = abs(arrK)+1/nPix + 0j # E
    arrPsf = nuift(arrK, arrDcf, (2*nPix)-1)/size(arrK) # P

    lstQuaErr = []
    for iP in range(arrP.size):
        p = arrP[iP]
        x = linspace(-nPix,nPix,2*nPix-1)
        # arrW = cos(pi/2 * x/nPix)**p
        arrW = 1 - (abs(x)/nPix)**p
        
        arrPsfStar = nuift(arrK, arrDcf/nufft(arrK, arrPsf*arrW), (2*nPix)-1)/size(arrK)
        arrErr = arrPsfStar * log2(1 + abs(linspace(-(nPix-1),nPix-1,2*nPix-1,1))) / abs(arrPsfStar[nPix-1])
        
        lstQuaErr.append(norm(arrErr)) # , ord=inf (min-max)
    
    if arrQuaErr is None: arrQuaErr = array(lstQuaErr)
    else: arrQuaErr = maximum(arrQuaErr, array(lstQuaErr))
    
arrQuaErr[isnan(arrQuaErr)] = max(arrQuaErr[~isnan(arrQuaErr)])
iPOpt = argmin(arrQuaErr)
    
# fig = figure(figsize=(5,3), dpi=150)
# ax = fig.add_subplot(111)
# lstC = ["tab:blue"]*arrP.size
# lstC[iPOpt] = "tab:red"
# ax.scatter(arrP, arrQuaErr, c=lstC[:])
# ax.set_xlabel(r"$p$")
# ax.set_ylabel(r"$\|P^\star_\mathrm{out}(\mathbf{x})\|$")
# ax.set_ylim(0,10)
# ax.text(arrP[iPOpt], arrQuaErr[iPOpt]-0.1, rf"$p={arrP[iPOpt]:.1f}$"+"\n"+rf"$\|P^\star_\mathrm{{out}}(\mathbf{x})\|={arrQuaErr[iPOpt]:.3f}$", horizontalalignment='center', verticalalignment='top')

wFPage = 4.77
fig = figure(figsize=(wFPage, wFPage*3/3), dpi=300)

ax = fig.add_subplot(211)
lstC = ["tab:blue"]*arrP.size
lstC[iPOpt] = "tab:red"
# sliDisp = slice(iPOpt%10,None,10)
ax.scatter(arrP[:], arrQuaErr[:], c=lstC[:])
ax.set_ylim(6,8)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\|P^\star_\mathrm{out}(\mathbf{x})\|$")
ax.text(arrP[iPOpt], arrQuaErr[iPOpt]-0.1, rf"$p={arrP[iPOpt]:.1f}$"+"\n"+rf"$\|P^\star_\mathrm{{out}}(\mathbf{{x}})\|={arrQuaErr[iPOpt]:.3f}$", horizontalalignment='center', verticalalignment='top')
ax.set_title("(A)", loc="left")

ax = fig.add_subplot(212)
x = linspace(0,1,1000)
y = 1-abs(x)**arrP[iPOpt]
ax.plot(x, y, "-")
ax.set_xlabel(r"$\|\mathbf{\bar{x}}\|$")
ax.set_ylabel(r"$W^\star(\mathbf{x})$")
ax.text(1,1, rf"$W^\star(\mathbf{{x}})=1-\|\mathbf{{\bar{{x}}}}\|^{{{arrP[iPOpt]:.1f}}}$", horizontalalignment='right', verticalalignment='top')
ax.set_title("(B)", loc="left")

fig.subplots_adjust(0.15,0.1,0.99,0.95, wspace=0.0, hspace=0.5)
fig.savefig("temp/figure/PSearch.png", dpi=300)

show()