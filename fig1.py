import numpy
from matplotlib.pyplot import subplots, colorbar
import pyFAI
from pyFAI.test.utilstest import UtilsTest
import fabio
from matplotlib.colors import LogNorm
import scipy.optimize


img = UtilsTest.getimage("Pilatus6M.cbf")
geo =  UtilsTest.getimage("Pilatus6M.poni")
method = ("no", "csr", "python")
npt = 500
targets = [87, 160]#, 240]
ai = pyFAI.load(geo)
fimg = fabio.open(img)
msk = fimg.data<=0
fixed = fimg.data.copy()
fixed[msk] = 1
fig,ax = subplots(2,2, figsize=(12,8))
fig.tight_layout(pad=3.0)
ln = LogNorm(1, fimg.data.max())
mimg = ax[0,0].imshow(fixed, norm=ln, interpolation="hanning", cmap="viridis")
ax[0,0].set_title("a) MX diffraction frame")

colorbar(mimg, ax=ax[0,0])
p0 = ai.sigma_clip_ng(fimg.data, npt, unit="r_mm", method=method, error_model="poisson", thres=0, max_iter=0,)
p1 = ai.sigma_clip_ng(fimg.data, npt, unit="r_mm", method=method, error_model="poisson", thres=0, max_iter=1,)
a0 = ai.sigma_clip_ng(fimg.data, npt, unit="r_mm", method=method, error_model="azimuthal", thres=0, max_iter=0,)
a1 = ai.sigma_clip_ng(fimg.data, npt, unit="r_mm", method=method, error_model="azimuthal", thres=0, max_iter=1,)
ax[0,1].plot(a0.radial, a0.intensity, label=r"Average")
ax[0,1].plot(a1.radial, a1.intensity, label=r"Average after clipping")
ax[0,1].legend()
ax[0,1].set_xlabel("Distance to the beam-center: "+a0.unit.label)
ax[0,1].set_ylabel("Intensity (count)")
ax[0,1].set_title("b) Azimuthal averaging ")

ax[1,1].plot(a0.radial, a0.std, alpha=0.7,label=r"Azimuthal deviation")
ax[1,1].plot(p0.radial, p0.std, alpha=0.7,label=r"Poissonian noise")
ax[1,1].plot(a0.radial, a1.std, alpha=0.7,label=r"Azimuthal deviation after clipping")
ax[1,1].plot(p0.radial, p1.std, alpha=0.7,label=r"Poissonian noise after clipping")
ax[1,1].set_xlabel("Distance to the beam-center: "+a0.unit.label)
ax[1,1].set_ylabel("Standard deviation for a pixel (count)")
ax[1,1].set_ylim(0, 20)
ax[1,1].set_title("c) Uncertainties measured")
ax[1,1].legend()

ax[1,0].set_ylabel("Number of pixels")
ax[1,0].set_xlabel("Intensity of pixels")
ax[1,0].set_title(f"d) Histogram of pixel intensities in {len(targets)} rings")

def gaussian(x, h, c, s):
    return h*numpy.exp(-(x-c)**2/(2*s*s))

arrowprops = dict(width=2,
                  headwidth=5,
                  headlength=5,
                  shrink=5)
                  #arrowstyle="->")
text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif'
               #, 'fontweight': 'bold'
               }
for target in targets:
    idx = numpy.argmin(abs(target-p0.radial))
    key = list(ai.engines.keys())[0]
    csr = ai.engines[key].engine.lut
    values = fimg.data.ravel()[csr[1][csr[2][idx]:csr[2][idx+1]]]
    ax[1,0].hist(values, 42, range=(-1, 40), label=f"ring @ r={target}mm", alpha=0.7)
    values = values[values>=0]
    v,c = numpy.histogram(values, values.max())
    c = numpy.argmax(v)
    s=1
    h = v.max()
    x = numpy.arange(values.max())+0.5
    res = scipy.optimize.curve_fit(gaussian, x, v, [h,c,s])[0]
    ax[1,0].plot(x, gaussian(x, *res), label=r"gauss($\mu=$%.2f, $\sigma=$%.2f)"%(res[1], res[2]))
    y_val = numpy.interp(target, a1.radial, a1.intensity)
    ax[0,1].annotate(f"r={target}mm", xy=(target, y_val), xytext=(target, -2),
            arrowprops=arrowprops, **text_params)
    y_val = numpy.interp(target, a1.radial, numpy.sqrt(p1.sum_variance/p1.sum_normalization))
    ax[1,1].annotate(f"r={target}mm", xy=(target, y_val), xytext=(target, 0.5),
            arrowprops=arrowprops, **text_params)


ax[1,0].set_xlim(-1, 40)
ax[1,0].legend()

fig.show()
fig.savefig("fig1.eps")
fig.savefig("fig1.png")
input("Finished !")
