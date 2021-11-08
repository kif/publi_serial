import numpy
from matplotlib.pyplot import subplots, colorbar
import pyFAI
from pyFAI.test.utilstest import UtilsTest
import fabio
from matplotlib.colors import LogNorm
import scipy.optimize
from scipy.special import erf

fig,ax = subplots(1, 1, figsize=(7,5))
#fig.tight_layout(pad=3.0)
#ln = LogNorm(1, fimg.data.max())
#mimg = ax[0,0].imshow(fixed, norm=ln, interpolation="bicubic")
#ax[0,0].set_title("a) MX diffraction frame")
#colorbar(mimg, ax=ax[0,0]

x = numpy.linspace(-3.5, 3.5, 1001)
g = lambda x: numpy.exp(-x*x/2)/numpy.sqrt(2*numpy.pi)
ax.plot(x, g(x))#, label="Normal distribution")
f=lambda k: "$\mu"+(str(k)+"\sigma$" if k<0 else ("+"+str(k)+"\sigma$" if k>0 else "$"))
ax.set_xticklabels([f(k) for k in range(-4,5)])
y1 = g(x)
for t in range(1,4):
    y1[x<t] = 0
    ax.fill(x, y1, alpha=0.5, label=f"$P(x>\mu+{t}\sigma)$ = {50*(1-erf(t/numpy.sqrt(2))):4.2f}%")
ax.set_ylabel("Probablity density")
ax.set_title("Normal distribution")
ax.legend(loc=1)
fig.savefig("distribution.eps")
fig.show()
input("finished")