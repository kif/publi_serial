%matplotlib widget
import numpy
import pyFAI, pyFAI.test.utilstest
import fabio
from pyFAI.gui import jupyter
from matplotlib.pyplot import subplots

img = fabio.open(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus6M.cbf")).data
ai = pyFAI.load(pyFAI.test.utilstest.UtilsTest.getimage("Pilatus6M.poni"))
ai

npt=1000
unit="r_mm"
polarization=1
method=("no","csr","opencl")
error_model="hybrid"

avg = ai.sigma_clip_ng(img, npt, polarization_factor=polarization, error_model="hybrid", method=method, unit=unit)
jupyter.plot1d(avg)

bg = numpy.random.poisson(ai.calcfrom1d(avg.radial, avg.intensity, mask=ai.detector.mask, dim1_unit=avg.unit, polarization_factor=polarization))
jupyter.display(bg)

res_azim = ai.integrate1d(bg, 1000, error_model="azimuthal", method=("no","csr","cython"),unit=unit, polarization_factor=1)
res_pois = ai.integrate1d(bg, 1000, error_model="poisson", method=("no","csr","cython"),unit=unit, polarization_factor=1)

fig, ax = subplots(1,2, figsize=(10,5))
ax[0].imshow(bg)
ax[0].set_title("a. Poissonnian background image")
ax[1].plot(res_azim.radial,res_azim.std, "-", label="std azimuthal")
ax[1].plot(res_pois.radial,res_pois.std, "-",label="std Poisson")
ax[1].set_xlabel(res_pois.unit.label)
ax[1].set_title("b. Standard deviation of pixel values")
ax[1].set_ylabel("Counts")
ax[1].legend()
fig.savefig("fig_std.png")
fig.savefig("fig_std.pdf")
fig.savefig("fig_std.eps")