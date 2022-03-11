import glob
import sys
import os
import posixpath
import time
import numpy
from matplotlib.pyplot import subplots, colorbar
import pyFAI, pyFAI.units
from pyFAI.test.utilstest import UtilsTest
import fabio
from matplotlib.colors import LogNorm
import scipy.optimize
from pyFAI.opencl.peak_finder import OCL_PeakFinder
import gc
import shutil


#Installation of a local copy of the Cython-bound peakfinder8
targeturl = "https://github.com/kif/peakfinder8"
targetdir = posixpath.split(targeturl)[-1]
if os.path.exists(targetdir):
    shutil.rmtree(targetdir, ignore_errors=True)
pwd = os.getcwd()
try:
    os.system("git clone " + targeturl)
    os.chdir(targetdir)
    os.system(sys.executable + " setup.py build")
except exception as err:
    print(err)
finally:
    os.chdir(pwd)
sys.path.append(pwd+"/"+glob.glob(f"{targetdir}/build/lib*")[0])

from ssc.peakfinder8_extension import peakfinder_8

img = UtilsTest.getimage("Pilatus6M.cbf")
geo =  UtilsTest.getimage("Pilatus6M.poni")
method = ("no", "csr", "cython")
unit = pyFAI.units.to_unit("q_nm^-1")

dummy = -2
ddummy=1.5
npt = 500
repeat = 10
SNR=3
noise=1.0
nb = 2
him = 4
hiM = 999
max_num_peaks = 10000
polarization_factor = 0.90

ai = pyFAI.load(geo)
fimg = fabio.open(img)
msk = fimg.data<=0
fixed = fimg.data.copy()
fixed[msk] = 1
polarization = ai.polarization(factor=polarization_factor)

fig,ax = subplots( figsize=(12,8))
#fig.tight_layout(pad=3.0)
ln = LogNorm(1, fimg.data.max())
mimg = ax.imshow(fixed, norm=ln, interpolation="nearest", cmap="magma")#bicubic")

ai.integrate1d(fimg.data, npt, unit=unit, method=method)
m = list(ai.engines.keys())[0]
integrator = ai.engines[m].engine
r2d = ai._cached_array[unit.name.split("_")[0] + "_center"]
r2dp = (r2d/ai.detector.pixel1).astype(numpy.float32)
data = fimg.data.astype(numpy.float32)
pmsk = (1-msk).astype(numpy.int8)
kwargs_pf = {"max_num_peaks":max_num_peaks,
             "data":data,
             "mask":pmsk,
             "pix_r":r2dp,
             "asic_nx":ai.detector.shape[1],
             "asic_ny":ai.detector.shape[0],
             "nasics_x":1,
             "nasics_y":1,
             "adc_thresh":noise,
             "hitfinder_min_snr":SNR,
             "hitfinder_min_pix_count":him,
             "hitfinder_max_pix_count":hiM,
             "hitfinder_local_bg_radius":nb}
res1 = peakfinder_8(**kwargs_pf)

kwargs_py = {"data":fimg.data,
             "dummy": dummy, "delta_dummy":ddummy,
             "error_model": "azimuthal",
             "cutoff_clip":0,
             "cycle":3,
             "noise":noise,
             "cutoff_pick":SNR,
             "patch_size":2*nb+1,
             "connected":him,
             "polarization": polarization
            }


print(f"Len of Cheetah result: {len(res1[0])}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res1 = peakfinder_8(**kwargs_pf)
t1 =  time.perf_counter()
gc.enable()
print(f"Execution_time for Cheetah: {1000*(t1-t0)/repeat:.3f}ms")

pf = OCL_PeakFinder(integrator.lut,
                        image_size=fimg.shape[0] * fimg.shape[1],
                        empty=0,
                        unit=unit,
                        bin_centers=integrator.bin_centers,
                        radius=ai._cached_array[unit.name.split("_")[0] + "_center"],
                        mask=msk.astype("int8"),
                        profile=True)
print(pf)
res = pf.peakfinder8(**kwargs_py)
print(f"Len of pyFAI result: {len(res)}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res = pf.peakfinder8(**kwargs_py)
t1 =  time.perf_counter()
gc.enable()
print("\n".join(pf.log_profile(1)))
print(f"Execution_time for pyFAI: {1000*(t1-t0)/repeat:.3f}ms")

ax.plot(res["pos1"], res["pos0"], "1", label="pyFAI")
ax.plot(res1[0], res1[1], "2", label="Cheetah")
ax.legend()
fig.savefig("peakfinder.eps")
fig.savefig("peakfinder.png")
fig.show()
input("finish")
