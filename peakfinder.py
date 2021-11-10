import glob
import sys
import os
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

if not os.path.isdir("peakfinder8"):
    os.system("git clone https://github.com/tjlane/peakfinder8")
    cwd = os.getcwd()
    os.chdir("peakfinder8")
    os.system(sys.executable +" setup.py build")
    d = glob.glob("build/lib*")[0]
    print(d)
    sys.path.append(os.path.join(os.getcwd(), d, "ssc"))
    os.chdir(cwd)

import peakfinder8_extension

img = UtilsTest.getimage("Pilatus6M.cbf")
geo =  UtilsTest.getimage("Pilatus6M.poni")
method = ("no", "csr", "cython")
unit = pyFAI.units.to_unit("r_mm")
dummy = -2
ddummy=1.5
npt = 500
repeat = 100
SNR=4
noise=2
nb = 5
him = 2
hiM = 100

ai = pyFAI.load(geo)
fimg = fabio.open(img)
msk = fimg.data<=0
fixed = fimg.data.copy()
fixed[msk] = 1

fig,ax = subplots( figsize=(12,8))
#fig.tight_layout(pad=3.0)
ln = LogNorm(1, fimg.data.max())
mimg = ax.imshow(fixed, norm=ln, interpolation="bicubic")

ai.integrate1d(fimg.data, npt, unit=unit, method=method)
m = list(ai.engines.keys())[0]
integrator = ai.engines[m].engine
r2d = ai._cached_array[unit.name.split("_")[0] + "_center"]
r2dp = (r2d/ai.detector.pixel1).astype(numpy.float32)
data = fimg.data.astype(numpy.float32)
pmsk = (1-msk).astype(numpy.int8)
res1 = peakfinder8_extension.peakfinder_8(max_num_peaks=1000000,
                                          data=data,
                                          mask=pmsk,
                                          pix_r=r2dp,
                                          asic_nx=ai.detector.shape[1],
                                          asic_ny=ai.detector.shape[0],
                                          nasics_x=1,
                                          nasics_y=1,
                                          adc_thresh=noise,
                                          hitfinder_min_snr=SNR,
                                          hitfinder_min_pix_count=him,
                                          hitfinder_max_pix_count=hiM,
                                          hitfinder_local_bg_radius=nb)

print(f"Len of Cheetah result: {len(res1[0])}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res1 = peakfinder8_extension.peakfinder_8(max_num_peaks=1000000,
                                          data=data,
                                          mask=pmsk,
                                          pix_r=r2dp,
                                          asic_nx=ai.detector.shape[1],
                                          asic_ny=ai.detector.shape[0],
                                          nasics_x=1,
                                          nasics_y=1,
                                          adc_thresh=noise,
                                          hitfinder_min_snr=SNR,
                                          hitfinder_min_pix_count=him,
                                          hitfinder_max_pix_count=hiM,
                                          hitfinder_local_bg_radius=nb)
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
res = pf.peakfinder8(fimg.data, dummy=dummy, delta_dummy=ddummy, error_model="azimuthal", cutoff_clip=0, cycle=5, noise=noise, cutoff_pick=SNR, patch_size=3, connected=him)
print(f"Len of pyFAI result: {len(res)}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res = pf.peakfinder8(fimg.data, dummy=dummy, delta_dummy=ddummy, error_model="azimuthal", cutoff_clip=0, cycle=5, noise=noise, cutoff_pick=SNR, patch_size=3, connected=him)
t1 =  time.perf_counter()
gc.enable()
print("\n".join(pf.log_profile(1)))
print(f"Execution_time for pyFAI: {1000*(t1-t0)/repeat:.3f}ms")
 
ax.plot(res1[0], res1[1], "2g", label="Cheetah")
ax.plot(res["pos1"], res["pos0"], "1r", label="pyFAI")
ax.legend()
fig.savefig("peakfinder.eps")
fig.show()
input("finish")
