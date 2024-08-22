import glob
import sys
import os
import posixpath
import time
import numpy
from math import pi
from matplotlib.pyplot import subplots, colorbar
import pyFAI, pyFAI.units
from pyFAI.test.utilstest import UtilsTest
import fabio
from matplotlib.colors import LogNorm
import scipy.optimize
from pyFAI.opencl.peak_finder import OCL_PeakFinder
import gc
import shutil
from pyFAI.ext.bilinear import Bilinear

pyfai_color = "limegreen"
onda_color = "orange"

def parse_stream(fname):
    indexed = []
    with open(fname) as f:
        started = False
        for line in f:
            if line.startswith("Reflections measured after indexing"):
                started = True
            if started:
                if line.startswith("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel"):
                    continue
                elif line.startswith("End of reflections"):
                    started = False
                else:
                    words = line.split()
                    x, y = words[-3:-1]
                    try:
                        indexed.append((float(x), float(y)))
                    except Exception as e:
                        print(e)
    return numpy.array(indexed)

indexed = parse_stream("pilatus6m.stream")



# Installation of a local copy of the Cython-bound peakfinder8
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
sys.path.append(pwd + "/" + glob.glob(f"{targetdir}/build/lib*")[0])

from ssc.peakfinder8_extension import peakfinder_8

img = UtilsTest.getimage("Pilatus6M.cbf")
geo = UtilsTest.getimage("Pilatus6M.poni")
method = ("no", "csr", "cython")
unit = pyFAI.units.to_unit("q_nm^-1")

dummy = -2
ddummy = 1.5
npt = 500
repeat = 10
SNR = 3
noise = 1.0
nb = 2
him = 4
hiM = 999
max_num_peaks = 10000
polarization_factor = 0.90

ai = pyFAI.load(geo)
print(ai)
fimg = fabio.open(img)
msk = fimg.data <= 0
fixed = fimg.data.copy()
fixed[msk] = 1
polarization = ai.polarization(factor=polarization_factor)

fig, ax = subplots(figsize=(12, 8))
# fig.tight_layout(pad=3.0)
ln = LogNorm(1, fimg.data.max())
mimg = ax.imshow(fixed, norm=ln, interpolation="hanning", cmap="viridis")

int1d = ai.integrate1d(fimg.data, npt, unit=unit, method=method)
m = list(ai.engines.keys())[0]
integrator = ai.engines[m].engine
r2d = ai._cached_array[unit.name.split("_")[0] + "_center"]
r2dp = (r2d / ai.detector.pixel1).astype(numpy.float32)
data = fimg.data.astype(numpy.float32)
pmsk = (1 - msk).astype(numpy.int8)
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
             "error_model": "hybrid", #azimuthal",
             "cutoff_clip":0,
             "cycle":3,
             "noise":noise,
             # "cutoff_pick":SNR,
             "cutoff_peak":SNR,
             "patch_size":2*nb+1,
             "connected":him,
             "polarization": polarization
            }


print(f"Len of Cheetah result: {len(res1[0])}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res1 = peakfinder_8(**kwargs_pf)
t1 = time.perf_counter()
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
print(pf, pf.ctx.devices[0])
res = pf.peakfinder(**kwargs_py)
print(f"Len of pyFAI result: {len(res)}")
gc.disable()
t0 = time.perf_counter()
for i in range(repeat):
    res = pf.peakfinder(**kwargs_py)
t1 = time.perf_counter()
gc.enable()
print("\n".join(pf.log_profile(1)))
print(f"Execution_time for pyFAI: {1000*(t1-t0)/repeat:.3f}ms")

kwargs_hy = kwargs_py.copy()
kwargs_hy["error_model"] = "hybrid"
reh = pf.peakfinder(**kwargs_hy)

ax.plot(reh["pos1"], reh["pos0"], "1", color="red", label="pyFAI-hybrid")
ax.plot(res["pos1"], res["pos0"], "1", color=pyfai_color, label="pyFAI")
ax.plot(res1[0], res1[1], "2", color=onda_color, label="Onda")
#ax.plot(indexed[:,0], indexed[:,1], ",", color="white", label="Xgandalf")


ax.legend()
fig.savefig("peakfinder.eps")
fig.savefig("peakfinder.png")
fig.show()

# peaks Valid, within 2 pixels from indexed: 
pyFAI_valid = ((numpy.subtract.outer(res["pos1"],indexed[:,0])**2+numpy.subtract.outer(res["pos0"],indexed[:,1])**2).min(axis=-1)<=4)
onda_valid = ((numpy.subtract.outer(res1[0],indexed[:,0])**2+numpy.subtract.outer(res1[1],indexed[:,1])**2).min(axis=-1)<=4)

print("# Histogram")
fig, ax = subplots(figsize=(12, 8))

rmax = 45
interp = Bilinear(r2d)
r_ch = [interp(i) for i in zip(res1[1], res1[0])]
r_py = [interp(i) for i in zip(res["pos0"], res["pos1"])]
v_ch = [interp(i) for i in zip(numpy.array(res1[1])[onda_valid], numpy.array(res1[0])[onda_valid])]
v_py = [interp(i) for i in zip(res["pos0"][pyFAI_valid], res["pos1"][pyFAI_valid])]

# ax.hist(r_py, rmax+1, range=(0, rmax), label="pyFAI", alpha=0.8)
# ax.hist(r_ch, rmax+1, range=(0, rmax), label="Cheetah", alpha=0.8)
hpy = numpy.histogram(r_py, rmax + 1, range=(0, rmax))
hch = numpy.histogram(r_ch, rmax + 1, range=(0, rmax))
vpy = numpy.histogram(v_py, rmax + 1, range=(0, rmax))
vch = numpy.histogram(v_ch, rmax + 1, range=(0, rmax))

#ax.plot(0.5 * (hpy[1][1:] + hpy[1][:-1]), hpy[0], "-", color=pyfai_color, label="pyFAI all peaks")
#ax.plot(0.5 * (vpy[1][1:] + vpy[1][:-1]), vpy[0], "--", color=pyfai_color, label="pyFAI indexed peaks")
#ax.plot(0.5 * (hch[1][1:] + hch[1][:-1]), hch[0], "-", color=onda_color, label="Onda all peaks")
#ax.plot(0.5 * (vch[1][1:] + vch[1][:-1]), vch[0], "--", color=onda_color, label="Onda indexed peaks")
# Rebin histogram by factor 2
ax.plot(hpy[1][1::2], hpy[0].reshape(-1,2).sum(axis=-1),"-", color=pyfai_color, label="pyFAI all peaks")
ax.plot(vpy[1][1::2], vpy[0].reshape(-1,2).sum(axis=-1), "--", color=pyfai_color, label="pyFAI indexed peaks")
ax.plot(hch[1][1::2], hch[0].reshape(-1,2).sum(axis=-1), "-", color=onda_color, label="Onda all peaks")
ax.plot(vch[1][1::2], vch[0].reshape(-1,2).sum(axis=-1), "--", color=onda_color, label="Onda indexed peaks")
# ax.set_xlabel(int1d.unit.label)
ax.set_xlabel("Resolution $d$-spacing ($\\AA$)")
ax.set_ylabel("Number of peaks per ring")
ax.set_title("Density of peaks per ring")
ax.legend()
#
q1 = ax.get_xticks()
# new_labels = [ f"{d:.4f}" for d in 20*pi/flabel]
d1 = 20 * pi / q1
d2 = numpy.linspace(len(d1) + int(abs(d1).min()), int(abs(d1).min()), len(d1) + 1)
q2 = 20 * pi / d2
new_labels = [str(int(i)) for  i in d2]
ax.set_xticks(q2)
ax.set_xticklabels(new_labels)
ax.set_xlim(0, rmax+1)
fig.show()
# fig.canvas.draw()
#################
fig.savefig("peak_per_ring.eps")
fig.savefig("peak_per_ring.png")

# Third figure:
fig, ax = subplots(figsize=(12, 8))

rmax = 44
interp = Bilinear(r2d)
ax.plot(0.5 * (hch[1][1:] + hch[1][:-1]), hch[0], "-", color=onda_color, label="Onda")

for model in "poisson", "azimuthal", "hybrid":
    kwargs_py["error_model"] = model
    res = pf.peakfinder8(**kwargs_py)
    print(f"Model: {model} {len(res)} peaks")
    pf.reset_log()
    gc.disable()
    t0 = time.perf_counter()
    for i in range(repeat):
        res = pf.peakfinder8(**kwargs_py)
    t1 = time.perf_counter()
    gc.enable()
    print(f"Execution_time for pyFAI, model {model}: {1000*(t1-t0)/repeat:.3f}ms")
    print("\n".join(pf.log_profile(1)))

    r_py = [interp(i) for i in zip(res["pos0"], res["pos1"])]
    hpy = numpy.histogram(r_py, rmax + 1, range=(0, rmax))
    ax.plot(0.5 * (hpy[1][1:] + hpy[1][:-1]), hpy[0], "-", label=f"pyFAI-{model}")

# ax.set_xlabel(int1d.unit.label)
ax.set_xlabel("Resolution $d$-spacing ($\\AA$)")
ax.set_ylabel("Number of Bragg peaks")
ax.set_title("Density of Bragg peaks per ring")
ax.legend()
#
q1 = ax.get_xticks()
# new_labels = [ f"{d:.4f}" for d in 20*pi/flabel]
d1 = 20 * pi / q1
d2 = numpy.linspace(len(d1) + int(abs(d1).min()), int(abs(d1).min()), len(d1) + 1)
q2 = 20 * pi / d2
new_labels = [str(int(i)) for  i in d2]
ax.set_xticks(q2)
ax.set_xticklabels(new_labels)
ax.set_xlim(0, rmax + 1)
fig.show()
# fig.canvas.draw()
#################
# fig.savefig("peak_per_ring.eps")
fig.savefig("model_comparison.png")

input("finish")
