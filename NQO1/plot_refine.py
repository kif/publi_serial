#!/usr/bin/env python
from matplotlib.pyplot import subplots,rcParams
import numpy
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

prop_cycle = rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def parse_phenix(fn):
    with open(fn) as f:
        res = {}
        keys = []
        started = False
        for l in f:
            if started:
                if not l.strip():
                    started = False
                else:
                    data = ([float(i) for i in l.split() if i!="-"])
                    if len(data) == len(keys):
                        for k,v in zip(keys, data):
                            res[k].append(v)

            elif "BIN  RESOLUTION RANGE  COMPL.    NWORK NFREE   RWORK  RFREE  CCWORK CCFREE" in l:
                started = True
                keys = l.split()
                for i in keys:
                    res[i] = []
    for k in res:
        res[k] = numpy.array(res[k])
    return res

res_dense = parse_phenix('Refine_dense/nqo1_refine_009_info.txt')
res_1_0si = parse_phenix('Refine_1sigma/nqo1_refine_006_info.txt')
res_1_4si = parse_phenix('Refine_1p4sigma/nqo1_refine_008_info.txt')
res = res_dense
q = (1/res["RESOLUTION"]+1/res["RANGE"])/2
d = numpy.array([f"{i:.2f}" for i in 1/q])
m=[0,1,2,4,6,9]

fig,ax = subplots(3, figsize=(6.4, 6.4))
ax[0].plot(q, res_dense["COMPL."], linestyle="solid", label="Initial")
ax[0].plot(q, res_1_0si["COMPL."],linestyle="solid", label="Sparse 1.0$\\sigma$")
ax[0].plot(q, res_1_4si["COMPL."],linestyle="solid", label="Sparse 1.4$\\sigma$")
ax[0].legend()
ax[0].set_ylabel(r"Completeness")
#ax[0].set_xlabel(r"d-spacing ($\AA$)")
#ax[0].set_xticks(q[m], d[m])
#ax[0].set_xlabel("")
ax[0].set_xticks(q[m], ["" for i in q[m]])

########
ax[1].plot(q, res_dense["RWORK"], linestyle="dotted", label="R$_{work}$ Init.")
ax[1].plot(q, res_1_0si["RWORK"], linestyle="dotted", label="R$_{work}$ 1.0$\\sigma$")
ax[1].plot(q, res_1_4si["RWORK"], linestyle="dotted", label="R$_{work}$ 1.4$\\sigma$")
ax[1].plot(q, res_dense["RFREE"], linestyle="dashed", color=colors[0], label="R$_{free}$ Init.")
ax[1].plot(q, res_1_0si["RFREE"], linestyle="dashed", color=colors[1],label="R$_{free}$ 1.0$\\sigma$")
ax[1].plot(q, res_1_4si["RFREE"], linestyle="dashed", color=colors[2], label="R$_{free}$ 1.4$\\sigma$")
ax[1].legend(ncol=2)
ax[1].set_ylabel("R Work/Free")
#ax[1].set_ylabel("CC½ Work/Free")
#ax[0].set_xlabel(r"d-spacing ($\AA$)")
ax[1].set_xticks(q[m], ["" for i in d[m]])

########
ax[2].plot(q, res_dense["CCWORK"], linestyle="dotted", label="CC$_{work}$ Init.")
ax[2].plot(q, res_1_0si["CCWORK"], linestyle="dotted", label="CC$_{work}$ 1.0$\\sigma$")
ax[2].plot(q, res_1_4si["CCWORK"], linestyle="dotted", label="CC$_{work}$ 1.4$\\sigma$")
ax[2].plot(q, res_dense["CCFREE"], linestyle="dashed", color=colors[0],label="CC$_{free}$ Init.")
ax[2].plot(q, res_1_0si["CCFREE"], linestyle="dashed", color=colors[1],label="CC$_{free}$ 1.0$\\sigma$")
ax[2].plot(q, res_1_4si["CCFREE"], linestyle="dashed", color=colors[2],label="CC$_{free}$ 1.4$\\sigma$")
ax[2].legend(ncol=2)
ax[2].set_ylabel(r"CC½ Work/Free")
ax[2].set_xlabel(r"d-spacing ($\AA$)")
ax[2].set_xticks(q[m], d[m])


#lim10=root_scalar(lambda x:interp1d(q, snr_dense)(x)-1.0, x0=1.92, x1=3.3).root
#lim14=root_scalar(lambda x:interp1d(q, snr_dense)(x)-1.4, x0=1.92, x1=3.3).root
lim10, lim14 = 0.33183076923076924, 0.32135294117647057

ax[0].set_ylim(0.7, 1.05)
ax[1].set_ylim(0.1,0.5)
ax[2].set_ylim(0,1)

ax[0].vlines((lim10,lim14), 0.7, 1.05, colors=(colors[1], colors[2]), label=("1$\sigma$","1.4$\sigma$"))
ax[1].vlines((lim10,lim14), 0.1, 0.5, colors=(colors[1], colors[2]))
ax[2].vlines((lim10,lim14), 0, 1, colors=(colors[1], colors[2]))

ax[0].set_title("(a)")
ax[1].set_title("(b)")
ax[2].set_title("(c)")

fig.savefig("NQO1_refine.png")
fig.savefig("NQO1_refine.eps")
