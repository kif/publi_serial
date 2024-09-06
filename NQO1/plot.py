#!/usr/bin/env python
from matplotlib.pyplot import subplots
import numpy
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

q,cc_dense,_,d,_,_=numpy.loadtxt("CC_run_dense.dat",skiprows=1).T
q,cc_s10,_,d,_,_=numpy.loadtxt("CC_run_1sigma.dat",skiprows=1).T
q,cc_s14,_,d,_,_=numpy.loadtxt("CC_run_1p4sigma.dat",skiprows=1).T

m=[0,1,2,4,6,8,11,15,19]

fig,ax = subplots(3)
ax[2].plot(q, cc_dense, label="Initial")
ax[2].plot(q, cc_s10, label="Sparse 1.0$\\sigma$")
ax[2].plot(q, cc_s14, label="Sparse 1.4$\\sigma$")
ax[2].legend()
ax[2].set_ylabel("CC½")
ax[2].set_xlabel(r"d-spacing ($\AA$)")
ax[2].set_xticks(q[m], d[m])
#ax[0].set_xlabel("")
#ax[0].set_xticks(q[m], ["" for i in q[m]])

########
q,_,_,_,_,_,snr_dense,_,d,_,_=numpy.loadtxt("snr_run_dense.dat",skiprows=1).T
q,_,_,_,_,_,snr_s10,_,d,_,_=numpy.loadtxt("snr_run_1sigma.dat",skiprows=1).T
q,_,_,_,_,_,snr_s14,_,d,_,_=numpy.loadtxt("snr_run_1p4sigma.dat",skiprows=1).T
ax[0].plot(q, snr_dense, label="Initial")
ax[0].plot(q, snr_s10, label="Sparse 1.0$\\sigma$")
ax[0].plot(q, snr_s14, label="Sparse 1.4$\\sigma$")
#ax[0].legend()
ax[0].set_ylabel("$<I>/\sigma$")
#ax[0].set_xlabel(r"d-spacing ($\AA$)")
ax[0].set_xticks(q[m], ["" for i in d[m]])

########
q,Rsplit_dense,_,d,_,_=numpy.loadtxt("Rsplit_run_dense.dat",skiprows=1).T
q,Rsplit_s10,_,d,_,_=numpy.loadtxt("Rsplit_run_1sigma.dat",skiprows=1).T
q,Rsplit_s14,_,d,_,_=numpy.loadtxt("Rsplit_run_1p4sigma.dat",skiprows=1).T
ax[1].plot(q, Rsplit_dense, label="Initial")
ax[1].plot(q[:15], Rsplit_s10[:15], label="Sparse 1.0$\\sigma$")
ax[1].plot(q[:15], Rsplit_s14[:15], label="Sparse 1.4$\\sigma$")
#ax[1].legend()
ax[1].set_ylabel(r"Rsplit (%)")
#ax[1].set_xlabel(r"d-spacing ($\AA$)")
ax[1].set_xticks(q[m], ["" for i in d[m]])

lim10=root_scalar(lambda x:interp1d(q, snr_dense)(x)-1.0, x0=1.92, x1=3.3).root
lim14=root_scalar(lambda x:interp1d(q, snr_dense)(x)-1.4, x0=1.92, x1=3.3).root

ax[0].set_ylim(0,8)
ax[1].set_ylim(0,100)
ax[2].set_ylim(0,1)

ax[0].vlines((lim10,lim14), 0, 8, colors=("orange", "green"), label=("1$\sigma$","1.4$\sigma$"))
ax[1].vlines((lim10,lim14), 0, 100, colors=("orange", "green"))
ax[2].vlines((lim10,lim14), 0, 1, colors=("orange", "green"))

ax[0].set_title("(a)")
ax[1].set_title("(b)")
ax[2].set_title("(c)")

fig.savefig("NQO1.png")
fig.savefig("NQO1.eps")
