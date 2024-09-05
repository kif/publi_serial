#!/bin/env python

import matplotlib
from matplotlib.pyplot import subplots
labels = ["initial", "sparse 0.8$\\sigma$","sparse 1.0$\\sigma$","sparse 1.4$\\sigma$","sparse 2.0$\\sigma$"]
Rwork = [29.11,29.98,30.03,30.61, 31.23]
Rfree = [34.41,34.58,35.15,35.70,36.94]

fig,ax=subplots()
ax.plot(Rwork, label="Rwork")
ax.plot(Rfree, label="Rfree")
ax.set_xticks(range(5), labels)
ax.set_ylabel("R-factor (%)")
ax.set_ylim(25, 40)
ax.legend()
fig.savefig("fig_Rfree.eps")
fig.savefig("fig_Rfree.png")
fig.show()
