#!/bin/env python

import matplotlib
from matplotlib.pyplot import subplots
labels = ["initial", "sparse 0.8$\\sigma$","sparse 1.0$\\sigma$","sparse 1.4$\\sigma$","sparse 2.0$\\sigma$"]
Rwork = [29.11,29.98,30.03,30.61, 31.23]
Rfree = [34.41,34.58,35.15,35.70,36.94]
compression = [1.0,
 1.3360896819151762,
 1.6289966404472827,
 2.6298891396823607,
 5.856469946283841]
# Updated values on Nicolas's work:
compression = [1.0,
               2.0754537490671,
               2.5743029975898457,
            #1.2:3.3268870018221746
               4.353055919259339,
               10.454538113428335]


fig,ax=subplots()
lns1 = ax.plot(Rwork, label="R$_{work}$ (%)")
lns2 = ax.plot(Rfree, label="R$_{free}$ (%)")
ax.set_xticks(range(5), labels)
ax.set_ylabel("R-factor (%)")
ax.set_ylim(25, 40)
ax2 = ax.twinx()
lns3 = ax2.plot(compression, "tab:green", label=r"Compression ratio ($\times$)")
ax2.set_ylabel(r"Compression ratio ($\times$)")

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="upper left")
fig.savefig("fig_Rfree.eps")
fig.savefig("fig_Rfree.png")
fig.show()
