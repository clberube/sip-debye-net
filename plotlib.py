#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# My style sheet
plt.style.use("seg.mplstyle")


# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = "%.1f"  # Show 2 decimals


def plot_learning_curves(losses, save=None):
    fig, ax = plt.subplots()
    idx = losses["train"] != 0

    nll = losses["NLL"][idx].astype(float)
    nll = nll - nll.min()  # + 1e-12n
    nll_plot = nll.copy()
    nll_plot[nll_plot == 0] = np.nan
    kld = losses["KLD"][idx].astype(float)

    ax.plot(
        range(1, sum(idx) + 1),
        nll_plot,
        ls="-",
        color="0.5",
        label=r"$\mathcal{L}_\textrm{NLL}$",
    )
    ax.plot(
        range(1, sum(idx) + 1),
        kld,
        ls="--",
        color="0.0",
        label=r"$D_\mathrm{KL}$",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(ncol=2)

    if save:
        plt.savefig(f"{save}/LC.pdf")
    else:
        plt.show()
