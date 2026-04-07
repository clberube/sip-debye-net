#
# Author: Charles L. Bérubé
# Created on: Tue Sep 10 2024
#
# Copyright (c) 2024 CL Bérubé JL Gagnon & S Gagnon
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utilities import restore_minor_ticks_log_plot


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
        label=r"$\mathcal{L}_\mathrm{NLL}$",
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


def plot_fit(all_results, samples=None, geom_factors=None, save=False):
    # Assume:
    # data_all, err_all: tensors (n_samples, n_freq)
    # Z_mu_pred, Z_l95_pred_real, Z_u95_pred_real, Z_l95_pred_imag, Z_u95_pred_imag: np arrays (n_samples, n_freq)
    # freq_all: torch tensor (n_samples, n_freq)
    # f_plot identical across samples (if same freq grid)
    f_plot = all_results["frequencies"].squeeze()
    # Mean and confidence intervals across repetitions
    Z_mu_pred = all_results["Z_pred"].mean(axis=1)
    Z_l95_pred_real = np.quantile(all_results["Z_pred"].real, 0.025, axis=1)
    Z_u95_pred_real = np.quantile(all_results["Z_pred"].real, 0.975, axis=1)
    Z_l95_pred_imag = np.quantile(all_results["Z_pred"].imag, 0.025, axis=1)
    Z_u95_pred_imag = np.quantile(all_results["Z_pred"].imag, 0.975, axis=1)

    data_all = all_results["data_all"]
    err_all = all_results["err_all"]

    if samples is not None:
        sample_list = samples
    else:
        sample_list = range(data_all.shape[0])

    fig, axs = plt.subplots(2, 1, sharex=True)

    model_labels = [None, None, None, None]
    data_labels = [
        r"Pyrite $< 0.5$ mm $10\,\%$",
        r"Graphite 0.044 mm $2.5\,\%$",
        r"Canadian Malartic field",
        r"Highland Valley core",
    ]
    data_markers = [".", "H", "X", "^"]

    z_orders = [5, 4, 6, 7]

    if geom_factors is None:
        geom_factors = np.ones(len(samples))

    for j, i in enumerate(sample_list):

        rho_factor = (
            all_results["rho0_pred"].mean(1).squeeze()
            * all_results["R0"].squeeze()
            * geom_factors[j]
        )

        # ---- Real part ----
        ax = axs[0]
        ax.errorbar(
            f_plot,
            data_all[i].real * rho_factor[i],
            err_all[i].real * rho_factor[i],
            marker=data_markers[j],
            mfc="w",
            color="k",
            linestyle="none",
            zorder=z_orders[j],
            label=data_labels[j],
        )
        ax.plot(
            f_plot,
            Z_mu_pred[i].real * rho_factor[i],
            color="0.5",
            label=model_labels[j],
        )
        ax.plot(
            f_plot,
            Z_l95_pred_real[i] * rho_factor[i],
            color="0.5",
            lw=0.5,
            ls=":",
        )
        ax.plot(
            f_plot,
            Z_u95_pred_real[i] * rho_factor[i],
            color="0.5",
            lw=0.5,
            ls=":",
        )
        ax.set_ylabel(r"$\rho^{\prime} (\Omega\,\mathrm{m})$")

        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )

        ax.set_ylim([3e1, 1.5e4])

        # ---- Imag part ----
        ax = axs[1]
        ax.errorbar(
            f_plot,
            -data_all[i].imag * rho_factor[i],
            err_all[i].imag * rho_factor[i],
            marker=data_markers[j],
            color="k",
            mfc="w",
            linestyle="none",
            zorder=z_orders[j],
        )
        ax.plot(
            f_plot,
            -Z_mu_pred[i].imag * rho_factor[i],
            color="0.5",
        )
        ax.plot(
            f_plot,
            -Z_l95_pred_imag[i] * rho_factor[i],
            color="0.5",
            lw=0.5,
            ls=":",
        )
        ax.plot(
            f_plot,
            -Z_u95_pred_imag[i] * rho_factor[i],
            color="0.5",
            lw=0.5,
            ls=":",
        )
        ax.set_ylabel(r"$-\rho^{\prime\prime} (\Omega\,\mathrm{m})$")
        ax.set_xlabel(r"$f$ (Hz)")

    for a in axs:
        a.set_xscale("log")
        a.set_yscale("log")

    restore_minor_ticks_log_plot(axs[1], axis="x")
    restore_minor_ticks_log_plot(axs[1], axis="y")

    # ---- Save or show ----
    if save:
        fig_path = f"{save}/FIT-examples.pdf"
        plt.savefig(fig_path)
    else:
        plt.show()
