import os
import matplotlib.pyplot as plt
import numpy as np 
from chainconsumer import ChainConsumer

import logging
# Turn off unconstrained parameter warnings
logging.getLogger("chainconsumer").setLevel(logging.CRITICAL) #(logging.WARNING)

# import warnings
# warnings.filterwarnings("ignore", module="chainconsumer\..*")

DPI = 400

def plot_fisher_matrix(
    alpha,
    main_covariance,
    other_covariance=None,
    parameter_names=None,
    title=None,
    filename=None
):
    c = ChainConsumer()

    alpha = np.asarray(alpha)
    main_covariance = np.asarray(main_covariance)

    if other_covariance is not None:
        other_covariance = np.asarray(other_covariance)
        c.add_covariance(
            alpha, 
            other_covariance, 
            name="$F_{Gaussian}$",
            color="blue", 
            parameters=parameter_names,
            linewidth=1.,
            shade_alpha=0.2
        )
    c.add_covariance(
        alpha, 
        main_covariance, 
        name="$F_{IMNN}$",
        color="green", 
        parameters=parameter_names,
        linewidth=1.,
        shade_alpha=0.2
    )
    c.configure(
        tick_font_size=10, 
        label_font_size=12, 
        flip=False, 
        sigma2d=False, 
        sigmas=[1, 2], 
        shade=True, 
        shade_alpha=0.2, 
        bar_shade=True, 
        smooth=1
    )
    fig = c.plotter.plot(
        # filename=filename, 
        figsize=(4., 4.), #"GROW",
        legend=True, 
        truth=alpha
    )
    if title is not None:
        fig.suptitle(title)
    fig.set_dpi(DPI)     
    plt.savefig(filename)
    plt.close()


def plot_losses(
    epoch,
    metrics,
    filename,
    targets=None
):
    """
        NOTE: plot logdet|F| (train and valid) against
        analytic target logdet|F| values... plottting
        the loss includes the regularisation of |C(x)|.
    """
    plotter = lambda ax: ax.plot
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.), dpi=100)
    (
        L_F, L_C, 
        L_valid_F, L_valid_C, 
        detC_train, detCinv_train, 
        detC_valid, detCinv_valid, 
        r
    ) = metrics[:epoch].T
    epoch_range = np.linspace(0., 1., len(metrics[:epoch])) * epoch
    # Plot losses
    ax = axs[0]
    ax.set_title(r"$\log|F|$")
    plotter(ax)(
        epoch_range, 
        L_F, 
        color="navy", 
        label=fr"$\log|F|_{{train}}$ target = {L_F[-1]:.1f}"
    )
    plotter(ax)(
        epoch_range, 
        L_valid_F, 
        color="goldenrod", 
        label=fr"$\log|F|_{{valid}}$ target = {L_valid_F[-1]:.1f}"
    )
    if len(targets) == 2:
        targets = [None] + targets
    if targets is not None:
        for target, color, label in zip(
            targets, 
            ["m", "k", "r"], 
            [
                r"$\log|F|$" + "(moments)= {:.1f}", 
                r"$\log|F|$" + "(bulk)= {:.1f}", 
                r"$\log|F|$" + "(bulk+tails)= {:.1f}"
            ]
        ):
            if target is not None:
                ax.axhline(
                    target, 
                    linestyle="--", 
                    color=color, 
                    label=label.format(target)
                )
    ax.legend()
    ax = axs[1]
    ax.set_title("$|C_f|$")
    ax.semilogy(epoch_range, detC_train, color="navy", label=r"$|C|=$" + f"{detC_train[-1]:.3f} (train)", linestyle="--")
    ax.semilogy(epoch_range, detCinv_train, color="darkorange", label=r"$|C^{-1}|=$" + f"{detCinv_train[-1]:.3f} (train)", linestyle="--")
    ax.semilogy(epoch_range, detC_valid, color="navy", label=r"$|C|=$" + f"{detC_valid[-1]:.3f} (valid)")
    ax.semilogy(epoch_range, detCinv_valid, color="darkorange", label=r"$|C^{-1}|=$" + f"{detCinv_valid[-1]:.3f} (valid)")
    ax.axhline(1., color="black", linestyle="--")
    ax.set_ylim(0.8, 1.2)
    ax.legend()
    # Plot covariance regularisation
    ax = axs[2]
    ax.set_title(r"$\Lambda(|C_f|)$")# [$r=${r:.2f}]")
    ax.semilogy(epoch_range, L_C, color="navy", label=f"{L_C[-1]:.3f} (train)")
    ax.semilogy(epoch_range, L_valid_C, color="goldenrod", label=f"{L_valid_C[-1]:.3f} (valid)")
    ax.semilogy(epoch_range, r, color="firebrick", label=r"$r_{\Lambda}$="+f"{r[-1]:.3f}")
    ax.legend()
    plt.savefig(filename)
    plt.close()


def plot_latins(alpha, x, parameters, filename):
    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(9.,3.))
    # fig, axs = plt.subplots(2, 3, dpi=100, figsize=(9.,6.))
    fig.suptitle(r"$f: d \rightarrow x$")

    ax = axs[0]
    ax.hist(x[:,0], bins=32) 
    ax.vlines(alpha[0], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),

    ax = axs[1]
    ax.hist(x[:,1], bins=32) 
    ax.vlines(alpha[1], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),

    # ax = axs[1, 0]
    # ax.hist(parameters[:, 0], bins=32)
    # ax.vlines(alpha[0], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),
    
    # ax = axs[1, 1]
    # ax.hist(parameters[:, 1], bins=32),
    # ax.vlines(alpha[1], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),

    ax = axs[2]
    ax.scatter(*x.T, s=0.5)
    # ax.hlines(alpha[1], 0, 1, transform=ax.get_yaxis_transform(), zorder=5, color="orange")
    # ax.vlines(alpha[0], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange")
    ax.axhline(alpha[1], zorder=5, color="orange")
    ax.axvline(alpha[0], zorder=5, color="orange")

    plt.savefig(filename)
    plt.close()


def plot_fiducials(alpha, x, filename):
    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(9.,3.))
    fig.suptitle(r"$f: d \rightarrow x$")

    ax = axs[0]
    ax.hist(x[:,0], bins=32, zorder=1) 
    ax.vlines(alpha[0], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),

    ax = axs[1]
    ax.hist(x[:,1], bins=32, zorder=1) 
    ax.vlines(alpha[1], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange"),

    ax = axs[2]
    ax.scatter(*x.T, s=0.5)
    # ax.hlines(alpha[1], 0, 1, transform=ax.get_yaxis_transform(), zorder=5, color="orange")
    # ax.vlines(alpha[0], 0, 1, transform=ax.get_xaxis_transform(), zorder=5, color="orange")
    ax.axhline(alpha[1], zorder=5, color="orange")
    ax.axvline(alpha[0], zorder=5, color="orange")

    plt.savefig(filename)
    plt.close()


def plot_summaries(alpha, x, names, chain_name=None, filename=None):
    c = ChainConsumer()
    c.add_chain(
        x, 
        name=chain_name if chain_name is not None else "",
        parameters=names
    ) 
    c.add_marker(
        alpha, 
        name=r"$\alpha$", 
        color="m",
        marker_size=40,
        parameters=names
    )
    fig = c.plotter.plot(
        filename=filename, 
        figsize=(4., 4.)
    )
    fig.set_dpi(DPI)     
    plt.savefig(filename)
    plt.close(fig=fig)


def summary_alpha_scatter(alphas, summaries, alpha_0=None, x_0=None, filename=None):
    fig, axs = plt.subplots(1, 3, figsize=(9., 3.), dpi=DPI)
    ax = axs[0]
    ax.set_title(r"$x$")
    ax.scatter(*summaries.T, s=0.25, label=r"$x$")
    if x_0 is not None:
        ax.scatter(*x_0.T, s=0.25, label=r"$x_0$")

    ax = axs[1]
    ax.set_title(r"$\Omega_m - x$")
    ax.scatter(alphas[:, 0], summaries[:, 0], s=0.25, label=r"$x_{\Omega_m}$")
    if alpha_0 is not None and x_0 is not None:
        ax.scatter(alpha_0[:, 0], x_0[:, 0], s=0.25, label=r"$x_{\Omega_m},0$")
    
    ax = axs[2]
    ax.set_title(r"$\sigma_8 - x$")
    ax.scatter(alphas[:, 4], summaries[:, 1], s=0.25, label=r"$x_{\sigma_8}$")
    if alpha_0 is not None and x_0 is not None:
        ax.scatter(alpha_0[:, 4], x_0[:, 1], s=0.25, label=r"$x_{\sigma_8},0$")

    for ax in axs: ax.legend()

    plt.savefig(filename)#os.path.join(os.getcwd(), results_dir, filename))
    plt.close()