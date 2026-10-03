"""
Visualisation helpers for Bayesian Optimisation runs on 1D and 2D input
functions. Designed to plug directly into the objects produced by
`bayes_tools.generate_next_point` (a fitted GP, an acquisition function,
and the proposed next point).

Usage
-----
    from bayes_tools import generate_next_point, ucb_acquisition
    from viz_tools import plot_1d_bo, plot_2d_bo

    x_next, gp = generate_next_point(X, y, bounds, acquisition="ucb", kappa=5.0)

    # 1D problem
    plot_1d_bo(X, y, bounds, gp, ucb_acquisition, x_next,
               acq_kwargs={"kappa": 5.0, "maximize": True})

    # 2D problem
    plot_2d_bo(X, y, bounds, gp, ucb_acquisition, x_next,
               acq_kwargs={"kappa": 5.0, "maximize": True})
"""

import numpy as np
import matplotlib.pyplot as plt

from bayes_tools import normalize


def _check_dim(bounds, expected):
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    if d != expected:
        raise ValueError(
            f"This plotting function is for {expected}D inputs, but bounds "
            f"implies dimension {d}."
        )
    return bounds


def plot_1d_bo(
    X,
    y,
    bounds,
    gp,
    acquisition_fn,
    x_next,
    acq_kwargs=None,
    n_grid=300,
    true_optimum=None,
    figsize=(8, 6),
):
    """
    Plot GP posterior (mean + 95% CI) with observations and the proposed
    next point on top, and the acquisition function on a panel below.

    Parameters
    ----------
    X, y : observed data, shapes (n, 1) and (n,)
    bounds : array-like, shape (1, 2)
    gp : fitted GaussianProcessRegressor (from bayes_tools.fit_gp)
    acquisition_fn : callable(X_norm, gp, **acq_kwargs) -> scores
        e.g. bayes_tools.ucb_acquisition or max_variance_acquisition
    x_next : array-like, shape (1,)
        Proposed next point, in original (unnormalised) units.
    acq_kwargs : dict, optional
        Extra kwargs passed to acquisition_fn (e.g. {"kappa": 5.0}).
    n_grid : int
        Resolution of the plotting grid.
    true_optimum : float, optional
        If known (e.g. for a synthetic benchmark), draws a vertical
        reference line.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    bounds = _check_dim(bounds, 1)
    acq_kwargs = acq_kwargs or {}

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_next = np.asarray(x_next, dtype=float).ravel()

    x_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_grid).reshape(-1, 1)
    x_grid_norm = normalize(x_grid, bounds)

    mu, sigma = gp.predict(x_grid_norm, return_std=True)
    acq_vals = acquisition_fn(x_grid_norm, gp, **acq_kwargs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(x_grid, mu, "b-", lw=2, label="GP mean")
    ax1.fill_between(
        x_grid.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma,
        alpha=0.2, color="blue", label="95% CI",
    )

    # Colour observations by iteration order to show search trajectory.
    order = np.arange(len(y))
    sc = ax1.scatter(
        X.ravel(), y, c=order, cmap="autumn_r", edgecolor="black",
        zorder=5, label="Observations", s=50,
    )
    if len(y) > 1:
        cb = fig.colorbar(sc, ax=ax1, pad=0.01)
        cb.set_label("Iteration order")

    ax1.axvline(x_next[0], color="red", linestyle="--", lw=1.5, label="Next point")
    if true_optimum is not None:
        ax1.axvline(true_optimum, color="green", linestyle=":", lw=1.5, label="True optimum")

    ax1.set_ylabel("f(x)")
    ax1.legend(loc="best", fontsize=9)
    ax1.set_title("GP posterior and observations")

    ax2.plot(x_grid, acq_vals, "g-", lw=2)
    ax2.axvline(x_next[0], color="red", linestyle="--", lw=1.5)
    ax2.fill_between(x_grid.ravel(), acq_vals, alpha=0.15, color="green")
    ax2.set_ylabel("Acquisition")
    ax2.set_xlabel("x")
    ax2.set_title("Acquisition function")

    plt.tight_layout()
    return fig


def plot_2d_bo(
    X,
    y,
    bounds,
    gp,
    acquisition_fn,
    x_next,
    acq_kwargs=None,
    n_grid=100,
    true_optimum=None,
    figsize=(16, 5),
):
    """
    Plot GP mean, GP uncertainty, and acquisition function as filled
    contours over the 2D input domain, with observations and the proposed
    next point overlaid on each panel.

    Parameters
    ----------
    X, y : observed data, shapes (n, 2) and (n,)
    bounds : array-like, shape (2, 2)
    gp : fitted GaussianProcessRegressor (from bayes_tools.fit_gp)
    acquisition_fn : callable(X_norm, gp, **acq_kwargs) -> scores
    x_next : array-like, shape (2,)
        Proposed next point, in original (unnormalised) units.
    acq_kwargs : dict, optional
    n_grid : int
        Resolution per axis of the plotting grid (n_grid^2 GP predictions).
    true_optimum : array-like, shape (2,), optional
        If known, marked with a green star.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    bounds = _check_dim(bounds, 2)
    acq_kwargs = acq_kwargs or {}

    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_next = np.asarray(x_next, dtype=float).ravel()

    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    grid_norm = normalize(grid, bounds)

    mu, sigma = gp.predict(grid_norm, return_std=True)
    mu = mu.reshape(X1.shape)
    sigma = sigma.reshape(X1.shape)
    acq = acquisition_fn(grid_norm, gp, **acq_kwargs).reshape(X1.shape)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (mu, "GP mean", "viridis"),
        (sigma, "GP std (uncertainty)", "magma"),
        (acq, "Acquisition", "plasma"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        cf = ax.contourf(X1, X2, data, levels=30, cmap=cmap)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

        # Observations coloured by y value, same colormap as the mean
        # panel for the GP-mean plot, otherwise white for contrast.
        if title == "GP mean":
            ax.scatter(
                X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="black",
                s=50, zorder=5, vmin=data.min(), vmax=data.max(),
            )
        else:
            ax.scatter(
                X[:, 0], X[:, 1], c="white", edgecolor="black",
                s=50, zorder=5,
            )

        ax.scatter(
            x_next[0], x_next[1], c="red", marker="*", s=300,
            edgecolor="black", zorder=6, label="Next point",
        )
        if true_optimum is not None:
            true_optimum = np.asarray(true_optimum, dtype=float)
            ax.scatter(
                true_optimum[0], true_optimum[1], c="lime", marker="*", s=300,
                edgecolor="black", zorder=6, label="True optimum",
            )

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    return fig
