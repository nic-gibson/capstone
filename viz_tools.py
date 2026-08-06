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


# ---------------------------------------------------------------------------
# History-level diagnostics (no true optimum required)
# ---------------------------------------------------------------------------
#
# These consume the history dict produced by
# bayes_tools.compute_iteration_diagnostics (built by replaying your
# ordered X/y -- initial batch plus every append_observations call -- no
# persisted state involved), and answer "is this actually converging?"
# across iterations rather than "what does the model believe right now?"
# (which plot_2d_bo already covers for a single snapshot).
#
# Fields like acq_value/domain_mean_std are NaN for the initial-batch rows
# (nothing was "proposed" for those, they just arrived) and are excluded
# automatically by the has_* masks below.

def _iter_mask_observed(history):
    return ~np.isnan(history["y"])


def plot_convergence(history, maximize=True, figsize=(6, 4)):
    """
    Best observed y so far vs. iteration. No true optimum needed -- look
    for the curve flattening out as the signal that search has converged.
    """
    obs = _iter_mask_observed(history)
    it = history["iteration"][obs]
    y = history["y"][obs]
    order = np.argsort(it)
    it, y = it[order], y[order]

    best_so_far = np.maximum.accumulate(y) if maximize else np.minimum.accumulate(y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(it, best_so_far, marker="o", color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best y observed so far")
    ax.set_title("Convergence (best-so-far)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_acquisition_decay(history, figsize=(6, 4)):
    """
    Acquisition value at each *proposed* point vs. iteration. A decaying
    trend means the model believes there's less and less to be gained
    anywhere in the domain -- the closest thing to a "regret" signal you
    get without a known true optimum. Excludes the initial batch (nothing
    was proposed for those points, so there's no acquisition value).
    """
    has_acq = ~np.isnan(history["acq_value"])
    it = history["iteration"][has_acq]
    acq = history["acq_value"][has_acq]
    order = np.argsort(it)
    it, acq = it[order], acq[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(it, acq, marker="o", color="tab:green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acquisition value at proposed point")
    ax.set_title("Acquisition decay")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_uncertainty_shrinkage(history, figsize=(6, 4)):
    """
    Average GP predictive std over the whole domain, at each proposal
    time. Should trend downward as observations accumulate; a plateau
    suggests the GP isn't learning much more from new points (either
    because it's converged, or because new points are too close together
    to reduce uncertainty elsewhere).
    """
    has_val = ~np.isnan(history["domain_mean_std"])
    it = history["iteration"][has_val]
    std = history["domain_mean_std"][has_val]
    order = np.argsort(it)
    it, std = it[order], std[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(it, std, marker="o", color="tab:purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean predictive std over domain")
    ax.set_title("Uncertainty shrinkage")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sample_trajectory(history, bounds, figsize=(6, 6)):
    """
    2D scatter of every point evaluated so far (initial batch + proposals),
    coloured by iteration order with a line connecting them in sequence.
    Shows whether the search is exploring broadly, clustering into a
    region, or oscillating between distant regions.
    """
    bounds = _check_dim(bounds, 2)
    obs = _iter_mask_observed(history)
    X = history["X"][obs]
    it = history["iteration"][obs]
    order = np.argsort(it)
    X, it = X[order], it[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X[:, 0], X[:, 1], "-", color="gray", alpha=0.4, lw=1, zorder=1)
    sc = ax.scatter(
        X[:, 0], X[:, 1], c=it, cmap="autumn_r", edgecolor="black",
        s=60, zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Iteration order")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Sample trajectory")
    plt.tight_layout()
    return fig


def plot_step_distance(history, figsize=(6, 4)):
    """
    Euclidean distance between consecutive evaluated points vs. iteration.
    A shrinking trend indicates the search is localising (exploitation);
    persistent large jumps mean the acquisition function keeps exploring
    far-apart regions -- worth checking against kappa/xi if unintended.
    """
    obs = _iter_mask_observed(history)
    X = history["X"][obs]
    it = history["iteration"][obs]
    order = np.argsort(it)
    X, it = X[order], it[order]

    dist = np.linalg.norm(np.diff(X, axis=0), axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(it[1:], dist, marker="o", color="tab:orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance from previous point")
    ax.set_title("Step distance between consecutive points")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bo_diagnostics(history, bounds, maximize=True, figsize=(12, 9)):
    """
    Convenience dashboard combining convergence, acquisition decay,
    uncertainty shrinkage, and sample trajectory in one 2x2 figure.
    """
    bounds = _check_dim(bounds, 2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    obs = _iter_mask_observed(history)
    it_obs = history["iteration"][obs]
    y_obs = history["y"][obs]
    order = np.argsort(it_obs)
    it_obs, y_obs = it_obs[order], y_obs[order]
    best_so_far = np.maximum.accumulate(y_obs) if maximize else np.minimum.accumulate(y_obs)

    ax = axes[0, 0]
    ax.plot(it_obs, best_so_far, marker="o", color="tab:blue")
    ax.set_title("Convergence (best-so-far)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best y so far")
    ax.grid(alpha=0.3)

    has_acq = ~np.isnan(history["acq_value"])
    it_acq = history["iteration"][has_acq]
    acq = history["acq_value"][has_acq]
    o = np.argsort(it_acq)
    ax = axes[0, 1]
    ax.plot(it_acq[o], acq[o], marker="o", color="tab:green")
    ax.set_title("Acquisition decay")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acquisition value")
    ax.grid(alpha=0.3)

    has_std = ~np.isnan(history["domain_mean_std"])
    it_std = history["iteration"][has_std]
    std = history["domain_mean_std"][has_std]
    o = np.argsort(it_std)
    ax = axes[1, 0]
    ax.plot(it_std[o], std[o], marker="o", color="tab:purple")
    ax.set_title("Uncertainty shrinkage")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean predictive std")
    ax.grid(alpha=0.3)

    X_obs = history["X"][obs][order]
    ax = axes[1, 1]
    ax.plot(X_obs[:, 0], X_obs[:, 1], "-", color="gray", alpha=0.4, lw=1, zorder=1)
    sc = ax.scatter(
        X_obs[:, 0], X_obs[:, 1], c=it_obs, cmap="autumn_r",
        edgecolor="black", s=50, zorder=2,
    )
    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046).set_label("Iteration")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title("Sample trajectory")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Higher-dimensional (4D-8D) diagnostics
# ---------------------------------------------------------------------------
#
# Past 3D you can no longer plot the whole surface, so these answer the
# same questions as plot_1d_bo / plot_2d_bo in a way that scales: 1D
# slices through the region you care about (plot_nd_slices), and a
# numerical check on whether the surrogate model itself is trustworthy
# (plot_loo_calibration), rather than a direct picture of the full
# D-dimensional surface.

def plot_nd_slices(
    X, y, bounds, gp, acquisition_fn, x_next,
    acq_kwargs=None, center=None, dims=None, n_grid=200,
    maximize=True, ncols=4, obs_tol_frac=0.1, figsize=None,
):
    """
    Generalises plot_1d_bo to D > 3: hold every dimension fixed at
    `center` except one, sweep that one dimension across its bounds, and
    plot the GP mean +/- 95% CI (left axis) with the acquisition function
    overlaid (right axis, green). One panel per dimension in `dims`.

    This is a *partial* view -- it shows what the GP believes along each
    axis near the region you care about, not interactions between
    dimensions -- but it's the direct generalisation of what plot_1d_bo
    and plot_2d_bo already show you.

    Parameters
    ----------
    X, y : observed data, shapes (n, D) and (n,)
    bounds : array-like, shape (D, 2)
    gp : fitted GaussianProcessRegressor
    acquisition_fn : callable(X_norm, gp, **acq_kwargs) -> scores
    x_next : array-like, shape (D,)
        Proposed next point -- marked with a red dashed line on each slice.
    center : array-like, shape (D,), optional
        Point to hold fixed while slicing each dimension. Defaults to the
        best observed point so far (the region you actually care about),
        rather than an arbitrary point like the domain centre.
    dims : list of int, optional
        Which dimensions to slice. Defaults to all of them. Use
        `bayes_tools.get_length_scales` to rank dimensions by sensitivity
        and pass only the most sensitive ones here if D is large and you
        don't want 8 panels.
    obs_tol_frac : float
        Observations get overlaid on a slice only if every *other*
        dimension is within `obs_tol_frac` of that dimension's range from
        `center` -- otherwise a point evaluated somewhere completely
        different in the other D-1 dimensions would misleadingly appear
        "on" this slice.

    Returns
    -------
    fig : matplotlib Figure
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_next = np.asarray(x_next, dtype=float).ravel()
    acq_kwargs = acq_kwargs or {}

    if center is None:
        center = X[np.argmax(y)] if maximize else X[np.argmin(y)]
    center = np.asarray(center, dtype=float).ravel()

    if dims is None:
        dims = list(range(d))

    n_panels = len(dims)
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, dim in zip(axes_flat, dims):
        grid_vals = np.linspace(bounds[dim, 0], bounds[dim, 1], n_grid)
        grid = np.tile(center, (n_grid, 1))
        grid[:, dim] = grid_vals
        grid_norm = normalize(grid, bounds)

        mu, sigma = gp.predict(grid_norm, return_std=True)
        acq_vals = acquisition_fn(grid_norm, gp, **acq_kwargs)

        ax_acq = ax.twinx()
        ax.plot(grid_vals, mu, "b-", lw=1.8, zorder=3)
        ax.fill_between(grid_vals, mu - 1.96 * sigma, mu + 1.96 * sigma,
                         alpha=0.2, color="blue", zorder=2)
        ax_acq.plot(grid_vals, acq_vals, "g-", lw=1.2, alpha=0.8, zorder=1)
        ax_acq.fill_between(grid_vals, acq_vals, alpha=0.1, color="green", zorder=1)

        ax.axvline(center[dim], color="black", ls=":", lw=1, zorder=4)
        ax.axvline(x_next[dim], color="red", ls="--", lw=1.3, zorder=4)

        # Only overlay observations that are actually close to this slice
        # in every *other* dimension -- otherwise a point evaluated far
        # away in other axes would misleadingly look like it sits on
        # this line.
        other_dims = [i for i in range(d) if i != dim]
        if other_dims:
            span = bounds[other_dims, 1] - bounds[other_dims, 0]
            close = np.all(
                np.abs(X[:, other_dims] - center[other_dims]) < obs_tol_frac * span,
                axis=1,
            )
        else:
            close = np.ones(len(y), dtype=bool)
        if close.any():
            ax.scatter(X[close, dim], y[close], c="black", s=25, zorder=5)

        ax.set_title(f"x{dim}", fontsize=10)
        ax.set_ylabel("f(x)", fontsize=8, color="tab:blue")
        ax_acq.set_ylabel("acq", fontsize=8, color="tab:green")
        ax.tick_params(labelsize=8)
        ax_acq.tick_params(labelsize=8, colors="tab:green")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"1D slices through center = {np.round(center, 3)}", fontsize=11)
    plt.tight_layout()
    return fig


def plot_loo_calibration(y, pred_mean, pred_std, figsize=(10, 4.5)):
    """
    Leave-one-out calibration check for the GP surrogate (feed it the
    output of `bayes_tools.loo_predictions`). This is the main way to
    validate the model once D is too large to visually inspect the
    fitted surface (roughly 4D+).

    Left panel: predicted vs. actual, with the LOO-predicted std as
    error bars. Points should scatter around the diagonal, with error
    bars mostly covering it -- systematic offset from the diagonal means
    the GP's *mean* predictions are biased; error bars that consistently
    miss the diagonal mean its *uncertainty* estimates aren't trustworthy
    either, which matters a lot since the acquisition function leans on
    sigma directly.

    Right panel: standardized residuals z = (actual - predicted) /
    predicted_std. Should look roughly like a standard normal (centred
    near 0, most mass within +/-2) if the GP's uncertainty is well
    calibrated, not just its mean.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Actual observed values.
    pred_mean, pred_std : np.ndarray, shape (n,)
        LOO-predicted mean/std at each point (from `loo_predictions`).
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    y = np.asarray(y, dtype=float).ravel()
    pred_mean = np.asarray(pred_mean, dtype=float).ravel()
    pred_std = np.asarray(pred_std, dtype=float).ravel()
    resid_z = (y - pred_mean) / np.maximum(pred_std, 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    lo = min(y.min(), pred_mean.min())
    hi = max(y.max(), pred_mean.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="Perfect calibration")
    ax1.errorbar(
        y, pred_mean, yerr=1.96 * pred_std, fmt="o", ecolor="gray",
        elinewidth=1, capsize=2, markersize=5, color="tab:blue",
    )
    ax1.set_xlabel("Actual y")
    ax1.set_ylabel("LOO-predicted mean (95% CI)")
    ax1.set_title("Predicted vs. actual")
    ax1.legend(fontsize=8)

    n_bins = min(15, max(5, len(resid_z) // 2))
    ax2.hist(resid_z, bins=n_bins, color="tab:orange", edgecolor="black", alpha=0.85)
    ax2.axvline(0, color="black", ls="--", lw=1)
    ax2.set_xlabel("Standardized residual  (actual - predicted) / std")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual calibration")

    plt.tight_layout()
    return fig
