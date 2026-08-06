import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

def normalize(X, bounds):
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (np.asarray(X, dtype=float) - lo) / (hi - lo)


def denormalize(X_norm, bounds):
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return np.asarray(X_norm, dtype=float) * (hi - lo) + lo

def ucb_acquisition(X_norm, gp, kappa=5.0, maximize=True):
    """
    Upper Confidence Bound. Exploration-heavy by default (large kappa).
    Returns higher values for points that should be evaluated next.
    """
    mu, sigma = gp.predict(X_norm, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    if maximize:
        return mu + kappa * sigma
    else:
        return -mu + kappa * sigma


def max_variance_acquisition(X_norm, gp):
    """
    Pure exploration: score is just the predictive standard deviation.
    Ignores the predicted mean entirely.
    """
    _, sigma = gp.predict(X_norm, return_std=True)
    return sigma


def exploit_acquisition(X_norm, gp, maximize=True):
    """
    Pure exploitation: score is just the predicted mean. Ignores
    uncertainty entirely -- the counterpart to `max_variance_acquisition`.
    Always pushes toward the GP's current best guess rather than toward
    unexplored regions, so use this once you trust the surrogate (e.g.
    late in optimisation, or when evaluations are expensive and you want
    to converge rather than keep exploring).
    """
    mu, _ = gp.predict(X_norm, return_std=True)
    return mu if maximize else -mu


def pi_acquisition(X_norm, gp, y_best, xi=0.01, maximize=True):
    """
    Probability of Improvement. Returns the probability, under the GP's
    posterior, that a point beats the best observation so far by at
    least `xi`.

    Compared to UCB, PI is more exploitative: it cares about the *chance*
    of improving at all, not *how much* you might improve by, so once a
    region is fairly confidently better than the current best it gets
    close to a score of 1 regardless of how much sigma remains -- it
    won't keep chasing extra uncertainty the way UCB or max_variance do.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalised input points to score.
    gp : fitted GaussianProcessRegressor
    y_best : float
        Best observed y value so far (max(y) if maximizing, min(y) if
        minimizing). This is data, not a GP prediction, so it must be
        passed in explicitly -- there's no way to infer it from `gp` or
        `X_norm` alone.
    xi : float
        Small "improvement margin" (in raw y-units). Higher xi demands a
        more decisive improvement before a point scores well, which
        nudges PI to explore a bit more; xi=0 makes PI purely greedy
        about any improvement, however marginal.
    maximize : bool
        Whether the underlying function is being maximised.

    Returns
    -------
    scores : np.ndarray
        Probability of improvement at each point, in [0, 1].
    """
    mu, sigma = gp.predict(X_norm, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    if maximize:
        z = (mu - y_best - xi) / sigma
    else:
        z = (y_best - mu - xi) / sigma
    return norm.cdf(z)


def ei_acquisition(X_norm, gp, y_best, xi=0.01, maximize=True):
    """
    Expected Improvement. The classic "balanced" acquisition function --
    unlike PI, which only asks *whether* a point might improve on y_best,
    EI weighs that by *how much* it might improve by. A point with a
    small but near-certain gain and a point with a large but uncertain
    gain can score similarly, so EI naturally trades exploration against
    exploitation without a manual weight like UCB's `kappa`.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalised input points to score.
    gp : fitted GaussianProcessRegressor
    y_best : float
        Best observed y value so far (max(y) if maximizing, min(y) if
        minimizing) -- data, not a GP prediction, so pass it explicitly.
    xi : float
        Small "improvement margin" (in raw y-units), same role as in
        `pi_acquisition`: higher xi demands a more decisive improvement
        and nudges EI toward more exploration.
    maximize : bool
        Whether the underlying function is being maximised.

    Returns
    -------
    scores : np.ndarray
        Expected improvement at each point, in raw y-units (>= 0).
    """
    mu, sigma = gp.predict(X_norm, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    if maximize:
        imp = mu - y_best - xi
    else:
        imp = y_best - mu - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


_ACQUISITIONS = {
    "ucb": ucb_acquisition,
    "max_variance": max_variance_acquisition,
    "exploit": exploit_acquisition,
    "pi": pi_acquisition,
    "ei": ei_acquisition,
}




def initial_bounds(X, pad_fraction=1.0, lower_limit=None, upper_limit=None):
    """
    Construct a generous starting box from observed data alone, for use
    when you have no domain knowledge about valid input ranges.

    Note: pad_fraction defaults to 1.0 (i.e. the box is roughly double the
    observed range) rather than a small pad like 0.1. With no domain
    knowledge, a small pad just guarantees the box is wrong almost
    immediately -- better to start wide and rely on `expand_bounds_if_needed`
    to keep growing it, than to start tight and hope you got lucky.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, D)
    pad_fraction : float
        Fraction of the observed range to add on *each* side.
    lower_limit, upper_limit : float or array-like, optional
        Hard floor/ceiling the padded bounds are not allowed to cross,
        applied per-dimension after padding (broadcasts if a scalar).
        Use this when you have external domain knowledge about valid
        input ranges -- e.g. lower_limit=0.0 if X is known to never be
        negative -- since data-derived padding alone has no way to know
        that and will happily propose a lower bound below what's
        actually physically achievable. Without this, `generate_next_point`
        can and will propose x_next values outside the true valid domain.

    Returns
    -------
    bounds : np.ndarray, shape (D, 2)
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    low = X.min(axis=0)
    high = X.max(axis=0)
    span = high - low
    low_pad = low - pad_fraction * span
    high_pad = high + pad_fraction * span
    if lower_limit is not None:
        low_pad = np.maximum(low_pad, lower_limit)
    if upper_limit is not None:
        high_pad = np.minimum(high_pad, upper_limit)
    return np.column_stack([low_pad, high_pad])


def clip_bounds(bounds, lower_limit=None, upper_limit=None):
    """
    Clip an existing bounds array to a known hard floor/ceiling per
    dimension, without recomputing it from data. Useful for a manually
    specified bounds array, or one that's already been through several
    rounds of widening, that needs to respect a known domain constraint
    -- e.g. clip_bounds(bounds, lower_limit=0.0) if X is known to never
    be negative.

    Parameters
    ----------
    bounds : array-like, shape (D, 2)
    lower_limit, upper_limit : float or array-like, optional

    Returns
    -------
    bounds : np.ndarray, shape (D, 2)
    """
    bounds = np.asarray(bounds, dtype=float).copy()
    if lower_limit is not None:
        bounds[:, 0] = np.maximum(bounds[:, 0], lower_limit)
    if upper_limit is not None:
        bounds[:, 1] = np.minimum(bounds[:, 1], upper_limit)
    return bounds


def validate_bounds_consistency(X, bounds):
    """
    Sanity-check that every observed X point actually falls within
    `bounds`. Catches the case where padding, a manual bounds override,
    or a known domain constraint (e.g. "X >= 0") has drifted out of sync
    with the data actually being fed to the GP -- worth calling any time
    bounds are constructed or modified before passing them into
    `generate_next_point`, `fit_gp`, or the plotting tools.

    Raises
    ------
    ValueError
        If any observed X value falls outside `bounds`, naming which
        dimension(s) are affected.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    bounds = np.asarray(bounds, dtype=float)
    below = X < bounds[:, 0]
    above = X > bounds[:, 1]
    if below.any() or above.any():
        bad_dims = sorted(set(np.where(below | above)[1]))
        raise ValueError(
            f"Some observed X values fall outside `bounds` in dimension(s) "
            f"{bad_dims}. Check that bounds reflect the true valid domain "
            "(e.g. a known floor/ceiling clipped via clip_bounds/initial_bounds)."
        )



def generate_next_point(
    X,
    y,
    bounds,
    acquisition="ucb",
    kappa=5.0,
    xi=0.01,
    maximize=True,
    n_restarts=25,
    random_state=None,
    gp_kwargs=None,
):
    """
    Given existing observations (X, y) for an unknown function of dimension
    D, fit (or reuse) a GP and propose the next point to evaluate by
    maximising an exploration-weighted acquisition function.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, D)
    y : np.ndarray, shape (n_samples,)
    bounds : array-like, shape (D, 2)
        [(low, high), ...] per dimension.
    acquisition : {"ucb", "max_variance", "exploit", "pi", "ei"}
    kappa : float
        Exploration weight, only used for "ucb". Higher = more exploration.
    xi : float
        Improvement margin, used by "pi" and "ei" (Probability/Expected
        Improvement). Higher xi demands a more decisive improvement
        before a point scores well, nudging both toward more exploration;
        xi=0 makes them purely greedy about any improvement, however small.
    maximize : bool
        Whether the underlying black-box function is being maximised
        (set False if you are minimising it). Used by "ucb", "exploit",
        "pi", and "ei"; not used for "max_variance" (which ignores the
        mean entirely).
    n_restarts : int
        Number of random multi-starts for optimising the acquisition
        function (acquisition surfaces can be multimodal, especially in
        higher dimensions, so more restarts helps in 6D-8D problems).
    random_state : int or None
    gp_kwargs : dict, optional
        Extra keyword arguments forwarded to `fit_gp`, e.g.
        {"noise_level": 0.1} if you know your function is noisy, or
        {"n_restarts_optimizer": 20} for trickier likelihood surfaces.
        See `fit_gp` for the full list of options.

    Returns
    -------
    x_next : np.ndarray, shape (D,)
        Next point to evaluate, in the original (unnormalised) units.
    gp : GaussianProcessRegressor
        The fitted GP (useful for diagnostics, e.g. checking predicted
        mean/std at x_next, or inspecting gp.kernel_ to see the learned
        noise level and length-scale).
    """
    if acquisition not in _ACQUISITIONS:
        raise ValueError(f"Unknown acquisition '{acquisition}'. Choose from {list(_ACQUISITIONS)}.")

    rng = np.random.default_rng(random_state)
    bounds_arr = np.asarray(bounds, dtype=float)
    d = bounds_arr.shape[0]

    X = np.atleast_2d(np.asarray(X, dtype=float))

    gp_kwargs = gp_kwargs or {}
    gp = fit_gp(X, y, bounds_arr, random_state=random_state, **gp_kwargs)

    # Build the kwargs each acquisition function actually accepts, then
    # dispatch generically -- avoids a hardcoded if/else per acquisition
    # that would need editing every time a new one is added.
    if acquisition == "ucb":
        acq_kwargs = {"kappa": kappa, "maximize": maximize}
    elif acquisition == "exploit":
        acq_kwargs = {"maximize": maximize}
    elif acquisition in ("pi", "ei"):
        y_arr = np.asarray(y, dtype=float).ravel()
        y_best = y_arr.max() if maximize else y_arr.min()
        acq_kwargs = {"y_best": y_best, "xi": xi, "maximize": maximize}
    else:  # max_variance
        acq_kwargs = {}

    acq_fn = _ACQUISITIONS[acquisition]

    def neg_acq(x_norm):
        return -acq_fn(x_norm.reshape(1, -1), gp, **acq_kwargs)[0]

    # Multi-start L-BFGS-B over the normalised unit hypercube [0, 1]^d.
    best_x_norm, best_val = None, np.inf
    starts = rng.uniform(0.0, 1.0, size=(n_restarts, d))
    for x0 in starts:
        res = minimize(neg_acq, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if res.fun < best_val:
            best_val, best_x_norm = res.fun, res.x

    x_next = denormalize(best_x_norm, bounds_arr)
    return x_next, gp



def fit_gp(
    X,
    y,
    bounds,
    alpha=1e-8,
    n_restarts_optimizer=10,
    random_state=None,
    noise_level=1e-2,
    noise_level_bounds=(1e-6, 1e1),
    length_scale=0.3,
    length_scale_bounds=(1e-2, 10.0),
    nu=2.5,
):
    """
    Fit a Gaussian Process surrogate to observed data for a function of
    arbitrary dimensionality D (inferred automatically from X.shape[1]).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, D)
        Observed input points.
    y : np.ndarray, shape (n_samples,)
        Observed function values.
    bounds : array-like, shape (D, 2)
        [(low, high), ...] per input dimension. Used to normalise inputs.
    alpha : float
        Numerical jitter added to the kernel diagonal for stability. This
        is NOT the observation noise level any more -- that is now learned
        automatically via the WhiteKernel term (see `noise_level`). Keep
        this small (1e-10 to 1e-6); it exists only to avoid numerical
        issues when points are close together.
    n_restarts_optimizer : int
        Restarts for the GP's internal kernel hyperparameter optimisation.
        Noisy data makes the marginal-likelihood surface itself noisier,
        so more restarts (e.g. 15-20) reduces the risk of landing in a
        bad local optimum.
    random_state : int or None
    noise_level : float
        Initial guess for the observation noise variance, in normalised
        y-units. This is a starting point for optimisation, not a fixed
        assumption -- sklearn will adjust it to fit the data via maximum
        marginal likelihood. Raise the starting guess if you know your
        function evaluations are quite noisy.
    noise_level_bounds : tuple
        (low, high) bounds the noise level is allowed to move within
        during optimisation.
    length_scale : float
        Initial guess for the RBF length-scale, in normalised [0, 1]
        input units.
    length_scale_bounds : tuple
        (low, high) bounds the length-scale is allowed to move within.
        The lower bound matters most for noisy data: without a floor,
        the optimiser can shrink the length-scale toward zero and simply
        interpolate through every noisy point instead of learning the
        underlying trend.
    nu : float
        Smoothness parameter for the Matern kernel. Common choices are
        0.5 (very rough, equivalent to an Ornstein-Uhlenbeck process),
        1.5, 2.5 (default here -- once-differentiable, a common default
        for physical/black-box functions that aren't perfectly smooth),
        or np.inf (recovers the RBF kernel exactly). Lower nu assumes
        less smoothness and is more forgiving of sharp, local features;
        it does NOT need to be re-optimised since it's fixed rather than
        a hyperparameter sklearn tunes.

    Returns
    -------
    gp : fitted GaussianProcessRegressor

    Notes
    -----
    The Matern length-scale is fit with ARD (Automatic Relevance
    Determination): a separate length-scale per input dimension rather
    than one shared value. This matters most once D gets too large to
    visually inspect the fitted surface (4D+) -- a short learned
    length-scale on a given dimension means the function varies quickly
    along that axis (it matters), while a length-scale pinned near the
    upper bound means the GP found that axis close to irrelevant. See
    `get_length_scales` to pull these out after fitting, and
    `viz_tools.plot_nd_slices` to use them to decide which dimensions
    are worth visualising directly.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    d = X.shape[1]

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 observations to fit a useful GP.")

    X_norm = normalize(X, bounds)

    # Broadcast a scalar starting length-scale to one-per-dimension (ARD).
    # If the caller already passed an array (e.g. from a previous fit,
    # to warm-start), leave it as-is.
    length_scale = np.atleast_1d(np.asarray(length_scale, dtype=float))
    if length_scale.shape[0] == 1 and d > 1:
        length_scale = np.full(d, length_scale[0])

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(X_norm, y)
    return gp




def _extract_kernel_params(gp):
    """
    Pull the Matern length-scale and WhiteKernel noise level out of a GP
    fitted by `fit_gp`'s kernel structure: ConstantKernel * Matern + WhiteKernel.
    Returns (mean_length_scale, noise_level); mean is taken in case the
    length-scale is anisotropic (one value per input dimension).
    """
    kernel = gp.kernel_
    length_scale = np.mean(np.atleast_1d(kernel.k1.k2.length_scale))
    noise_level = kernel.k2.noise_level
    return float(length_scale), float(noise_level)


def _acq_dispatch(acquisition, gp, y_obs, kappa, xi, maximize):
    """Build (acq_fn, acq_kwargs) the same way generate_next_point does,
    so the logged acquisition value matches what was actually optimised."""
    if acquisition == "ucb":
        return ucb_acquisition, {"kappa": kappa, "maximize": maximize}
    elif acquisition == "exploit":
        return exploit_acquisition, {"maximize": maximize}
    elif acquisition in ("pi", "ei"):
        y_best = y_obs.max() if maximize else y_obs.min()
        fn = pi_acquisition if acquisition == "pi" else ei_acquisition
        return fn, {"y_best": y_best, "xi": xi, "maximize": maximize}
    else:
        return max_variance_acquisition, {}



def get_length_scales(gp):
    """
    Per-dimension Matern length-scales learned by an ARD-fitted GP (see
    `fit_gp`). Shape (D,) regardless of whether the kernel ended up
    isotropic or anisotropic.

    A short length-scale means the function varies quickly along that
    axis (the GP thinks it matters); a length-scale sitting near the
    upper `length_scale_bounds` means the GP found that axis close to
    irrelevant -- output barely changes as you move along it. Useful in
    4D+ problems to decide which dimensions are worth a slice plot
    (`viz_tools.plot_nd_slices`) rather than plotting all of them.
    """
    return np.atleast_1d(gp.kernel_.k1.k2.length_scale).astype(float)


def loo_predictions(X, y, bounds, gp_kwargs=None):
    """
    Leave-one-out cross-validated GP predictions: refit the GP once per
    observation, each time leaving that observation out, and predict at
    the left-out point using the rest. This is the main way to validate
    the surrogate model once D is too large to visually inspect the
    fitted surface (roughly 4D+) -- if LOO predictions track the actual
    values well, with residuals falling inside the predicted uncertainty,
    the GP is a trustworthy stand-in for the true function even though
    you can't see its shape directly.

    Note this refits the GP n times, so for larger datasets (dozens of
    points) it costs roughly n times a normal fit -- fine for the modest
    dataset sizes typical in a BO loop, but worth being aware of.

    Parameters
    ----------
    X : np.ndarray, shape (n, D)
    y : np.ndarray, shape (n,)
    bounds : array-like, shape (D, 2)
    gp_kwargs : dict, optional
        Forwarded to `fit_gp` for every refit (e.g. {"n_restarts_optimizer": 20}).

    Returns
    -------
    pred_mean, pred_std : np.ndarray, shape (n,)
        LOO-predicted mean/std at each point, same order as X/y.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(
            "Need at least 4 observations for a meaningful LOO-CV check "
            f"(got {n})."
        )
    gp_kwargs = gp_kwargs or {}
    bounds_arr = np.asarray(bounds, dtype=float)

    pred_mean = np.full(n, np.nan)
    pred_std = np.full(n, np.nan)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[:] = True
        mask[i] = False
        gp_i = fit_gp(X[mask], y[mask], bounds_arr, **gp_kwargs)
        x_i_norm = normalize(X[i:i + 1], bounds_arr)
        mu, sigma = gp_i.predict(x_i_norm, return_std=True)
        pred_mean[i], pred_std[i] = mu[0], sigma[0]

    return pred_mean, pred_std


# ---------------------------------------------------------------------------
# Appending new observations
# ---------------------------------------------------------------------------

def append_observations(X, y, new_X, new_y):
    """
    Append newly-arrived observations to the existing dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n, D)
        Existing inputs (e.g. the initial batch, or the result of a
        previous append_observations call).
    y : np.ndarray, shape (n,)
        Existing observed values.
    new_X : np.ndarray, shape (m, D) or (D,)
        Newly observed input(s). A single point of shape (D,) is
        accepted and reshaped automatically.
    new_y : np.ndarray, shape (m,) or scalar
        Newly observed value(s), matching new_X row for row.

    Returns
    -------
    X_updated, y_updated : np.ndarray
        The concatenated arrays, in the order given (existing rows
        first, then new_X/new_y appended in the order provided). Several
        of the diagnostic tools below (see `compute_iteration_diagnostics`)
        assume this row order reflects the actual chronological order
        points were evaluated in, so don't reorder rows after appending.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    new_X = np.atleast_2d(np.asarray(new_X, dtype=float))
    new_y = np.atleast_1d(np.asarray(new_y, dtype=float)).ravel()

    if new_X.shape[0] != new_y.shape[0]:
        raise ValueError(
            f"new_X has {new_X.shape[0]} row(s) but new_y has {new_y.shape[0]} "
            "value(s) -- they must match one-for-one."
        )
    if new_X.shape[1] != X.shape[1]:
        raise ValueError(
            f"new_X has dimension {new_X.shape[1]}, but existing X has "
            f"dimension {X.shape[1]}."
        )

    X_updated = np.vstack([X, new_X])
    y_updated = np.append(y, new_y)
    return X_updated, y_updated


# ---------------------------------------------------------------------------
# Iteration-level diagnostics (no persisted log required)
# ---------------------------------------------------------------------------

def compute_iteration_diagnostics(
    X, y, bounds, n_initial,
    acquisition="ucb", kappa=5.0, xi=0.01, maximize=True,
    gp_kwargs=None, domain_grid_n=40,
):
    """
    Reconstruct the same per-iteration diagnostics the old history log
    used to persist incrementally -- but purely by replaying the ordered
    (X, y) you already have, with no file or saved state involved.

    For every point after the first `n_initial` (assumed to be the
    initial batch, arrived together rather than chosen by an acquisition
    function), this refits the GP on everything *before* that point and
    computes what the acquisition value, GP hyperparameters, and
    domain-wide uncertainty would have been at the moment it was
    proposed -- then compares that to the value actually observed.

    IMPORTANT: this assumes the row order in X/y is the actual
    chronological order points were evaluated in (initial batch first,
    then every subsequent observation in the order it arrived -- exactly
    what you get by repeatedly calling `append_observations`). Reordering
    rows will silently produce meaningless diagnostics.

    Parameters
    ----------
    X : np.ndarray, shape (n, D)
    y : np.ndarray, shape (n,)
    bounds : array-like, shape (D, 2)
    n_initial : int
        Number of rows (from the start) making up the initial batch --
        these are tagged iteration 0 and skipped for acquisition/std
        diagnostics, since nothing was "proposed" for them. Must be >= 2.
    acquisition, kappa, xi, maximize : as in `generate_next_point` --
        should match whatever acquisition was actually used to choose
        each point, for the replayed acquisition value to mean anything.
    gp_kwargs : dict, optional
        Forwarded to `fit_gp` on every refit.
    domain_grid_n : int
        Resolution per axis for the domain-average-uncertainty grid
        (see `propose_and_log`'s old docstring for what this measures --
        same idea, just recomputed here instead of cached).

    Returns
    -------
    history : dict of np.ndarray
        Same field names as the old persisted log (X, y, iteration,
        acq_value, length_scale, noise_level, pred_mean, pred_std,
        domain_mean_std) -- feeds directly into plot_convergence,
        plot_acquisition_decay, plot_uncertainty_shrinkage,
        plot_sample_trajectory, plot_step_distance, plot_bo_diagnostics
        without any changes to those functions.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    bounds_arr = np.asarray(bounds, dtype=float)
    d = bounds_arr.shape[0]
    gp_kwargs = gp_kwargs or {}

    if n_initial < 2:
        raise ValueError("n_initial must be >= 2 (fit_gp needs at least 2 points).")
    if n_initial > n:
        raise ValueError(f"n_initial ({n_initial}) exceeds the number of rows in X ({n}).")

    iteration = np.zeros(n, dtype=float)
    acq_value = np.full(n, np.nan)
    length_scale = np.full(n, np.nan)
    noise_level = np.full(n, np.nan)
    pred_mean = np.full(n, np.nan)
    pred_std = np.full(n, np.nan)
    domain_mean_std = np.full(n, np.nan)

    grids = np.meshgrid(*[np.linspace(0, 1, domain_grid_n) for _ in range(d)])
    grid_norm = np.column_stack([g.ravel() for g in grids])

    for i in range(n_initial, n):
        X_prev, y_prev = X[:i], y[:i]
        gp = fit_gp(X_prev, y_prev, bounds_arr, **gp_kwargs)

        x_i_norm = normalize(X[i:i + 1], bounds_arr)
        mu, sigma = gp.predict(x_i_norm, return_std=True)
        pred_mean[i], pred_std[i] = mu[0], sigma[0]

        acq_fn, acq_kwargs = _acq_dispatch(acquisition, gp, y_prev, kappa, xi, maximize)
        acq_value[i] = float(acq_fn(x_i_norm, gp, **acq_kwargs)[0])

        _, grid_sigma = gp.predict(grid_norm, return_std=True)
        domain_mean_std[i] = float(grid_sigma.mean())

        length_scale[i], noise_level[i] = _extract_kernel_params(gp)
        iteration[i] = i - n_initial + 1

    return {
        "X": X,
        "y": y,
        "iteration": iteration,
        "acq_value": acq_value,
        "length_scale": length_scale,
        "noise_level": noise_level,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "domain_mean_std": domain_mean_std,
    }
