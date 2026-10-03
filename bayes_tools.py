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




def initial_bounds(X, pad_fraction=1.0):
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

    Returns
    -------
    bounds : np.ndarray, shape (D, 2)
    """
    low = X.min(axis=0)
    high = X.max(axis=0)
    span = high - low
    low_pad = low - pad_fraction * span
    high_pad = high + pad_fraction * span
    return np.column_stack([low_pad, high_pad])



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
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    d = X.shape[1]

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 observations to fit a useful GP.")

    X_norm = normalize(X, bounds)

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

