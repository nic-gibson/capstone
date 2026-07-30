import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
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



_ACQUISITIONS = {
    "ucb": ucb_acquisition,
    "max_variance": max_variance_acquisition,
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
    maximize=True,
    n_restarts=25,
    random_state=None,
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
    acquisition : {"ucb", "max_variance"}
    kappa : float
        Exploration weight, only used for "ucb". Higher = more exploration.
    maximize : bool
        Whether the underlying black-box function is being maximised
        (set False if you are minimising it). Not used for "max_variance".
    n_restarts : int
        Number of random multi-starts for optimising the acquisition
        function (acquisition surfaces can be multimodal, especially in
        higher dimensions, so more restarts helps in 6D-8D problems).
    random_state : int or None


    Returns
    -------
    x_next : np.ndarray, shape (D,)
        Next point to evaluate, in the original (unnormalised) units.
    gp : GaussianProcessRegressor
        The fitted GP (useful for diagnostics, e.g. checking predicted
        mean/std at x_next).
    """
    if acquisition not in _ACQUISITIONS:
        raise ValueError(f"Unknown acquisition '{acquisition}'. Choose from {list(_ACQUISITIONS)}.")

    rng = np.random.default_rng(random_state)
    bounds_arr = np.asarray(bounds, dtype=float)
    d = bounds_arr.shape[0]

    X = np.atleast_2d(np.asarray(X, dtype=float))

    gp = fit_gp(X, y, bounds_arr, random_state=random_state)

    if acquisition == "ucb":
        def neg_acq(x_norm):
            return -ucb_acquisition(x_norm.reshape(1, -1), gp, kappa=kappa, maximize=maximize)[0]
    else:  # max_variance
        def neg_acq(x_norm):
            return -max_variance_acquisition(x_norm.reshape(1, -1), gp)[0]

    # Multi-start L-BFGS-B over the normalised unit hypercube [0, 1]^d.
    best_x_norm, best_val = None, np.inf
    starts = rng.uniform(0.0, 1.0, size=(n_restarts, d))
    for x0 in starts:
        res = minimize(neg_acq, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if res.fun < best_val:
            best_val, best_x_norm = res.fun, res.x

    x_next = denormalize(best_x_norm, bounds_arr)
    return x_next, gp



def fit_gp(X, y, bounds, alpha=1e-6, n_restarts_optimizer=10, random_state=None):
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
        Assumed observation noise level (variance). Increase if your
        function evaluations are noisy.
    n_restarts_optimizer : int
        Restarts for the GP's internal kernel hyperparameter optimisation.
    random_state : int or None

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

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(X_norm, y)
    return gp

