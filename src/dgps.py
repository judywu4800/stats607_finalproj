import numpy as np
from scipy.stats import multivariate_normal,f

# Generate data under the null
def generate_null_data(n, p, mu=None, sigma=1.0, rng=None):
    """
    Generate synthetic observations under the null hypothesis of no clustering structure.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Dimension of each observation.
    mu : float or array-like of shape (p,), optional
        Mean vector of the data-generating distribution. If None, defaults to a zero mean.
    sigma : float or array-like of shape (p,p), optional
        Standard deviation of the Gaussian noise (unknown parameter)
        Default is 1.0.
    rng : numpy.random.Generator or None, optional
        A NumPy random number generator instance for reproducibility. If None, defaults
        to `np.random.default_rng()`.

    Returns
    -------
    X : ndarray of shape (n, p)
        Simulated data matrix generated from a multivariate normal distribution with
        mean `mu` and isotropic covariance `sigma^2 I_p` or non-isotropic covariance sigma
    """
    if rng is None:
        rng = np.random.default_rng()
    if mu is None:
        mu = np.zeros(p)
    mu = np.asarray(mu)

    # handle sigma
    if np.isscalar(sigma):
        Sigma = (sigma ** 2) * np.eye(p)
    else:
        Sigma = np.asarray(sigma)
        if Sigma.shape != (p, p):
            raise ValueError(f"Covariance matrix must be {p}×{p}, got {Sigma.shape}.")

    if mu.shape[0] != p:
        raise ValueError(f"Mean vector length {mu.shape[0]} does not match p={p}.")

    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=n)
    return X


def generate_alternative_data(n_each, delta, sigma, n_clusters=3, true_mean=False, rng=None):
    """
    Generate synthetic observations of shape (n_each*3, 2) under the alternative hypothesis,
    where there exists 2 or 3 true clusters

    Parameters
    ----------
    n_each : int
        Number of observations in each cluster.
    p : int
        Dimension of each observation.
    sigma : float or array-like of shape (2,2), optional
        Standard deviation of the Gaussian noise (unknown parameter)
        Default is 1.0.
    n_clusters : int
        Number of clustering structure exists in the data
    true_mean: bool, optional
        Whether to return the true mean or not
    rng : numpy.random.Generator or None, optional
        A NumPy random number generator instance for reproducibility. If None, defaults
        to `np.random.default_rng()`.

    Returns
    -------
    X : ndarray of shape (n, p)
        Simulated data matrix generated from a multivariate normal distribution with
        mean `mu` and isotropic covariance `sigma^2 I_p` or non-isotropic covariance sigma
    """
    if np.isscalar(sigma):
        cov = np.eye(2) * (sigma ** 2)
    else:
        cov = np.asarray(sigma)
        if cov.shape != (2, 2):
            raise ValueError(f"Expected 2x2 covariance matrix, got shape {cov.shape}")

    if rng is None:
        rng = np.random.default_rng()

    if n_clusters == 2:
        mus = [np.array([0, 0]),
               np.array([delta, 0])]
    elif n_clusters == 3:
        mus = [np.array([0, 0]),
               np.array([delta, 0]),
               np.array([delta / 2, np.sqrt(delta ** 2 - (delta ** 2) / 4)])]

    else:
        raise ValueError("n_clusters must be 2 or 3.")

    X_parts, labels_parts = [], []
    for i, mu in enumerate(mus, start=1):
        Xi = rng.multivariate_normal(mean=mu, cov=cov, size=n_each)
        X_parts.append(Xi)
        labels_parts.append(np.ones(n_each) * i)

    X = np.vstack(X_parts)
    labels = np.concatenate(labels_parts)

    mu = np.vstack(mus)[labels.astype(int) - 1]

    if true_mean:
        return X, labels, mu
    else:
        return X, labels