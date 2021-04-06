import numpy as np

__all__ = [
    'rbf_kernel',
    'polynomial_kernel',
    'linear_kernel',
]


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    """Calculate the pairwise values of the RBF kernel over the given arrays.

    Parameters
    ----------
    x1 : numpy array
        first array, shape (n1, n_features)
    x2 : numpy array
        second array, shape (n2, n_features)
    gamma : float

    Returns
    -------
    numpy array
        kernel values, shape (n1, n2)
    """

    squared_diff = ((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2).sum(axis=2)

    return np.exp(- gamma * squared_diff)


def polynomial_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float, degree: int, coef0: float) -> np.ndarray:
    """Calculate the pairwise values of the polynomial kernel over the given arrays.

    Parameters
    ----------
    x1 : numpy array
        first array, shape (n1, n_features)
    x2 : numpy array
        second array, shape (n2, n_features)
    gamma : float
    degree : int
    coef0 : float

    Returns
    -------
    numpy array
        kernel values, shape (n1, n2)
    """

    return (gamma * x1.dot(x2.T) + coef0) ** degree


def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Calculate the pairwise values of the linear kernel (i.e. dot-product)
    over the given arrays.

    Parameters
    ----------
    x1 : numpy array
        first array, shape (n1, n_features)
    x2 : numpy array
        second array, shape (n2, n_features)

    Returns
    -------
    numpy array
        kernel values, shape (n1, n2)
    """

    return x1.dot(x2.T)
