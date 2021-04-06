from typing import List, Union, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

from lasvm.utils import cumulative_mean
from lasvm.base_kernel_method import BaseKernelMethod

__all__ = [
    'LaSVM'
]


class LaSVM(BaseKernelMethod):
    """The LaSVM online learning algorithm for binary classification
    as described here:
    https://leon.bottou.org/papers/bordes-ertekin-weston-bottou-2005

    Example
    -------
    x = np.array([[ 1.84, -1.7 ],
                  [-0.52,  0.27],
                  [-0.23, -0.26],
                  [-1.42,  0.17],
                  [ 1.  , -1.  ],
                  [ 0.01,  1.71],
                  [-0.53,  1.7 ],
                  [-0.27,  0.06]])

    y = np.array([1, 1, 0, 0, 1, 0, 0, 1])

    pos_samples = x[:2]
    neg_samples = x[2:4]

    lasvm = LaSVM()
    lasvm.initialize(pos_samples, neg_samples)
    lasvm.fit(x, y, finalize=True)

    (lasvm.predict(x) == y).mean()  # 0.875

    Parameters
    ----------
    c : float, default 1
        regularization parameter
    kernel : {'rbf', 'linear', 'poly'}, default 'rbf'
        kernel name
    gamma : 'scale' or float, default 'scale'
        rbf and polynomial kernel gamma parameter (ignored for linear
        kernel). If 'scale' is passed, the actual value of gamma is
        calculated on initialization as::
            1 / x.shape[1] / x.var()
    degree : int, default 3
        polynomial kernel degree parameter (ignored for the rest of
        the kernels)
    coef0 : float, default 0
        polynomial kernel coef0 parameter (ignored for the rest of
        the kernels)
    tol : float
        tolerance parameter
    niter : int
        number of iterations on final step

    Attributes
    ----------
    c : float
        regularization parameter
    kernel : {'rbf', 'linear', 'poly'}
        kernel name
    gamma : 'scale' or float
        rbf and polynomial kernel gamma parameter
    gamma_ : float
        calculated gamma parameter
    degree : int, default 3
        polynomial kernel degree parameter
    coef0 : float, default 0
        polynomial kernel coef0 parameter
    tol : float
        tolerance parameter
    niter : int
        number of iterations on final step
    support_vectors : numpy array
        support vectors, shape (n_vectors, n_features)
    alpha : numpy array
        model coefficients, shape (n_vectors,)
    intercept : float
        model intercept
    target : numpy array
        support vector class labels, shape (n_vectors,)
    kernel_mx : numpy array
        pair-wise kernel values of support vectors,
        shape (n_vectors, n_vectors)
    gradient : numpy array
        gradient, shape (n_vectors,)
    a : numpy array
        lower bounds, shape (n_vectors,)
    b : numpy array
        upper bounds, shape (n_vectors,)
    delta : float
        current delta
    initialized : bool
        whether the model has been initialized

    Methods
    -------
    initialize(pos_samples, neg_samples)
        initialize the model before fitting
    fit(x, y)
        fit the model on the training data
    finalize()
        finalize the training process after fitting is done
    predict(x)
        predict class labels of the test data

    Properties
    ----------
    history : dict
        training history of form::

            {
                'acc': [...],  # model accuracy on each incoming vector
                'sv': [...]  # number of support vectors after each iteration
            }

    coef_ : numpy array
        coefficients of the separating hyperplane in the linear
        kernel case, shape (n_vectors,)

    """

    ERR = 0.00001

    class NotInitializedError(Exception):
        def __init__(self):
            super().__init__('model not initialized')

    class GradientPairNotFoundError(Exception):
        def __init__(self):
            super().__init__('could not find a maximum gradient pair')

    def __init__(
            self,
            c: float = 1,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
            tol: float = 0.0001,
            niter: int = 10000,
    ) -> None:
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        self.c = c

        self.tol = tol

        self.target = np.empty(shape=(0,))
        self.kernel_mx = np.empty(shape=(0, 0))
        self.gradient = np.empty(shape=(0,))
        self.a = np.empty(shape=(0,))
        self.b = np.empty(shape=(0,))

        self.delta = None

        self.niter = niter

        self.initialized = False

        self._history_pred = []
        self._history_sv = []

    def initialize(self, pos_samples: np.ndarray, neg_samples: np.ndarray) -> 'LaSVM':
        """Initialize model by adding some positive and negative samples as
        support vectors.

        This is a mandatory step before fitting.

        Parameters
        ----------
        pos_samples : numpy array
            positive samples, shape (n_vectors, n_features)
        neg_samples : numpy array
            negative samples, shape (n_vectors, n_features)

        Returns
        -------
        self
        """

        self._remove_all_support_vectors()

        self._history_pred = []
        self._history_sv = []

        pos_samples = pos_samples.copy()
        neg_samples = neg_samples.copy()

        if self.gamma == 'scale':
            self.gamma_ = self._scaled_gamma(np.vstack([pos_samples, neg_samples]))

        self.support_vectors = np.empty(shape=(0, pos_samples.shape[1]))

        self._add_support_vectors(pos_samples, y=np.ones(pos_samples.shape[0]), predict=False)
        self._add_support_vectors(neg_samples, y=- np.ones(neg_samples.shape[0]), predict=False)

        i, j = self._find_maximum_gradient_pair()
        self.intercept = (self.gradient[i] + self.gradient[j]) / 2
        self.delta = self.gradient[i] - self.gradient[j]

        self.initialized = True

        return self

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
            finalize: bool = False,
    ) -> 'LaSVM':
        """Fit model to the given data.

        Array `x` must be 2-dimensional of shape (n_vectors, n_features).
        Array `y` must be 1-dimensional of shape (n_vectors,) and contain only
        class labels 0 or 1.

        Parameters
        ----------
        x : numpy array
            data, shape (n_vectors, n_features)
        y : numpy array
            class labels, shape (n_vectors,), valued 0 or 1
        shuffle : bool
            shuffle data before fitting
        verbose : bool
            set verbosity
        finalize : bool
            perform finalizing step after fitting

        Returns
        -------
        self
        """

        if not self.initialized:
            raise self.NotInitializedError

        x = x.copy()
        y = self._prepare_targets(y)

        ids = np.arange(x.shape[0])

        if shuffle:
            np.random.shuffle(ids)

        for i in tqdm(ids, disable=not verbose):
            x0 = x[[i]]
            y0 = y[[i]]

            self._process(x=x0, y=y0)
            self._reprocess()

            self._history_sv.append(self.support_vectors.shape[0])

        if finalize:
            if verbose:
                print('Finalizing')

            self.finalize(verbose=verbose)

        return self

    def finalize(self, niter: Optional[int] = None, verbose: bool = False) -> 'LaSVM':
        """Perform the final step of learning (i.e. attempt to get delta
        below tau).

        Parameters
        ----------
        niter : int
            number of finalizing iterations
        verbose : bool
            set verbosity

        Returns
        -------
        self
        """

        niter = niter or self.niter

        tqdm_bar = tqdm(total=niter) if verbose else None
        broke_loop = False

        for _ in range(niter):
            self._reprocess()

            self._history_sv.append(self.support_vectors.shape[0])

            if self.delta <= self.tol:
                broke_loop = True
                break

            if verbose:
                tqdm_bar.update(1)

        if not broke_loop:
            print(f'Warning: delta did not converge below tol in {niter} iterations')

        if verbose:
            tqdm_bar.update(tqdm_bar.total - tqdm_bar.n)
            tqdm_bar.close()

        return self

    @property
    def history(self) -> Dict[str, list]:
        return {
            'acc': cumulative_mean(self._history_pred),
            'sv': self._history_sv,
        }

    def __repr__(self):
        return 'LaSVM()'

    def _add_support_vectors(self, x: np.ndarray, y: np.ndarray, predict: bool) -> None:
        """Add support vectors with zero coefficients.

        Parameters
        ----------
        x : numpy array
            data, shape (n_vectors, n_features)
        y : numpy array
            class labels, shape (n_vectors, n_features)
        predict : bool
            try predicting on input
        """

        n_vectors = x.shape[0]

        self.support_vectors = np.vstack([self.support_vectors, x])
        self.alpha = np.append(self.alpha, np.zeros(n_vectors))
        self.target = np.append(self.target, y)

        new_kernel_values = self._kernel(x, self.support_vectors)

        self.kernel_mx = np.vstack([self.kernel_mx, new_kernel_values[:, :-n_vectors]])
        self.kernel_mx = np.hstack([self.kernel_mx, new_kernel_values.T])

        dot = new_kernel_values.dot(self.alpha)

        if predict:
            pred = dot + self.intercept
            pred[pred > 0] = 1
            pred[pred <= 0] = -1
            pred = (pred == y).astype(int).tolist()

            self._history_pred.extend(pred)

        gradient = y - dot
        self.gradient = np.append(self.gradient, gradient)

        a = y * self.c
        a[a > 0] = 0
        self.a = np.append(self.a, a)

        b = y * self.c
        b[b < 0] = 0
        self.b = np.append(self.b, b)

    def _remove_support_vectors(self, vector_ids: List[int]) -> None:
        """Remove support vectors with given ids.

        Parameters
        ----------
        vector_ids : list
            ids of vectors to remove
        """

        self.support_vectors = np.delete(self.support_vectors, vector_ids, axis=0)
        self.alpha = np.delete(self.alpha, vector_ids)
        self.kernel_mx = np.delete(self.kernel_mx, vector_ids, axis=0)
        self.kernel_mx = np.delete(self.kernel_mx, vector_ids, axis=1)
        self.target = np.delete(self.target, vector_ids)
        self.gradient = np.delete(self.gradient, vector_ids)
        self.a = np.delete(self.a, vector_ids)
        self.b = np.delete(self.b, vector_ids)

    def _remove_all_support_vectors(self) -> None:
        n_sv = self.support_vectors.shape[0]

        if n_sv == 0:
            return

        to_remove = list(range(n_sv))
        self._remove_support_vectors(to_remove)

    def _is_violating_pair(self, i: int, j: int) -> bool:
        return self.alpha[i] < self.b[i] \
               and self.alpha[j] > self.a[j] \
               and self.gradient[i] - self.gradient[j] > self.tol

    def _find_max_gradient_id(self) -> int:
        """Find id of the vector with conditionally maximal gradient.

        Returns
        -------
        int
            vector id
        """

        mask = self.alpha < self.b
        mask_ids = np.where(mask)[0]
        i = mask_ids[np.argmax(self.gradient[mask])]

        return i

    def _find_min_gradient_id(self) -> int:
        """Find id of the vector with conditionally minimal gradient.

        Returns
        -------
        int
            vector id
        """

        mask = self.alpha > self.a
        mask_ids = np.where(mask)[0]
        j = mask_ids[np.argmin(self.gradient[mask])]

        return j

    def _find_maximum_gradient_pair(self) -> Tuple[int, int]:
        return self._find_max_gradient_id(), self._find_min_gradient_id()

    def _update_parameters(self, i: int, j: int) -> None:
        lambda_ = min(
            (self.gradient[i] - self.gradient[j]) / (
                        self.kernel_mx[i, i] + self.kernel_mx[j, j] - 2 * self.kernel_mx[i, j]),
            self.b[i] - self.alpha[i],
            self.alpha[j] - self.a[j],
        )

        self.alpha[i] = self.alpha[i] + lambda_
        self.alpha[j] = self.alpha[j] - lambda_

        self.gradient = self.gradient - lambda_ * (self.kernel_mx[i] - self.kernel_mx[j])

    def _process(self, x: np.ndarray, y: np.ndarray) -> None:
        """Process an object-target pair.

        Parameters
        ----------
        x : numpy array
            feature vector, shape (1, n_features)
        y : numpy array
            class label, shape (1,)
        """

        if (((self.support_vectors - x) ** 2).sum(axis=1) ** 0.5 < self.ERR).any():
            return

        self._add_support_vectors(x, y, predict=True)

        if y[0] == 1:
            i = self.support_vectors.shape[0] - 1
            j = self._find_min_gradient_id()

        else:
            j = self.support_vectors.shape[0] - 1
            i = self._find_max_gradient_id()

        if not self._is_violating_pair(i, j):
            return

        self._update_parameters(i=i, j=j)

    def _reprocess(self) -> None:
        """Reprocess.
        """

        i, j = self._find_maximum_gradient_pair()

        if not self._is_violating_pair(i, j):
            return

        self._update_parameters(i=i, j=j)

        i, j = self._find_maximum_gradient_pair()
        to_remove = []

        for k in np.where(np.abs(self.alpha) < self.ERR)[0]:
            if (self.target[k] == -1 and self.gradient[k] >= self.gradient[i]) \
                    or (self.target[k] == 1 and self.gradient[k] <= self.gradient[j]):
                to_remove.append(k)

        self.intercept = (self.gradient[i] + self.gradient[j]) / 2
        self.delta = self.gradient[i] - self.gradient[j]

        self._remove_support_vectors(to_remove)
