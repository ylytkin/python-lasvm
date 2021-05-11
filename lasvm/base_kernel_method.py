from typing import Union

import numpy as np

from lasvm.kernels import rbf_kernel, polynomial_kernel, linear_kernel

__all__ = [
    'BaseKernelMethod',
]


class BaseKernelMethod:
    class NotFittedError(Exception):
        def __init__(self):
            super().__init__('model not fitted')

    def __init__(
            self,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
    ) -> None:
        self.kernel = kernel

        self.gamma = gamma
        self.gamma_ = gamma
        self.degree = degree
        self.coef0 = coef0

        self.support_vectors = np.empty(shape=(0, 0))
        self.alpha = np.empty(shape=(0,))
        self.intercept = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.support_vectors.shape[0] == 0:
            y_pred = np.zeros(shape=(x.shape[0],))
        else:
            y_pred = self._kernel(x, self.support_vectors).dot(self.alpha)

        y_pred = y_pred + self.intercept

        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0

        return y_pred

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(x) == y).mean()

    @property
    def coef_(self) -> np.ndarray:
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a linear kernel')

        return self.alpha.dot(self.support_vectors)

    @staticmethod
    def _scaled_gamma(x: np.ndarray) -> float:
        return 1 / x.shape[1] / x.var()

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == 'rbf':
            return rbf_kernel(x1, x2, gamma=self.gamma_)
        elif self.kernel == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel == 'poly':
            return polynomial_kernel(x1, x2, gamma=self.gamma_, degree=self.degree, coef0=self.coef0)
        else:
            raise ValueError(f"Available kernels: 'rbf', 'linear', 'poly'. Got {repr(self.kernel)}")

    @staticmethod
    def _prepare_targets(y: np.ndarray) -> np.ndarray:
        if (y * (1 - y)).any():
            raise ValueError(f'class labels must be 0 or 1')

        y = y.copy()
        y[y == 0] = -1

        return y
