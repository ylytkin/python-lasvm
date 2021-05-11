from typing import Union

import numpy as np
from tqdm import tqdm

from lasvm.base_kernel_method import BaseKernelMethod

__all__ = [
    'KernelPerceptron',
    'BudgetKernelPerceptron',
]


class KernelPerceptron(BaseKernelMethod):
    def __init__(
            self,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
    ) -> None:
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    def partial_fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'KernelPerceptron':
        x = x.copy()
        y = self._prepare_targets(y)

        if self.support_vectors.shape[1] == 0:
            self.support_vectors = np.empty(shape=(0, x.shape[1]))

        if self.gamma_ == 'scale':
            self.gamma_ = self._scaled_gamma(x)

        ids = np.arange(x.shape[0])

        if shuffle:
            np.random.shuffle(ids)

        for i in tqdm(ids, disable=not verbose):
            x0 = x[[i]]
            y0 = y[[i]]

            yhat = self._kernel(x0, self.support_vectors).dot(self.alpha)

            if y0[0] * yhat[0] <= 0:
                self.support_vectors = np.vstack([self.support_vectors, x0])
                self.alpha = np.hstack([self.alpha, y0])

        return self

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'KernelPerceptron':
        return self.partial_fit(x, y, verbose=verbose, shuffle=shuffle)


class BudgetKernelPerceptron(BaseKernelMethod):
    def __init__(
            self,
            beta: float = 0,
            n: int = 50,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
    ) -> None:
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        self.beta = beta
        self.n = n

        self.kernel_mx = np.empty(shape=(0, 0))

    def partial_fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'BudgetKernelPerceptron':
        x = x.copy()
        y = self._prepare_targets(y)

        if self.support_vectors.shape[1] == 0:
            self.support_vectors = np.empty(shape=(0, x.shape[1]))

        if self.gamma_ == 'scale':
            self.gamma_ = self._scaled_gamma(x)

        ids = np.arange(x.shape[0])

        if shuffle:
            np.random.shuffle(ids)

        for i in tqdm(ids, disable=not verbose):
            x0 = x[[i]]
            y0 = y[[i]]

            yhat = self._kernel(x0, self.support_vectors).dot(self.alpha)

            m = y0[0] * yhat[0]

            if m <= self.beta:
                if self.support_vectors.shape[0] >= self.n:
                    self._remove_support_vector()

                self.support_vectors = np.vstack([self.support_vectors, x0])
                self.alpha = np.hstack([self.alpha, y0])

                new_kernel_values = self._kernel(x0, self.support_vectors)
                self.kernel_mx = np.vstack([self.kernel_mx, new_kernel_values[:, :-1]])
                self.kernel_mx = np.hstack([self.kernel_mx, new_kernel_values.T])

        return self

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'BudgetKernelPerceptron':
        return self.partial_fit(x, y, verbose=verbose, shuffle=shuffle)

    def _remove_support_vector(self) -> None:
        yhat = self.kernel_mx.dot(self.alpha)
        diag = np.diagonal(self.kernel_mx)
        values = self.alpha * (yhat - self.alpha * diag)
        j = [np.argmax(values)]

        self.support_vectors = np.delete(self.support_vectors, j, axis=0)
        self.alpha = np.delete(self.alpha, j)
        self.kernel_mx = np.delete(self.kernel_mx, j, axis=0)
        self.kernel_mx = np.delete(self.kernel_mx, j, axis=1)
