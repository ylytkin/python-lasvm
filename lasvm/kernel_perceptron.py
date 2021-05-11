from typing import Union, Dict

import numpy as np
from tqdm import tqdm

from lasvm.utils import cumulative_mean
from lasvm.base_kernel_method import BaseKernelMethod


class KernelPerceptron(BaseKernelMethod):
    def __init__(
            self,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
    ) -> None:
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        self._history_pred = []
        self._history_sv = []

    def fit(
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

                self._history_pred.append(0)

            else:
                self._history_pred.append(1)

            self._history_sv.append(self.support_vectors.shape[0])

        return self

    @property
    def history(self) -> Dict[str, list]:
        return {
            'acc': cumulative_mean(self._history_pred),
            'sv': self._history_sv,
        }
