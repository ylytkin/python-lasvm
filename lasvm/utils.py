from typing import List

import numpy as np

__all__ = [
    'cumulative_mean',
]


def cumulative_mean(ar: List[float]) -> List[float]:
    return (np.cumsum(ar) / np.arange(1, len(ar) + 1)).tolist()
