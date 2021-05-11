# python-lasvm

A Python implementation of the LaSVM online learning algorithm [1].

[1] [Fast Kernel Classifiers with Online and Active Learning](https://leon.bottou.org/papers/bordes-ertekin-weston-bottou-2005)

Installation:
```bash
pip install git+https://github.com/ylytkin/python-lasvm
```

Usage:

```python
import numpy as np

from lasvm import LaSVM

x = np.array([[1.84, -1.7],
              [-0.52, 0.27],
              [-0.23, -0.26],
              [-1.42, 0.17],
              [1., -1.],
              [0.01, 1.71],
              [-0.53, 1.7],
              [-0.27, 0.06]])

y = np.array([1, 1, 0, 0, 1, 0, 0, 1])

pos_samples = x[:2]
neg_samples = x[2:4]

lasvm = LaSVM(pos_samples, neg_samples)  # some initial support vectors are required
lasvm.fit(x, y)

lasvm.score(x, y)  # 0.875
```

Also implemented are the really simple but elegant Kernel Perceptron and Budget Kernel Perceptron models (see `lasvm/kernel_perceptron.py`).

***

TODO:

* When using the linear kernel, sometimes delta does not converge below tau in the final stage.
* Not clear how much seeded examples are needed on initialization (when using the linear kernel and 2 initial examples of both classes, sometimes all support vectors get removed, which causes the model to crash).
* History values are not alligned with iterations.
