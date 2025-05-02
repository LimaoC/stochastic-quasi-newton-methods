from typing import Callable

import numpy as np
from scipy.special import expit


def logistic_f(A, b, reg_param=0):
    """Logistic loss function with l2 regularisation"""
    n, d = A.shape

    def phi(t):
        """Computes phi(t) = ln(1 + e^t) using the log-sum-exp trick"""
        c = max(0, t)
        return c + np.log(np.exp(-c) + np.exp(t - c))

    return lambda x: sum(
        phi(A[i].dot(x)) - b[i].item() * A[i].dot(x) for i in range(n)
    ) + reg_param / 2 * (np.linalg.norm(x) ** 2)


def logistic_grad_f(A, b, reg_param=0) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic loss gradient function with l2 regularisation"""
    n, d = A.shape

    grad_phi = expit

    return (
        lambda x: sum(A[i] * grad_phi(A[i].dot(x)) - b[i] * A[i] for i in range(n))
        + reg_param * x
    )
