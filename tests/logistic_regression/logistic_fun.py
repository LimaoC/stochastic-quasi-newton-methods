from typing import Callable

import numpy as np
from scipy.special import expit


def logistic_f(A, b, reg_param=0):
    """Logistic loss function with l2 regularisation"""

    def phi(t):
        """Computes phi(t) = ln(1 + e^t) using the log-sum-exp trick"""
        c = np.maximum(0, t)
        return c + np.log(np.exp(-c) + np.exp(t - c))

    def loss(x):
        Ax = A @ x
        reg_term = (reg_param / 2) * (np.linalg.norm(x) ** 2)
        return np.sum(phi(Ax) - b.flatten() * Ax) + reg_term

    return loss


def logistic_grad_f(A, b, reg_param=0) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic loss gradient function with l2 regularisation"""
    grad_phi = expit

    def loss(x):
        return A.T @ (grad_phi(A @ x) - b.flatten()) + reg_param * x

    return loss
