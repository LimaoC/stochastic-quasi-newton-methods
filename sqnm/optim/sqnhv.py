"""
Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

REF: Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. (2015). A Stochastic
    Quasi-Newton Method for Large-Scale Optimization.
"""

import logging
from typing import Callable

import torch
from torch import Tensor
from torch.autograd.functional import hvp

from ..line_search.prob_line_search import prob_line_search
from ..line_search.strong_wolfe_line_search import strong_wolfe_line_search
from .sqn_base import SQNBase

logger = logging.getLogger(__name__)


class SQNHv(SQNBase):
    LINE_SEARCH_FNS = ["strong_wolfe", "prob_wolfe"]

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        line_search_fn: str | None = None,
        history_size: int = 20,
        skip: int = 10,
    ):
        """
        Hessian-Vector Stochastic Quasi-Newton (SQN-Hv)

        Parameters:
            params: iterable of parameters to optimize
            lr: learning rate, ignored if line_search_fn is not None
            line_search_fn: line search function to use, either None for fixed step
                size, or one of OLBFGS.LINE_SEARCH_FNS
            history_size: history size, usually 2 <= m <= 30
            skip: number of iterations between curvature estimates
        """
        if line_search_fn is not None and line_search_fn not in self.LINE_SEARCH_FNS:
            raise ValueError(f"SQN-Hv only supports one of: {self.LINE_SEARCH_FNS}")

        defaults = dict(
            lr=lr,
            history_size=history_size,
            line_search_fn=line_search_fn,
            skip=skip,
        )
        super().__init__(params, defaults)

        state = self.state[self._params[0]]
        state["num_iters"] = 1  # Algorithm in paper starts from k = 1
        state["xt"] = [
            torch.zeros_like(self._get_param_vector()),
            torch.zeros_like(self._get_param_vector()),
        ]
        # Used for probabilistic LS
        state["alpha_start"] = 1.0
        state["alpha_running_avg"] = state["alpha_start"]

    def _two_loop_recursion(self, grad: Tensor) -> Tensor:
        """
        Two loop recursion for computing H_k * grad

        This differs from the standard two loop recursion in that the (s, y) pairs are
        indexed by t not k (as the curvature pair computations are decoupled from the
        stochastic gradient computations).
        """
        group = self.param_groups[0]
        m = group["history_size"]

        state = self.state[self._params[0]]
        # The paper's t index is off by 1 compared to the convention, account for this
        # They define s_t = x_t - x_{t-1} instead of s_{t-1} = x_t - x_{t-1}
        t = state["num_sy_pairs"] - 1
        sy_history = state["sy_history"]

        q = grad.clone()
        alphas = torch.zeros(m)
        s_prev, y_prev = sy_history[t % m]
        history_idxs = range(max(t - m, 0), t)
        for i in reversed(history_idxs):
            s_prev, y_prev = sy_history[i % m]
            alphas[i - (t - m)] = s_prev.dot(q) / s_prev.dot(y_prev)
            q -= alphas[i - (t - m)] * y_prev
        r = (s_prev.dot(y_prev) / y_prev.dot(y_prev)) * q
        for i in history_idxs:
            s_prev, y_prev = sy_history[i % m]
            beta = y_prev.dot(r) / s_prev.dot(y_prev)
            r += (alphas[i - (t - m)] - beta) * s_prev
        return r

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float],
        fn: Callable[[Tensor, bool], Tensor] | None = None,
        curvature_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> float:
        """
        Perform a single SQN-Hv iteration.

        Parameters:
            closure: A closure that re-evaluates the model and returns the loss.
            fn: A pure function that computes the loss for a given input. Required if
                line_search_fn == "prob_wolfe". The function should take a boolean
                parameter which, if True, also returns the gradient, loss variance, and
                gradient variance.
            curvature_fn: A pure function that computes the loss for a given input. The
                function should be provided every `skip` iterations.
        """
        # Get state and hyperparameter variables
        group = self.param_groups[0]
        m = group["history_size"]
        lr = group["lr"]
        line_search_fn = group["line_search_fn"]
        skip = group["skip"]  # L

        state = self.state[self._params[0]]
        k = state["num_iters"]
        sy_history = state["sy_history"]
        # Note index t for curvature pairs, which are decoupled from gradient estimates
        xt = state["xt"]
        alpha_start = state["alpha_start"]
        alpha_running_avg = state["alpha_running_avg"]

        if line_search_fn is not None and fn is None:
            raise ValueError("fn parameter is needed for line search")

        if k % skip != 0 and curvature_fn is not None:
            logger.warning(f"Got curvature_fn but didn't expect it on iteration {k}")
        if k % skip == 0 and curvature_fn is None:
            raise TypeError(f"Expected curvature_fn but didn't get it on iteration {k}")

        ################################################################################

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        orig_loss = closure()  # Populate gradients
        xk = self._get_param_vector()
        grad = self._get_grad_vector()

        # NOTE: Termination criterion?

        # Accumulate average over L iterations
        xt[1] += xk

        if k <= 2 * skip:
            # Stochastic gradient descent for first 2L iterations
            pk = -grad
        else:
            # NOTE: Can't reliably check if pk is a descent direction here
            pk = -self._two_loop_recursion(grad)

        if line_search_fn == "strong_wolfe":
            assert fn is not None
            # Choose step size to satisfy strong Wolfe conditions
            grad_fn = torch.func.grad(fn)
            alpha_k = strong_wolfe_line_search(fn, grad_fn, xk, pk)
        elif line_search_fn == "prob_wolfe":
            assert fn is not None
            f0, df0, var_f0, var_df0 = fn(xk, True)
            # Don't need function handle to return vars in line search
            alpha_k, alpha_start, alpha_running_avg = prob_line_search(
                lambda x: fn(x, False),
                xk,
                pk,
                f0,
                df0,
                var_f0,
                var_df0,
                alpha_running_avg,
                a0=alpha_start,
            )
            state["alpha_start"] = alpha_start
            state["running_avg"] = alpha_running_avg
        else:
            # Use fixed step size
            alpha_k = lr

        xk_next = xk + alpha_k * pk
        self._set_param_vector(xk_next)

        if k % skip == 0:
            # Compute curvature pairs every L iterations
            t = state["num_sy_pairs"] - 1
            xt[1] /= skip
            if t >= 0:
                st = xt[1] - xt[0]
                # Compute subsampled Hessian vector product on a different, larger
                # sample given by curvature_fn
                _, yt = hvp(curvature_fn, xt[1], v=st, strict=True)
                sy_history[t % m] = (st, yt)
            xt[0], xt[1] = xt[1], torch.zeros_like(xt[1])
            state["num_sy_pairs"] += 1

        state["num_iters"] += 1
        return orig_loss
