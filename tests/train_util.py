import logging
import time
from contextlib import contextmanager
from typing import Callable

import torch
import torch.func as ft
from torch import Tensor

from sqnm.utils.param import unflatten

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def get_device():
    return (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )


@contextmanager
def timing_context(name: str):
    start = time.time()
    logger.info(f"Training with {name}...")
    try:
        yield
    finally:
        logger.info(f"{name} took {time.time() - start:.3f} seconds")


def log_training_info(epoch, loss, grad_norm) -> None:
    logger.info(f"\tEpoch {epoch:4}, loss = {loss:.4f}, grad norm = {grad_norm:.4f}")


def create_closure(X, y, optimizer, model, loss_fn) -> Callable[[], float]:
    """
    Returns a closure that re-evaluates and returns the model loss, and populates
    gradients.

    This can be used to create a closure to pass to optimizer.step().
    """

    def closure() -> float:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        return loss.item()

    return closure


def create_loss_fn_pure(
    X, y, model, loss_fn, param_shapes
) -> Callable[[Tensor], Tensor]:
    """
    Returns a pure function that computes the model loss for a given (1D) input tensor.
    Unlike create_closure(), the returned function does not affect the model (notably,
    gradients are not populated).

    This can be used to create a closure to pass to optimizer.step() for optimisers
    that evaluate the object function multiple times, e.g. for a subroutine like line
    search.
    """

    def fn(inputs: Tensor) -> Tensor:
        """Pure function that computes the loss for a given input"""
        pred = ft.functional_call(model, unflatten(inputs, param_shapes), (X,))
        return loss_fn(pred, y)

    return fn


def create_loss_fn_with_vars_pure(
    X, y, model, loss_fn, ps_loss_fn, params, param_shapes
) -> Callable[[Tensor, bool], Tensor | tuple[float, Tensor, float, Tensor]]:
    """
    Returns a pure function that computes the model loss for a given (1D) input tensor.
    Unlike create_closure(), the returned function does not affect the model (notably,
    gradients are not populated).

    The pure function takes a second parameter `return_vars`. If True, the function
    will additionally return the gradient loss (d-dimensional), the variance of the
    loss (scalar), and the variance of the gradient (d-dimensional).

    This can be used to create a closure to pass to optimizer.step() for optimisers
    that use probabilistic line search (which needs the variances).
    """

    def compute_loss_one_sample(params, X_one, y_one):
        """Pure function to compute loss for a single sample"""
        # Model expects batch dimension
        X_as_batch = X_one.unsqueeze(0)
        y_as_batch = y_one.unsqueeze(0)

        pred = ft.functional_call(model, params, (X_as_batch,))
        loss = loss_fn(pred, y_as_batch)
        return loss

    def fn(
        inputs: Tensor, return_vars: bool
    ) -> Tensor | tuple[float, Tensor, float, Tensor]:
        inputs = unflatten(inputs, param_shapes)
        pred = ft.functional_call(model, inputs, (X,))

        if not return_vars:
            # Only need to compute loss
            loss = loss_fn(pred, y)
            return loss

        # Per-sample loss
        ps_loss = ps_loss_fn(pred, y)  # [b, 1], where b is batch size
        loss = ps_loss.mean().item()

        # Per-sample grad
        # Map over second (X_one) and third (y_one) dims, and not over first (params)
        ps_grad_fn = ft.vmap(ft.grad(compute_loss_one_sample), in_dims=(None, 0, 0))
        ps_grad = ps_grad_fn(params, X, y)

        flat_ps_grad = [g.reshape(g.shape[0], -1) for g in ps_grad.values()]
        grad_matrix = torch.cat(flat_ps_grad, dim=1)  # [b, d]
        grad_mean = grad_matrix.mean(dim=0)  # [d]

        # Variance of loss and grad, computed from samples
        loss_var = ps_loss.var(unbiased=True).item()
        grad_var = grad_matrix.var(dim=0, unbiased=True)

        return loss, grad_mean, loss_var, grad_var

    return fn
