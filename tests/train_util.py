import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.func as ft
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from sqnm.utils.param import unflatten

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class OverlappingBatchSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        overlap_size: int,
        shuffle: bool = True,
        generator: torch.Generator = None,
    ):
        self.data_source = data_source
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.overlap_size = overlap_size
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            indices = torch.randperm(self.data_size, generator=self.generator)
        else:
            indices = torch.arange(self.data_size)
        indices = indices.tolist()

        start = 0
        while start + self.batch_size <= self.data_size:
            yield indices[start : start + self.batch_size]
            start += self.batch_size - self.overlap_size

        if start < self.data_size:
            yield indices[start:]

    def __len__(self) -> int:
        return self.batch_size


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


def compute_loss(dataset: Dataset, model, loss_fn, batch_size=1000):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    num_batches = len(dataloader)
    loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            loss += loss_fn(model(X_batch), y_batch)
    return loss / num_batches


def log_training_info(epoch, epoch_loss, test_loss) -> None:
    logger.info(
        f"\tEpoch {epoch:4}, epoch loss = {epoch_loss:.4f}, test loss = {test_loss:.4f}"
    )


def num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def create_losses_plot(
    outs: list[dict[str, Any]],
    out_names: list[str],
    log_frequency,
    logy=True,
    skip_first=10,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    num_epochs = len(outs[0]["epoch_losses"])
    epochs = np.arange(skip_first, num_epochs)
    plots = []
    for out, out_name in zip(outs, out_names):
        (p,) = ax.plot(epochs, out["epoch_losses"][skip_first:], label=out_name)
        plots.append(p)

    epochs = np.arange(0, num_epochs, log_frequency) + log_frequency
    cs = [p.get_color() for p in plots]
    for i, out in enumerate(outs):
        ax.plot(epochs, out["test_losses"], color=cs[i], marker="x", alpha=0.3)

    ax.set_xlabel("Epoch")
    if logy:
        ax.set_ylabel("Log loss")
        ax.set_yscale("log")
    else:
        ax.set_ylabel("Loss")
    ax.legend()
    return fig, ax
