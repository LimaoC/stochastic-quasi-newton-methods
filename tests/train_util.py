import itertools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.func as ft
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torcheval.metrics import Mean
from torcheval.metrics.metric import Metric

from sqnm.utils.param import unflatten

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


################################################################################
# Logging
################################################################################


@contextmanager
def timing_context(name: str):
    start = time.time()
    logger.info(f"Training with {name}...")
    try:
        yield
    finally:
        logger.info(f"\tTook {time.time() - start:.3f} seconds")


def log_training_info(
    epoch: int, epoch_loss: float, test_loss: float, test_metrics: dict[str, Metric]
) -> None:
    msg = (
        f"\tEpoch {epoch:4}, epoch_loss = {epoch_loss:8.4f}, "
        f"test_loss = {test_loss:8.4f}"
    )
    for name, metric_vals in test_metrics.items():
        msg += f", {name} = {metric_vals[-1]:.4f}"
    logger.info(msg)


################################################################################
# Metrics
################################################################################


def compute_loss(dataset: Dataset, model: nn.Module, loss_fn, device, batch_size=1000):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss = Mean()  # keep metric on cpu
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss.update(loss_fn(model(X_batch), y_batch).cpu())
    return loss.compute().item()


def compute_metric(
    dataset: Dataset, model: nn.Module, metric: Metric, device, batch_size=1000
):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            metric.update(model(X_batch).cpu().squeeze(), y_batch.squeeze())
    return metric.compute().item()


################################################################################
# Objects
################################################################################


def save_lowest_test_loss_runs(outs, optimizer_name, step_size, output_dir, n=3):
    sorted_outs = sorted(outs, key=lambda out: out["min_test_loss"])
    for i, out in enumerate(sorted_outs[:n]):
        loss, params = out["min_test_loss"], out["params"]
        save = f"{optimizer_name}-{step_size}-{i+1}.pt"
        logger.info(f"Saving min. test loss {loss:.4f} run, params {params} ({save})")
        torch.save(out, output_dir + save)


################################################################################
# Plots
################################################################################


def create_losses_plot(
    ax,
    outs: list[dict[str, Any]],
    out_names: list[str],
    logy=True,
    skip_first=0,
):
    num_epochs = len(outs[0]["epoch_losses"])
    epochs = np.arange(skip_first, num_epochs)
    plots = []
    for out, out_name in zip(outs, out_names):
        (p,) = ax.plot(epochs, out["epoch_losses"][skip_first:], label=out_name)
        plots.append(p)

    log_frequency = outs[0]["log_frequency"]
    epochs = np.arange(0, num_epochs - 1, log_frequency) + log_frequency
    cs = [p.get_color() for p in plots]
    for i, out in enumerate(outs):
        ax.plot(epochs, out["test_losses"], color=cs[i], marker="x", alpha=0.3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if logy:
        ax.set_yscale("log")
    ax.legend()


################################################################################
# Training
################################################################################


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


def num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_grid(param_dict):
    """
    scikit-learn-like parameter grid
    """
    items = sorted(param_dict.items())  # consistent ordering
    keys, values = zip(*items)
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


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
