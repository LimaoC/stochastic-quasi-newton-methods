import logging
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.metric import Metric

from sqnm.optim.olbfgs import OLBFGS

from .train_util import (
    compute_loss,
    compute_metric,
    create_closure,
    create_loss_fn_pure,
    log_training_info,
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    optimizer: OLBFGS,
    scheduler: LRScheduler | None,
    model: nn.Module,
    loss_fn,
    device,
    metrics: dict[str, Metric] = dict(),
    generator: torch.Generator | None = None,
    num_epochs=1000,
    log_frequency=100,
    batch_size=100,
) -> dict[str, Any]:
    param_shapes = {name: param.shape for name, param in model.named_parameters()}
    line_search_fn = optimizer.param_groups[0]["line_search_fn"]

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    num_batches = len(dataloader)

    epoch_losses = []
    test_losses = []  # computed every log_frequency epochs
    test_metrics = defaultdict(list)  # problem-specific metrics
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)

            # Optimiser expects loss function handle if using line search
            if line_search_fn == "strong_wolfe":
                fn = create_loss_fn_pure(X_batch, y_batch, model, loss_fn, param_shapes)
            else:
                fn = None

            epoch_loss += optimizer.step(closure, fn)

        if scheduler is not None:
            scheduler.step()

        # Aggregate loss and grad norm across all batches in this epoch
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)

        if math.isnan(epoch_loss):
            logger.info(f"\tLoss hit NaN at epoch {epoch+1}, stopping early")
            break

        if epoch % log_frequency == log_frequency - 1:
            test_loss = compute_loss(test_dataset, model, loss_fn)
            test_losses.append(test_loss)
            for name, metric in metrics.items():
                metric_val = compute_metric(test_dataset, model, metric)
                test_metrics[name].append(metric_val)
            log_training_info(epoch + 1, epoch_loss, test_loss, test_metrics)

    out = {
        "epoch_losses": epoch_losses,
        "test_losses": test_losses,
        "min_test_loss": float("inf") if not test_losses else min(test_losses),
        "log_frequency": log_frequency,
    }
    for name, metric in test_metrics.items():
        out[name] = test_metrics[name]
    return out


# def train_with_prob_ls(
#     train_dataset: TensorDataset,
#     test_dataset: TensorDataset,
#     optimizer: OLBFGS,
#     model,
#     loss_fn,
#     ps_loss_fn,
#     device,
#     num_epochs=1000,
#     batch_size=100,
#     log_frequency=100,
# ) -> dict[str, Any]:
#     params = {name: param.detach() for name, param in model.named_parameters()}
#     param_shapes = {name: param.shape for name, param in model.named_parameters()}
#
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     num_batches = len(train_dataloader)
#     # X_test, y_test = test_dataset[:]
#
#     losses = []
#     grad_norms = []
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         epoch_grad_norm = 0.0
#
#         for X_batch, y_batch in train_dataloader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             closure = create_closure(X_batch, y_batch, optimizer, model, loss_fn)
#             fn = create_loss_fn_with_vars_pure(
#                 X_batch, y_batch, model, loss_fn, ps_loss_fn, params, param_shapes
#             )
#             epoch_loss += optimizer.step(closure, fn)
#             epoch_grad_norm += grad_vec(model.parameters()).norm()
#
#         # Aggregate loss and grad norm across all batches in this epoch
#         epoch_loss /= num_batches
#         epoch_grad_norm /= num_batches
#         losses.append(epoch_loss)
#         grad_norms.append(epoch_grad_norm)
#
#         if epoch % log_frequency == log_frequency - 1:
#             with torch.no_grad():
#                 training_loss = compute_loss(train_dataset, model, loss_fn)
#             log_training_info(epoch + 1, training_loss, epoch_grad_norm)
#
#     return {"losses": losses, "grad_norms": grad_norms}
